# queen.py
"""
Queen Drone - Command Center with AI Learning
Monitors warrior camera feeds, detects threats using YOLOv8,
and learns from operator decisions for improved threat assessment.
"""

import os
import time
import traceback
import logging
from logging.handlers import RotatingFileHandler

import numpy as np
import airsim

from swarm_state import swarm
from adaptive_learner import AdaptiveThreatLearner

# ----------------------------
# Logging for Queen
# ----------------------------
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("QUEEN")
logger.setLevel(logging.DEBUG)
fh = RotatingFileHandler(os.path.join(LOG_DIR, "queen.log"), maxBytes=2_000_000, backupCount=3)
fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"))
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] %(message)s", "%H:%M:%S"))
logger.addHandler(ch)
logger.propagate = False


class Queen:
    """
    Queen Drone - AI Command Center
    
    Responsibilities:
    - Monitor warrior camera feeds
    - Run YOLOv8 threat detection
    - Learn from operator decisions
    - Make autonomous decisions in jammer mode
    - Coordinate kamikaze strikes
    """
    
    def __init__(self):
        self.client = None
        self.model = None
        self.threat_classes = {
            0: 'person', 
            2: 'car', 
            5: 'bus', 
            7: 'truck',
            3: 'motorcycle'
        }
        self.ai_scan_count = 0
        self.last_threat_time = 0
        self.model_loaded = False

        # Initialize adaptive threat learner
        logger.info("Initializing AI Learning System...")
        try:
            self.learner = AdaptiveThreatLearner()
            learner_stats = self.learner.get_stats()
            logger.info("AI Learning System initialized successfully")
            logger.info(f"  Classifier available: {self.learner.classifier is not None}")
            logger.info(f"  Model trained: {self.learner.is_trained}")
            logger.info(f"  Experience buffer: {len(self.learner.experience_buffer)} examples")
        except Exception as e:
            logger.error(f"Failed to initialize AI learner: {e}")
            logger.error("Continuing without AI learning...")
            self.learner = None
        
        # Connect to AirSim
        try:
            logger.info("Queen: Connecting to AirSim...")
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            logger.info("Queen: Connected to AirSim")
            swarm.log("QUEEN", "Connected to AirSim", "INFO")
        except Exception as e:
            logger.exception("Queen: Cannot connect to AirSim")
            swarm.log("QUEEN", "ERROR: Cannot connect to AirSim!", "WARNING")
            raise Exception("AirSim not running - Start simulation first!") from e

        if self.learner:
            swarm.log("QUEEN", "AI LEARNING MODE - Command Center Active", "INFO")
        else:
            swarm.log("QUEEN", "BASIC MODE - Command Center Active (AI learning unavailable)", "WARNING")

    def load_model(self):
        """Load YOLOv8 model (lazy loading on first detection)"""
        if self.model_loaded:
            return

        swarm.log("QUEEN", "Initializing YOLOv8 model...", "INFO")
        try:
            from ultralytics import YOLO
            self.model = YOLO('yolov8n.pt')
            self.model_loaded = True
            swarm.log("QUEEN", "YOLOv8 model ready", "INFO")
            logger.info("YOLOv8 model loaded successfully")
        except Exception as e:
            self.model = None
            self.model_loaded = False
            swarm.log("QUEEN", f"Model load failed: {e}", "WARNING")
            logger.exception("Failed to load YOLO model")

    def get_warrior_camera(self):
        """
        Fetch camera feed from Warrior1 drone.
        
        Returns:
            tuple: (image_array, width, height) or (None, None, None) on failure
        """
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ], vehicle_name="Warrior1")
            
            if not responses:
                return None, None, None
                
            r = responses[0]
            if not r or len(r.image_data_uint8) == 0:
                return None, None, None
                
            img1d = np.frombuffer(r.image_data_uint8, dtype=np.uint8)
            img = img1d.reshape(r.height, r.width, 3)
            
            return img, r.width, r.height
            
        except Exception as e:
            if self.ai_scan_count % 50 == 0:
                swarm.log("QUEEN", f"Warrior camera error: {e}", "WARNING")
            return None, None, None

    def detect_threats_from_warrior(self):
        """
        Main threat detection function.
        
        Process:
        1. Get warrior camera feed
        2. Run YOLOv8 detection
        3. Filter for threat classes
        4. Use AI learner to assess threat level (if available)
        5. Return high-confidence threats
        
        Returns:
            dict: Threat data or None if no threat detected
        """
        self.ai_scan_count += 1

        # Periodic status logging
        if self.ai_scan_count % 30 == 0:
            if self.learner:
                stats = self.learner.get_stats()
                swarm.log("QUEEN", 
                         f"Scan #{self.ai_scan_count} | "
                         f"AI: {stats['total_detections']} experiences, "
                         f"{stats['confirmed_threats']} confirmed", 
                         "INFO")
            else:
                swarm.log("QUEEN", f"Scan #{self.ai_scan_count}", "INFO")

        # Get camera feed
        img, img_w, img_h = self.get_warrior_camera()
        if img is None:
            return None

        # Load YOLO model if not loaded
        if not self.model_loaded:
            self.load_model()
            if not self.model_loaded:
                return None

        try:
            # Run YOLOv8 detection
            results = self.model(img, verbose=False, conf=0.45)

            # Log all detections periodically
            detected = []
            for box in results[0].boxes:
                class_id = int(box.cls)
                name = self.model.names[class_id]
                conf = float(box.conf)
                detected.append(f"{name}:{int(conf*100)}%")

            if detected and self.ai_scan_count % 20 == 0:
                swarm.log("QUEEN", f"Warrior sees: {', '.join(detected[:4])}", "INFO")

            # Check for threat classes
            for box in results[0].boxes:
                class_id = int(box.cls)
                conf = float(box.conf)
                
                # Only process threat classes
                if class_id not in self.threat_classes:
                    continue
                    
                # Minimum YOLO confidence threshold
                if conf <= 0.5:
                    continue
                
                # Cooldown to prevent spam
                if time.time() - self.last_threat_time < 8:
                    continue

                # Extract bounding box info
                try:
                    xywh = box.xywh[0]
                    px = float(xywh[0])
                    py = float(xywh[1])
                    bbox_w = float(xywh[2])
                    bbox_h = float(xywh[3])
                    bbox_area = bbox_w * bbox_h
                except:
                    px = img_w / 2
                    py = img_h / 2
                    bbox_area = 1000.0

                # Get warrior position for world coordinates
                try:
                    warrior_pose = self.client.simGetVehiclePose("Warrior1").position
                    wx, wy = warrior_pose.x_val, warrior_pose.y_val
                except:
                    wx, wy = 0, 0

                # Convert pixel coordinates to world coordinates (rough approximation)
                scale = max(img_w / 50.0, 10.0)
                dx_m = (px - img_w/2) / scale
                dy_m = (py - img_h/2) / scale

                world_x = wx + dx_m
                world_y = wy + dy_m

                # Build threat data structure
                threat = {
                    'class': self.threat_classes[class_id],
                    'confidence': conf,
                    'world_pos': (world_x, world_y),
                    'bbox_area': bbox_area,
                    'timestamp': time.time()
                }

                # Use AI learner to assess threat (if available)
                if self.learner:
                    is_real_threat, ai_confidence = self.learner.predict_threat_level(threat)
                    
                    if is_real_threat:
                        self.last_threat_time = time.time()
                        
                        swarm.log("QUEEN",
                                 f"AI CONFIRMED THREAT: {threat['class']} "
                                 f"(YOLO: {int(conf*100)}%, AI: {int(ai_confidence*100)}%) "
                                 f"at ({world_x:.1f}, {world_y:.1f})",
                                 "CRITICAL")
                        
                        # Update swarm stats
                        swarm.update_learning_stats(confirmed=False, auto_mode=False, confidence=ai_confidence)
                        
                        return threat
                    else:
                        # Low threat - log occasionally
                        if self.ai_scan_count % 50 == 0:
                            swarm.log("QUEEN", 
                                     f"Low threat: {threat['class']} "
                                     f"(AI: {int(ai_confidence*100)}% confidence)", 
                                     "INFO")
                else:
                    # No AI learner - use basic confidence check
                    if conf > 0.7:
                        self.last_threat_time = time.time()
                        swarm.log("QUEEN",
                                 f"THREAT DETECTED: {threat['class']} "
                                 f"({int(conf*100)}%) at ({world_x:.1f}, {world_y:.1f})",
                                 "CRITICAL")
                        return threat

        except Exception as e:
            if self.ai_scan_count % 50 == 0:
                swarm.log("QUEEN", f"Detection error: {e}", "WARNING")
                logger.exception("Detection error")
            return None

    def handle_threat(self, threat):
        """
        Handle detected threat based on current mode.
        
        Args:
            threat: Dict containing threat information
            
        Returns:
            bool: True if strike authorized, False otherwise
        """
        # Check operating mode
        if swarm.queen_mode == "jammer":
            # AUTONOMOUS MODE - AI decides (if available)
            if self.learner:
                if self.learner.autonomous_decision(threat):
                    swarm.log("QUEEN", "AUTO-AUTHORIZED (AI Learning)", "CRITICAL")
                    
                    # Update stats for autonomous decision
                    _, ai_conf = self.learner.predict_threat_level(threat)
                    swarm.update_learning_stats(
                        confirmed=True, 
                        auto_mode=True,
                        confidence=ai_conf
                    )
                    
                    # Learn from autonomous decision
                    self.learner.learn_from_feedback(threat, user_confirmed_threat=True)
                    
                    return True
                else:
                    # AI decided not to strike
                    with swarm.lock:
                        swarm.threats.clear()
                        swarm.active_threat = None
                    swarm.threat_level = "YELLOW"
                    return False
            else:
                # No AI - auto-approve in jammer mode
                swarm.log("QUEEN", "AUTO-AUTHORIZED (Jammer Mode - No AI)", "CRITICAL")
                return True
        else:
            # NORMAL MODE - Request human authorization
            approved = swarm.request_permission()
            
            # LEARN from user decision (if learner available)
            if self.learner:
                self.learner.learn_from_feedback(threat, user_confirmed_threat=approved)
                
                # Update stats
                _, ai_conf = self.learner.predict_threat_level(threat)
                swarm.update_learning_stats(
                    confirmed=approved, 
                    auto_mode=False,
                    confidence=ai_conf
                )
            
            if approved:
                return True
            else:
                # User denied - clear threat
                with swarm.lock:
                    swarm.threats.clear()
                    swarm.active_threat = None
                swarm.threat_level = "YELLOW"
                return False
    def get_annotated_warrior_feed(self):
        """
        Return Warrior's camera with YOLO detections drawn on it.
        This is what Queen is "seeing" when analyzing threats.
        """
        import cv2
        
        img, img_w, img_h = self.get_warrior_camera()
        if img is None or not self.model_loaded:
            return None
        
        try:
            results = self.model(img, verbose=False, conf=0.45)
            annotated = img.copy()
            
            for box in results[0].boxes:
                class_id = int(box.cls)
                conf = float(box.conf)
                name = self.model.names[class_id]
                
                # Get bounding box
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Color coding: RED for threats, GREEN for safe objects
                is_threat = class_id in self.threat_classes
                color = (0, 0, 255) if is_threat else (0, 255, 0)
                thickness = 3 if is_threat else 2
                
                # Draw detection box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                
                # Label with confidence
                label = f"{name} {int(conf*100)}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1-th-10), (x1+tw, y1), color, -1)
                cv2.putText(annotated, label, (x1, y1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
            # Add AI status overlay
            status_text = f"AI Scan #{self.ai_scan_count}"
            cv2.putText(annotated, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            
            return annotated
            
        except Exception as e:
            logger.error(f"Annotation error: {e}")
            return None

    def run(self):
        """
        Main Queen drone loop.
        
        Process:
        1. Take off and position
        2. Monitor warrior feeds
        3. Detect threats
        4. Request authorization or decide autonomously
        5. Coordinate kamikaze strike
        """
        swarm.log("QUEEN", "Starting - AI Learning Active" if self.learner else "Starting", "INFO")

        # Log learner status
        if self.learner:
            stats = self.learner.get_stats()
            logger.info(f"AI Learner Status: trained={stats['is_trained']}, experiences={stats['total_detections']}")

        # Startup sequence
        try:
            self.client.enableApiControl(True, "Queen")
            self.client.armDisarm(True, "Queen")
            
            logger.info("Taking off...")
            future = self.client.takeoffAsync(vehicle_name="Queen")
            future.join()
            time.sleep(2)
            logger.info("Takeoff complete")
            
        except Exception as e:
            logger.warning(f"Takeoff error: {e}")

        swarm.log("QUEEN", "Airborne - Moving to command position", "INFO")
        
        # Move to command position
        try:
            logger.info("Moving to position (0, 0, -20)...")
            future = self.client.moveToPositionAsync(0, 0, -20, 5, vehicle_name="Queen")
            future.join()
            logger.info("Position reached")
            
        except Exception as e:
            logger.warning(f"Position move error: {e}")

        swarm.log("QUEEN", "MONITORING - AI Learning Active" if self.learner else "MONITORING", "WARNING")
        swarm.threat_level = "YELLOW"

        scan = 0

        # Main monitoring loop
        while not swarm.kamikaze_deployed:
            scan += 1
            swarm.queen_scans = scan

            # Update queen position for relative patrol
            try:
                q = self.client.simGetVehiclePose("Queen").position
                swarm.last_queen_pos = (q.x_val, q.y_val, q.z_val)
            except:
                pass

            # Periodic status update
            if scan % 40 == 0:
                if self.learner:
                    stats = self.learner.get_stats()
                    swarm.log("QUEEN", 
                             f"Command Center: Scan #{scan} | "
                             f"Learning: {stats['total_detections']} exp, "
                             f"{stats['confirmed_threats']} confirmed", 
                             "INFO")
                else:
                    swarm.log("QUEEN", f"Command Center: Scan #{scan}", "INFO")

            # Check for manual threats (spawned via UI)
            if swarm.active_threat and not swarm.kamikaze_deployed:
                t = swarm.active_threat
                swarm.log("QUEEN", f"MANUAL THREAT: {t['class']}", "CRITICAL")

                if self.handle_threat(t):
                    # Strike authorized
                    swarm.kamikaze_target = t['world_pos']
                    swarm.kamikaze_deployed = True
                    break

            # AI threat detection from warrior feed
            else:
                ai_threat = self.detect_threats_from_warrior()
                
                if ai_threat:
                    swarm.add_threat(ai_threat)
                    
                    if self.handle_threat(ai_threat):
                        # Strike authorized
                        swarm.kamikaze_target = ai_threat['world_pos']
                        swarm.kamikaze_deployed = True
                        break

            time.sleep(1)

        # Mission complete
        if self.learner:
            stats = self.learner.get_stats()
            swarm.log("QUEEN", 
                     f"Mission complete | AI Model: {stats['total_detections']} experiences", 
                     "INFO")
            
            # Save learned model
            self.learner.save_model()
        else:
            swarm.log("QUEEN", "Mission complete", "INFO")
        
        # Hover in place
        try:
            self.client.hoverAsync(vehicle_name="Queen").join()
        except:
            pass


def run():
    """Entry point for queen module"""
    try:
        Queen().run()
    except Exception as e:
        logger.exception("Queen run() failed")
        raise