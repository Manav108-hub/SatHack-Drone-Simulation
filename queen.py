# queen.py
# queen.py
import os
import time
import traceback
import logging
from logging.handlers import RotatingFileHandler
import sys  # ← ADD THIS LINE

import numpy as np
import airsim

from swarm_state import swarm

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

# ✅ UTF-8 FIX - REPLACE THESE 3 LINES:
ch = logging.StreamHandler(sys.stdout)
if hasattr(ch.stream, 'reconfigure'):
    ch.stream.reconfigure(encoding='utf-8')
ch.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] %(message)s", "%H:%M:%S"))
logger.addHandler(ch)


class Queen:
    def __init__(self):
        self.client = None
        self.model = None
        self.threat_classes = {0: 'person', 2: 'car', 5: 'bus', 7: 'truck'}
        self.ai_scan_count = 0
        self.last_threat_time = 0
        self.model_loaded = False

        try:
            logger.info("Queen: Connecting to AirSim...")
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            logger.info("Queen: Connected to AirSim")
            swarm.log("QUEEN", "Connected to AirSim", "INFO")
        except Exception as e:
            logger.exception("Queen: Cannot connect to AirSim")
            swarm.log("QUEEN", "QUEEN ERROR: Cannot connect to AirSim!", "WARNING")
            raise Exception("AirSim not running - Start simulation first!") from e

        swarm.log("QUEEN", "Initializing - COMMAND CENTER MODE", "INFO")

    def load_model(self):
        if self.model_loaded:
            return

        swarm.log("QUEEN", "Initializing AI model (may take a moment)...", "INFO")
        try:
            from ultralytics import YOLO
            self.model = YOLO('yolov8n.pt')
            self.model_loaded = True
            swarm.log("QUEEN", "AI model ready", "INFO")
            logger.info("YOLO model loaded")
        except Exception as e:
            self.model = None
            self.model_loaded = False
            swarm.log("QUEEN", f"Model load failed: {e}", "WARNING")
            logger.exception("Failed to load YOLO model")

    def get_warrior_camera(self):
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
        self.ai_scan_count += 1

        # More frequent status updates
        if self.ai_scan_count % 20 == 0:
            swarm.log("QUEEN", f"AI Monitoring Active... Scan #{self.ai_scan_count}", "INFO")

        img, img_w, img_h = self.get_warrior_camera()
        if img is None:
            return None

        if not self.model_loaded:
            self.load_model()
            if not self.model_loaded:
                return None

        try:
            results = self.model(img, verbose=False, conf=0.35)  # Lower confidence for more detections

            detected = []
            for box in results[0].boxes:
                class_id = int(box.cls)
                name = self.model.names[class_id]
                conf = float(box.conf)
                detected.append(f"{name}:{int(conf*100)}%")

            # Log ALL detections more frequently
            if detected and self.ai_scan_count % 10 == 0:
                swarm.log("QUEEN", f"Warrior sees: {', '.join(detected[:6])}", "INFO")

            # check threat classes
            for box in results[0].boxes:
                class_id = int(box.cls)
                conf = float(box.conf)
                
                # Check if it's a threat class
                if class_id in self.threat_classes and conf > 0.4:  # Lower threshold
                    # Reduced cooldown for testing
                    if time.time() - self.last_threat_time < 3:
                        continue

                    try:
                        px, py, _, _ = box.xywh[0]
                    except:
                        px = img_w / 2
                        py = img_h / 2

                    warrior_pose = self.client.simGetVehiclePose("Warrior1").position
                    wx, wy = warrior_pose.x_val, warrior_pose.y_val

                    scale = max(img_w / 50.0, 10.0)
                    dx_m = (px - img_w/2) / scale
                    dy_m = (py - img_h/2) / scale

                    world_x = wx + dx_m
                    world_y = wy + dy_m

                    threat = {
                        'class': self.threat_classes[class_id],
                        'confidence': conf,
                        'world_pos': (world_x, world_y),
                        'timestamp': time.time()
                    }

                    self.last_threat_time = time.time()
                    swarm.log("QUEEN",
                              f"⚠️ THREAT DETECTED: {threat['class']} {int(conf*100)}% at ({world_x:.1f}, {world_y:.1f})",
                              "CRITICAL")
                    return threat

        except Exception as e:
            if self.ai_scan_count % 50 == 0:
                logger.warning(f"Detection error: {e}")
            return None

    # -----------------------------------------------------
    # MAIN LOOP
    # -----------------------------------------------------
    def run(self):
        swarm.log("QUEEN", "Starting", "INFO")

        try:
            self.client.enableApiControl(True, "Queen")
            self.client.armDisarm(True, "Queen")
            future = self.client.takeoffAsync(vehicle_name="Queen")
            future.join()
        except:
            pass

        swarm.log("QUEEN", "Airborne - Command Center", "INFO")
        
        try:
            future = self.client.moveToPositionAsync(0, 0, -20, 5, vehicle_name="Queen")
            future.join()
        except:
            pass

        swarm.log("QUEEN", "👁️ Monitoring Warrior camera feed", "WARNING")
        
        # Slow rotation
        try:
            self.client.rotateByYawRateAsync(6, 60, vehicle_name="Queen")
            swarm.log("QUEEN", "Rotating slowly for surveillance", "INFO")
        except Exception as e:
            logger.warning(f"Rotation failed: {e}")
        
        swarm.threat_level = "YELLOW"

        scan = 0

        while not swarm.kamikaze_deployed:
            scan += 1
            swarm.queen_scans = scan

            # Broadcast Queen position
            try:
                q = self.client.simGetVehiclePose("Queen").position
                swarm.last_queen_pos = (q.x_val, q.y_val, q.z_val)
            except:
                pass

            if scan % 30 == 0:  # More frequent updates
                swarm.log("QUEEN", f"Command Center Active: Scan #{scan}", "INFO")

            # Manual threats
            if swarm.active_threat and not swarm.kamikaze_deployed:
                t = swarm.active_threat
                swarm.log("QUEEN", f"🎯 MANUAL THREAT: {t['class']}", "CRITICAL")

                approved = swarm.request_permission()
                if approved:
                    swarm.kamikaze_target = t['world_pos']
                    swarm.kamikaze_deployed = True
                    swarm.log("QUEEN", "✅ STRIKE AUTHORIZED", "CRITICAL")
                    break
                else:
                    with swarm.lock:
                        swarm.threats.clear()
                        swarm.active_threat = None
                    swarm.threat_level = "YELLOW"
                    swarm.log("QUEEN", "❌ STRIKE DENIED", "WARNING")

            # AI threats - scan faster
            else:
                ai_t = self.detect_threats_from_warrior()
                if ai_t:
                    swarm.add_threat(ai_t)
                    swarm.log("QUEEN", f"🚨 AI THREAT ADDED TO QUEUE", "CRITICAL")
                    
                    approved = swarm.request_permission()
                    if approved:
                        swarm.kamikaze_target = ai_t['world_pos']
                        swarm.kamikaze_deployed = True
                        swarm.log("QUEEN", "✅ AI STRIKE AUTHORIZED", "CRITICAL")
                        break
                    else:
                        with swarm.lock:
                            swarm.threats.clear()
                            swarm.active_threat = None
                        swarm.threat_level = "YELLOW"
                        swarm.log("QUEEN", "❌ AI STRIKE DENIED", "WARNING")

            time.sleep(0.5)  # Faster scanning

        swarm.log("QUEEN", "Mission complete", "INFO")
        try:
            future = self.client.hoverAsync(vehicle_name="Queen")
            future.join()
        except:
            pass


def run():
    Queen().run()