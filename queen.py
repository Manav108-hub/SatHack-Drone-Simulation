# queen.py
import os
import time
import traceback
import logging
from logging.handlers import RotatingFileHandler

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
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] %(message)s", "%H:%M:%S"))
logger.addHandler(ch)


class Queen:
    def __init__(self):
        self.client = None
        self.model = None
        self.threat_classes = {0: 'person', 2: 'car', 5: 'bus', 7: 'truck'}
        self.ai_scan_count = 0
        self.last_threat_time = 0  # Prevent spamming same detection
        self.model_loaded = False

        try:
            logger.info("Queen: Connecting to AirSim...")
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            logger.info("Queen: Connected to AirSim")
            swarm.log("QUEEN", "Connected to AirSim", "INFO")
        except Exception as e:
            logger.exception("Queen: Cannot connect to AirSim")
            swarm.log("QUEEN", "❌ QUEEN ERROR: Cannot connect to AirSim!", "WARNING")
            raise Exception("AirSim not running - Start simulation first!") from e

        swarm.log("QUEEN", "Initializing - COMMAND CENTER MODE", "INFO")

    def load_model(self):
        """Load YOLO model once and reuse it."""
        if self.model_loaded:
            return

        swarm.log("QUEEN", "📥 Initializing AI model (may take a moment)...", "INFO")
        try:
            from ultralytics import YOLO
            # Using yolov8n for speed — change to a different weights file if required
            self.model = YOLO('yolov8n.pt')
            self.model_loaded = True
            swarm.log("QUEEN", "✅ AI model ready", "INFO")
            logger.info("YOLO model loaded")
        except Exception as e:
            self.model = None
            self.model_loaded = False
            swarm.log("QUEEN", f"❌ Model load failed: {e}", "WARNING")
            logger.exception("Failed to load YOLO model")

    def get_warrior_camera(self):
        """Return BGR image from Warrior's camera or None."""
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
            # Log less frequently to avoid spam
            if self.ai_scan_count % 50 == 0:
                swarm.log("QUEEN", f"Warrior camera error: {e}", "WARNING")
                logger.warning("Warrior camera fetch error", exc_info=True)
            return None, None, None

    def detect_threats_from_warrior(self):
        """Analyze Warrior's camera feed for threats and return a threat dict or None."""
        self.ai_scan_count += 1

        # Low-rate informational log
        if self.ai_scan_count % 30 == 0:
            swarm.log("QUEEN", f"📡 Monitoring Warrior feed... Scan #{self.ai_scan_count}", "INFO")

        img, img_w, img_h = self.get_warrior_camera()
        if img is None:
            return None

        # Ensure model is loaded once
        if not self.model_loaded:
            self.load_model()
            if not self.model_loaded:
                return None  # cannot run detection

        try:
            # Run model (ultralytics accepts numpy images)
            results = self.model(img, verbose=False, conf=0.45)

            # Build a readable list of detected objects (for logging)
            detected_objects = []
            for box in results[0].boxes:
                class_id = int(box.cls)
                name = self.model.names[class_id] if hasattr(self.model, "names") else str(class_id)
                conf = float(box.conf)
                detected_objects.append(f"{name}:{int(conf*100)}%")

            if detected_objects and self.ai_scan_count % 20 == 0:
                swarm.log("QUEEN", f"👁️ Warrior sees: {', '.join(detected_objects[:4])}", "INFO")

            # Check for threat classes
            for box in results[0].boxes:
                class_id = int(box.cls)
                conf = float(box.conf)
                if class_id in self.threat_classes and conf > 0.5:
                    # Avoid spamming the same detection too often
                    if time.time() - self.last_threat_time < 8:
                        continue

                    # xywh: centerx, centery, w, h (model.provide)
                    # Use box.xywh[0] if available
                    try:
                        xywh = box.xywh[0].tolist()
                        px = float(xywh[0])
                        py = float(xywh[1])
                    except Exception:
                        # fallback: use image center
                        px = img_w / 2
                        py = img_h / 2

                    # Get warrior world pose and approximate world coordinates of detection
                    warrior_pose = self.client.simGetVehiclePose("Warrior1").position
                    wx, wy, wz = warrior_pose.x_val, warrior_pose.y_val, warrior_pose.z_val

                    # Map pixel offset to meters using a heuristic scale
                    # scale = how many pixels correspond to 1 meter at current altitude — heuristic
                    # We set scale proportional to image width and a constant factor (tweak if needed)
                    scale = max(img_w / 50.0, 10.0)  # px per meter approx; larger => smaller meter offset
                    # compute offset from image center in meters
                    dx_m = (px - (img_w / 2.0)) / scale
                    dy_m = (py - (img_h / 2.0)) / scale
                    # convert to world coordinates (assuming simple camera facing forward - tune as needed)
                    world_x = wx + dx_m
                    world_y = wy + dy_m

                    threat = {
                        'class': self.threat_classes[class_id],
                        'confidence': conf,
                        'world_pos': (world_x, world_y),
                        'timestamp': time.time()
                    }

                    self.last_threat_time = time.time()
                    swarm.log("QUEEN", f"🚨 WARRIOR SPOTTED: {threat['class']} {int(conf*100)}% at ({world_x:.1f}, {world_y:.1f})", "CRITICAL")
                    logger.info(f"Detected {threat['class']} conf={conf:.2f} approx_world=({world_x:.2f},{world_y:.2f})")
                    return threat

        except Exception as e:
            # occasionally log detection errors
            if self.ai_scan_count % 50 == 0:
                swarm.log("QUEEN", f"Detection error: {e}", "WARNING")
                logger.exception("Detection error")
        return None

    def run(self):
        swarm.log("QUEEN", "Starting", "INFO")
        logger.info("Queen run() starting")

        try:
            self.client.enableApiControl(True, "Queen")
            self.client.armDisarm(True, "Queen")
            self.client.takeoffAsync(vehicle_name="Queen").join()
        except Exception as e:
            logger.warning("Takeoff/enable control error (Queen)", exc_info=True)

        swarm.log("QUEEN", "✈️ Airborne - Command Center", "INFO")
        try:
            # position queen above the arena for an overview
            self.client.moveToPositionAsync(0, 0, -20, 5, vehicle_name="Queen").join()
        except Exception:
            pass

        swarm.log("QUEEN", "📡 Monitoring Warrior camera feed", "WARNING")
        swarm.threat_level = "YELLOW"

        scan = 0
        try:
            while not swarm.kamikaze_deployed:
                scan += 1
                swarm.queen_scans = scan

                # periodic logging
                if scan % 40 == 0:
                    swarm.log("QUEEN", f"Command Center: Scan #{scan}", "INFO")

                # PRIORITY 1: Manual / UI threats
                if swarm.active_threat and not swarm.kamikaze_deployed:
                    t = swarm.active_threat
                    swarm.log("QUEEN", f"📍 MANUAL THREAT: {t['class']}", "CRITICAL")
                    # Request permission (this will wait and auto-authorize on timeout)
                    approved = swarm.request_permission()
                    if approved:
                        swarm.log("QUEEN", "✅ Strike authorized", "CRITICAL")
                        swarm.kamikaze_target = t['world_pos']
                        swarm.kamikaze_deployed = True
                        break
                    else:
                        swarm.log("QUEEN", "❌ Strike denied", "WARNING")
                        with swarm.lock:
                            swarm.threats.clear()
                            swarm.active_threat = None
                        swarm.threat_level = "YELLOW"

                # PRIORITY 2: AI detection from Warrior's camera
                else:
                    ai_threat = self.detect_threats_from_warrior()
                    if ai_threat:
                        # Add threat into swarm and request permission
                        swarm.add_threat(ai_threat)
                        approved = swarm.request_permission()
                        if approved:
                            swarm.log("QUEEN", "✅ AI Strike authorized", "CRITICAL")
                            swarm.kamikaze_target = ai_threat['world_pos']
                            swarm.kamikaze_deployed = True
                            break
                        else:
                            swarm.log("QUEEN", "❌ AI Strike denied", "WARNING")
                            with swarm.lock:
                                swarm.threats.clear()
                                swarm.active_threat = None
                            swarm.threat_level = "YELLOW"

                time.sleep(1)

        except Exception as e:
            logger.exception("Queen main loop error")
            swarm.log("QUEEN", f"Fatal error: {e}", "CRITICAL")

        swarm.log("QUEEN", "✅ Mission complete", "INFO")
        try:
            self.client.hoverAsync(vehicle_name="Queen").join()
        except Exception:
            pass


def run():
    q = Queen()
    q.run()
