import airsim
import numpy as np
import time
from swarm_state import swarm

class Queen:
    def __init__(self):
        try:
            print("Queen: Connecting to AirSim...")
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            print("Queen: Connected!")
            swarm.log("QUEEN", "Connected to AirSim", "INFO")
        except Exception as e:
            print(f"❌ QUEEN ERROR: Cannot connect to AirSim!")
            print(f"   Make sure simulation is running first!")
            print(f"   Error: {e}")
            raise Exception("AirSim not running - Start simulation first!")
        
        swarm.log("QUEEN", "Initializing - COMMAND CENTER MODE", "INFO")
        self.model = None
        self.threat_classes = {0: 'person', 2: 'car', 5: 'bus', 7: 'truck'}
        self.ai_scan_count = 0
        self.last_threat_time = 0  # Prevent spam
        
    def load_model(self):
        """Load YOLO model - downloads if needed"""
        if self.model is None:
            swarm.log("QUEEN", "📥 Downloading AI model...", "WARNING")
            try:
                from ultralytics import YOLO
                self.model = YOLO('yolov8n.pt')
                swarm.log("QUEEN", "✅ AI model ready", "INFO")
            except Exception as e:
                swarm.log("QUEEN", f"❌ Model load failed: {e}", "WARNING")
                self.model = "failed"
        
    def get_warrior_camera(self):
        """Monitor WARRIOR's camera feed"""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ], "Warrior1")  # ← WATCHING WARRIOR!
            
            if responses and len(responses[0].image_data_uint8) > 0:
                img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                img = img1d.reshape(responses[0].height, responses[0].width, 3)
                return img
        except Exception as e:
            if self.ai_scan_count % 50 == 0:  # Log errors less frequently
                swarm.log("QUEEN", f"Warrior camera error: {e}", "WARNING")
        return None
    
    def detect_threats_from_warrior(self):
        """Analyze Warrior's camera feed for threats"""
        self.ai_scan_count += 1
        
        # Log every 30 scans (less spam)
        if self.ai_scan_count % 30 == 0:
            swarm.log("QUEEN", f"📡 Monitoring Warrior feed... Scan #{self.ai_scan_count}", "INFO")
        
        img = self.get_warrior_camera()
        if img is None:
            return None
        
        self.load_model()
        
        if self.model == "failed" or self.model is None:
            return None
        
        try:
            results = self.model(img, verbose=False, conf=0.5)  # 50% confidence minimum
            
            detected_objects = []
            for box in results[0].boxes:
                class_id = int(box.cls)
                class_name = self.model.names[class_id]
                conf = float(box.conf)
                detected_objects.append(f"{class_name}:{conf:.0%}")
            
            # Log what Warrior sees (every 20 scans)
            if detected_objects and self.ai_scan_count % 20 == 0:
                swarm.log("QUEEN", f"👁️ Warrior sees: {', '.join(detected_objects[:3])}", "INFO")
            
            # Check for HIGH-CONFIDENCE threats only
            for box in results[0].boxes:
                class_id = int(box.cls)
                if class_id in self.threat_classes:
                    conf = float(box.conf)
                    
                    # Only report if confidence > 50% AND not reported recently
                    if conf > 0.5 and (time.time() - self.last_threat_time) > 10:
                        x, y, w, h = box.xywh[0].tolist()
                        
                        # Get Warrior's position (where threat is)
                        warrior_pos = self.client.simGetVehiclePose("Warrior1").position
                        world_x = warrior_pos.x_val + (x - 640) / 30
                        world_y = warrior_pos.y_val + (y - 360) / 30
                        
                        threat = {
                            'class': self.threat_classes[class_id],
                            'confidence': conf,
                            'world_pos': (world_x, world_y),
                            'timestamp': time.time()
                        }
                        
                        self.last_threat_time = time.time()
                        swarm.log("QUEEN", f"🚨 WARRIOR SPOTTED: {threat['class']} {conf:.0%} at ({world_x:.0f}, {world_y:.0f})", "CRITICAL")
                        return threat
                        
        except Exception as e:
            if self.ai_scan_count % 50 == 0:
                swarm.log("QUEEN", f"Detection error: {e}", "WARNING")
        
        return None
        
    def run(self):
        swarm.log("QUEEN", "Starting", "INFO")
        
        self.client.enableApiControl(True, "Queen")
        self.client.armDisarm(True, "Queen")
        self.client.takeoffAsync(vehicle_name="Queen").join()
        
        swarm.log("QUEEN", "✈️ Airborne - Command Center", "INFO")
        self.client.moveToPositionAsync(0, 0, -20, 5, vehicle_name="Queen").join()
        
        swarm.log("QUEEN", "📡 Monitoring Warrior camera feed", "WARNING")
        swarm.threat_level = "YELLOW"
        
        scan = 0
        
        while True:
            scan += 1
            swarm.queen_scans = scan
            
            if scan % 40 == 0:
                swarm.log("QUEEN", f"Command Center: Scan #{scan}", "INFO")
            
            # PRIORITY 1: Manual threats from UI
            if swarm.active_threat and not swarm.kamikaze_deployed:
                threat = swarm.active_threat
                swarm.log("QUEEN", f"📍 MANUAL THREAT: {threat['class']}", "CRITICAL")
                
                if swarm.request_permission():
                    swarm.log("QUEEN", "✅ Strike authorized", "CRITICAL")
                    swarm.kamikaze_target = threat['world_pos']
                    swarm.kamikaze_deployed = True
                    break
                else:
                    swarm.log("QUEEN", "❌ Strike denied", "WARNING")
                    swarm.threats.clear()
                    swarm.active_threat = None
                    swarm.threat_level = "YELLOW"
            
            # PRIORITY 2: AI detection from Warrior's camera
            elif False:
                ai_threat = self.detect_threats_from_warrior()
                if ai_threat:
                    swarm.add_threat(ai_threat)
                    
                    if swarm.request_permission():
                        swarm.log("QUEEN", "✅ AI Strike authorized", "CRITICAL")
                        swarm.kamikaze_target = ai_threat['world_pos']
                        swarm.kamikaze_deployed = True
                        break
                    else:
                        swarm.log("QUEEN", "❌ AI Strike denied", "WARNING")
                        swarm.threats.clear()
                        swarm.active_threat = None
                        swarm.threat_level = "YELLOW"
            
            time.sleep(1)  # Slower scan = less CPU usage
        
        swarm.log("QUEEN", "✅ Mission complete", "INFO")
        self.client.hoverAsync(vehicle_name="Queen").join()

def run():
    queen = Queen()
    queen.run()