import airsim
import numpy as np
from ultralytics import YOLO
import time
from swarm_state import swarm

class Queen:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        print("📥 Loading AI model...")
        self.model = YOLO('yolov8n.pt')
        print("✅ AI loaded!")
        
        self.threat_classes = {0: 'person', 2: 'car', 7: 'truck'}
        
    def get_camera(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ], "Queen")
        
        if responses and len(responses[0].image_data_uint8) > 0:
            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            return img1d.reshape(responses[0].height, responses[0].width, 3)
        return None
    
    def detect_threats(self):
        img = self.get_camera()
        if img is None:
            return None
        
        results = self.model(img, verbose=False, conf=0.5)
        
        for box in results[0].boxes:
            class_id = int(box.cls)
            if class_id in self.threat_classes:
                conf = float(box.conf)
                x, y, w, h = box.xywh[0].tolist()
                
                queen_pos = self.client.simGetVehiclePose("Queen").position
                world_x = queen_pos.x_val + (x - 320) / 20
                world_y = queen_pos.y_val + (y - 240) / 20
                
                return {
                    'class': self.threat_classes[class_id],
                    'confidence': conf,
                    'world_pos': (world_x, world_y),
                    'timestamp': time.time()
                }
        return None
    
    def run(self):
        print("\n👑 QUEEN STARTING...")
        
        self.client.enableApiControl(True, "Queen")
        self.client.armDisarm(True, "Queen")
        self.client.takeoffAsync(vehicle_name="Queen").join()
        
        print("📍 Moving to observation position...")
        self.client.moveToPositionAsync(0, 0, -20, 5, vehicle_name="Queen").join()
        
        print("\n👁️  QUEEN MONITORING ACTIVE\n")
        
        scan = 0
        while True:
            scan += 1
            if scan % 10 == 0:
                print(f"🔍 Scan #{scan}")
            
            threat = self.detect_threats()
            
            if threat:
                swarm.add_threat(threat)
                
                if swarm.request_permission():
                    print("✅ Deploying kamikaze!")
                    swarm.kamikaze_target = threat['world_pos']
                    swarm.kamikaze_deployed = True
                    break
            
            time.sleep(0.5)
        
        print("👑 Queen mission complete")
        self.client.hoverAsync(vehicle_name="Queen").join()

def run():
    queen = Queen()
    queen.run()