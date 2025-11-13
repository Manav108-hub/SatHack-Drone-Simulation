from threading import Lock
import time

class SwarmState:
    def __init__(self):
        self.lock = Lock()
        self.threats = []
        self.active_threat = None
        self.queen_mode = "normal"  # "normal" or "jammer"
        self.kamikaze_deployed = False
        self.kamikaze_target = None
        
    def add_threat(self, threat_data):
        with self.lock:
            self.threats.append(threat_data)
            self.active_threat = threat_data
            print(f"🚨 THREAT: {threat_data['class']} at ({threat_data['world_pos'][0]:.1f}, {threat_data['world_pos'][1]:.1f})")
    
    def request_permission(self):
        if self.queen_mode == "jammer":
            print("⚡ JAMMER MODE - AUTO-APPROVED")
            return True
        
        print("\n📞 PERMISSION REQUEST - Deploy kamikaze?")
        print(f"   Threat: {self.active_threat['class']}")
        print("   Auto-approving in 3 seconds...")
        time.sleep(3)
        print("✅ APPROVED")
        return True

swarm = SwarmState()