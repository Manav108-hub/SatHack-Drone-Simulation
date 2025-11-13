from threading import Lock
import time
from datetime import datetime

class SwarmState:
    def __init__(self):
        self.lock = Lock()
        self.threats = []
        self.active_threat = None
        self.queen_mode = "normal"
        self.kamikaze_deployed = False
        self.kamikaze_target = None
        
        self.mission_logs = []
        self.warrior_reports = []
        self.queen_scans = 0
        self.threat_level = "GREEN"
        
        self.pending_permission = False
        self.user_response = None
        
        # Patrol settings
        self.patrol_center_x = 0
        self.patrol_center_y = 0
        self.patrol_radius = 30
        
    def log(self, source, message, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {
            'time': timestamp,
            'source': source,
            'message': message,
            'level': level
        }
        with self.lock:
            self.mission_logs.append(log_entry)
            if len(self.mission_logs) > 200:
                self.mission_logs.pop(0)
        print(f"[{timestamp}] [{source}] {message}")
    
    def add_threat(self, threat_data):
        with self.lock:
            self.threats.append(threat_data)
            self.active_threat = threat_data
            self.threat_level = "RED"
        self.log("QUEEN", f"🚨 THREAT: {threat_data['class']} at ({threat_data['world_pos'][0]:.1f}, {threat_data['world_pos'][1]:.1f})", "CRITICAL")
    
    def warrior_report(self, position):
        report = {
            'time': datetime.now().strftime("%H:%M:%S"),
            'position': position,
            'status': 'PATROLLING'
        }
        with self.lock:
            self.warrior_reports.append(report)
            if len(self.warrior_reports) > 20:
                self.warrior_reports.pop(0)
        
        if len(self.warrior_reports) % 5 == 0:
            self.log("WARRIOR", f"Pos: ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})", "INFO")
    
    def request_permission(self):
        if self.queen_mode == "jammer":
            self.log("SYSTEM", "⚡ JAMMER MODE - AUTO-AUTH", "WARNING")
            return True
        
        self.log("QUEEN", "📞 REQUESTING AUTHORIZATION", "WARNING")
        self.log("QUEEN", f"Target: {self.active_threat['class']} @ ({self.active_threat['world_pos'][0]:.1f}, {self.active_threat['world_pos'][1]:.1f})", "WARNING")
        
        self.pending_permission = True
        
        start = time.time()
        while time.time() - start < 5:
            if self.user_response is not None:
                approved = self.user_response
                self.user_response = None
                self.pending_permission = False
                
                if approved:
                    self.log("USER", "✅ AUTHORIZED", "CRITICAL")
                    return True
                else:
                    self.log("USER", "❌ DENIED", "WARNING")
                    return False
            time.sleep(0.1)
        
        self.log("SYSTEM", "⏱️ TIMEOUT - AUTO-AUTH", "WARNING")
        self.pending_permission = False
        return True
    
    def get_logs(self, limit=50):
        with self.lock:
            return self.mission_logs[-limit:]
    
    def get_warrior_status(self):
        with self.lock:
            return self.warrior_reports[-1] if self.warrior_reports else None
    
    def set_patrol_area(self, cx, cy, radius):
        with self.lock:
            self.patrol_center_x = cx
            self.patrol_center_y = cy
            self.patrol_radius = radius

swarm = SwarmState()