# swarm_state.py
from threading import Lock
import time
from datetime import datetime
import json
import logging
from logging.handlers import RotatingFileHandler
import os

# --- Logging setup for swarm/system (one file) ---
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("SWARM")
logger.setLevel(logging.DEBUG)
fh = RotatingFileHandler(os.path.join(LOG_DIR, "swarm.log"), maxBytes=2_000_000, backupCount=3)
fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"))
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] %(message)s", "%H:%M:%S"))
logger.addHandler(ch)
logger.propagate = False

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

        # Patrol mode: when True, patrol_center_x/y are offsets relative to Queen
        self.patrol_relative_to_queen = False

        # Telemetry
        self.last_warrior_pos = (None, None, None)
        self.last_warrior_update = 0.0
        self.last_patrol_update = 0.0

        # Persist file
        self.persist_file = os.path.join(LOG_DIR, "swarm_persist.json")
        self._load_persisted()

    # --- persistence ---
    def _load_persisted(self):
        try:
            if os.path.isfile(self.persist_file):
                with open(self.persist_file, "r") as f:
                    data = json.load(f)
                    with self.lock:
                        self.mission_logs = data.get("mission_logs", [])
                        if len(self.mission_logs) > 500:
                            self.mission_logs = self.mission_logs[-500:]
                        self.patrol_center_x = data.get("patrol_center_x", self.patrol_center_x)
                        self.patrol_center_y = data.get("patrol_center_y", self.patrol_center_y)
                        self.patrol_radius = data.get("patrol_radius", self.patrol_radius)
                        self.patrol_relative_to_queen = data.get("patrol_relative_to_queen", self.patrol_relative_to_queen)
                        logger.info("Loaded persisted state.")
        except Exception as e:
            logger.warning(f"Failed to load persisted swarm state: {e}")

    def _persist(self):
        try:
            with self.lock:
                data = {
                    "mission_logs": self.mission_logs[-500:],
                    "patrol_center_x": self.patrol_center_x,
                    "patrol_center_y": self.patrol_center_y,
                    "patrol_radius": self.patrol_radius,
                    "patrol_relative_to_queen": self.patrol_relative_to_queen
                }
            with open(self.persist_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist swarm state: {e}")

    # --- logging helper that writes to mission_logs and file logger ---
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
            if len(self.mission_logs) > 2000:
                self.mission_logs.pop(0)
        # also send to file logger with source tag
        if level == "CRITICAL":
            logger.error(f"[{source}] {message}")
        elif level == "WARNING":
            logger.warning(f"[{source}] {message}")
        else:
            logger.info(f"[{source}] {message}")

        # persist occasionally
        if len(self.mission_logs) % 20 == 0:
            self._persist()

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
            if len(self.warrior_reports) > 200:
                self.warrior_reports.pop(0)
            # update telemetry
            self.last_warrior_pos = position
            self.last_warrior_update = time.time()
        # log summary occasionally
        if len(self.warrior_reports) % 5 == 0:
            self.log("WARRIOR", f"Pos: ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})", "INFO")

    def request_permission(self):
        if self.queen_mode == "jammer":
            self.log("SYSTEM", "⚡ JAMMER MODE - AUTO-AUTH", "WARNING")
            return True
        
        self.log("QUEEN", "📞 REQUESTING AUTHORIZATION", "WARNING")
        if self.active_threat:
            self.log("QUEEN", f"Target: {self.active_threat['class']} @ ({self.active_threat['world_pos'][0]:.1f}, {self.active_threat['world_pos'][1]:.1f})", "WARNING")
        
        self.pending_permission = True
        
        start = time.time()
        timeout = 15  # 15 seconds
        last_countdown = 0
        
        while time.time() - start < timeout:
            elapsed = int(time.time() - start)
            remaining = timeout - elapsed
            
            if remaining != last_countdown and remaining % 5 == 0:
                self.log("SYSTEM", f"⏳ Waiting for authorization... {remaining}s remaining", "WARNING")
                last_countdown = remaining
            
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
            time.sleep(0.2)
        
        self.log("SYSTEM", "⏱️ TIMEOUT - AUTO-AUTH", "WARNING")
        self.pending_permission = False
        return True

    def get_logs(self, limit=50):
        with self.lock:
            return self.mission_logs[-limit:]
    
    def get_warrior_status(self):
        with self.lock:
            return self.warrior_reports[-1] if self.warrior_reports else None

    def set_patrol_area(self, cx, cy, radius, relative=False):
        """Update patrol area - 'relative' means the (cx,cy) are offsets from Queen when True."""
        with self.lock:
            self.patrol_center_x = cx
            self.patrol_center_y = cy
            self.patrol_radius = radius
            self.patrol_relative_to_queen = bool(relative)
            self.last_patrol_update = time.time()
        mode = "RELATIVE_TO_QUEEN" if relative else "ABSOLUTE"
        self.log("SYSTEM", f"📡 Patrol updated: ({cx:.0f}, {cy:.0f}) R={radius:.0f}m MODE={mode}", "WARNING")
        self._persist()

    def get_patrol_area(self):
        with self.lock:
            return (self.patrol_center_x, self.patrol_center_y, self.patrol_radius)

    def get_effective_patrol(self, queen_pose=None):
        """
        Return effective (center_x, center_y, radius).
        If patrol_relative_to_queen is True and queen_pose provided (x,y) then center = queen_pos + offset.
        """
        with self.lock:
            cx = self.patrol_center_x
            cy = self.patrol_center_y
            r = self.patrol_radius
            rel = self.patrol_relative_to_queen

        if rel:
            if queen_pose and len(queen_pose) >= 2 and (queen_pose[0] is not None):
                eff_x = queen_pose[0] + cx
                eff_y = queen_pose[1] + cy
                return (eff_x, eff_y, r)
            # fallback: return offsets as-is (caller may resolve)
            return (cx, cy, r)
        else:
            return (cx, cy, r)

# single instance
swarm = SwarmState()
