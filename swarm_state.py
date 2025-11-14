# swarm_state.py
from threading import Lock
import time
from datetime import datetime
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import sys

CONSOLE_SUPPORTS_UTF8 = sys.stdout.encoding.lower().startswith("utf")

# -------------------------------------------------------
# SAFE SERIALIZER (GLOBAL FUNCTION, NOT INSIDE CLASS)
# -------------------------------------------------------
def to_serializable(obj):
    """Convert tensors/numpy types into JSON-safe Python values."""
    # torch tensors
    try:
        if hasattr(obj, "item"):
            return float(obj.item())
    except:
        pass

    # list/tuple
    if isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]

    # dict
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}

    # numpy
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except:
        pass

    return obj
# -------------------------------------------------------


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
        self.patrol_relative_to_queen = False

        # Telemetry
        self.last_warrior_pos = (None, None, None)
        self.last_warrior_update = 0.0
        self.last_patrol_update = 0.0

        # Persist file
        self.persist_file = os.path.join(LOG_DIR, "swarm_persist.json")
        self._load_persisted()


    # -------------------------------------------------------
    # Persistence
    # -------------------------------------------------------
    def _load_persisted(self):
        try:
            if os.path.isfile(self.persist_file):
                with open(self.persist_file, "r") as f:
                    data = json.load(f)
                    with self.lock:
                        self.mission_logs = data.get("mission_logs", [])[-500:]
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

    # -------------------------------------------------------
    # Main logger (mission + file log)
    # -------------------------------------------------------
    def log(self, source, message, level="INFO"):
        message = to_serializable(message)

        log_entry = {
            'time': datetime.now().strftime("%H:%M:%S"),
            'source': source,
            'message': message,
            'level': level
        }

        with self.lock:
            self.mission_logs.append(log_entry)
            if len(self.mission_logs) > 2000:
                self.mission_logs.pop(0)

        # File logger
        safe_msg = message
        if not CONSOLE_SUPPORTS_UTF8:
            safe_msg = safe_msg.encode("ascii", "ignore").decode()

        # console output (no emojis if Windows)
        if level == "CRITICAL":
            logger.error(f"[{source}] {safe_msg}")
        elif level == "WARNING":
            logger.warning(f"[{source}] {safe_msg}")
        else:
            logger.info(f"[{source}] {safe_msg}")



    # -------------------------------------------------------
    # Threat handling
    # -------------------------------------------------------
    def add_threat(self, threat_data):
        with self.lock:
            self.threats.append(threat_data)
            self.active_threat = threat_data
            self.threat_level = "RED"

        self.log("QUEEN", 
                 f"🚨 THREAT: {threat_data['class']} at "
                 f"({threat_data['world_pos'][0]:.1f}, {threat_data['world_pos'][1]:.1f})",
                 "CRITICAL")

    # -------------------------------------------------------
    # Warrior status
    # -------------------------------------------------------
    def warrior_report(self, position):
        with self.lock:
            self.last_warrior_pos = position
            self.last_warrior_update = time.time()
            self.warrior_reports.append({
                'time': datetime.now().strftime("%H:%M:%S"),
                'position': position,
                'status': 'PATROLLING'
            })
            if len(self.warrior_reports) > 200:
                self.warrior_reports.pop(0)

        if len(self.warrior_reports) % 5 == 0:
            self.log("WARRIOR", f"Pos: ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})", "INFO")

    # -------------------------------------------------------
    # Permission request (15s countdown)
    # -------------------------------------------------------
    def request_permission(self):

        # --- REAL AUTH LOGIC STARTS HERE ---
        if self.queen_mode == "jammer":
            self.log("SYSTEM", "⚡ JAMMER MODE - AUTO-AUTH", "WARNING")
            return True

        self.log("QUEEN", "📞 REQUESTING AUTHORIZATION", "WARNING")

        if self.active_threat:
            self.log("QUEEN",
                    f"Target: {self.active_threat['class']} @ "
                    f"({self.active_threat['world_pos'][0]:.1f}, "
                    f"{self.active_threat['world_pos'][1]:.1f})",
                    "WARNING")

        self.pending_permission = True
        self.user_response = None

        start = time.time()
        timeout = 15

        while time.time() - start < timeout:
            if self.user_response is not None:
                approved = self.user_response
                self.pending_permission = False
                self.user_response = None

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


        # --- Real logic kept below for future use ----
        # ...

    # -------------------------------------------------------
    # Getters
    # -------------------------------------------------------
    def get_logs(self, limit=50):
        with self.lock:
            return self.mission_logs[-limit:]

    def get_warrior_status(self):
        with self.lock:
            return self.warrior_reports[-1] if self.warrior_reports else None

    # -------------------------------------------------------
    # Patrol area
    # -------------------------------------------------------
    def set_patrol_area(self, cx, cy, radius, relative=False):
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
        with self.lock:
            cx, cy, r, rel = (self.patrol_center_x, self.patrol_center_y,
                              self.patrol_radius, self.patrol_relative_to_queen)

        if rel and queen_pose and queen_pose[0] is not None:
            return (queen_pose[0] + cx, queen_pose[1] + cy, r)

        return (cx, cy, r)


# SINGLE INSTANCE
swarm = SwarmState()
