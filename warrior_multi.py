import os
import time
import math
import logging
from logging.handlers import RotatingFileHandler
import threading
import sys  # ← ADD THIS

import airsim
from swarm_state import swarm

# ----------------------------
# Logging
# ----------------------------
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("WARRIORS")
logger.setLevel(logging.DEBUG)
fh = RotatingFileHandler(os.path.join(LOG_DIR, "warriors.log"), maxBytes=5_000_000, backupCount=3)
fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"))
logger.addHandler(fh)

# ✅ FIX: UTF-8 encoding
ch = logging.StreamHandler(sys.stdout)
if hasattr(ch.stream, 'reconfigure'):
    ch.stream.reconfigure(encoding='utf-8')
ch.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] %(message)s", "%H:%M:%S"))
logger.addHandler(ch)

class Warrior:
    def __init__(self, vehicle_name="Warrior1", angle_offset=0):
        self.vehicle_name = vehicle_name
        self.angle_offset = angle_offset
        self.client = None  # Will be created in run()

    def _connect(self):
        """Create a dedicated client for this warrior"""
        try:
            logger.info(f"{self.vehicle_name}: Connecting to AirSim...")
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            logger.info(f"{self.vehicle_name}: Connected to AirSim")
            return True
        except Exception as e:
            logger.exception(f"{self.vehicle_name}: Connection to AirSim failed")
            swarm.log(self.vehicle_name, f"ERROR: {e}", "CRITICAL")
            return False

    def _safe_move(self, x, y, z=-15, speed=8):
        try:
            f = self.client.moveToPositionAsync(
                x, y, z, speed, vehicle_name=self.vehicle_name
            )
            f.join()  # Fixed: removed timeout parameter
            return True
        except Exception as e:
            logger.warning(f"{self.vehicle_name} move error: {e}")
            return False

    def _report_position(self):
        try:
            pose = self.client.simGetVehiclePose(self.vehicle_name).position
            pos = (pose.x_val, pose.y_val, pose.z_val)
            if self.vehicle_name == "Warrior1":
                swarm.warrior_report(pos)
        except Exception:
            pass

    def run(self):
        # Connect to AirSim FIRST - each warrior gets own client
        if not self._connect():
            logger.error(f"{self.vehicle_name}: Failed to connect, aborting")
            return
        
        swarm.log(self.vehicle_name, "Initializing", "INFO")
        logger.info(f"{self.vehicle_name} run() starting with angle offset {self.angle_offset}")

        # Startup sequence
        try:
            self.client.enableApiControl(True, self.vehicle_name)
            time.sleep(0.5)
            self.client.armDisarm(True, self.vehicle_name)
            time.sleep(0.5)
            
            # Fixed: Don't pass timeout to join()
            future = self.client.takeoffAsync(vehicle_name=self.vehicle_name)
            future.join()
            
            time.sleep(2)
            swarm.log(self.vehicle_name, f"Airborne at angle {self.angle_offset}", "INFO")
        except Exception as e:
            logger.error(f"{self.vehicle_name} startup failed: {e}")
            return

        angle = self.angle_offset
        last_patrol = None

        while not swarm.kamikaze_deployed:
            try:
                # Get queen position
                try:
                    qp = self.client.simGetVehiclePose("Queen").position
                    queen_xy = (qp.x_val, qp.y_val)
                except Exception:
                    queen_xy = None

                # Get effective patrol
                cx, cy, radius = swarm.get_effective_patrol(queen_xy)
                current_patrol = (cx, cy, radius)

                # Detect patrol change
                if last_patrol != current_patrol:
                    swarm.log(
                        self.vehicle_name,
                        f"Patrol: ({cx:.1f}, {cy:.1f}) R={radius:.1f}",
                        "INFO",
                    )
                    last_patrol = current_patrol
                    angle = self.angle_offset

                # Compute waypoint
                x = cx + radius * math.cos(math.radians(angle))
                y = cy + radius * math.sin(math.radians(angle))
                z = -15

                logger.debug(f"{self.vehicle_name} -> ({x:.1f}, {y:.1f})")
                self._safe_move(x, y, z=z, speed=8)

                # Report position
                for _ in range(2):
                    self._report_position()
                    time.sleep(1)

                # Update angle
                angle = (angle + 60) % 360

            except Exception as e:
                logger.error(f"{self.vehicle_name} error: {e}")
                time.sleep(2)

        # Mission complete
        swarm.log(self.vehicle_name, "RTB (hover)", "WARNING")
        try:
            self.client.hoverAsync(vehicle_name=self.vehicle_name)
        except:
            pass


def launch_warrior(name, angle_offset):
    """Launch a single warrior in a thread"""
    try:
        warrior = Warrior(vehicle_name=name, angle_offset=angle_offset)
        warrior.run()
    except Exception as e:
        logger.exception(f"Failed to launch {name}")


def run():
    """Launch all warriors from settings.json"""
    import json
    
    warriors_list = []
    try:
        if os.path.isfile("settings.json"):
            with open("settings.json", "r") as f:
                cfg = json.load(f)
            vehicles = cfg.get("Vehicles", {})
            
            for name in vehicles.keys():
                if "warrior" in name.lower():
                    warriors_list.append(name)
            
            warriors_list.sort()
            
        if not warriors_list:
            warriors_list = ["Warrior1"]
            
    except Exception as e:
        logger.exception("Failed to load settings.json")
        warriors_list = ["Warrior1"]
    
    logger.info(f"Launching {len(warriors_list)} warriors: {warriors_list}")
    print("\n" + "="*70)
    print(f"LAUNCHING {len(warriors_list)} WARRIORS")
    print("="*70)
    for w in warriors_list:
        print(f"   - {w}")
    print("="*70 + "\n")
    
    threads = []
    angle_step = 360 // len(warriors_list)
    
    for idx, warrior_name in enumerate(warriors_list):
        angle_offset = idx * angle_step
        t = threading.Thread(
            target=launch_warrior, 
            args=(warrior_name, angle_offset),
            daemon=True,
            name=warrior_name
        )
        t.start()
        threads.append(t)
        logger.info(f"Started thread for {warrior_name} at angle {angle_offset}")
        time.sleep(2)  # Increased stagger time
    
    logger.info(f"All {len(warriors_list)} warrior threads started")
    
    # Keep main thread alive
    try:
        while True:
            alive_count = sum(1 for t in threads if t.is_alive())
            if alive_count == 0:
                logger.warning("All warrior threads have stopped")
                break
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("Warriors interrupted by user")


if __name__ == "__main__":
    run()