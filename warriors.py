# warriors.py
import os
import time
import math
import logging
from logging.handlers import RotatingFileHandler

import airsim
from swarm_state import swarm

# ----------------------------
# Logging for Warrior
# ----------------------------
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("WARRIOR")
logger.setLevel(logging.DEBUG)
fh = RotatingFileHandler(os.path.join(LOG_DIR, "warrior.log"), maxBytes=2_000_000, backupCount=3)
fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"))
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] %(message)s", "%H:%M:%S"))
logger.addHandler(ch)


class Warrior:
    def __init__(self, vehicle_name="Warrior1"):
        self.vehicle_name = vehicle_name

        try:
            logger.info("Warrior: Connecting to AirSim...")
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            logger.info("Warrior: Connected to AirSim")

        except Exception as e:
            logger.exception("Warrior: Connection to AirSim failed")
            swarm.log("WARRIOR", f"WARRIOR ERROR: {e}", "CRITICAL")
            raise

    def _safe_move(self, x, y, z=-15, speed=8):
        """Move to position - FIXED: No timeout parameter"""
        try:
            future = self.client.moveToPositionAsync(
                x, y, z, speed, vehicle_name=self.vehicle_name
            )
            future.join()
            return True
        except Exception as e:
            logger.warning(f"Move error: {e}")
            logger.warning(f"Move error: {e}")
            swarm.log("WARRIOR", f"Move error: {e}", "WARNING")
            return False

    def _report_position(self):
        try:
            pose = self.client.simGetVehiclePose(self.vehicle_name).position
            pos = (pose.x_val, pose.y_val, pose.z_val)
            swarm.warrior_report(pos)
        except Exception:
            logger.debug("Failed to report warrior position")

    def run(self):
        swarm.log("WARRIOR", "Initializing", "INFO")
        logger.info("Warrior run() starting")

        # Startup sequence
        try:
            self.client.enableApiControl(True, self.vehicle_name)
            self.client.armDisarm(True, self.vehicle_name)
            
            future = self.client.takeoffAsync(vehicle_name=self.vehicle_name)
            future.join()
            time.sleep(2)
            
        except Exception as e:
            logger.warning(f"Startup error: {e}")

        angle = 0
        last_patrol = None

        while not swarm.kamikaze_deployed:

            try:
                try:
                    qp = self.client.simGetVehiclePose("Queen").position
                    queen_xy = (qp.x_val, qp.y_val)
                except Exception:
                    queen_xy = None

                cx, cy, radius = swarm.get_effective_patrol(queen_xy)
                current_patrol = (cx, cy, radius)

            except Exception as e:
                swarm.log("WARRIOR", f"Patrol calc error: {e}", "WARNING")
                time.sleep(1)
                continue

            if last_patrol != current_patrol:
                swarm.log(
                    "WARRIOR",
                    f"PATROL CHANGE: ({cx:.1f}, {cy:.1f}) R={radius:.1f}",
                    "WARNING",
                )
                last_patrol = current_patrol
                angle = 0

            x = cx + radius * math.cos(math.radians(angle))
            y = cy + radius * math.sin(math.radians(angle))
            z = -15

            swarm.log("WARRIOR", f"Moving to ({x:.1f}, {y:.1f})", "INFO")
            
            self._safe_move(x, y, z=z, speed=8)

            for _ in range(3):
                self._report_position()
                time.sleep(1)

            angle = (angle + 60) % 360

        swarm.log("WARRIOR", "RTB (hover)", "WARNING")
        try:
            self.client.hoverAsync(vehicle_name=self.vehicle_name)
        except:
            pass


def run():
    Warrior().run()