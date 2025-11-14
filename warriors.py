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
            swarm.log("WARRIOR", f"❌ WARRIOR ERROR: {e}", "CRITICAL")
            raise

    # ------------------------------------------------
    # Safe movement wrapper
    # ------------------------------------------------
    def _safe_move(self, x, y, z=-15, speed=8, timeout_sec=15):
        try:
            f = self.client.moveToPositionAsync(
                x, y, z, speed, vehicle_name=self.vehicle_name
            )
            f.join(timeout_sec)
            return True
        except Exception as e:
            logger.warning(f"Move error: {e}")
            swarm.log("WARRIOR", f"Move error: {e}", "WARNING")
            return False

    # ------------------------------------------------
    # Report position
    # ------------------------------------------------
    def _report_position(self):
        try:
            pose = self.client.simGetVehiclePose(self.vehicle_name).position
            pos = (pose.x_val, pose.y_val, pose.z_val)
            swarm.warrior_report(pos)
        except Exception:
            logger.debug("Failed to report warrior position")

    # ------------------------------------------------
    # Main loop
    # ------------------------------------------------
    def run(self):
        swarm.log("WARRIOR", "Initializing", "INFO")
        logger.info("Warrior run() starting")

        # Startup sequence
        try:
            self.client.enableApiControl(True, self.vehicle_name)
            self.client.armDisarm(True, self.vehicle_name)
            self.client.takeoffAsync(vehicle_name=self.vehicle_name).join(10)
        except Exception:
            pass

        angle = 0
        last_patrol = None

        while not swarm.kamikaze_deployed:

            # ------------------------------------------------------------------
            # ALWAYS compute patrol center the SAME WAY
            # ------------------------------------------------------------------
            try:
                # Try to get queen
                try:
                    qp = self.client.simGetVehiclePose("Queen").position
                    queen_xy = (qp.x_val, qp.y_val)
                except Exception:
                    queen_xy = None

                # Effective patrol center (ABSOLUTE or RELATIVE internally handled)
                cx, cy, radius = swarm.get_effective_patrol(queen_xy)
                current_patrol = (cx, cy, radius)

            except Exception as e:
                swarm.log("WARRIOR", f"Patrol calc error: {e}", "WARNING")
                time.sleep(1)
                continue

            # ------------------------------------------------------------------
            # Detect patrol change
            # ------------------------------------------------------------------
            if last_patrol != current_patrol:
                swarm.log(
                    "WARRIOR",
                    f"🎯 PATROL CHANGE: ({cx:.1f}, {cy:.1f}) R={radius:.1f}",
                    "WARNING",
                )
                last_patrol = current_patrol
                angle = 0

            # ------------------------------------------------------------------
            # Compute next waypoint on patrol circle
            # ------------------------------------------------------------------
            x = cx + radius * math.cos(math.radians(angle))
            y = cy + radius * math.sin(math.radians(angle))
            z = -15

            swarm.log("WARRIOR", f"→ ({x:.1f}, {y:.1f})", "INFO")
            self._safe_move(x, y, z=z, speed=8, timeout_sec=20)

            # Report position a few times for smooth UI
            for _ in range(3):
                self._report_position()
                time.sleep(1)

            # Update angle
            angle = (angle + 60) % 360

        # ------------------------------------------------------------------
        # Mission complete
        # ------------------------------------------------------------------
        swarm.log("WARRIOR", "RTB (hover)", "WARNING")
        try:
            self.client.hoverAsync(vehicle_name=self.vehicle_name)
        except:
            pass


def run():
    Warrior().run()
