# warriors.py
import os
import time
import math
import traceback
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
        self.client = None
        try:
            logger.info("Warrior: Connecting to AirSim...")
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            logger.info("Warrior: Connected to AirSim")
        except Exception as e:
            logger.exception("Warrior: Connection to AirSim failed")
            swarm.log("WARRIOR", f"❌ WARRIOR ERROR: {e}", "CRITICAL")
            raise

    def _safe_move(self, x, y, z=-15, speed=8, timeout_sec=15):
        """Move to position and return True if command executed (not necessarily reached)."""
        try:
            # non-blocking call then wait with timeout
            fut = self.client.moveToPositionAsync(x, y, z, speed, vehicle_name=self.vehicle_name)
            fut.join(timeout_sec)
            return True
        except Exception as e:
            logger.warning(f"Move error: {e}")
            swarm.log("WARRIOR", f"Move error: {e}", "WARNING")
            return False

    def _report_position(self):
        """Get vehicle pose and report to swarm_state (warrior_report)."""
        try:
            pose = self.client.simGetVehiclePose(self.vehicle_name).position
            pos_tuple = (pose.x_val, pose.y_val, pose.z_val)
            swarm.warrior_report(pos_tuple)
            # also log a concise line for operator
            logger.debug(f"Warrior pos reported: ({pos_tuple[0]:.2f}, {pos_tuple[1]:.2f}, {pos_tuple[2]:.2f})")
        except Exception as e:
            logger.debug("Failed to report warrior position", exc_info=True)

    def run(self):
        swarm.log("WARRIOR", "Initializing", "INFO")
        logger.info("Warrior run() starting")

        try:
            self.client.enableApiControl(True, self.vehicle_name)
            self.client.armDisarm(True, self.vehicle_name)
            self.client.takeoffAsync(vehicle_name=self.vehicle_name).join(timeout=10)
        except Exception as e:
            logger.warning("Takeoff/enable control had issues (continuing): %s", e)

        # initial variables
        angle = 0
        last_patrol = None
        heartbeat_time = time.time()

        # main loop
        while not swarm.kamikaze_deployed:
            try:
                current_patrol = swarm.get_patrol_area()
                cx, cy, radius = current_patrol

                # If patrol changed, reset pattern
                if last_patrol != current_patrol:
                    swarm.log("WARRIOR", f"🎯 PATROL CHANGE: ({cx:.0f}, {cy:.0f}) R={radius:.0f}m", "WARNING")
                    logger.info(f"Patrol updated from {last_patrol} -> {current_patrol}")
                    last_patrol = current_patrol
                    angle = 0

                # Compute waypoint on circle
                x = cx + radius * math.cos(math.radians(angle))
                y = cy + radius * math.sin(math.radians(angle))
                z = -15  # fixed altitude for patrol; tune as needed

                swarm.log("WARRIOR", f"→ ({x:.1f}, {y:.1f})", "INFO")
                self._safe_move(x, y, z=z, speed=8, timeout_sec=20)

                # After move attempt, report position multiple times to ensure UI gets updates
                for i in range(3):
                    self._report_position()
                    time.sleep(1)

                # increment angle for next waypoint
                angle = (angle + 60) % 360

                # Heartbeat: if UI hasn't seen warrior recently, push a status log
                if time.time() - heartbeat_time > 10:
                    heartbeat_time = time.time()
                    logger.debug("Heartbeat - reporting warrior status")
                    self._report_position()

            except Exception as e:
                logger.exception("Warrior main loop error")
                swarm.log("WARRIOR", f"Error: {e}", "WARNING")
                # Sleep briefly to avoid tight error loop
                time.sleep(1)

        # When kamikaze_deployed flag is true, we return to base / hover
        swarm.log("WARRIOR", "RTB (return to base/hover)", "WARNING")
        try:
            self.client.hoverAsync(vehicle_name=self.vehicle_name).join(timeout=5)
        except Exception:
            pass
        logger.info("Warrior exiting run()")

def run():
    w = Warrior()
    w.run()
