# kamikaze.py
import time
import logging
from logging.handlers import RotatingFileHandler
import os
import airsim
from swarm_state import swarm

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("KAMIKAZE")
logger.setLevel(logging.DEBUG)
fh = RotatingFileHandler(os.path.join(LOG_DIR, "kamikaze.log"), maxBytes=2_000_000, backupCount=3)
fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"))
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] %(message)s", "%H:%M:%S"))
logger.addHandler(ch)
logger.propagate = False


class Kamikaze:
    def __init__(self, vehicle_name="Kamikaze1"):
        self.vehicle_name = vehicle_name
        try:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            logger.info("Kamikaze: connected to AirSim")
        except Exception as e:
            logger.warning(f"Kamikaze AirSim connection failed: {e}")
            self.client = None

    def run(self):
        swarm.log("KAMIKAZE", "Initializing", "INFO")

        if self.client:
            try:
                self.client.enableApiControl(True, self.vehicle_name)
                self.client.armDisarm(True, self.vehicle_name)
                self.client.takeoffAsync(vehicle_name=self.vehicle_name).join(timeout=8)
            except:
                pass
        else:
            swarm.log("KAMIKAZE", "No AirSim client (standby)", "WARNING")

        while True:
            if swarm.kamikaze_deployed and swarm.kamikaze_target:
                target = swarm.kamikaze_target
                swarm.log("KAMIKAZE", f"🔥 KAMIKAZE STRIKE -> {target}", "CRITICAL")

                if self.client:
                    try:
                        tx, ty = float(target[0]), float(target[1])
                        self.client.moveToPositionAsync(tx, ty, -10, 10,
                                vehicle_name=self.vehicle_name).join()

                        swarm.log("KAMIKAZE", 
                                  f"Reached strike target ({tx:.1f},{ty:.1f})",
                                  "CRITICAL")
                        self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
                    except Exception as e:
                        swarm.log("KAMIKAZE", f"Strike error: {e}", "WARNING")

                swarm.kamikaze_deployed = False
                swarm.kamikaze_target = None

            time.sleep(1)


def run():
    k = Kamikaze()
    k.run()
