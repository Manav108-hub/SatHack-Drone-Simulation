import time
import logging
from logging.handlers import RotatingFileHandler
import os
import threading
import sys  # ← ADD THIS
import airsim
from swarm_state import swarm

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("KAMIKAZE")
logger.setLevel(logging.DEBUG)
fh = RotatingFileHandler(os.path.join(LOG_DIR, "kamikaze.log"), maxBytes=2_000_000, backupCount=3)
fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"))
logger.addHandler(fh)

# ✅ FIX: UTF-8 encoding
ch = logging.StreamHandler(sys.stdout)
if hasattr(ch.stream, 'reconfigure'):
    ch.stream.reconfigure(encoding='utf-8')
ch.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] %(message)s", "%H:%M:%S"))
logger.addHandler(ch)
logger.propagate = False


class Kamikaze:
    def __init__(self, vehicle_name="Kamikaze1"):
        self.vehicle_name = vehicle_name
        self.client = None
        
    def _connect(self):
        """Create dedicated client"""
        try:
            logger.info(f"{self.vehicle_name}: Connecting to AirSim...")
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            logger.info(f"{self.vehicle_name}: Connected")
            return True
        except Exception as e:
            logger.error(f"{self.vehicle_name}: Connection failed: {e}")
            return False

    def run(self):
        # Connect first
        if not self._connect():
            swarm.log(self.vehicle_name, "Connection failed (standby)", "WARNING")
            # Still stay alive to listen for commands
            while True:
                time.sleep(1)
            return
            
        swarm.log(self.vehicle_name, "Initializing", "INFO")
        
        # Startup
        try:
            self.client.enableApiControl(True, self.vehicle_name)
            time.sleep(0.5)
            self.client.armDisarm(True, self.vehicle_name)
            time.sleep(0.5)
            
            # Takeoff
            future = self.client.takeoffAsync(vehicle_name=self.vehicle_name)
            future.join()
            
            swarm.log(self.vehicle_name, "Standby", "INFO")
        except Exception as e:
            logger.error(f"{self.vehicle_name} startup failed: {e}")
            swarm.log(self.vehicle_name, f"Startup error: {e}", "WARNING")

        # Main loop - wait for strike command
        while True:
            if swarm.kamikaze_deployed and swarm.kamikaze_target:
                target = swarm.kamikaze_target
                swarm.log(self.vehicle_name, f"STRIKE -> {target}", "CRITICAL")
                
                try:
                    tx, ty = float(target[0]), float(target[1])
                    
                    # Move to target
                    future = self.client.moveToPositionAsync(
                        tx, ty, -10, 15, 
                        vehicle_name=self.vehicle_name
                    )
                    future.join()
                    
                    swarm.log(self.vehicle_name, f"Reached target ({tx:.1f},{ty:.1f})", "CRITICAL")
                    
                    # Hover after strike
                    self.client.hoverAsync(vehicle_name=self.vehicle_name)
                    
                except Exception as e:
                    swarm.log(self.vehicle_name, f"Strike error: {e}", "WARNING")
                    logger.exception("Strike failed")
                
                # Reset (only first kamikaze resets the flag)
                if self.vehicle_name == "Kamikaze1":
                    swarm.kamikaze_deployed = False
                    swarm.kamikaze_target = None
                
                break  # Mission complete for this kamikaze
                
            time.sleep(0.5)


def launch_kamikaze(name):
    """Launch a single kamikaze in a thread"""
    try:
        kamikaze = Kamikaze(vehicle_name=name)
        kamikaze.run()
    except Exception as e:
        logger.exception(f"Failed to launch {name}")


def run():
    """Launch all kamikazes from settings.json"""
    import json
    
    kamikazes_list = []
    try:
        if os.path.isfile("settings.json"):
            with open("settings.json", "r") as f:
                cfg = json.load(f)
            vehicles = cfg.get("Vehicles", {})
            
            for name in vehicles.keys():
                if "kamikaze" in name.lower():
                    kamikazes_list.append(name)
            
            kamikazes_list.sort()
            
        if not kamikazes_list:
            kamikazes_list = ["Kamikaze1"]
            
    except Exception as e:
        logger.exception("Failed to load settings.json")
        kamikazes_list = ["Kamikaze1"]
    
    logger.info(f"Launching {len(kamikazes_list)} kamikazes: {kamikazes_list}")
    print("\n" + "="*70)
    print(f"LAUNCHING {len(kamikazes_list)} KAMIKAZES")
    print("="*70)
    for k in kamikazes_list:
        print(f"   - {k}")
    print("="*70 + "\n")
    
    threads = []
    
    for kamikaze_name in kamikazes_list:
        t = threading.Thread(
            target=launch_kamikaze, 
            args=(kamikaze_name,),
            daemon=True,
            name=kamikaze_name
        )
        t.start()
        threads.append(t)
        logger.info(f"Started thread for {kamikaze_name}")
        time.sleep(1)  # Stagger launches
    
    logger.info(f"All {len(kamikazes_list)} kamikaze threads started")
    
    # Keep main thread alive
    try:
        while True:
            alive_count = sum(1 for t in threads if t.is_alive())
            if alive_count == 0:
                logger.warning("All kamikaze threads have stopped")
                break
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("Kamikazes interrupted by user")


if __name__ == "__main__":
    run()