import airsim
import time
from swarm_state import swarm

class Queen:
    def __init__(self):
        try:
            print("Queen: Connecting to AirSim...")
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            print("Queen: Connected!")
            swarm.log("QUEEN", "Connected to AirSim", "INFO")
        except Exception as e:
            print(f"❌ QUEEN ERROR: Cannot connect to AirSim!")
            print(f"   Make sure simulation is running first!")
            print(f"   Error: {e}")
            raise Exception("AirSim not running - Start simulation first!")
        
        swarm.log("QUEEN", "Initializing - MANUAL MODE", "INFO")
        
    def run(self):
        swarm.log("QUEEN", "Starting", "INFO")
        
        self.client.enableApiControl(True, "Queen")
        self.client.armDisarm(True, "Queen")
        self.client.takeoffAsync(vehicle_name="Queen").join()
        
        swarm.log("QUEEN", "Airborne - Awaiting manual threats", "INFO")
        self.client.moveToPositionAsync(0, 0, -20, 5, vehicle_name="Queen").join()
        
        swarm.log("QUEEN", "Monitoring active", "INFO")
        swarm.threat_level = "YELLOW"
        
        scan = 0
        while True:
            scan += 1
            swarm.queen_scans = scan
            
            if scan % 20 == 0:
                swarm.log("QUEEN", f"Scan #{scan} - Standing by", "INFO")
            
            # Check if manual threat was added
            if swarm.active_threat and not swarm.kamikaze_deployed:
                threat = swarm.active_threat
                swarm.log("QUEEN", f"Manual threat detected: {threat['class']}", "CRITICAL")
                
                if swarm.request_permission():
                    swarm.log("QUEEN", "Strike authorized", "CRITICAL")
                    swarm.kamikaze_target = threat['world_pos']
                    swarm.kamikaze_deployed = True
                    break
                else:
                    swarm.log("QUEEN", "Strike denied", "WARNING")
                    swarm.threats.clear()
                    swarm.active_threat = None
                    swarm.threat_level = "YELLOW"
            
            time.sleep(0.5)
        
        swarm.log("QUEEN", "Mission complete", "INFO")
        self.client.hoverAsync(vehicle_name="Queen").join()

def run():
    queen = Queen()
    queen.run()