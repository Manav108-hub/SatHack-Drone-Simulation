import airsim
import time
from swarm_state import swarm

class Kamikaze:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        
    def run(self):
        print("\n💤 KAMIKAZE STARTING...")
        
        self.client.enableApiControl(True, "Kamikaze1")
        self.client.armDisarm(True, "Kamikaze1")
        self.client.takeoffAsync(vehicle_name="Kamikaze1").join()
        
        print("📍 Moving to standby...")
        self.client.moveToPositionAsync(-20, -20, -10, 3, vehicle_name="Kamikaze1").join()
        
        print("💤 KAMIKAZE ON STANDBY\n")
        
        while not swarm.kamikaze_deployed:
            time.sleep(0.5)
        
        target = swarm.kamikaze_target
        
        print(f"\n💥 KAMIKAZE STRIKE AUTHORIZED!")
        print(f"🎯 Target: ({target[0]:.1f}, {target[1]:.1f})")
        
        self.client.moveToPositionAsync(
            target[0], target[1], -5, 
            velocity=20, 
            vehicle_name="Kamikaze1"
        ).join()
        
        print("💥💥💥 TARGET ELIMINATED!")
        
        time.sleep(1)
        self.client.landAsync(vehicle_name="Kamikaze1").join()

def run():
    kamikaze = Kamikaze()
    kamikaze.run()