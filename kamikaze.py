import airsim
import time
from swarm_state import swarm

class Kamikaze:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        
    def run(self):
        swarm.log("KAMIKAZE", "Initializing", "INFO")
        
        self.client.enableApiControl(True, "Kamikaze1")
        self.client.armDisarm(True, "Kamikaze1")
        self.client.takeoffAsync(vehicle_name="Kamikaze1").join()
        
        swarm.log("KAMIKAZE", "Standby", "INFO")
        
        while not swarm.kamikaze_deployed:
            time.sleep(0.5)
        
        target = swarm.kamikaze_target
        
        swarm.log("KAMIKAZE", f"STRIKE AUTHORIZED!", "CRITICAL")
        swarm.log("KAMIKAZE", f"Target: ({target[0]:.1f}, {target[1]:.1f})", "CRITICAL")
        
        self.client.moveToPositionAsync(target[0], target[1], -3, velocity=25, vehicle_name="Kamikaze1").join()
        
        print("\n" + "="*60)
        for i in range(3):
            print("💥 " * 20)
            time.sleep(0.2)
        print("="*60)
        print("🔥 TARGET DESTROYED! 🔥")
        print("="*60 + "\n")
        
        swarm.log("KAMIKAZE", "TARGET ELIMINATED", "CRITICAL")
        
        self.client.simSetVehiclePose(
            airsim.Pose(airsim.Vector3r(target[0], target[1], 0), airsim.to_quaternion(0, 0, 0)),
            True, "Kamikaze1"
        )
        
        time.sleep(1)

def run():
    kamikaze = Kamikaze()
    kamikaze.run()