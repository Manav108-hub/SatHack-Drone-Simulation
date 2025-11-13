import airsim
import time
import math
from swarm_state import swarm

class Warrior:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        
    def run(self):
        print("\n🛸 WARRIOR STARTING...")
        
        self.client.enableApiControl(True, "Warrior1")
        self.client.armDisarm(True, "Warrior1")
        self.client.takeoffAsync(vehicle_name="Warrior1").join()
        
        print("📍 Moving to patrol zone...")
        self.client.moveToPositionAsync(20, 20, -15, 5, vehicle_name="Warrior1").join()
        
        print("🎯 WARRIOR PATROL ACTIVE\n")
        
        radius = 25
        angle = 0
        count = 0
        
        while not swarm.kamikaze_deployed:
            x = radius * math.cos(math.radians(angle))
            y = radius * math.sin(math.radians(angle))
            
            if angle % 90 == 0:
                count += 1
                print(f"🛸 Warrior waypoint {count}")
            
            self.client.moveToPositionAsync(x, y, -15, 8, vehicle_name="Warrior1")
            angle = (angle + 30) % 360
            time.sleep(2)
        
        print("🛸 Warrior returning to base")
        self.client.hoverAsync(vehicle_name="Warrior1").join()

def run():
    warrior = Warrior()
    warrior.run()