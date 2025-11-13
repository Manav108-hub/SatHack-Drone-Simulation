import airsim
import time
import math
from swarm_state import swarm

class Warrior:
    def __init__(self):
        try:
            print("Warrior: Connecting to AirSim...")
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            print("Warrior: Connected!")
        except Exception as e:
            print(f"❌ WARRIOR ERROR: {e}")
            raise
        
    def run(self):
        swarm.log("WARRIOR", "Initializing", "INFO")
        
        self.client.enableApiControl(True, "Warrior1")
        self.client.armDisarm(True, "Warrior1")
        self.client.takeoffAsync(vehicle_name="Warrior1").join()
        
        angle = 0
        last_patrol = None
        
        while not swarm.kamikaze_deployed:
            try:
                # Get CURRENT patrol settings
                current_patrol = swarm.get_patrol_area()
                cx, cy, radius = current_patrol
                
                # DEBUG: Print every loop
                print(f"[DEBUG] Current patrol: {current_patrol}, Last: {last_patrol}")
                
                # Check if changed
                if last_patrol != current_patrol:
                    swarm.log("WARRIOR", f"🎯 PATROL CHANGE: ({cx:.0f}, {cy:.0f}) R={radius:.0f}m", "WARNING")
                    last_patrol = current_patrol
                    angle = 0
                
                # Calculate waypoint
                x = cx + radius * math.cos(math.radians(angle))
                y = cy + radius * math.sin(math.radians(angle))
                
                # Move
                swarm.log("WARRIOR", f"→ ({x:.0f}, {y:.0f})", "INFO")
                self.client.moveToPositionAsync(x, y, -15, 8, vehicle_name="Warrior1").join()
                
                # Scan
                for i in range(3):
                    pos = self.client.simGetVehiclePose("Warrior1").position
                    swarm.warrior_report((pos.x_val, pos.y_val, pos.z_val))
                    time.sleep(1)
                
                angle = (angle + 60) % 360
                
            except Exception as e:
                swarm.log("WARRIOR", f"Error: {e}", "WARNING")
                time.sleep(1)
        
        swarm.log("WARRIOR", "RTB", "WARNING")
        self.client.hoverAsync(vehicle_name="Warrior1").join()

def run():
    warrior = Warrior()
    warrior.run()