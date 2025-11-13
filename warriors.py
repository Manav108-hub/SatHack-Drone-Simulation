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
        
        # Store last known patrol settings to detect changes
        self.last_center_x = None
        self.last_center_y = None
        self.last_radius = None
        
    def check_patrol_update(self):
        """Check if patrol area was updated"""
        if (self.last_center_x != swarm.patrol_center_x or 
            self.last_center_y != swarm.patrol_center_y or 
            self.last_radius != swarm.patrol_radius):
            
            self.last_center_x = swarm.patrol_center_x
            self.last_center_y = swarm.patrol_center_y
            self.last_radius = swarm.patrol_radius
            
            swarm.log("WARRIOR", f"Patrol updated: ({self.last_center_x:.0f}, {self.last_center_y:.0f}) R={self.last_radius:.0f}m", "WARNING")
            return True
        return False
        
    def scan_position(self, x, y):
        """Move to position and scan for 3 seconds"""
        swarm.log("WARRIOR", f"→ ({x:.0f}, {y:.0f})", "INFO")
        
        # Move and WAIT for completion
        self.client.moveToPositionAsync(x, y, -15, 8, vehicle_name="Warrior1").join()
        
        # Scan for 3 seconds
        for i in range(3):
            pos = self.client.simGetVehiclePose("Warrior1").position
            swarm.warrior_report((pos.x_val, pos.y_val, pos.z_val))
            time.sleep(1)
        
    def run(self):
        swarm.log("WARRIOR", "Initializing", "INFO")
        
        self.client.enableApiControl(True, "Warrior1")
        self.client.armDisarm(True, "Warrior1")
        self.client.takeoffAsync(vehicle_name="Warrior1").join()
        
        # Initialize patrol settings
        self.check_patrol_update()
        
        swarm.log("WARRIOR", f"Patrol: ({swarm.patrol_center_x:.0f}, {swarm.patrol_center_y:.0f}) R={swarm.patrol_radius:.0f}m", "INFO")
        
        angle = 0
        
        while not swarm.kamikaze_deployed:
            try:
                # Check for patrol updates EVERY loop
                self.check_patrol_update()
                
                cx = swarm.patrol_center_x
                cy = swarm.patrol_center_y
                radius = swarm.patrol_radius
                
                # Calculate patrol waypoint
                x = cx + radius * math.cos(math.radians(angle))
                y = cy + radius * math.sin(math.radians(angle))
                
                # Scan this position
                self.scan_position(x, y)
                
                # Next waypoint (60 degrees = 6 points per circle)
                angle = (angle + 60) % 360
                
            except Exception as e:
                swarm.log("WARRIOR", f"Error: {e}", "WARNING")
                time.sleep(1)
        
        swarm.log("WARRIOR", "RTB", "WARNING")
        self.client.hoverAsync(vehicle_name="Warrior1").join()

def run():
    warrior = Warrior()
    warrior.run()