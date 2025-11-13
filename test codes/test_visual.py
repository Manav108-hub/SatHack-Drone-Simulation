import airsim
import time

client = airsim.MultirotorClient()
client.confirmConnection()
print("✅ Connected!")

# Enable all drones
for drone in ["Queen", "Warrior1", "Kamikaze1"]:
    client.enableApiControl(True, drone)
    client.armDisarm(True, drone)
    print(f"✅ {drone} ready")

# Set camera to external view
camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0),
                         airsim.to_quaternion(0.3, 0, 0))
client.simSetCameraOrientation("0", camera_pose, vehicle_name="Queen")

# Takeoff all drones
print("\n🚁 Taking off all drones...")
client.takeoffAsync(vehicle_name="Queen")
client.takeoffAsync(vehicle_name="Warrior1")
client.takeoffAsync(vehicle_name="Kamikaze1").join()

time.sleep(3)  # Wait for stabilization
print("✅ All airborne!")

# Move them HIGHER and FARTHER apart (more visible)
print("\n📍 Moving to visible formation...")
client.moveToPositionAsync(0, 0, -20, 5, vehicle_name="Queen")
client.moveToPositionAsync(30, 0, -20, 5, vehicle_name="Warrior1")
client.moveToPositionAsync(-30, 0, -20, 5, vehicle_name="Kamikaze1").join()
print("✅ Formation complete!")

print("\n⏱️ Hovering for 15 seconds...")
print("👀 PRESS F1 to switch to Queen view!")
print("👀 PRESS \\ (backslash) to change camera!")
print("👀 PRESS - (minus) to zoom out!")

for i in range(15, 0, -1):
    print(f"   {i} seconds remaining...")
    time.sleep(1)

# Do a visible maneuver
print("\n🎯 FORMATION DANCE - WATCH THIS!")

# Circle pattern
for angle in range(0, 360, 30):
    import math
    x = 20 * math.cos(math.radians(angle))
    y = 20 * math.sin(math.radians(angle))
    
    client.moveToPositionAsync(0, 0, -20, 3, vehicle_name="Queen")
    client.moveToPositionAsync(x, y, -20, 5, vehicle_name="Warrior1")
    client.moveToPositionAsync(-x, -y, -20, 5, vehicle_name="Kamikaze1")
    
    time.sleep(1)

print("✅ Dance complete!")

# Land all
print("\n🛬 Landing...")
client.landAsync(vehicle_name="Queen")
client.landAsync(vehicle_name="Warrior1")
client.landAsync(vehicle_name="Kamikaze1").join()
print("✅ All landed!")