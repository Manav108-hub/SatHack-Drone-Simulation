import airsim
import time

client = airsim.MultirotorClient()
client.confirmConnection()
print("✅ Connected!")

# Enable all
for drone in ["Queen", "Warrior1", "Kamikaze1"]:
    client.enableApiControl(True, drone)
    client.armDisarm(True, drone)

# Takeoff
client.takeoffAsync(vehicle_name="Queen")
client.takeoffAsync(vehicle_name="Warrior1")
client.takeoffAsync(vehicle_name="Kamikaze1").join()

print("✅ All airborne!")

# Formation
client.moveToPositionAsync(0, 0, -20, 5, vehicle_name="Queen")
client.moveToPositionAsync(30, 0, -20, 5, vehicle_name="Warrior1")
client.moveToPositionAsync(-30, 0, -20, 5, vehicle_name="Kamikaze1").join()

print("✅ Formation complete! Hovering 10 seconds...")
time.sleep(10)

# Land
client.landAsync(vehicle_name="Queen")
client.landAsync(vehicle_name="Warrior1")
client.landAsync(vehicle_name="Kamikaze1").join()

print("✅ Test complete!")