import sys
import airsim
import time

print("=" * 60)
print("🚁 SIMPLE CONNECTION TEST")
print("=" * 60)

try:
    print("\nConnecting...")
    
    # Use different connection method
    client = airsim.MultirotorClient()
    client.confirmConnection()
    
    print("✅ CONNECTED!")
    print(f"AirSim Version: {client.getServerVersion()}")
    
    # Enable and arm
    client.enableApiControl(True, "Queen")
    client.armDisarm(True, "Queen")
    print("✅ Queen armed!")
    
    # Takeoff
    print("\nTaking off...")
    client.takeoffAsync(vehicle_name="Queen").join()
    print("✅ Airborne!")
    
    time.sleep(2)
    
    # Get state
    state = client.getMultirotorState(vehicle_name="Queen")
    print(f"✅ Position: {state.kinematics_estimated.position}")
    
    # Land
    print("\nLanding...")
    client.landAsync(vehicle_name="Queen").join()
    print("✅ Landed!")
    
    print("\n" + "=" * 60)
    print("✅ SUCCESS! READY FOR AI PHASE!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()