from threading import Thread
import time
import queen
import warriors
import kamikaze


print("\n" + "="*70)
print("🚁 AUTONOMOUS DRONE SWARM - HIVE INTELLIGENCE")
print("="*70)
print("\n📋 DRONES:")
print("   👑 Queen: AI threat detection (YOLOv8)")
print("   🛸 Warrior: Autonomous patrol")
print("   💥 Kamikaze: Strike on command")
print("="*70 + "\n")

input("Press ENTER to launch...")

print("\n🚀 LAUNCHING IN 3...")
time.sleep(1)
print("2...")
time.sleep(1)
print("1...\n")

# Import after user confirms

# Launch threads
t1 = Thread(target=queen.run, daemon=True)
t2 = Thread(target=warriors.run, daemon=True)
t3 = Thread(target=kamikaze.run, daemon=True)

t1.start()
time.sleep(2)
t2.start()
time.sleep(2)
t3.start()

# Wait for completion
t1.join()
t2.join()
t3.join()

print("\n" + "="*70)
print("✅ MISSION COMPLETE")
print("="*70)