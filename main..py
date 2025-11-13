from threading import Thread
import time
import sys

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

# Force flush
sys.stdout.flush()

def run_with_catch(func, name):
    try:
        func()
    except Exception as e:
        print(f"ERROR in {name}: {e}")

t1 = Thread(target=lambda: run_with_catch(queen.run, "Queen"), daemon=True)
t2 = Thread(target=lambda: run_with_catch(warriors.run, "Warrior"), daemon=True)
t3 = Thread(target=lambda: run_with_catch(kamikaze.run, "Kamikaze"), daemon=True)

t1.start()
time.sleep(3)
t2.start()
time.sleep(3)
t3.start()

# Keep main alive
try:
    while t1.is_alive() or t2.is_alive() or t3.is_alive():
        time.sleep(1)
except KeyboardInterrupt:
    print("\n\n⚠️  Interrupted by user")

print("\n" + "="*70)
print("✅ MISSION COMPLETE")
print("="*70)