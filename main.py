# main.py (launcher)
from threading import Thread
import time
import sys

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

sys.stdout.flush()

# modules must be present in same folder: queen.py, warriors.py, kamikaze.py
import queen
import warriors
import kamikaze

# import datacenter (Flask UI)
try:
    import datacenter
    have_datacenter = True
except Exception as e:
    print("⚠️ Warning: could not import datacenter.py (Flask UI). Run UI separately if needed.")
    print("   Import error:", e)
    have_datacenter = False

def run_with_catch(func, name):
    try:
        func()
    except Exception as e:
        print(f"ERROR in {name}: {e}")

threads = []

# Start the datacenter (web UI) in a daemon thread so it shares swarm_state
if have_datacenter:
    t_web = Thread(target=lambda: run_with_catch(lambda: datacenter.run_web(host='0.0.0.0', port=5000), "Datacenter"), daemon=True)
    t_web.start()
    threads.append(t_web)
    print("🌐 Datacenter (web UI) thread started (http://localhost:5000)")

# Start queen/warrior/kamikaze threads
t1 = Thread(target=lambda: run_with_catch(queen.run, "Queen"), daemon=True)
t2 = Thread(target=lambda: run_with_catch(warriors.run, "Warrior"), daemon=True)
t3 = Thread(target=lambda: run_with_catch(kamikaze.run, "Kamikaze"), daemon=True)

t1.start()
time.sleep(1)
t2.start()
time.sleep(1)
t3.start()

threads.extend([t1, t2, t3])

try:
    # Keep main alive while threads run
    while any(t.is_alive() for t in threads):
        time.sleep(0.5)
except KeyboardInterrupt:
    print("\n\n⚠️  Interrupted")

print("\n" + "="*70)
print("✅ MISSION COMPLETE")
print("="*70)
