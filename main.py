# main.py
import os
import sys
import time
import logging
from threading import Thread
import traceback

# Setup a small logger for main
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("MAIN")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] %(message)s", "%H:%M:%S"))
logger.addHandler(ch)

# Import the modules (they should exist in the same folder)
# datacenter defines run_web(host, port)
import datacenter
import queen
import warriors
import kamikaze

def run_with_catch(func, name):
    try:
        logger.info(f"Thread starting: {name}")
        func()
        logger.info(f"Thread finished normally: {name}")
    except Exception as e:
        logger.error(f"Unhandled exception in {name}: {e}")
        traceback.print_exc()

def start_thread(target, name, daemon=True, delay=0):
    t = Thread(target=lambda: run_with_catch(target, name), daemon=daemon, name=name)
    t.start()
    if delay:
        time.sleep(delay)
    return t

def main():
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

    # Start datacenter web UI in a daemon thread (so it shares the same memory & swarm instance)
    try:
        web_thread = Thread(target=lambda: datacenter.run_web(host='0.0.0.0', port=5000),
                            daemon=True, name="Datacenter-Web")
        web_thread.start()
        logger.info("🌐 Datacenter (web UI) thread started (http://localhost:5000)")
    except Exception as e:
        logger.error(f"Failed to start datacenter thread: {e}")
        traceback.print_exc()

    # Small delay so web UI begins before drones (helps when UI polls immediately)
    time.sleep(1.2)

    # Start Queen, Warrior, Kamikaze as daemon threads using the run() entrypoints
    t1 = start_thread(queen.run, "Queen", daemon=True)
    time.sleep(2)
    t2 = start_thread(warriors.run, "Warrior", daemon=True)
    time.sleep(2)
    t3 = start_thread(kamikaze.run, "Kamikaze", daemon=True)

    try:
        # Keep main thread alive while any child threads are running
        while True:
            alive = any(t.is_alive() for t in [web_thread, t1, t2, t3] if t is not None)
            if not alive:
                logger.info("All threads have stopped. Exiting.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received — shutting down.")
    except Exception as e:
        logger.error(f"Main loop error: {e}")
        traceback.print_exc()
    finally:
        print("\n" + "="*70)
        print("✅ SHUTTING DOWN")
        print("="*70 + "\n")
        # Threads are daemon — process will exit cleanly

if __name__ == '__main__':
    main()
