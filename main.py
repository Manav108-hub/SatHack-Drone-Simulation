# main.py
import os
import sys
import time
import logging
import traceback
import threading
from threading import Thread, Event

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("MAIN")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] %(message)s", "%H:%M:%S"))
logger.addHandler(ch)

import datacenter
import queen
import warriors
import kamikaze
from swarm_state import swarm

# Global stop event for threads
stop_event = Event()

# Global thread references
drone_threads = {
    'queen': None,
    'warrior': None,
    'kamikaze': None
}

def run_with_catch(func, name, stop_event):
    """Run function with error handling and stop event"""
    try:
        logger.info(f"Thread starting: {name}")
        # Pass stop_event to function if it supports it
        func()
        logger.info(f"Thread finished normally: {name}")
    except Exception as e:
        if not stop_event.is_set():
            logger.error(f"Unhandled exception in {name}: {e}")
            traceback.print_exc()

def start_drone_threads():
    """Start all drone threads"""
    global drone_threads, stop_event
    
    stop_event.clear()
    
    # Start Queen
    drone_threads['queen'] = Thread(
        target=lambda: run_with_catch(queen.run, "Queen", stop_event),
        daemon=True,
        name="Queen"
    )
    drone_threads['queen'].start()
    time.sleep(2)
    
    # Start Warrior
    drone_threads['warrior'] = Thread(
        target=lambda: run_with_catch(warriors.run, "Warrior", stop_event),
        daemon=True,
        name="Warrior"
    )
    drone_threads['warrior'].start()
    time.sleep(2)
    
    # Start Kamikaze
    drone_threads['kamikaze'] = Thread(
        target=lambda: run_with_catch(kamikaze.run, "Kamikaze", stop_event),
        daemon=True,
        name="Kamikaze"
    )
    drone_threads['kamikaze'].start()
    
    logger.info("All drone threads started")

def stop_drone_threads():
    """Stop all drone threads gracefully"""
    global drone_threads, stop_event
    
    logger.info("Stopping drone threads...")
    stop_event.set()
    
    # Wait for threads to finish (with timeout)
    for name, thread in drone_threads.items():
        if thread and thread.is_alive():
            logger.info(f"Waiting for {name} thread to stop...")
            thread.join(timeout=3)
            if thread.is_alive():
                logger.warning(f"{name} thread did not stop gracefully")
    
    logger.info("All drone threads stopped")

def press_backspace_in_airsim():
    """Simulate backspace press in AirSim to reset drones"""
    try:
        logger.info("Sending backspace to AirSim to reset drones...")
        
        # Use pyautogui or keyboard library to press backspace
        try:
            import keyboard
            keyboard.press_and_release('backspace')
            logger.info("Backspace sent successfully (keyboard library)")
        except ImportError:
            try:
                import pyautogui
                pyautogui.press('backspace')
                logger.info("Backspace sent successfully (pyautogui)")
            except ImportError:
                logger.warning("Neither 'keyboard' nor 'pyautogui' installed. Manual reset required.")
                logger.warning("Install with: pip install keyboard")
                return False
        
        return True
    except Exception as e:
        logger.error(f"Failed to send backspace: {e}")
        return False

def reset_mission():
    """Complete mission reset: Stop threads, reset AirSim, restart threads"""
    logger.info("="*70)
    logger.info("MISSION RESET INITIATED")
    logger.info("="*70)
    
    # Step 1: Stop drone threads
    stop_drone_threads()
    time.sleep(1)
    
    # Step 2: Reset swarm state
    swarm.reset_mission()
    time.sleep(0.5)
    
    # Step 3: Press backspace in AirSim
    if press_backspace_in_airsim():
        logger.info("Waiting for AirSim to reset drones...")
        time.sleep(5)  # Wait for AirSim reset
    else:
        logger.warning("Could not auto-reset AirSim. Please press Backspace manually.")
        logger.warning("Waiting 5 seconds...")
        time.sleep(5)
    
    # Step 4: Restart drone threads
    logger.info("Restarting drone threads...")
    start_drone_threads()
    
    logger.info("="*70)
    logger.info(f"MISSION #{swarm.mission_count} READY")
    logger.info("="*70)

# Make reset function available to datacenter
datacenter.reset_mission_handler = reset_mission

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

    # Start datacenter web server (runs forever)
    try:
        web_thread = Thread(
            target=lambda: datacenter.run_web(host='0.0.0.0', port=5000),
            daemon=True,
            name="Datacenter-Web"
        )
        web_thread.start()
        logger.info("🌐 Datacenter (web UI) thread started (http://localhost:5000)")
    except Exception as e:
        logger.error(f"Failed to start datacenter thread: {e}")
        traceback.print_exc()
        return

    time.sleep(1.2)

    # Start drone threads
    start_drone_threads()

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
            
            # Check if all drone threads are dead (mission complete)
            all_dead = all(
                not thread or not thread.is_alive() 
                for thread in drone_threads.values()
            )
            
            if all_dead and not swarm.kamikaze_deployed:
                logger.warning("All drone threads stopped unexpectedly!")
                break
                
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received — shutting down.")
    except Exception as e:
        logger.error(f"Main loop error: {e}")
        traceback.print_exc()
    finally:
        print("\n" + "="*70)
        print("✅ SHUTTING DOWN")
        print("="*70 + "\n")

if __name__ == '__main__':
    main()