# datacenter.py
import os
import time
import traceback
import logging
from logging.handlers import RotatingFileHandler
import json
import sys  # ← ADD THIS

from flask import Flask, Response, render_template_string, jsonify, request
import airsim
import numpy as np
import cv2

from swarm_state import swarm

# ----------------------------
# Logging for datacenter
# ----------------------------
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("DATACENTER")
logger.setLevel(logging.DEBUG)
fh = RotatingFileHandler(os.path.join(LOG_DIR, "datacenter.log"), maxBytes=2_000_000, backupCount=3)
fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"))
logger.addHandler(fh)

# ✅ FIX: UTF-8 encoding
ch = logging.StreamHandler(sys.stdout)
if hasattr(ch.stream, 'reconfigure'):
    ch.stream.reconfigure(encoding='utf-8')
ch.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] %(message)s", "%H:%M:%S"))
logger.addHandler(ch)
logger.propagate = False

app = Flask(__name__)

# One AirSim client per drone name (reused)
clients = {}


def get_client(drone: str):
    """
    Return a cached AirSim client for a drone name.
    Robust to connection errors.
    """
    if drone in clients and clients[drone] is not None:
        return clients[drone]

    try:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        clients[drone] = client
        logger.info(f"Connected AirSim client for {drone}")
        return client
    except Exception as e:
        logger.warning(f"Could not create AirSim client for {drone}: {e}")
        clients[drone] = None
        return None


# ============================================
#  FRAME GRABBER & STREAM GENERATOR
# ============================================
def get_frame_bytes(drone: str, feed_style="normal"):
    """
    Safely fetches a single camera frame.
    Never crashes if AirSim is unavailable.
    Returns JPEG bytes or None.
    """
    try:
        client = get_client(drone)
        if client is None:
            return None

        # # ✅ ADD THIS: Check if drone is actually initialized
        # try:
        #     state = client.getMultirotorState(vehicle_name=drone)
        #     if not state.landed_state:  # If state is invalid
        #         return None
        # except:
        #     pass

        responses = client.simGetImages(
            [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)],
            vehicle_name=drone,
        )

        if not responses or len(responses[0].image_data_uint8) == 0:
            return None

        img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img = img1d.reshape(responses[0].height, responses[0].width, 3)
        img = np.copy(img)

        # FEED VISUAL MODES
        if feed_style == "thermal":
            img = cv2.applyColorMap(img, cv2.COLORMAP_AUTUMN)
        elif feed_style == "nightvision":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            img[:, :, 1] = np.clip(img[:, :, 1] * 1.4, 0, 255).astype(np.uint8)

        # OVERLAY TELEMETRY (SAFE)
        try:
            cv2.rectangle(img, (0, 0), (350, 90), (0, 0, 0), -1)
            cv2.putText(
                img,
                drone,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 200, 0),
                2,
            )

            try:
                pose = client.simGetVehiclePose(vehicle_name=drone)
                pos = pose.position
                txt = f"X:{pos.x_val:.1f} Y:{pos.y_val:.1f} Z:{pos.z_val:.1f}"
                cv2.putText(
                    img,
                    txt,
                    (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 255, 200),
                    1,
                )
            except Exception:
                pass
        except Exception:
            pass

        # Encode JPEG
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes()
    except Exception as e:
        logger.debug(f"[FRAME ERROR] {drone}: {e}")
        return None


def gen_stream(drone: str, feed_style="normal"):
    """
    MJPEG Generator — ALWAYS outputs a frame.
    Never breaks stream job.
    """
    failures = 0
    is_kamikaze = "kamikaze" in drone.lower()
    
    while True:
        try:
            frame = get_frame_bytes(drone, feed_style)

            if frame:
                failures = 0
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            else:
                failures += 1
                blank = np.zeros((240, 320, 3), dtype=np.uint8)
                
                # ✅ Special message for kamikazes
                if is_kamikaze:
                    cv2.putText(blank, drone, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 140, 0), 2)
                    cv2.putText(blank, "STANDBY MODE", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                    cv2.putText(blank, "Waiting for orders...", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                else:
                    cv2.putText(blank, drone, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.putText(blank, "Waiting for camera...", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                
                _, buf = cv2.imencode(".jpg", blank)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                )

                if failures > 5:
                    time.sleep(1)

            time.sleep(0.12)
        except GeneratorExit:
            break
        except Exception as e:
            logger.debug(f"[STREAM ERROR] {drone}: {e}")
            time.sleep(0.5)


# --------------------------------------------
#  DRONE REGISTRATION
# --------------------------------------------
registered_drones = {
    "queen": [],
    "warriors": [],
    "kamikazes": [],
    "others": [],
}


def build_registered_drones(settings_path="settings.json"):
    """
    Loads settings.json and categorizes drones.
    Fully safe — always produces valid lists.
    """
    global registered_drones

    try:
        if not os.path.isfile(settings_path):
            logger.warning("settings.json not found — using default drones")
            registered_drones = {
                "queen": ["Queen"],
                "warriors": ["Warrior1"],
                "kamikazes": ["Kamikaze1"],
                "others": [],
            }
            return

        with open(settings_path, "r") as f:
            cfg = json.load(f)

        vehicles = cfg.get("Vehicles", {})
        reg = {"queen": [], "warriors": [], "kamikazes": [], "others": []}

        for name in vehicles.keys():
            lname = name.lower()
            if "queen" in lname:
                reg["queen"].append(name)
            elif "warrior" in lname:
                reg["warriors"].append(name)
            elif "kamikaze" in lname:
                reg["kamikazes"].append(name)
            else:
                reg["others"].append(name)

        if not reg["queen"]:
            reg["queen"] = ["Queen"]
        if not reg["warriors"]:
            reg["warriors"] = ["Warrior1"]
        if not reg["kamikazes"]:
            reg["kamikazes"] = ["Kamikaze1"]

        registered_drones = reg
        logger.info(f"REGISTERED DRONES = {registered_drones}")
    except Exception as e:
        logger.error(f"[SETTINGS ERROR] {e}")
        traceback.print_exc()
        registered_drones = {
            "queen": ["Queen"],
            "warriors": ["Warrior1"],
            "kamikazes": ["Kamikaze1"],
            "others": [],
        }


# DON'T auto-call on import - let run_web() handle it
# build_registered_drones()


# ----------------------------
# Flask endpoints
# ----------------------------
@app.route("/feed/<drone>")
def feed_drone(drone):
    style = "normal"
    dl = drone.lower()
    if dl.startswith("queen"):
        style = "thermal"
    elif "warrior" in dl:
        style = "nightvision"
    return Response(
        gen_stream(drone, feed_style=style),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/queen")
def queen_feed():
    name = registered_drones["queen"][0] if registered_drones["queen"] else "Queen"
    return feed_drone(name)


@app.route("/warrior")
def warrior_feed():
    name = (
        registered_drones["warriors"][0]
        if registered_drones["warriors"]
        else "Warrior1"
    )
    return feed_drone(name)


@app.route("/kamikaze")
def kamikaze_feed():
    name = (
        registered_drones["kamikazes"][0]
        if registered_drones["kamikazes"]
        else "Kamikaze1"
    )
    return feed_drone(name)


@app.route("/drones")
def drones_list():
    try:
        return jsonify(registered_drones)
    except Exception as e:
        logger.exception("Drones list error")
        return jsonify({"error": str(e)}), 500


@app.route("/logs")
def logs():
    try:
        return jsonify(swarm.get_logs(200))
    except Exception as e:
        logger.exception("Error returning logs")
        return jsonify([])


def safe_pose_to_list(pose):
    try:
        p = pose.position
        return [float(p.x_val), float(p.y_val), float(p.z_val)]
    except Exception:
        return None


def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(x) for x in obj]
    if isinstance(obj, tuple):
        return [make_serializable(x) for x in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


@app.route("/status")
def status():
    try:
        # Patrol area
        cx, cy, r = 0.0, 0.0, 30.0
        try:
            area = swarm.get_patrol_area()
            if isinstance(area, (list, tuple)) and len(area) >= 3:
                cx, cy, r = float(area[0]), float(area[1]), float(area[2])
        except Exception:
            pass

        warrior_status = swarm.get_warrior_status()

        # Queen pose - use cached position from swarm instead of AirSim
        queen_pose = None
        try:
            if swarm.last_queen_pos and swarm.last_queen_pos[0] is not None:
                queen_pose = [float(x) for x in swarm.last_queen_pos]
            else:
                # Only fetch from AirSim if cache is empty (startup)
                q_name = registered_drones["queen"][0] if registered_drones["queen"] else "Queen"
                q_client = get_client(q_name)
                if q_client:
                    pose = q_client.simGetVehiclePose(vehicle_name=q_name)
                    queen_pose = safe_pose_to_list(pose)
        except Exception as e:
            logger.debug(f"Queen pose error: {e}")
            queen_pose = None

        # Active threat
        active_threat = None
        if getattr(swarm, "active_threat", None):
            at = (
                swarm.active_threat.copy()
                if isinstance(swarm.active_threat, dict)
                else {"class": str(swarm.active_threat)}
            )
            wp = at.get("world_pos")
            if wp is not None:
                try:
                    at["world_pos"] = [float(wp[0]), float(wp[1])]
                except Exception:
                    at["world_pos"] = None
            active_threat = {
                k: v
                for k, v in at.items()
                if isinstance(v, (str, int, float, list, bool, type(None)))
            }

        status_data = {
            "threat_level": str(getattr(swarm, "threat_level", "GREEN")),
            "pending_permission": bool(getattr(swarm, "pending_permission", False)),
            "active_threat": active_threat,
            "warrior_status": warrior_status,
            "kamikaze_deployed": bool(getattr(swarm, "kamikaze_deployed", False)),
            "queen_scans": int(getattr(swarm, "queen_scans", 0)),
            "patrol_area": {
                "center_x": float(cx),
                "center_y": float(cy),
                "radius": float(r),
            },
            "patrol_relative": bool(getattr(swarm, "patrol_relative_to_queen", False)),
            "last_patrol_update": float(getattr(swarm, "last_patrol_update", 0.0)),
            "last_warrior_update": float(getattr(swarm, "last_warrior_update", 0.0)),
            "queen_pose": queen_pose,
            "registered_drones": registered_drones,
            "ai_scan_count": int(getattr(swarm, "ai_scan_count", 0)) if hasattr(swarm, "ai_scan_count") else 0,
        }

        return jsonify(make_serializable(status_data))
    except Exception as e:
        logger.exception("Status error")
        return jsonify({"error": str(e)}), 500


@app.route("/approve", methods=["POST"])
def approve():
    try:
        swarm.user_response = True
        swarm.pending_permission = False
        swarm.log("USER", "AUTHORIZED", "CRITICAL")
        return jsonify({"status": "approved"})
    except Exception as e:
        logger.exception("Approve error")
        return jsonify({"error": str(e)}), 500


@app.route("/deny", methods=["POST"])
def deny():
    try:
        swarm.user_response = False
        swarm.pending_permission = False
        swarm.log("USER", "DENIED", "WARNING")
        return jsonify({"status": "denied"})
    except Exception as e:
        logger.exception("Deny error")
        return jsonify({"error": str(e)}), 500


@app.route("/spawn_threat", methods=["POST"])
def spawn_threat():
    try:
        data = request.json or {}
        x = float(data.get("x", 10))
        y = float(data.get("y", 10))
        threat_type = data.get("type", "person")
        threat = {
            "class": threat_type,
            "confidence": float(data.get("confidence", 1.0)),
            "world_pos": (x, y),
            "timestamp": time.time(),
        }
        swarm.add_threat(threat)
        logger.info(f"Manual threat spawned: {threat_type} at ({x}, {y})")
        return jsonify(
            {
                "status": "spawned",
                "threat": {
                    "class": threat_type,
                    "confidence": threat["confidence"],
                    "world_pos": [x, y],
                },
            }
        )
    except Exception as e:
        logger.exception("Spawn threat error")
        return jsonify({"error": str(e)}), 500


@app.route("/set_patrol", methods=["POST"])
def set_patrol():
    try:
        data = request.json or {}
        cx = float(data.get("x", 0))
        cy = float(data.get("y", 0))
        radius = float(data.get("radius", 30))
        relative = bool(data.get("relative", False))
        swarm.set_patrol_area(cx, cy, radius, relative=relative)
        logger.info(
            f"Patrol updated via UI: ({cx}, {cy}) R={radius} relative={relative}"
        )
        return jsonify(
            {
                "center_x": float(cx),
                "center_y": float(cy),
                "radius": float(radius),
                "relative": bool(relative),
            }
        )
    except Exception as e:
        logger.exception("Set patrol error")
        return jsonify({"error": str(e)}), 500


@app.route("/expand_patrol", methods=["POST"])
def expand_patrol():
    """Expand patrol radius by multiplier (body: {'mult':1.5})"""
    try:
        data = request.json or {}
        mult = float(data.get("mult", 1.5))
        new_r = swarm.expand_patrol(mult)
        return jsonify({"new_radius": float(new_r)})
    except Exception as e:
        logger.exception("Expand patrol error")
        return jsonify({"error": str(e)}), 500


@app.route("/distribute", methods=["POST"])
def distribute():
    """Return N patrol points to distribute to units (body: {'n':3,'spread':1.0})"""
    try:
        data = request.json or {}
        n = int(data.get("n", 1))
        spread = float(data.get("spread", 1.0))
        pts = swarm.distribute_patrol_points(n, spread)
        pts_out = [[float(x), float(y)] for (x, y) in pts]
        return jsonify({"points": pts_out})
    except Exception as e:
        logger.exception("Distribute error")
        return jsonify({"error": str(e)}), 500


@app.route("/get_patrol")
def get_patrol():
    try:
        area = swarm.get_patrol_area()
        if isinstance(area, (list, tuple)) and len(area) >= 3:
            cx, cy, r = float(area[0]), float(area[1]), float(area[2])
        else:
            cx, cy, r = 0.0, 0.0, 30.0
        return jsonify(
            {
                "center_x": cx,
                "center_y": cy,
                "radius": r,
                "relative": swarm.patrol_relative_to_queen,
            }
        )
    except Exception as e:
        logger.exception("Get patrol error")
        return jsonify({"error": str(e)}), 500


@app.route("/set_queen_pose", methods=["POST"])
def set_queen_pose():
    try:
        data = request.json or {}
        x = float(data.get("x", 0))
        y = float(data.get("y", 0))
        z = float(data.get("z", -20))
        q_name = registered_drones["queen"][0] if registered_drones["queen"] else "Queen"
        client = get_client(q_name)
        if client is None:
            return jsonify({"error": "No AirSim client for Queen"}), 500
        pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0, 0, 0))
        client.simSetVehiclePose(pose, True, vehicle_name=q_name)
        swarm.log(
            "SYSTEM", f"Queen teleported to ({x:.1f}, {y:.1f}, {z:.1f})", "INFO"
        )
        return jsonify({"status": "ok", "x": float(x), "y": float(y), "z": float(z)})
    except Exception as e:
        logger.exception("Set queen pose error")
        return jsonify({"error": str(e)}), 500


@app.route("/move_queen", methods=["POST"])
def move_queen():
    try:
        data = request.json or {}
        x = float(data.get("x", 0))
        y = float(data.get("y", 0))
        z = float(data.get("z", -20))
        speed = float(data.get("speed", 5.0))
        q_name = registered_drones["queen"][0] if registered_drones["queen"] else "Queen"
        client = get_client(q_name)
        if client is None:
            return jsonify({"error": "No AirSim client for Queen"}), 500
        client.enableApiControl(True, q_name)
        client.armDisarm(True, q_name)
        client.moveToPositionAsync(x, y, z, speed, vehicle_name=q_name)
        swarm.log(
            "SYSTEM",
            f"Queen moving to ({x:.1f}, {y:.1f}, {z:.1f}) speed={speed}",
            "INFO",
        )
        return jsonify(
            {
                "status": "moving",
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "speed": float(speed),
            }
        )
    except Exception as e:
        logger.exception("Move queen error")
        return jsonify({"error": str(e)}), 500


@app.route("/favicon.ico")
def favicon():
    return "", 204


# ----------------------------
# UI Template
# ----------------------------
UI_TEMPLATE = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>🚁 AUTONOMOUS DRONE COMMAND CENTER</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    :root { --bg:#060606; --accent:#00ff66; --warn:#ff6600; --danger:#ff4444; --muted:#888; --panel:#0f0f0f; }
    html,body{height:100%;margin:0;background:linear-gradient(#040404,#070707);color:var(--accent);font-family: "Consolas","Courier New",monospace;}
    .header{display:flex;justify-content:space-between;align-items:center;padding:10px 16px;border-bottom:2px solid rgba(0,255,102,0.06);}
    h1{margin:0;font-size:1.1rem;color:var(--accent);}
    .stat{background:rgba(0,0,0,0.4);padding:6px 8px;border-radius:6px;font-size:0.9rem;color:var(--accent);}
    .container{display:grid;grid-template-columns:1fr 420px;gap:12px;padding:12px;height:calc(100% - 64px);box-sizing:border-box;}
    .left{display:flex;flex-direction:column;gap:12px;}
    .feeds{display:flex;gap:8px;}
    .feed{flex:1;border:2px solid rgba(255,255,255,0.03);border-radius:6px;height:220px;overflow:hidden;background:#000;position:relative;transition:all 0.3s cubic-bezier(0.4,0,0.2,1);cursor:zoom-in;}
    .feed img{width:100%;height:100%;object-fit:cover;display:block;}
    .feed-label{position:absolute;top:6px;left:8px;background:rgba(0,0,0,0.6);padding:6px 8px;border-radius:4px;font-size:0.8rem;color:var(--warn);z-index:10;}
    .panel{background:var(--panel);border:2px solid rgba(0,255,102,0.06);padding:10px;border-radius:8px;color:var(--accent);}
    .grid-feeds{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:8px;}
    .logs{height:260px;overflow:auto;background:#060606;padding:8px;border-radius:6px;border:1px solid rgba(255,255,255,0.02);color:#bfffbf;font-size:0.9rem;}
    .log-entry{font-family:monospace;padding:4px 0;border-bottom:1px dashed rgba(255,255,255,0.02);}
    .controls{display:flex;gap:8px;margin-top:8px;}
    input[type=number]{width:120px;padding:6px;background:#000;color:var(--accent);border:1px solid rgba(0,255,102,0.06);border-radius:6px;}
    label{color:#cfe8cf;font-size:0.9rem;}
    button.btn{padding:8px 10px;border-radius:6px;border:1px solid rgba(255,255,255,0.03);background:transparent;color:var(--accent);cursor:pointer;}
    .small{font-size:0.85rem;color:var(--muted);}
    
    /* Authorization Panel */
    .authorization-panel {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: rgba(26, 0, 0, 0.98);
        border: 4px solid #ff0000;
        border-radius: 12px;
        padding: 24px;
        z-index: 99999;
        min-width: 500px;
        box-shadow: 0 0 80px rgba(255, 0, 0, 0.8);
        animation: urgentPulse 0.8s infinite;
        display: none;
    }
    .authorization-panel.active {
        display: block !important;
    }
    @keyframes urgentPulse {
        0%, 100% { 
            box-shadow: 0 0 80px rgba(255, 0, 0, 0.8);
            border-color: #ff0000;
        }
        50% { 
            box-shadow: 0 0 120px rgba(255, 0, 0, 1);
            border-color: #ff6666;
        }
    }
    .threat-info-box {
        background: #000;
        padding: 16px;
        border: 2px solid #ff0000;
        border-radius: 8px;
        margin: 16px 0;
        font-size: 1.1rem;
    }
    .threat-info-box p {
        margin: 8px 0;
        color: #ffff00;
    }
    .threat-info-box strong {
        color: #ff0000;
    }
    .auth-buttons {
        display: flex;
        gap: 16px;
        margin-top: 20px;
    }
    .auth-buttons button {
        flex: 1;
        padding: 16px;
        font-size: 1.2rem;
        font-weight: bold;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s;
        text-transform: uppercase;
        font-family: "Consolas","Courier New",monospace;
    }
    .btn-auth-approve {
        background: #001a00;
        color: #00ff00;
        border: 3px solid #00ff00;
    }
    .btn-auth-approve:hover {
        background: #00ff00;
        color: #000;
        transform: scale(1.05);
    }
    .btn-auth-deny {
        background: #1a0000;
        color: #ff0000;
        border: 3px solid #ff0000;
    }
    .btn-auth-deny:hover {
        background: #ff0000;
        color: #000;
        transform: scale(1.05);
    }
    
    /* Zoom styles */
    .feed.zoomed, .feed.zoomed img {
        box-shadow: 0 0 60px rgba(0,255,102,0.6) !important;
        border-color: rgba(0,255,102,0.8) !important;
        cursor: zoom-out;
    }
    .feed.zoomed img {
        object-fit: contain !important;
        width: 100% !important;
        height: 100% !important;
    }
  </style>
</head>
<body>
  <div class="header">
    <h1>🚁 AUTONOMOUS DRONE COMMAND CENTER</h1>
    <div style="display:flex;gap:10px;align-items:center;">
      <div class="stat">Scans: <span id="scans">0</span></div>
      <div class="stat">Threat: <span id="threatLevel">GREEN</span></div>
    </div>
  </div>

  <!-- AUTHORIZATION PANEL (OVERLAY) -->
  <div class="authorization-panel" id="authPanel">
      <h2 style="color: #ff0000; text-align: center; margin-top: 0; font-size: 1.8rem;">
          ⚠️ STRIKE AUTHORIZATION REQUIRED ⚠️
      </h2>
      <div class="threat-info-box" id="authThreatInfo">
          <p><strong>⚠️ IMMEDIATE ACTION REQUIRED ⚠️</strong></p>
          <p><strong>Threat Type:</strong> <span id="authThreatType">Unknown</span></p>
          <p><strong>Location:</strong> <span id="authThreatPos">Unknown</span></p>
          <p><strong>Confidence:</strong> <span id="authThreatConf">0%</span></p>
          <p style="color: #ff0000; font-weight: bold; font-size: 1.3rem; margin-top: 16px;">
              ⏰ YOU HAVE 15 SECONDS TO DECIDE!
          </p>
      </div>
      <div class="auth-buttons">
          <button class="btn-auth-approve" id="btnAuthApprove">✅ AUTHORIZE STRIKE</button>
          <button class="btn-auth-deny" id="btnAuthDeny">❌ DENY STRIKE</button>
      </div>
  </div>

  <div class="container">
    <div class="left">
      <div class="panel">
        <h3 style="margin:4px 0;">Queen Feed</h3>
        <div class="feed" style="height:260px;">
          <div class="feed-label">🔥 QUEEN (THERMAL)</div>
          <img id="imgQueen" src="/queen" alt="Queen feed">
        </div>
      </div>

      <div class="panel">
        <h3 style="margin:4px 0;">Unit Feeds</h3>
        <div id="unitGrid" class="grid-feeds"></div>
        <div style="margin-top:8px;">
          <label>Auto-refresh feeds</label>
          <button id="btnRefreshFeeds" class="btn">Refresh Drone List
          </button>
        </div>
      </div>

      <div class="panel">
        <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
          <div>
            <h3 style="margin:0 0 6px 0;">Patrol Controls</h3>
            <div>Patrol Center: <span id="patrolCenter">0,0</span></div>
            <div>Radius: <span id="patrolRadius">30</span> m</div>
            <div style="margin-top:6px;">
              <label>Expand Radius x</label><br>
              <input id="expandMult" type="number" value="1.5" step="0.1" min="1">
              <button id="btnExpand" class="btn">Expand Patrol</button>
            </div>
            <div style="margin-top:6px;">
              <label>Distribute to N units</label><br>
              <input id="distN" type="number" value="3" step="1" min="1">
              <input id="distSpread" type="number" value="1.0" step="0.1" min="0.1">
              <button id="btnDist" class="btn">Get Patrol Points</button>
            </div>
          </div>

          <div style="flex:1;">
            <h3 style="margin:0 0 6px 0;">Telemetry</h3>
            <div>Warrior Position: <span id="warriorPos">—</span></div>
            <div>Last update: <span id="lastUpdate">—</span></div>
          </div>
        </div>
      </div>

      <div class="panel">
        <h3 style="margin-top:0;">Mission Logs</h3>
        <div id="logs" class="logs"></div>
      </div>
    </div>

    <div>
      <div class="panel" style="width:400px;">
        <h3>Manual Threat Spawner / Authorization</h3>
        <div>
          <label>Type</label><br>
          <select id="threatType" style="width:100%; padding:6px; border-radius:6px; margin-bottom:6px;">
            <option value="person">👤 Person</option>
            <option value="car">🚗 Car</option>
            <option value="bus">🚌 Bus</option>
            <option value="truck">🚚 Truck</option>
          </select>
        </div>
        <div>
          <label>X</label><br>
          <input id="threatX" type="number" value="50" step="1">
        </div>
        <div style="margin-top:6px;">
          <label>Y</label><br>
          <input id="threatY" type="number" value="50" step="1">
        </div>

        <div style="margin-top:10px;display:flex;gap:8px;">
          <button id="btnSpawn" class="btn">🎯 Spawn</button>
          <button id="btnApprove" class="btn">✅ Authorize</button>
          <button id="btnDeny" class="btn">❌ Deny</button>
        </div>

        <div style="margin-top:12px;">
          <h4 style="margin:6px 0 4px 0;">System Info</h4>
          <div class="small">Datacenter logs saved to <code>/logs/datacenter.log</code></div>
        </div>
      </div>

      <div class="panel" style="width:400px;margin-top:12px;">
        <h3>Active Threat (AI chooses highest only)</h3>
        <div id="activeThreat" style="background:#000;padding:8px;border-radius:6px;color:#ffd;">No threat</div>
      </div>
    </div>
  </div>

<script>
document.addEventListener('DOMContentLoaded', () => {
  const unitGrid = document.getElementById('unitGrid');
  const btnRefresh = document.getElementById('btnRefreshFeeds');
  const btnSpawn = document.getElementById('btnSpawn');
  const btnApprove = document.getElementById('btnApprove');
  const btnDeny = document.getElementById('btnDeny');
  const logsDiv = document.getElementById('logs');
  const activeThreatDiv = document.getElementById('activeThreat');
  const authPanel = document.getElementById('authPanel');

  let allWarriors = [];
  let currentWarriorIndex = 0;
  let isUpdatingStatus = false;
  let lastLogsFetch = 0;

  async function renderUnitFeeds() {
    try {
      const res = await fetch('/status');
      const st = await res.json();
      const regs = st.registered_drones || {};
      
      allWarriors = regs.warriors || [];
      const units = [].concat(regs.warriors || [], regs.kamikazes || [], regs.others || []);
      
      unitGrid.innerHTML = '';
      
      if (units.length === 0) {
        unitGrid.innerHTML = '<div class="small">No unit feeds found (check settings.json)</div>';
        return;
      }
      
      units.forEach((name, idx) => {
        const tile = document.createElement('div');
        tile.className = 'feed';
        tile.style.height = '150px';
        tile.style.position = 'relative';
        tile.style.cursor = 'pointer';
        
        if (allWarriors.includes(name)) {
          tile.onclick = () => {
            currentWarriorIndex = allWarriors.indexOf(name);
            focusOnWarrior(name);
          };
        }
        
        tile.innerHTML = `
          <div class="feed-label" style="${allWarriors.includes(name) ? 'background: rgba(0,255,102,0.2);' : ''}">${name}</div>
          <img src="/feed/${encodeURIComponent(name)}" alt="${name}">
        `;
        unitGrid.appendChild(tile);
      });
      
      if (allWarriors.length > 0) {
        const info = document.createElement('div');
        info.style.marginTop = '8px';
        info.style.color = '#00ff66';
        info.style.fontSize = '0.85rem';
        info.innerHTML = `
          <div>${allWarriors.length} Warriors active | <span id="currentWarriorName">${allWarriors[0]}</span></div>
          <button id="btnCycleWarrior" class="btn" style="margin-top:6px;">🔄 Next Warrior</button>
        `;
        unitGrid.appendChild(info);
        document.getElementById('btnCycleWarrior').addEventListener('click', cycleWarrior);
      }
      
    } catch (e) {
      console.error('renderUnitFeeds', e);
      unitGrid.innerHTML = '<div class="small">Failed to load unit feeds</div>';
    }
  }

  function cycleWarrior() {
    if (allWarriors.length === 0) return;
    currentWarriorIndex = (currentWarriorIndex + 1) % allWarriors.length;
    const warriorName = allWarriors[currentWarriorIndex];
    focusOnWarrior(warriorName);
  }

  function focusOnWarrior(name) {
    const nameSpan = document.getElementById('currentWarriorName');
    if (nameSpan) {
      nameSpan.textContent = name;
      nameSpan.style.color = '#00ff66';
      nameSpan.style.fontWeight = 'bold';
    }
    
    const tiles = unitGrid.querySelectorAll('.feed');
    tiles.forEach(tile => {
      const label = tile.querySelector('.feed-label');
      if (label && label.textContent === name) {
        tile.style.border = '2px solid #00ff66';
        tile.style.boxShadow = '0 0 15px rgba(0,255,102,0.5)';
      } else if (allWarriors.includes(label?.textContent)) {
        tile.style.border = '2px solid rgba(255,255,255,0.03)';
        tile.style.boxShadow = 'none';
      }
    });
    
    console.log(`Focused on ${name}`);
  }

  btnRefresh.addEventListener('click', () => {
    buildUI();
  });

  btnSpawn.addEventListener('click', async () => {
    const type = document.getElementById('threatType').value;
    const x = parseFloat(document.getElementById('threatX').value);
    const y = parseFloat(document.getElementById('threatY').value);
    try {
      const res = await fetch('/spawn_threat', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({x,y,type,confidence: 0.95})
      });
      const j = await res.json();
      if (j.error) alert('Error: '+j.error); else alert('✅ Threat spawned at ('+x+','+y+')');
      updateStatus();
    } catch (e) { console.error(e); alert('Failed to spawn'); }
  });

  btnApprove.addEventListener('click', async () => {
    await fetch('/approve', {method:'POST'});
    alert('✅ Strike authorized');
    updateStatus();
  });
  
  btnDeny.addEventListener('click', async () => {
    await fetch('/deny', {method:'POST'});
    alert('❌ Strike denied');
    updateStatus();
  });

  document.getElementById('btnAuthApprove').addEventListener('click', async () => {
    console.log('✅ User approved strike from AUTH PANEL');
    await fetch('/approve', {method: 'POST'});
    authPanel.classList.remove('active');
    updateStatus();
  });

  document.getElementById('btnAuthDeny').addEventListener('click', async () => {
    console.log('❌ User denied strike from AUTH PANEL');
    await fetch('/deny', {method: 'POST'});
    authPanel.classList.remove('active');
    updateStatus();
  });

  document.getElementById('btnExpand').addEventListener('click', async () => {
    const mult = parseFloat(document.getElementById('expandMult').value) || 1.5;
    const res = await fetch('/expand_patrol', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({mult})});
    const j = await res.json();
    if (j.error) alert('Error: '+j.error);
    else {
      document.getElementById('patrolRadius').textContent = j.new_radius.toFixed(1);
      alert('✅ Patrol expanded to radius: ' + j.new_radius.toFixed(1));
      updateStatus();
    }
  });

  document.getElementById('btnDist').addEventListener('click', async () => {
    const n = parseInt(document.getElementById('distN').value) || 1;
    const spread = parseFloat(document.getElementById('distSpread').value) || 1.0;
    const res = await fetch('/distribute', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({n, spread})});
    const j = await res.json();
    if (j.error) { alert('Error: '+j.error); return; }
    const pts = j.points || [];
    alert('📍 Patrol points:\n\n' + pts.map((p,i) => `${i+1}. (${p[0].toFixed(1)}, ${p[1].toFixed(1)})`).join('\n'));
  });

  async function updateStatus() {
    if (isUpdatingStatus) {
      return;
    }
    isUpdatingStatus = true;

    try {
      // ✅ ADD TIMEOUT
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 sec timeout
      
      const res = await fetch('/status', { signal: controller.signal });
      clearTimeout(timeoutId);
      
      const status = await res.json();
      if (status.error) return;
      document.getElementById('scans').textContent = status.queen_scans || 0;
      document.getElementById('threatLevel').textContent = status.threat_level || 'GREEN';
      document.getElementById('patrolCenter').textContent =
        `${status.patrol_area.center_x.toFixed(1)},${status.patrol_area.center_y.toFixed(1)}`;
      document.getElementById('patrolRadius').textContent =
        `${status.patrol_area.radius.toFixed(1)}`;

      if (status.warrior_status && status.warrior_status.position) {
        const p = status.warrior_status.position;
        document.getElementById('warriorPos').textContent =
          `${p[0].toFixed(2)}, ${p[1].toFixed(2)}, ${p[2].toFixed(2)}`;
        document.getElementById('lastUpdate').textContent =
          status.warrior_status.time || new Date().toLocaleTimeString();
      } else {
        document.getElementById('warriorPos').textContent = '—';
      }

      if (status.pending_permission && status.active_threat) {
        const at = status.active_threat;
        authPanel.classList.add('active');
        document.getElementById('authThreatType').textContent = at.class.toUpperCase();
        document.getElementById('authThreatPos').textContent =
          `X:${at.world_pos[0].toFixed(1)}m, Y:${at.world_pos[1].toFixed(1)}m`;
        document.getElementById('authThreatConf').textContent =
          `${Math.round(at.confidence * 100)}%`;
        console.log('🚨 AUTHORIZATION PANEL SHOWN');
      } else {
        authPanel.classList.remove('active');
      }

      if (status.active_threat) {
        const at = status.active_threat;
        activeThreatDiv.innerHTML =
          `<strong style="color:#ff4444;">${at.class}</strong> — confidence ` +
          `${Math.round((at.confidence || 0) * 100)}% at (` +
          `${at.world_pos ? at.world_pos.map(v => v.toFixed(1)).join(',') : '—'})`;
        activeThreatDiv.style.border = '2px solid #ff4444';
        activeThreatDiv.style.animation = 'blink 1s infinite';
      } else {
        activeThreatDiv.textContent = 'No active threat';
        activeThreatDiv.style.border = '1px solid rgba(255,255,255,0.1)';
        activeThreatDiv.style.animation = 'none';
      }

      const now = Date.now();
      if (now - lastLogsFetch > 3000) {
        const logsRes = await fetch('/logs');
        const logs = await logsRes.json();
        renderLogs(logs || []);
        lastLogsFetch = now;
      }
    } catch (e) {
      // ✅ Better error handling - don't spam console
      if (e.name === 'AbortError') {
        console.warn('Status request timeout - retrying...');
      } else {
        console.error('updateStatus error:', e.message);
      }
    } finally {
      isUpdatingStatus = false;
    }
  }

  function renderLogs(logs) {
    const formatted = logs.reverse().slice(0,200).map(l => {
      let color = '#bfffbf';
      if (l.level === 'CRITICAL') color = '#ff4444';
      else if (l.level === 'WARNING') color = '#ff6600';
      return `<div class="log-entry" style="color:${color};">[${l.time}] [${l.source}] ${l.message}</div>`;
    }).join('');
    logsDiv.innerHTML = formatted;
    logsDiv.scrollTop = logsDiv.scrollHeight;
  }

  function setupFeedZoom() {
    const allFeeds = document.querySelectorAll('.feed');
    allFeeds.forEach(feed => {
      feed.addEventListener('dblclick', function(e) {
        e.stopPropagation();
        
        if (this.classList.contains('zoomed')) {
          this.classList.remove('zoomed');
          this.style.position = '';
          this.style.top = '';
          this.style.left = '';
          this.style.width = '';
          this.style.height = '';
          this.style.zIndex = '';
          this.style.transform = '';
          this.style.cursor = 'zoom-in';
        } else {
          this.classList.add('zoomed');
          this.style.position = 'fixed';
          this.style.top = '50%';
          this.style.left = '50%';
          this.style.width = '85vw';
          this.style.height = '85vh';
          this.style.zIndex = '99998';
          this.style.transform = 'translate(-50%, -50%)';
          this.style.cursor = 'zoom-out';
        }
      });
    });
  }

  async function buildUI() {
    await renderUnitFeeds();
    await updateStatus();
    setupFeedZoom();
  }

  const style = document.createElement('style');
  style.textContent = '@keyframes blink { 0%, 50% { opacity: 1; } 25%, 75% { opacity: 0.5; } }';
  document.head.appendChild(style);

  buildUI();
  setInterval(updateStatus, 1000);
});
</script>

</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(UI_TEMPLATE)


def run_web(host="0.0.0.0", port=5000):
    """
    Run Flask app — use_reloader=False so it can be started in-thread safely.
    This is the ONLY place where datacenter initializes and starts.
    """
    try:
        # Initialize drones when web server actually starts
        build_registered_drones()
        
        logger.info(f"Starting datacenter web UI on {host}:{port}")
        app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)
    except Exception:
        logger.exception("Flask run failed")


# ============================================
# NO AUTO-EXECUTION BLOCK
# Datacenter should ONLY run when explicitly
# called from main.py via run_web()
# ============================================