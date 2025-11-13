# datacenter.py
import os
import time
import traceback
import logging
from logging.handlers import RotatingFileHandler

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
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] %(message)s", "%H:%M:%S"))
logger.addHandler(ch)
logger.propagate = False

app = Flask(__name__)

# One AirSim client per drone name (reused)
clients = {}

def get_client(drone):
    """Return a cached AirSim client for a drone name. Robust to connection errors."""
    if drone in clients:
        return clients[drone]
    try:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        clients[drone] = client
        logger.info(f"Connected AirSim client for {drone}")
        return client
    except Exception as e:
        logger.warning(f"Could not create AirSim client for {drone}: {e}")
        return None

def get_frame_bytes(drone, feed_style="normal", use_ai=False):
    """Grab a single camera frame from AirSim for a named drone and return JPEG bytes or None."""
    try:
        client = get_client(drone)
        if client is None:
            return None
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ], vehicle_name=drone)
        if not responses or len(responses[0].image_data_uint8) == 0:
            return None
        img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img = img1d.reshape(responses[0].height, responses[0].width, 3)
        img = np.copy(img)

        # visual styles
        if feed_style == "thermal":
            img = cv2.applyColorMap(img, cv2.COLORMAP_AUTUMN)
        elif feed_style == "nightvision":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            img[:, :, 1] = np.clip(img[:, :, 1] * 1.4, 0, 255).astype(np.uint8)

        # overlay drone name and telemetry if available
        color = (255, 255, 255)
        try:
            colors = {"Queen": (0, 165, 255), "Warrior1": (0, 255, 0), "Kamikaze1": (0, 0, 255)}
            color = colors.get(drone, (255, 255, 255))
            cv2.rectangle(img, (0, 0), (300, 80), (0, 0, 0), -1)
            cv2.putText(img, drone, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            # position text if available
            try:
                pos = client.simGetVehiclePose(drone).position
                pos_text = f"X:{pos.x_val:.1f} Y:{pos.y_val:.1f} Z:{pos.z_val:.1f}"
                cv2.putText(img, pos_text, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            except Exception:
                pass
        except Exception:
            pass

        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buffer.tobytes()
    except Exception as e:
        logger.debug(f"Frame fetch error for {drone}: {e}")
        return None

def gen_stream(drone, feed_style="normal"):
    """Generator for multipart JPEG stream (MJPEG)."""
    while True:
        try:
            frame = get_frame_bytes(drone, feed_style)
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                blank = np.zeros((120, 160, 3), dtype=np.uint8)
                _, b = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, 40])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + b.tobytes() + b'\r\n')
            time.sleep(0.12)
        except GeneratorExit:
            break
        except Exception as e:
            logger.debug(f"Stream generator error for {drone}: {e}")
            time.sleep(0.5)

# --- Flask endpoints ---
@app.route('/queen')
def queen_feed():
    return Response(gen_stream("Queen", feed_style="thermal"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/warrior')
def warrior_feed():
    return Response(gen_stream("Warrior1", feed_style="nightvision"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/kamikaze')
def kamikaze_feed():
    return Response(gen_stream("Kamikaze1", feed_style="normal"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs')
def logs():
    try:
        return jsonify(swarm.get_logs(200))
    except Exception as e:
        logger.exception("Error returning logs")
        return jsonify([])

@app.route('/status')
def status():
    try:
        cx, cy, r = swarm.get_patrol_area()
        warrior_status = swarm.get_warrior_status()

        # --- GET QUEEN POSE FOR RELATIVE PATROL ---
        queen_pose = None
        try:
            q_client = get_client("Queen")
            if q_client:
                pos = q_client.simGetVehiclePose("Queen").position
                queen_pose = (pos.x_val, pos.y_val, pos.z_val)
        except Exception:
            queen_pose = None

        status_data = {
            'threat_level': swarm.threat_level,
            'pending_permission': swarm.pending_permission,
            'active_threat': swarm.active_threat,
            'warrior_status': warrior_status,
            'kamikaze_deployed': swarm.kamikaze_deployed,
            'queen_scans': swarm.queen_scans,
            'patrol_area': {
                'center_x': cx,
                'center_y': cy,
                'radius': r
            },
            'patrol_relative': swarm.patrol_relative_to_queen,
            'last_patrol_update': getattr(swarm, 'last_patrol_update', 0),
            'last_warrior_update': getattr(swarm, 'last_warrior_update', 0),

            # <<< FIXED LINE >>>
            'queen_pose': queen_pose
        }
        return jsonify(status_data)
    except Exception as e:
        logger.exception("Status error")
        return jsonify({'error': str(e)}), 500


@app.route('/approve', methods=['POST'])
def approve():
    try:
        swarm.user_response = True
        swarm.pending_permission = False
        swarm.log("USER", "✅ AUTHORIZED", "CRITICAL")
        return jsonify({'status': 'approved'})
    except Exception as e:
        logger.exception("Approve error")
        return jsonify({'error': str(e)}), 500

@app.route('/deny', methods=['POST'])
def deny():
    try:
        swarm.user_response = False
        swarm.pending_permission = False
        swarm.log("USER", "❌ DENIED", "WARNING")
        return jsonify({'status': 'denied'})
    except Exception as e:
        logger.exception("Deny error")
        return jsonify({'error': str(e)}), 500

@app.route('/spawn_threat', methods=['POST'])
def spawn_threat():
    try:
        data = request.json or {}
        x = float(data.get('x', 10))
        y = float(data.get('y', 10))
        threat_type = data.get('type', 'person')
        threat = {
            'class': threat_type,
            'confidence': 1.0,
            'world_pos': (x, y),
            'timestamp': time.time()
        }
        swarm.add_threat(threat)
        logger.info(f"Manual threat spawned: {threat_type} at ({x}, {y})")
        return jsonify({'status': 'spawned', 'threat': threat})
    except Exception as e:
        logger.exception("Spawn threat error")
        return jsonify({'error': str(e)}), 500

@app.route('/set_patrol', methods=['POST'])
def set_patrol():
    try:
        data = request.json or {}
        cx = float(data.get('x', 0))
        cy = float(data.get('y', 0))
        radius = float(data.get('radius', 30))
        relative = bool(data.get('relative', False))
        swarm.set_patrol_area(cx, cy, radius, relative=relative)
        logger.info(f"Patrol updated via UI: ({cx}, {cy}) R={radius} relative={relative}")
        return jsonify({'center_x': cx, 'center_y': cy, 'radius': radius, 'relative': relative})
    except Exception as e:
        logger.exception("Set patrol error")
        return jsonify({'error': str(e)}), 500

@app.route('/get_patrol')
def get_patrol():
    try:
        cx, cy, r = swarm.get_patrol_area()
        return jsonify({'center_x': cx, 'center_y': cy, 'radius': r, 'relative': swarm.patrol_relative_to_queen})
    except Exception as e:
        logger.exception("Get patrol error")
        return jsonify({'error': str(e)}), 500

@app.route('/set_queen_pose', methods=['POST'])
def set_queen_pose():
    """Teleport Queen to given pose (x,y,z). Immediate change — useful for testing."""
    try:
        data = request.json or {}
        x = float(data.get('x', 0))
        y = float(data.get('y', 0))
        z = float(data.get('z', -20))
        client = get_client("Queen")
        if client is None:
            return jsonify({'error': 'No AirSim client for Queen'}), 500
        pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0,0,0))
        client.simSetVehiclePose(pose, True, vehicle_name="Queen")
        swarm.log("SYSTEM", f"Queen teleported to ({x:.1f}, {y:.1f}, {z:.1f})", "INFO")
        return jsonify({'status': 'ok', 'x': x, 'y': y, 'z': z})
    except Exception as e:
        logger.exception("Set queen pose error")
        return jsonify({'error': str(e)}), 500

@app.route('/move_queen', methods=['POST'])
def move_queen():
    """Command Queen to fly smoothly to a target (x,y,z) using moveToPositionAsync."""
    try:
        data = request.json or {}
        x = float(data.get('x', 0))
        y = float(data.get('y', 0))
        z = float(data.get('z', -20))
        speed = float(data.get('speed', 5.0))
        client = get_client("Queen")
        if client is None:
            return jsonify({'error': 'No AirSim client for Queen'}), 500
        # non-blocking command; return immediately
        client.enableApiControl(True, "Queen")
        client.armDisarm(True, "Queen")
        client.moveToPositionAsync(x, y, z, speed, vehicle_name="Queen")
        swarm.log("SYSTEM", f"Queen moving to ({x:.1f}, {y:.1f}, {z:.1f}) speed={speed}", "INFO")
        return jsonify({'status': 'moving', 'x': x, 'y': y, 'z': z, 'speed': speed})
    except Exception as e:
        logger.exception("Move queen error")
        return jsonify({'error': str(e)}), 500

@app.route('/favicon.ico')
def favicon():
    return '', 204

# ----------------------------
# UI Template (full) - same as before but with relative toggle and queen controls
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
    h1{margin:0;font-size:1.1rem;color:var(--accent);text-shadow:0 0 8px rgba(0,255,102,0.06);}
    .stat{background:rgba(0,0,0,0.4);padding:6px 8px;border-radius:6px;font-size:0.9rem;color:var(--accent);}
    .container{display:grid;grid-template-columns:1fr 420px;gap:12px;padding:12px;height:calc(100% - 64px);box-sizing:border-box;}
    .left{display:flex;flex-direction:column;gap:12px;}
    .feeds{display:flex;gap:8px;}
    .feed{flex:1;border:2px solid rgba(255,255,255,0.03);border-radius:6px;height:220px;overflow:hidden;background:#000;position:relative;}
    .feed img{width:100%;height:100%;object-fit:cover;display:block;}
    .feed-label{position:absolute;top:6px;left:8px;background:rgba(0,0,0,0.6);padding:6px 8px;border-radius:4px;font-size:0.8rem;color:var(--warn);border:1px solid rgba(255,102,0,0.12);}
    .panel{background:var(--panel);border:2px solid rgba(0,255,102,0.06);padding:10px;border-radius:8px;color:var(--accent);}
    canvas#mapCanvas{width:100%;height:auto;border-radius:6px;display:block;background:#040404;}
    .logs{height:260px;overflow:auto;background:#060606;padding:8px;border-radius:6px;border:1px solid rgba(255,255,255,0.02);color:#bfffbf;font-size:0.9rem;}
    .log-entry{font-family:monospace;padding:4px 0;border-bottom:1px dashed rgba(255,255,255,0.02);}
    .controls{display:flex;gap:8px;margin-top:8px;}
    input[type=number]{width:120px;padding:6px;background:#000;color:var(--accent);border:1px solid rgba(0,255,102,0.06);border-radius:6px;}
    label{color:#cfe8cf;font-size:0.9rem;}
    .permission-panel{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:#120000;border:3px solid var(--danger);padding:16px;border-radius:8px;z-index:9999;display:none;width:420px;}
    .permission-panel.active{display:block;}
    .permission-overlay{position:fixed;top:0;left:0;right:0;bottom:0;background: rgba(0,0,0,0.6);display:none;z-index:9998;}
    .permission-overlay.active{display:block;}
    .small{font-size:0.85rem;color:var(--muted);}
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

  <div class="container">
    <div class="left">
      <div class="feeds panel">
        <div style="display:flex;gap:8px;">
          <div class="feed" style="flex:1">
            <div class="feed-label">🔥 THERMAL (Queen sees)</div>
            <img id="imgQueen" src="/queen" alt="Queen feed">
          </div>
          <div class="feed" style="flex:1">
            <div class="feed-label">🌙 NIGHT VISION (Warrior)</div>
            <img id="imgWarrior" src="/warrior" alt="Warrior feed">
          </div>
          <div class="feed" style="width:220px;">
            <div class="feed-label">KAMIKAZE</div>
            <img id="imgKamikaze" src="/kamikaze" alt="Kamikaze feed">
          </div>
        </div>
      </div>

      <div class="panel">
        <div style="display:flex;gap:12px;align-items:flex-start;">
          <div style="width:420px;">
            <h3 style="margin:0 0 8px 0;">Telemetry</h3>
            <div>Patrol Center (stored): <span id="patrolCenter">0, 0</span></div>
            <div>Patrol Radius: <span id="patrolRadius">30</span> m</div>
            <div>Patrol Mode: <span id="patrolMode">ABSOLUTE</span></div>
            <div>Warrior Position: <span id="warriorPos">—</span></div>
            <div>Last warrior update: <span id="lastUpdate">—</span></div>

            <div style="margin-top:8px;">
              <label>Center X</label><br>
              <input id="centerX" type="number" value="0" step="1">
              <label style="margin-left:8px;"><input id="relativeToggle" type="checkbox"> Relative to Queen</label>
            </div>

            <div style="margin-top:6px;">
              <label>Center Y</label><br>
              <input id="centerY" type="number" value="0" step="1">
            </div>

            <div style="margin-top:6px;">
              <label>Radius</label><br>
              <input id="radius" type="number" value="30" step="1" min="5">
            </div>

            <div style="margin-top:8px;">
              <button id="btnUpdate" class="btn">🔄 UPDATE PATROL</button>
              <button id="btnSpawn" class="btn" style="background:rgba(255,102,0,0.08);color:var(--warn);">🎯 SPAWN THREAT</button>
            </div>

            <div style="margin-top:12px;">
              <h4 style="margin:6px 0 4px 0;">Queen Controls</h4>
              <label>X</label><br><input id="queenX" type="number" value="0" step="1">
              <label>Y</label><br><input id="queenY" type="number" value="0" step="1">
              <label>Z</label><br><input id="queenZ" type="number" value="-20" step="1"><br><br>
              <button id="btnQueenSet" class="btn">Teleport Queen</button>
              <button id="btnQueenMove" class="btn">Command Queen to Fly</button>
            </div>
          </div>

          <div>
            <h3 style="margin-top:0;">Mini Map</h3>
            <canvas id="mapCanvas" width="420" height="300"></canvas>
            <div class="small">Green = Warrior • Orange = Patrol center • Circle = Patrol radius • Blue = Queen</div>
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
        <h3>Manual Threat Spawner</h3>
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

        <div style="margin-top:10px;">
          <button id="btnApprove" class="btn">✅ AUTHORIZE</button>
          <button id="btnDeny" class="btn" style="background:rgba(255,0,0,0.06);color:var(--danger);">❌ DENY</button>
        </div>

        <div style="margin-top:12px;">
          <h4 style="margin:6px 0 4px 0;">System Info</h4>
          <div class="small">Datacenter logs saved to <code>/logs/datacenter.log</code></div>
        </div>
      </div>
    </div>
  </div>

  <div id="permissionOverlay" class="permission-overlay"></div>
  <div id="permissionPanel" class="permission-panel">
    <h3>⚠️ STRIKE AUTHORIZATION REQUIRED ⚠️</h3>
    <div id="threatInfo" style="background:#000;padding:8px;border-radius:6px;color:#ffd;"></div>
    <div style="margin-top:10px;display:flex;gap:8px;">
      <button id="panelApprove" class="btn">✅ Authorize</button>
      <button id="panelDeny" class="btn" style="background:rgba(255,0,0,0.06);color:var(--danger);">❌ Deny</button>
    </div>
  </div>

<script>
document.addEventListener('DOMContentLoaded', () => {
  const overlay = document.getElementById('permissionOverlay');
  const panel = document.getElementById('permissionPanel');
  const threatInfo = document.getElementById('threatInfo');

  const centerXInput = document.getElementById('centerX');
  const centerYInput = document.getElementById('centerY');
  const radiusInput  = document.getElementById('radius');
  const relativeToggle = document.getElementById('relativeToggle');

  const queenX = document.getElementById('queenX');
  const queenY = document.getElementById('queenY');
  const queenZ = document.getElementById('queenZ');

  const map = document.getElementById('mapCanvas');
  const ctx = map.getContext('2d');

  let buttonsDisabled = false;

  function clearMap(){ ctx.fillStyle = "#040404"; ctx.fillRect(0,0,map.width,map.height); }

  function drawMap(patrol, warriorStatus, queenPos) {
    clearMap();
    if(!patrol) return;
    const cx = patrol.center_x, cy = patrol.center_y, r = patrol.radius;
    const pxPerMeter = (Math.min(map.width, map.height) * 0.35) / Math.max(1, r);
    const centerScreen = {x: map.width/2, y: map.height/2};

    // patrol circle
    ctx.beginPath();
    ctx.strokeStyle = "#ff6600";
    ctx.lineWidth = 2;
    ctx.arc(centerScreen.x, centerScreen.y, r*pxPerMeter, 0, Math.PI*2);
    ctx.stroke();
    // center marker
    ctx.fillStyle = "#ff6600";
    ctx.fillRect(centerScreen.x-5, centerScreen.y-5, 10, 10);

    // warrior position
    if(warriorStatus && warriorStatus.position){
      const pos = warriorStatus.position;
      const wx = pos[0] - cx;
      const wy = pos[1] - cy;
      const sx = centerScreen.x + wx*pxPerMeter;
      const sy = centerScreen.y - wy*pxPerMeter;
      ctx.fillStyle = "#00ff66";
      ctx.beginPath();
      ctx.arc(sx, sy, 6, 0, Math.PI*2);
      ctx.fill();
      ctx.strokeStyle = "#000";
      ctx.stroke();
    }

    // Queen marker
    if (queenPos && queenPos.length>=2 && !isNaN(queenPos[0])) {
      const qx = queenPos[0] - cx;
      const qy = queenPos[1] - cy;
      const sx = centerScreen.x + qx*pxPerMeter;
      const sy = centerScreen.y - qy*pxPerMeter;
      ctx.fillStyle = "#66ccff";
      ctx.beginPath();
      ctx.arc(sx, sy, 6, 0, Math.PI*2);
      ctx.fill();
      ctx.strokeStyle = "#000";
      ctx.stroke();
    }
  }

  function renderLogs(logs) {
    const logsDiv = document.getElementById('logs');
    logsDiv.innerHTML = logs.reverse().slice(0,200).map(l => {
      const msg = `[${l.time}] [${l.source}] ${l.message}`;
      return `<div class="log-entry">${msg}</div>`;
    }).join('');
    logsDiv.scrollTop = logsDiv.scrollHeight;
  }

  // Show/hide permission modal
  function showPermission(threat) {
    if (!threat) threat = {class:'unknown', world_pos:[0,0]};
    threatInfo.innerHTML = `
      <div><strong>Threat:</strong> ${threat.class || 'unknown'}</div>
      <div><strong>Confidence:</strong> ${threat.confidence ? (Math.round(threat.confidence*100) + '%') : '—'}</div>
      <div><strong>Coords:</strong> ${threat.world_pos ? `${threat.world_pos[0].toFixed(1)}, ${threat.world_pos[1].toFixed(1)}` : '—'}</div>
      <div style="margin-top:8px;" class="small">You have the option to Authorize (strike) or Deny. If you do nothing, the system will auto-authorize after timeout.</div>
    `;
    overlay.classList.add('active');
    panel.classList.add('active');
    // enable buttons
    document.getElementById('panelApprove').disabled = false;
    document.getElementById('panelDeny').disabled = false;
    buttonsDisabled = false;
  }

  function hidePermission() {
    overlay.classList.remove('active');
    panel.classList.remove('active');
    threatInfo.innerHTML = '';
    buttonsDisabled = false;
  }

  // Approve / Deny with disable-once-click to avoid duplicate requests
  function approve() {
    if (buttonsDisabled) return;
    buttonsDisabled = true;
    document.getElementById('panelApprove').disabled = true;
    document.getElementById('panelDeny').disabled = true;
    fetch('/approve', { method: 'POST' })
      .then(r => r.json()).then(() => {
        hidePermission();
      }).catch(e => { console.error(e); hidePermission(); });
  }
  function deny() {
    if (buttonsDisabled) return;
    buttonsDisabled = true;
    document.getElementById('panelApprove').disabled = true;
    document.getElementById('panelDeny').disabled = true;
    fetch('/deny', { method: 'POST' })
      .then(r => r.json()).then(() => {
        hidePermission();
      }).catch(e => { console.error(e); hidePermission(); });
  }

  document.getElementById('panelApprove').addEventListener('click', approve);
  document.getElementById('panelDeny').addEventListener('click', deny);

  // UI handlers (patrol, spawn, queen)
  function spawnThreatUI() {
    const type = document.getElementById('threatType').value;
    const x = parseFloat(document.getElementById('threatX').value);
    const y = parseFloat(document.getElementById('threatY').value);
    fetch('/spawn_threat', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({x,y,type}) })
      .then(r => r.json()).then(d => { if(d.error) alert('Error: '+d.error); else { alert('Spawned'); } })
      .catch(e => { console.error(e); alert('Failed to spawn'); });
  }

  function updatePatrolUI() {
    const x = parseFloat(centerXInput.value);
    const y = parseFloat(centerYInput.value);
    const radius = parseFloat(radiusInput.value);
    const relative = document.getElementById('relativeToggle').checked;
    fetch('/set_patrol', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({x,y,radius, relative})
    }).then(r => r.json()).then(d => {
      if (d.error) alert('Error: '+d.error); else {
        setTimeout(updateStatus, 500);
        alert(`Patrol updated: (${x},${y}) R=${radius} (relative=${relative})`);
      }
    }).catch(e => { console.error(e); alert('Failed to update patrol'); });
  }

  document.getElementById('btnUpdate').addEventListener('click', updatePatrolUI);
  document.getElementById('btnSpawn').addEventListener('click', spawnThreatUI);

  // Queen controls
  document.getElementById('btnQueenSet').addEventListener('click', () => {
    const x = parseFloat(queenX.value); const y = parseFloat(queenY.value); const z = parseFloat(queenZ.value);
    fetch('/set_queen_pose', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({x,y,z}) })
      .then(r => r.json()).then(d => { if(d.error) alert('Error: '+d.error); else { alert('Queen teleported'); setTimeout(updateStatus,400); } })
      .catch(e => { console.error(e); alert('Failed to teleport queen'); });
  });

  document.getElementById('btnQueenMove').addEventListener('click', () => {
    const x = parseFloat(queenX.value); const y = parseFloat(queenY.value); const z = parseFloat(queenZ.value);
    fetch('/move_queen', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({x,y,z,speed:5.0}) })
      .then(r => r.json()).then(d => { if(d.error) alert('Error: '+d.error); else { alert('Queen commanded to fly'); setTimeout(updateStatus,400); } })
      .catch(e => { console.error(e); alert('Failed to move queen'); });
  });

  // Main polling - get status + logs and show permission modal when pending
  async function updateStatus() {
    try {
      const res = await fetch('/status');
      const status = await res.json();
      if (status.error) { console.warn("Status error:", status.error); return; }

      document.getElementById('scans').textContent = status.queen_scans || 0;
      document.getElementById('threatLevel').textContent = status.threat_level || 'GREEN';
      document.getElementById('patrolCenter').textContent = `${status.patrol_area.center_x.toFixed(1)}, ${status.patrol_area.center_y.toFixed(1)}`;
      document.getElementById('patrolRadius').textContent = `${status.patrol_area.radius.toFixed(1)}`;
      document.getElementById('patrolMode').textContent = status.patrol_relative ? 'RELATIVE' : 'ABSOLUTE';
      // Only update UI controls if the user is NOT actively editing them
      if (document.activeElement !== centerXInput) {
        centerXInput.value = Number(status.patrol_area.center_x).toFixed(1);
      }
      if (document.activeElement !== centerYInput) {
        centerYInput.value = Number(status.patrol_area.center_y).toFixed(1);
      }
      if (document.activeElement !== radiusInput) {
        radiusInput.value = Number(status.patrol_area.radius).toFixed(1);
      }
      // For checkbox, avoid overwriting if user just clicked it (it may get focus)
      if (document.activeElement !== relativeToggle) {
        relativeToggle.checked = Boolean(status.patrol_relative);
      }


      if (status.warrior_status && status.warrior_status.position) {
        const p = status.warrior_status.position;
        document.getElementById('warriorPos').textContent = `${p[0].toFixed(2)}, ${p[1].toFixed(2)}, ${p[2].toFixed(2)}`;
        document.getElementById('lastUpdate').textContent = status.warrior_status.time || new Date().toLocaleTimeString();
      } else {
        document.getElementById('warriorPos').textContent = '—';
      }

      // render logs
      const logsRes = await fetch('/logs');
      const logs = await logsRes.json();
      renderLogs(logs || []);
      // draw map (no queen pose exposed yet)
      drawMap(status.patrol_area, status.warrior_status || null, status.queen_pose);


      // permission modal handling
      if (status.pending_permission) {
        // show modal with threat info
        showPermission(status.active_threat);
      } else {
        // hide if not pending
        hidePermission();
      }
    } catch (e) {
      console.error("updateStatus error:", e);
    }
  }

  updateStatus();
  setInterval(updateStatus, 900);
});
</script>

</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(UI_TEMPLATE)

# run helper for in-thread launching
def run_web(host='0.0.0.0', port=5000):
    """Run Flask app — use_reloader=False so it can be started in-thread safely."""
    try:
        logger.info(f"Starting datacenter web UI on {host}:{port}")
        app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)
    except Exception as e:
        logger.exception("Flask run failed")

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌐 DATACENTER - COMMAND CENTER WEB UI")
    print("="*70)
    print("Starting web UI at http://localhost:5000")
    run_web()
