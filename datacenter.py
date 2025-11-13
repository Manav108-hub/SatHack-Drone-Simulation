# main.py
import airsim
import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify, request
import time
import traceback
from swarm_state import swarm

app = Flask(__name__)

# Create ONE client per drone (reuse it)
clients = {}

def get_client(drone):
    if drone not in clients:
        clients[drone] = airsim.MultirotorClient()
        clients[drone].confirmConnection()
    return clients[drone]

def get_frame(drone, use_ai=False, feed_style="normal"):
    try:
        client = get_client(drone)
        
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ], drone)
        
        if not responses or len(responses[0].image_data_uint8) == 0:
            return None
        
        img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img = img1d.reshape(responses[0].height, responses[0].width, 3)
        img = np.copy(img)
        
        # Apply visual effects to differentiate feeds
        if feed_style == "thermal":
            img = cv2.applyColorMap(img, cv2.COLORMAP_AUTUMN)
        elif feed_style == "nightvision":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            img[:,:,1] = np.clip(img[:,:,1] * 1.4, 0, 255).astype(np.uint8)
        
        if use_ai:
            try:
                from ultralytics import YOLO
                model = YOLO('yolov8n.pt')
                results = model(img, verbose=False, conf=0.5)
                threat_classes = {0: 'person', 2: 'car', 5: 'bus', 7: 'truck'}
                
                for box in results[0].boxes:
                    if int(box.cls) in threat_classes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf)
                        label = f"{threat_classes[int(box.cls)]} {conf:.0%}"
                        
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(img, (x1, y1-h-10), (x1+w, y1), (0, 0, 255), -1)
                        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        
                        cx, cy = (x1+x2)//2, (y1+y2)//2
                        cv2.drawMarker(img, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 30, 3)
            except Exception as ai_error:
                print(f"AI detection error: {ai_error}")
        
        colors = {"Queen": (255, 102, 0), "Warrior1": (0, 255, 0), "Kamikaze1": (255, 0, 0)}
        color = colors.get(drone, (255, 255, 255))
        
        cv2.rectangle(img, (0, 0), (250, 75), (0, 0, 0), -1)
        cv2.putText(img, drone, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        
        try:
            pos = client.simGetVehiclePose(drone).position
            pos_text = f"X:{pos.x_val:.0f} Y:{pos.y_val:.0f} Z:{pos.z_val:.0f}"
            cv2.putText(img, pos_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception:
            pass
        
        timestamp = time.strftime("%H:%M:%S")
        cv2.rectangle(img, (img.shape[1]-150, 0), (img.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(img, timestamp, (img.shape[1]-140, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes()
        
    except Exception as e:
        print(f"Frame error {drone}: {e}")
        return None

def gen(drone, use_ai=False, feed_style="normal"):
    while True:
        try:
            frame = get_frame(drone, use_ai, feed_style)
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
        except Exception as e:
            print(f"Generator error {drone}: {e}")
            time.sleep(1)

@app.route('/queen')
def queen():
    return Response(gen("Queen", True, "thermal"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/warrior')
def warrior():
    return Response(gen("Warrior1", True, "nightvision"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/kamikaze')
def kamikaze():
    return Response(gen("Kamikaze1", False, "normal"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs')
def logs():
    try:
        return jsonify(swarm.get_logs(100))
    except Exception as e:
        print(f"Logs error: {e}")
        return jsonify([])

@app.route('/status')
def status():
    try:
        cx, cy, r = swarm.get_patrol_area()
        status_data = {
            'threat_level': swarm.threat_level,
            'pending_permission': swarm.pending_permission,
            'active_threat': swarm.active_threat,
            'warrior_status': swarm.get_warrior_status(),
            'kamikaze_deployed': swarm.kamikaze_deployed,
            'queen_scans': swarm.queen_scans,
            'patrol_area': {
                'center_x': cx,
                'center_y': cy,
                'radius': r
            }
        }
        if status_data['pending_permission']:
            print(f"[STATUS] ⚠️ pending={status_data['pending_permission']}, threat={status_data['active_threat'] is not None}")
        return jsonify(status_data)
    except Exception as e:
        print(f"Status error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)})

@app.route('/approve', methods=['POST'])
def approve():
    try:
        swarm.user_response = True
        print("✅ USER APPROVED STRIKE")
        return jsonify({'status': 'approved'})
    except Exception as e:
        print(f"Approve error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/deny', methods=['POST'])
def deny():
    try:
        swarm.user_response = False
        print("❌ USER DENIED STRIKE")
        return jsonify({'status': 'denied'})
    except Exception as e:
        print(f"Deny error: {e}")
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
        print(f"🚨 Manual threat spawned: {threat_type} at ({x}, {y})")
        
        return jsonify({'status': 'spawned', 'threat': threat})
    except Exception as e:
        print(f"Spawn threat error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/set_patrol', methods=['POST'])
def set_patrol():
    try:
        data = request.json or {}
        cx = float(data.get('x', 0))
        cy = float(data.get('y', 0))
        radius = float(data.get('radius', 30))
        
        swarm.set_patrol_area(cx, cy, radius)
        print(f"🎯 Patrol updated via UI: ({cx}, {cy}) R={radius}")
        
        return jsonify({'center_x': cx, 'center_y': cy, 'radius': radius})
    except Exception as e:
        print(f"Set patrol error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_patrol')
def get_patrol():
    try:
        cx, cy, r = swarm.get_patrol_area()
        return jsonify({'center_x': cx, 'center_y': cy, 'radius': r})
    except Exception as e:
        print(f"Get patrol error: {e}")
        return jsonify({'error': str(e)})

# Simple favicon handler to avoid 404 noise
@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/')
def index():
    return render_template_string('''<!DOCTYPE html>
<html>
<head>
    <title>🚁 COMMAND CENTER</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            background: #0a0a0a; 
            color: #0f0; 
            font-family: 'Courier New', monospace; 
            overflow-x: hidden;
        }
        .header {
            background: linear-gradient(180deg, #1a1a1a 0%, #0a0a0a 100%);
            padding: 8px 12px;
            border-bottom: 2px solid #0f0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 1000;
        }
        h1 { color: #0f0; text-shadow: 0 0 10px #0f0; font-size: 1.1em; }
        .stats {
            display: flex;
            gap: 12px;
            align-items: center;
        }
        .stat {
            padding: 4px 8px;
            border: 2px solid #0f0;
            border-radius: 5px;
            font-size: 0.85em;
        }
        .threat-level {
            padding: 4px 12px;
            border: 2px solid;
            border-radius: 5px;
            font-weight: bold;
            font-size: 0.85em;
        }
        .threat-level.GREEN { color: #0f0; border-color: #0f0; }
        .threat-level.YELLOW { color: #ff0; border-color: #ff0; }
        .threat-level.RED { color: #f00; border-color: #f00; animation: blink 0.5s infinite; }
        @keyframes blink { 0%, 50% { opacity: 1; } 25%, 75% { opacity: 0.3; } }
        
        .container {
            display: grid;
            grid-template-columns: 1.5fr 1fr;
            gap: 8px;
            padding: 8px;
        }
        .feeds {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .feed {
            border: 2px solid;
            border-radius: 6px;
            overflow: hidden;
            background: #000;
            height: 280px;
            position: relative;
        }
        .feed.queen { 
            border-color: #ff6600; 
            box-shadow: 0 0 10px #ff6600; 
        }
        .feed.warrior { 
            border-color: #00ff00; 
            box-shadow: 0 0 10px #00ff00; 
        }
        .feed.kamikaze { border-color: #f00; }
        .feed img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }
        .feed-label {
            position: absolute;
            top: 5px;
            right: 5px;
            background: rgba(0,0,0,0.9);
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.75em;
            z-index: 10;
            font-weight: bold;
        }
        .feed.queen .feed-label { 
            color: #ff6600; 
            border: 1px solid #ff6600; 
        }
        .feed.warrior .feed-label { 
            color: #00ff00; 
            border: 1px solid #00ff00; 
        }
        
        .control-panel {
            display: flex;
            flex-direction: column;
            gap: 8px;
            max-height: calc(100vh - 60px);
            overflow-y: auto;
            padding-right: 5px;
        }
        
        .permission-panel {
            background: #1a0000;
            border: 3px solid #f00;
            border-radius: 6px;
            padding: 15px;
            text-align: center;
            display: none;
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            z-index: 9999;
            width: 400px;
            max-width: 90%;
        }
        .permission-panel.active { 
            display: block !important; 
            animation: urgent 0.5s infinite; 
        }
        @keyframes urgent { 0%, 100% { box-shadow: 0 0 30px #f00; } 50% { box-shadow: 0 0 50px #f00; } }
        .permission-panel h3 { 
            color: #f00; 
            margin-bottom: 10px; 
            font-size: 1.1em; 
            text-transform: uppercase;
        }
        .threat-details {
            background: #000;
            padding: 12px;
            border: 2px solid #f00;
            border-radius: 4px;
            margin: 10px 0;
            text-align: left;
            font-size: 0.9em;
        }
        .threat-details p { 
            margin: 5px 0; 
            color: #ff0; 
        }
        .threat-details strong { color: #f00; }
        .btn-group { 
            display: flex; 
            gap: 10px; 
            margin-top: 12px; 
        }
        button {
            flex: 1;
            padding: 12px;
            font-size: 0.95em;
            font-family: 'Courier New', monospace;
            font-weight: bold;
            border: 2px solid;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
            text-transform: uppercase;
        }
        .btn-approve { 
            background: #001a00; 
            color: #0f0; 
            border-color: #0f0; 
        }
        .btn-approve:hover { 
            background: #0f0; 
            color: #000; 
            transform: scale(1.05); 
        }
        .btn-deny { 
            background: #1a0000; 
            color: #f00; 
            border-color: #f00; 
        }
        .btn-deny:hover { 
            background: #f00; 
            color: #000; 
            transform: scale(1.05); 
        }
        
        .threat-spawner {
            background: #1a0000;
            border: 2px solid #f00;
            border-radius: 6px;
            padding: 10px;
        }
        .threat-spawner h3 {
            color: #f00;
            margin-bottom: 6px;
            border-bottom: 1px solid #f00;
            padding-bottom: 4px;
            font-size: 0.95em;
        }
        .btn-spawn {
            width: 100%;
            padding: 10px;
            background: #1a0000;
            color: #f00;
            border: 2px solid #f00;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-weight: bold;
            cursor: pointer;
            margin-top: 6px;
            font-size: 0.9em;
            transition: all 0.2s;
        }
        .btn-spawn:hover { 
            background: #f00; 
            color: #000; 
            transform: scale(1.02); 
        }
        select {
            width: 100%;
            padding: 5px;
            background: #000;
            border: 1px solid #f00;
            color: #f00;
            font-family: 'Courier New', monospace;
            border-radius: 3px;
            font-size: 0.85em;
        }
        
        .patrol-control {
            background: #001a00;
            border: 2px solid #0f0;
            border-radius: 6px;
            padding: 10px;
        }
        .patrol-control h3 {
            color: #0f0;
            margin-bottom: 6px;
            border-bottom: 1px solid #0f0;
            padding-bottom: 4px;
            font-size: 0.95em;
        }
        .input-group { margin: 6px 0; }
        .input-group label {
            display: block;
            color: #0f0;
            margin-bottom: 2px;
            font-size: 0.8em;
        }
        input[type="number"] {
            width: 100%;
            padding: 5px;
            background: #000;
            border: 1px solid #0f0;
            color: #0f0;
            font-family: 'Courier New', monospace;
            border-radius: 3px;
            font-size: 0.85em;
        }
        .btn-update {
            width: 100%;
            padding: 8px;
            background: #001a00;
            color: #0f0;
            border: 2px solid #0f0;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-weight: bold;
            cursor: pointer;
            margin-top: 6px;
            font-size: 0.85em;
            transition: all 0.2s;
        }
        .btn-update:hover { 
            background: #0f0; 
            color: #000; 
            transform: scale(1.02); 
        }
        
        .logs-panel {
            background: #111;
            border: 2px solid #0f0;
            border-radius: 6px;
            padding: 8px;
            max-height: 300px;
            overflow-y: auto;
        }
        .logs-panel::-webkit-scrollbar { width: 6px; }
        .logs-panel::-webkit-scrollbar-track { background: #000; }
        .logs-panel::-webkit-scrollbar-thumb { background: #0f0; border-radius: 3px; }
        .logs-panel h3 {
            color: #0f0;
            margin-bottom: 6px;
            border-bottom: 1px solid #0f0;
            padding-bottom: 4px;
            font-size: 0.95em;
        }
        .log-entry {
            padding: 2px;
            margin: 1px 0;
            border-left: 2px solid;
            padding-left: 4px;
            font-size: 0.7em;
        }
        .log-entry.INFO { border-color: #0f0; color: #0f0; }
        .log-entry.WARNING { border-color: #ff0; color: #ff0; }
        .log-entry.CRITICAL { border-color: #f00; color: #f00; font-weight: bold; }
        .log-time { opacity: 0.7; margin-right: 4px; }
        
        .instructions {
            background: #111;
            border: 2px solid #ff0;
            border-radius: 6px;
            padding: 10px;
            font-size: 0.75em;
            color: #ff0;
        }
        .instructions h3 { margin-bottom: 5px; font-size: 0.95em; }
        .instructions ol { margin-left: 15px; line-height: 1.4; }
                                  
        .permission-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.85);
            z-index: 9998;                          
        }
        .permission-overlay.active {
            display: block !important;
        }
    </style>
</head>
<body>
    <div class="permission-overlay" id="permissionOverlay"></div>                              
    <div class="header">
        <h1>🚁 AUTONOMOUS DRONE COMMAND CENTER</h1>
        <div class="stats">
            <div class="stat">Scans: <span id="scans">0</span></div>
            <div class="threat-level" id="threatLevel">GREEN</div>
        </div>
    </div>
    <div class="container">
        <div class="feeds">
            <div class="feed queen">
                <div class="feed-label">🔥 THERMAL VIEW</div>
                <div style="position: absolute; bottom: 10px; left: 10px; background: rgba(0,0,0,0.9); padding: 10px; border: 2px solid #ff6600; border-radius: 5px; z-index: 20;">
                    <div style="color: #ff6600; font-weight: bold; font-size: 0.9em;">📡 SOURCE: WARRIOR1 CAMERA</div>
                    <div style="color: #ff0; font-size: 0.75em; margin-top: 5px;">Queen is monitoring Warrior's view</div>
                </div>
                <img src="/queen" alt="Queen Feed" />
            </div>
            <div class="feed warrior">
                <div class="feed-label">🌙 NIGHT VISION</div>
                <img src="/warrior" alt="Warrior Feed" />
            </div>
            <div class="feed kamikaze">
                <img src="/kamikaze" alt="Kamikaze Feed" />
            </div>
        </div>
        <div class="control-panel">
            <div class="permission-panel" id="permissionPanel">
                <h3>⚠️ STRIKE AUTHORIZATION REQUIRED ⚠️</h3>
                <div class="threat-details" id="threatInfo"></div>
                <div class="btn-group">
                    <button id="btnApprove" class="btn-approve">✅ Authorize</button>
                    <button id="btnDeny" class="btn-deny">❌ Deny</button>
                </div>
            </div>
            
            <div class="instructions">
                <h3>📖 SYSTEM STATUS:</h3>
                <ol>
                    <li>Queen monitors Warrior's camera (thermal)</li>
                    <li>Warrior patrols with night vision</li>
                    <li>AI detects threats autonomously</li>
                    <li>You have 15 seconds to authorize strikes</li>
                </ol>
            </div>
            
            <div class="threat-spawner">
                <h3>🚨 MANUAL THREAT SPAWNER</h3>
                <div class="input-group">
                    <label>Threat Type</label>
                    <select id="threatType">
                        <option value="person">👤 Person</option>
                        <option value="car">🚗 Car</option>
                        <option value="bus">🚌 Bus</option>
                        <option value="truck">🚚 Truck</option>
                    </select>
                </div>
                <div class="input-group">
                    <label>X Coordinate (meters)</label>
                    <input type="number" id="threatX" value="50" step="10">
                </div>
                <div class="input-group">
                    <label>Y Coordinate (meters)</label>
                    <input type="number" id="threatY" value="50" step="10">
                </div>
                <button id="btnSpawn" class="btn-spawn">🎯 SPAWN THREAT</button>
            </div>
            
            <div class="patrol-control">
                <h3>🛸 WARRIOR PATROL AREA</h3>
                <div class="input-group">
                    <label>Center X (meters)</label>
                    <input type="number" id="centerX" value="0" step="10">
                </div>
                <div class="input-group">
                    <label>Center Y (meters)</label>
                    <input type="number" id="centerY" value="0" step="10">
                </div>
                <div class="input-group">
                    <label>Radius (meters)</label>
                    <input type="number" id="radius" value="30" step="5" min="10" max="100">
                </div>
                <button id="btnUpdate" class="btn-update">🔄 UPDATE PATROL</button>
            </div>
            
            <div class="logs-panel">
                <h3>📋 MISSION LOGS</h3>
                <div id="logs"></div>
            </div>
        </div>
    </div>
    
    <script>
    // Wrap everything in DOMContentLoaded so elements exist and the script won't run partially
    document.addEventListener('DOMContentLoaded', () => {
        let lastThreatId = null;
        const overlay = document.getElementById('permissionOverlay');
        const panel = document.getElementById('permissionPanel');
        const threatInfo = document.getElementById('threatInfo');

        function safeAddClass(el, cls){ if(el) el.classList.add(cls); }
        function safeRemoveClass(el, cls){ if(el) el.classList.remove(cls); }

        function updateLogs() {
            fetch('/logs').then(r => r.json()).then(logs => {
                try {
                    const logsDiv = document.getElementById('logs');
                    const wasAtBottom = logsDiv.scrollHeight - logsDiv.clientHeight <= logsDiv.scrollTop + 1;
                    
                    logsDiv.innerHTML = logs.reverse().slice(0, 100).map(log => 
                        `<div class="log-entry ${log.level}"><span class="log-time">[${log.time}]</span><span>[${log.source}]</span> ${log.message}</div>`
                    ).join('');
                    
                    if (wasAtBottom) {
                        logsDiv.scrollTop = logsDiv.scrollHeight;
                    }
                } catch (err) {
                    console.error('updateLogs render error:', err);
                }
            }).catch(err => console.error('Logs error:', err));
        }

        function updateStatus() {
            fetch('/status').then(r => r.json()).then(status => {
                try {
                    console.log('[STATUS CHECK]', {
                        pending: status.pending_permission, 
                        threat: status.active_threat !== null
                    });
                    
                    document.getElementById('threatLevel').textContent = status.threat_level || 'GREEN';
                    document.getElementById('threatLevel').className = 'threat-level ' + (status.threat_level || 'GREEN');
                    document.getElementById('scans').textContent = status.queen_scans || 0;
                    
                    if (status.pending_permission && status.active_threat) {
                        const threatId = JSON.stringify(status.active_threat);
                        
                        if (threatId !== lastThreatId) {
                            lastThreatId = threatId;
                            console.log('🚨 AUTHORIZATION REQUIRED!', status.active_threat);
                        }
                        
                        if (panel) panel.style.display = 'block';
                        safeAddClass(panel, 'active');
                        safeAddClass(overlay, 'active');
                        
                        if (threatInfo && status.active_threat) {
                            const ap = status.active_threat;
                            const worldPos = (ap.world_pos && ap.world_pos.length >= 2) ? ap.world_pos : [0,0];
                            threatInfo.innerHTML = `
                                <p><strong>⚠️ IMMEDIATE ACTION REQUIRED ⚠️</strong></p>
                                <p><strong>Target:</strong> ${String(ap.class || 'UNKNOWN').toUpperCase()}</p>
                                <p><strong>Location:</strong> X:${(worldPos[0] || 0).toFixed(1)}m, Y:${(worldPos[1] || 0).toFixed(1)}m</p>
                                <p><strong>Confidence:</strong> ${((ap.confidence || 0) * 100).toFixed(0)}%</p>
                                <p style="color: #f00; font-weight: bold; margin-top: 10px; font-size: 1.2em;">⏰ 15 SECONDS TO DECIDE!</p>
                            `;
                        }
                    } else {
                        if (panel) panel.style.display = 'none';
                        safeRemoveClass(panel, 'active');
                        safeRemoveClass(overlay, 'active');
                        lastThreatId = null;
                    }
                } catch (err) {
                    console.error('updateStatus render error:', err);
                }
            }).catch(err => console.error('Status error:', err));
        }

        function approve() { 
            console.log('✅ User authorized strike');
            fetch('/approve', {method: 'POST'})
                .then(() => alert('✅ STRIKE AUTHORIZED\n\nKamikaze deploying!'))
                .catch(err => console.error('Approve error:', err));
        }
        
        function deny() { 
            console.log('❌ User denied strike');
            fetch('/deny', {method: 'POST'})
                .then(() => alert('❌ STRIKE DENIED\n\nContinuing patrol.'))
                .catch(err => console.error('Deny error:', err));
        }

        function spawnThreat() {
            const x = parseFloat(document.getElementById('threatX').value);
            const y = parseFloat(document.getElementById('threatY').value);
            const type = document.getElementById('threatType').value;
            
            console.log('Spawning threat:', type, 'at', x, y);
            
            fetch('/spawn_threat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({x, y, type})
            }).then(r => r.json()).then(data => {
                if (data.error) {
                    alert('❌ ERROR: ' + data.error);
                } else {
                    alert(`🚨 THREAT SPAWNED!\n\nType: ${type.toUpperCase()}\nLocation: (${x}m, ${y}m)`);
                }
            }).catch(err => {
                console.error('Spawn error:', err);
                alert('❌ Failed to spawn threat');
            });
        }

        function updatePatrol() {
            const x = parseFloat(document.getElementById('centerX').value);
            const y = parseFloat(document.getElementById('centerY').value);
            const radius = parseFloat(document.getElementById('radius').value);
            
            console.log('Updating patrol:', {x, y, radius});
            
            fetch('/set_patrol', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({x, y, radius})
            })
            .then(r => r.json())
            .then(data => {
                console.log('Patrol updated:', data);
                alert(`✅ PATROL UPDATED!\n\nCenter: (${x}m, ${y}m)\nRadius: ${radius}m\n\nWarrior will move on next waypoint.`);
            })
            .catch(err => {
                console.error('Patrol error:', err);
                alert('❌ Failed to update patrol');
            });
        }

        // wire up buttons (no inline onclick)
        const btnApprove = document.getElementById('btnApprove');
        const btnDeny = document.getElementById('btnDeny');
        const btnSpawn = document.getElementById('btnSpawn');
        const btnUpdate = document.getElementById('btnUpdate');

        if (btnApprove) btnApprove.addEventListener('click', approve);
        if (btnDeny) btnDeny.addEventListener('click', deny);
        if (btnSpawn) btnSpawn.addEventListener('click', spawnThreat);
        if (btnUpdate) btnUpdate.addEventListener('click', updatePatrol);

        // Init: load patrol + start polling
        fetch('/get_patrol')
            .then(r => r.json())
            .then(data => {
                if (data) {
                    try {
                        document.getElementById('centerX').value = data.center_x;
                        document.getElementById('centerY').value = data.center_y;
                        document.getElementById('radius').value = data.radius;
                    } catch (e) { console.warn('Failed to init patrol fields:', e); }
                }
            })
            .catch(err => console.error('Get patrol error:', err));

        // Start polling
        updateLogs();
        updateStatus();
        setInterval(updateLogs, 500);
        setInterval(updateStatus, 200);

        console.log('✅ Command Center initialized and monitoring');
    }); // end DOMContentLoaded
    </script>
</body>
</html>
''')

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌐 COMMAND CENTER WEB UI")
    print("="*70)
    print("\n📡 Starting server...")
    print("   URL: http://localhost:5000")
    print("\n⚠️  IMPORTANT:")
    print("   1. Open http://localhost:5000 in browser")
    print("   2. Press F12 to open console (for debugging)")
    print("   3. Then run: python main.py")
    print("="*70 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
    except Exception as e:
        print(f"\n❌ FATAL ERROR:")
        print(f"   {e}")
        print("\n💡 SOLUTION:")
        print("   - If port 5000 is in use, change to port 5001")
        print("   - Make sure swarm_state.py exists in same folder")
