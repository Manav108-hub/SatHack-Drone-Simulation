import airsim
import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify, request
import time
from swarm_state import swarm

app = Flask(__name__)

# Create ONE client per drone (reuse it)
clients = {}

def get_client(drone):
    if drone not in clients:
        clients[drone] = airsim.MultirotorClient()
        clients[drone].confirmConnection()
    return clients[drone]

def get_frame(drone, use_ai=False):
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
        
        if use_ai:
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')
            results = model(img, verbose=False, conf=0.4)
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
        
        colors = {"Queen": (255, 215, 0), "Warrior1": (0, 255, 0), "Kamikaze1": (255, 0, 0)}
        color = colors.get(drone, (255, 255, 255))
        
        cv2.rectangle(img, (0, 0), (250, 75), (0, 0, 0), -1)
        cv2.putText(img, drone, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        
        try:
            pos = client.simGetVehiclePose(drone).position
            pos_text = f"X:{pos.x_val:.0f} Y:{pos.y_val:.0f} Z:{pos.z_val:.0f}"
            cv2.putText(img, pos_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except:
            pass
        
        timestamp = time.strftime("%H:%M:%S")
        cv2.rectangle(img, (img.shape[1]-150, 0), (img.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(img, timestamp, (img.shape[1]-140, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes()
        
    except Exception as e:
        print(f"Frame error {drone}: {e}")
        return None

def gen(drone, use_ai=False):
    while True:
        frame = get_frame(drone, use_ai)
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.1)

@app.route('/queen')
def queen():
    return Response(gen("Queen", True), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/warrior')
def warrior():
    return Response(gen("Warrior1", False), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/kamikaze')
def kamikaze():
    return Response(gen("Kamikaze1", False), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs')
def logs():
    return jsonify(swarm.get_logs())

@app.route('/status')
def status():
    return jsonify({
        'threat_level': swarm.threat_level,
        'pending_permission': swarm.pending_permission,
        'active_threat': swarm.active_threat,
        'warrior_status': swarm.get_warrior_status(),
        'kamikaze_deployed': swarm.kamikaze_deployed,
        'queen_scans': swarm.queen_scans,
        'patrol_area': {
            'center_x': swarm.patrol_center_x,
            'center_y': swarm.patrol_center_y,
            'radius': swarm.patrol_radius
        }
    })

@app.route('/approve', methods=['POST'])
def approve():
    swarm.user_response = True
    return jsonify({'status': 'approved'})

@app.route('/deny', methods=['POST'])
def deny():
    swarm.user_response = False
    return jsonify({'status': 'denied'})

@app.route('/set_patrol', methods=['POST'])
def set_patrol():
    data = request.json
    cx = float(data.get('x', 0))
    cy = float(data.get('y', 0))
    radius = float(data.get('radius', 30))
    
    swarm.set_patrol_area(cx, cy, radius)
    swarm.log("SYSTEM", f"Patrol: ({cx:.0f}, {cy:.0f}) R:{radius:.0f}m", "INFO")
    
    return jsonify({'center_x': cx, 'center_y': cy, 'radius': radius})

@app.route('/get_patrol')
def get_patrol():
    return jsonify({
        'center_x': swarm.patrol_center_x,
        'center_y': swarm.patrol_center_y,
        'radius': swarm.patrol_radius
    })

@app.route('/spawn_threat', methods=['POST'])
def spawn_threat():
    """Manually spawn a threat at specified coordinates"""
    data = request.json
    x = float(data.get('x', 10))
    y = float(data.get('y', 10))
    threat_type = data.get('type', 'person')
    
    # Create manual threat
    threat = {
        'class': threat_type,
        'confidence': 1.0,  # 100% confidence for manual threats
        'world_pos': (x, y),
        'timestamp': time.time()
    }
    
    swarm.add_threat(threat)
    swarm.log("MANUAL", f"Threat spawned: {threat_type} at ({x}, {y})", "CRITICAL")
    
    return jsonify({'status': 'spawned', 'threat': threat})

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🚁 COMMAND CENTER</title>
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
            }
            .feed.queen { border-color: #ffd700; }
            .feed.warrior { border-color: #0f0; }
            .feed.kamikaze { border-color: #f00; }
            .feed img {
                width: 100%;
                height: 100%;
                object-fit: contain;
                display: block;
            }
            
            .control-panel {
                display: flex;
                flex-direction: column;
                gap: 8px;
                max-height: calc(100vh - 60px);
                overflow-y: auto;
                padding-right: 5px;
            }
            
            .control-panel::-webkit-scrollbar {
                width: 8px;
            }
            .control-panel::-webkit-scrollbar-track {
                background: #0a0a0a;
            }
            .control-panel::-webkit-scrollbar-thumb {
                background: #0f0;
                border-radius: 4px;
            }
            
            .permission-panel {
                background: #1a0000;
                border: 3px solid #f00;
                border-radius: 6px;
                padding: 12px;
                text-align: center;
                display: none;
            }
            .permission-panel.active { display: block; animation: urgent 0.5s infinite; }
            @keyframes urgent { 0%, 100% { box-shadow: 0 0 20px #f00; } 50% { box-shadow: 0 0 40px #f00; } }
            .permission-panel h3 { color: #f00; margin-bottom: 8px; font-size: 0.95em; }
            .threat-details {
                background: #000;
                padding: 8px;
                border: 1px solid #f00;
                border-radius: 4px;
                margin: 8px 0;
                text-align: left;
                font-size: 0.8em;
            }
            .threat-details p { margin: 2px 0; color: #ff0; }
            .btn-group { display: flex; gap: 8px; margin-top: 8px; }
            button {
                flex: 1;
                padding: 8px;
                font-size: 0.85em;
                font-family: 'Courier New', monospace;
                font-weight: bold;
                border: 2px solid;
                border-radius: 4px;
                cursor: pointer;
                transition: all 0.2s;
            }
            .btn-approve { background: #001a00; color: #0f0; border-color: #0f0; }
            .btn-approve:hover { background: #0f0; color: #000; }
            .btn-deny { background: #1a0000; color: #f00; border-color: #f00; }
            .btn-deny:hover { background: #f00; color: #000; }
            
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
            .btn-spawn:hover { background: #f00; color: #000; transform: scale(1.02); }
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
            .btn-update:hover { background: #0f0; color: #000; transform: scale(1.02); }
            
            .logs-panel {
                background: #111;
                border: 2px solid #0f0;
                border-radius: 6px;
                padding: 8px;
                max-height: 300px;
                overflow-y: auto;
            }
            .logs-panel::-webkit-scrollbar {
                width: 6px;
            }
            .logs-panel::-webkit-scrollbar-track {
                background: #000;
            }
            .logs-panel::-webkit-scrollbar-thumb {
                background: #0f0;
                border-radius: 3px;
            }
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
            .instructions h3 {
                margin-bottom: 5px;
                font-size: 0.95em;
            }
            .instructions ol {
                margin-left: 15px;
                line-height: 1.4;
            }
        </style>
    </head>
    <body>
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
                    <img src="/queen" alt="Queen Feed" />
                </div>
                <div class="feed warrior">
                    <img src="/warrior" alt="Warrior Feed" />
                </div>
                <div class="feed kamikaze">
                    <img src="/kamikaze" alt="Kamikaze Feed" />
                </div>
            </div>
            <div class="control-panel">
                <div class="permission-panel" id="permissionPanel">
                    <h3>⚠️ STRIKE AUTHORIZATION REQUIRED</h3>
                    <div class="threat-details" id="threatInfo"></div>
                    <div class="btn-group">
                        <button class="btn-approve" onclick="approve()">✅ AUTHORIZE</button>
                        <button class="btn-deny" onclick="deny()">❌ DENY</button>
                    </div>
                </div>
                
                <div class="instructions">
                    <h3>📖 HOW TO USE:</h3>
                    <ol>
                        <li>Set patrol area coordinates</li>
                        <li>Click "UPDATE PATROL"</li>
                        <li>Spawn a threat at desired location</li>
                        <li>Authorize or deny strike</li>
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
                    <button class="btn-spawn" onclick="spawnThreat()">🎯 SPAWN THREAT</button>
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
                    <button class="btn-update" onclick="updatePatrol()">🔄 UPDATE PATROL</button>
                </div>
                
                <div class="logs-panel">
                    <h3>📋 MISSION LOGS</h3>
                    <div id="logs"></div>
                </div>
            </div>
        </div>
        
        <script>
            function updateLogs() {
                fetch('/logs').then(r => r.json()).then(logs => {
                    const logsDiv = document.getElementById('logs');
                    const wasAtBottom = logsDiv.scrollHeight - logsDiv.clientHeight <= logsDiv.scrollTop + 1;
                    
                    logsDiv.innerHTML = logs.reverse().slice(0, 100).map(log => 
                        `<div class="log-entry ${log.level}"><span class="log-time">[${log.time}]</span><span>[${log.source}]</span> ${log.message}</div>`
                    ).join('');
                    
                    if (wasAtBottom) {
                        logsDiv.scrollTop = logsDiv.scrollHeight;
                    }
                });
            }
            
            function updateStatus() {
                fetch('/status').then(r => r.json()).then(status => {
                    document.getElementById('threatLevel').textContent = status.threat_level;
                    document.getElementById('threatLevel').className = 'threat-level ' + status.threat_level;
                    document.getElementById('scans').textContent = status.queen_scans || 0;
                    
                    const panel = document.getElementById('permissionPanel');
                    if (status.pending_permission && status.active_threat) {
                        panel.classList.add('active');
                        document.getElementById('threatInfo').innerHTML = `
                            <p><strong>Target Type:</strong> ${status.active_threat.class}</p>
                            <p><strong>Location:</strong> X:${status.active_threat.world_pos[0].toFixed(1)}m Y:${status.active_threat.world_pos[1].toFixed(1)}m</p>
                            <p><strong>Confidence:</strong> ${(status.active_threat.confidence * 100).toFixed(0)}%</p>
                        `;
                    } else {
                        panel.classList.remove('active');
                    }
                });
            }
            
            function approve() { 
                fetch('/approve', {method: 'POST'})
                    .then(() => console.log('Strike authorized'));
            }
            
            function deny() { 
                fetch('/deny', {method: 'POST'})
                    .then(() => console.log('Strike denied'));
            }
            
            function spawnThreat() {
                const x = parseFloat(document.getElementById('threatX').value);
                const y = parseFloat(document.getElementById('threatY').value);
                const type = document.getElementById('threatType').value;
                
                fetch('/spawn_threat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({x, y, type})
                }).then(r => r.json()).then(data => {
                    alert(`🚨 THREAT SPAWNED!

Type: ${type.toUpperCase()}
Location: (${x}m, ${y}m)

Watch the authorization panel!`);
                });
            }
            
            function updatePatrol() {
                const x = parseFloat(document.getElementById('centerX').value);
                const y = parseFloat(document.getElementById('centerY').value);
                const radius = parseFloat(document.getElementById('radius').value);
                
                fetch('/set_patrol', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({x, y, radius})
                }).then(() => {
                    alert(`✅ PATROL AREA UPDATED!

Center: (${x}m, ${y}m)
Radius: ${radius}m

Warrior will move to new area on next waypoint.`);
                });
            }
            
            // Load initial patrol settings
            fetch('/get_patrol').then(r => r.json()).then(data => {
                document.getElementById('centerX').value = data.center_x;
                document.getElementById('centerY').value = data.center_y;
                document.getElementById('radius').value = data.radius;
            });
            
            setInterval(updateLogs, 500);
            setInterval(updateStatus, 200);
            updateLogs();
            updateStatus();
        </script>
    </body>
    </html>
    ''')

if __name__ == '__main__':
    print("🌐 http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)