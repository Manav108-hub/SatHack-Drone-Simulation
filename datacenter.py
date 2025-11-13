import airsim
import cv2
import numpy as np
from flask import Flask, Response, render_template_string
import time
from ultralytics import YOLO

app = Flask(__name__)

def get_frame(drone, use_ai=False):
    try:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ], drone)
        
        if responses and len(responses[0].image_data_uint8) > 0:
            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img = img1d.reshape(responses[0].height, responses[0].width, 3)
            
            # Make writable copy
            img = np.copy(img)
            
            # AI detection for Queen
            if use_ai:
                model = YOLO('yolov8n.pt')
                results = model(img, verbose=False, conf=0.5)
                threat_classes = {0: 'person', 2: 'car', 7: 'truck'}
                
                for box in results[0].boxes:
                    if int(box.cls) in threat_classes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf)
                        label = f"{threat_classes[int(box.cls)]} {conf:.2f}"
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Drone name
            color = (255, 215, 0) if drone == "Queen" else (0, 255, 0) if drone == "Warrior1" else (255, 0, 0)
            cv2.putText(img, drone, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return buffer.tobytes()
    except Exception as e:
        print(f"Error {drone}: {e}")
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

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🚁 DATACENTER</title>
        <style>
            body { margin: 0; background: #000; color: #0f0; font-family: monospace; }
            .header { background: #111; padding: 20px; text-align: center; border-bottom: 3px solid #0f0; }
            h1 { margin: 0; color: #0f0; text-shadow: 0 0 10px #0f0; }
            .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; padding: 20px; }
            .feed { border: 3px solid; border-radius: 10px; overflow: hidden; }
            .feed.queen { border-color: #ffd700; }
            .feed.warrior { border-color: #0f0; }
            .feed.kamikaze { border-color: #f00; }
            .feed h2 { margin: 0; padding: 15px; background: #0a0a0a; text-align: center; }
            .feed.queen h2 { color: #ffd700; }
            .feed.warrior h2 { color: #0f0; }
            .feed.kamikaze h2 { color: #f00; }
            .feed img { width: 100%; height: 400px; object-fit: cover; background: #000; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚁 DRONE SWARM DATACENTER</h1>
        </div>
        <div class="grid">
            <div class="feed queen">
                <h2>👑 QUEEN</h2>
                <img src="/queen" />
            </div>
            <div class="feed warrior">
                <h2>🛸 WARRIOR</h2>
                <img src="/warrior" />
            </div>
            <div class="feed kamikaze">
                <h2>💥 KAMIKAZE</h2>
                <img src="/kamikaze" />
            </div>
        </div>
    </body>
    </html>
    ''')

if __name__ == '__main__':
    print("🌐 http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)