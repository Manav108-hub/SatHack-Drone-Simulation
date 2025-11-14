# 🚁 Autonomous Drone Swarm System - Hive Intelligence

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![AirSim](https://img.shields.io/badge/AirSim-Simulation-green.svg)](https://github.com/Microsoft/AirSim)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-AI%20Detection-red.svg)](https://github.com/ultralytics/ultralytics)
[![Flask](https://img.shields.io/badge/Flask-Web%20UI-lightgrey.svg)](https://flask.palletsprojects.com/)

An advanced autonomous drone swarm system featuring hierarchical command structure, AI-powered threat detection, and real-time web-based mission control. Built for a hackathon project demonstrating swarm intelligence and human-machine collaboration.

---

## 🎯 Project Overview

This system implements a three-tier drone hierarchy inspired by nature's hive structures:

- **👑 Queen Drone**: Command center with YOLOv8 AI for real-time threat detection
- **🛸 Warrior Drones**: Autonomous patrol units providing surveillance and reconnaissance
- **💥 Kamikaze Drones**: Strike-capable units deployed on authorized threats

### Key Features

✅ **Autonomous Swarm Coordination** - Multi-drone communication via shared state management  
✅ **AI-Powered Threat Detection** - YOLOv8n neural network for real-time object recognition  
✅ **Human-in-the-Loop Control** - Web-based authorization for critical strike decisions  
✅ **Real-Time Camera Feeds** - Live video streams from all drones with AI detection overlays  
✅ **Dynamic Patrol Patterns** - Multi-altitude scanning with configurable patrol zones  
✅ **Plug-and-Play Architecture** - Modular design compatible with various drone hardware

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    COMMAND CENTER                        │
│            (Flask Web Interface - Port 5000)             │
│  • Live camera feeds from all drones                     │
│  • Strike authorization panel                            │
│  • Mission logs and threat alerts                        │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              SHARED STATE MANAGER (swarm_state.py)       │
│  • Thread-safe inter-drone communication                 │
│  • Threat queue management                               │
│  • Mission status synchronization                        │
└─────────────────────────────────────────────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌─────────────┐    ┌──────────────────┐   ┌──────────────┐
│ QUEEN DRONE │    │ WARRIOR DRONES   │   │  KAMIKAZES   │
│ (AI Brain)  │◄───┤  (Patrol Units)  │   │ (Strike Unit)│
│             │    │                  │   │              │
│ • YOLOv8    │    │ • Multi-altitude │   │ • Standby    │
│ • Detection │    │ • Circular scan  │   │ • Fast strike│
│ • Command   │    │ • Feed relay     │   │ • Precision  │
└─────────────┘    └──────────────────┘   └──────────────┘
```

---

## 📋 Project Structure

```
DroneSwarm/
├── main.py                    # Main orchestrator - launches all drones
├── swarm_state.py             # Shared state management for inter-drone comms
├── queen.py                   # Queen drone AI controller
├── warrior_multi.py           # Multi-warrior patrol system
├── kamikaze_multi.py          # Multi-kamikaze strike system
├── datacenter.py              # Flask web interface server
├── requirements.txt           # Python dependencies
└── Documents/
    └── AirSim/
        └── settings.json      # AirSim drone configuration
```

---

## 🚀 Quick Start Guide

### Prerequisites

**Required Software:**
- Python 3.8 or higher
- AirSim/Colosseum (Unreal Engine-based simulator)
- Windows 10/11 (for AirSim compatibility)

**Hardware Requirements:**
- 16GB RAM minimum (32GB recommended for smooth simulation)
- NVIDIA GPU recommended (for AI model and UE5 rendering)
- 50GB free disk space

### Installation Steps

#### 1. Clone or Download the Project

```bash
git clone https://github.com/yourusername/drone-swarm-system.git
cd drone-swarm-system
```

#### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install airsim opencv-python numpy ultralytics flask
```

#### 3. Install AirSim/Colosseum

**Option A: Pre-built Binary (Recommended)**
1. Download AirSim Neighbourhood environment from [AirSim Releases](https://github.com/microsoft/AirSim/releases)
2. Extract to `D:\AirSim\` (or your preferred location)
3. Run the `.exe` file to launch simulation

**Option B: Build from Source**
1. Install Unreal Engine 5.7.1
2. Clone Colosseum repository
3. Run `build.cmd`
4. Open environment in Unreal Editor

#### 4. Configure AirSim Settings

Create/edit: `C:\Users\[YourUsername]\Documents\AirSim\settings.json`

```json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "Vehicles": {
    "Queen": {
      "VehicleType": "SimpleFlight",
      "X": 0, "Y": 0, "Z": 0,
      "Cameras": {
        "front_center": {
          "CaptureSettings": [{"ImageType": 0, "Width": 1280, "Height": 720}]
        }
      }
    },
    "Warrior1": {
      "VehicleType": "SimpleFlight",
      "X": 30, "Y": 0, "Z": 0,
      "Cameras": {
        "front_center": {
          "CaptureSettings": [{"ImageType": 0, "Width": 1280, "Height": 720}]
        }
      }
    },
    "Warrior2": {
      "VehicleType": "SimpleFlight",
      "X": -30, "Y": 30, "Z": 0,
      "Cameras": {
        "front_center": {
          "CaptureSettings": [{"ImageType": 0, "Width": 1280, "Height": 720}]
        }
      }
    },
    "Kamikaze1": {
      "VehicleType": "SimpleFlight",
      "X": -30, "Y": -30, "Z": 0,
      "Cameras": {
        "front_center": {
          "CaptureSettings": [{"ImageType": 0, "Width": 640, "Height": 480}]
        }
      }
    }
  }
}
```

#### 5. Download AI Model

The YOLOv8n model will download automatically on first run, or manually:

```bash
# This happens automatically when running queen.py
pip install ultralytics
```

---

## 🎮 Running the System

### Step 1: Launch AirSim Simulation

1. Navigate to your AirSim installation folder
2. Run the executable (e.g., `AirSimNH.exe`)
3. Wait for the environment to fully load
4. Press **Play** in Unreal Engine (if using source build)

### Step 2: Start the Drone Swarm

Open terminal in project directory:

```bash
# Activate virtual environment
venv\Scripts\activate

# Launch all drones
python main.py
```

**Expected Output:**
```
======================================================================
🚁 AUTONOMOUS DRONE SWARM - HIVE INTELLIGENCE
======================================================================
📋 DRONES:
   👑 Queen: AI threat detection (YOLOv8)
   🛸 Warriors: Autonomous patrol (2 units)
   💥 Kamikazes: Strike on command (1 unit)
======================================================================
Press ENTER to launch...
🚀 LAUNCHING IN 3...
2...
1...

Connected to AirSim!
📥 Loading AI model...
✅ AI loaded!
👑 QUEEN STARTING...
🛸 WARRIOR1 STARTING...
🛸 WARRIOR2 STARTING...
💤 KAMIKAZE1 STARTING...
```

### Step 3: Open Command Center Interface

Open a second terminal:

```bash
python datacenter.py
```

Then navigate to: **http://localhost:5000**

---

## 🖥️ Web Interface Guide

The command center provides real-time monitoring and control:

### Dashboard Layout

```
┌─────────────────────────────────────────────────────────┐
│               🚁 DRONE SWARM COMMAND CENTER              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │  QUEEN   │  │ WARRIOR1 │  │KAMIKAZE1 │              │
│  │  [FEED]  │  │  [FEED]  │  │  [FEED]  │              │
│  └──────────┘  └──────────┘  └──────────┘              │
│                                                          │
│  📊 MISSION STATUS                                       │
│  • Patrol Active: YES                                    │
│  • Threats Detected: 0                                   │
│  • Kamikazes Available: 1                                │
│                                                          │
│  📜 MISSION LOG                                          │
│  [10:23:45] Warrior1 scanning sector A3                 │
│  [10:23:47] Queen analyzing feed from Warrior2          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Threat Authorization Panel

When a threat is detected:

```
┌─────────────────────────────────────────┐
│  🚨 THREAT DETECTED                      │
│  Target: Person (85% confidence)         │
│  Location: (X:25.0, Y:20.0)              │
│  Source: Warrior1                        │
│                                          │
│  [ AUTHORIZE STRIKE ]  [ DISMISS ]       │
└─────────────────────────────────────────┘
```

### Interactive Features

- **Double-click any feed** to zoom to full screen
- **Click "Next Warrior"** to cycle through warrior feeds
- **Authorize Strike** button appears when threats detected
- **Mission logs** auto-scroll with new events

---

## 🧠 How It Works

### 1. Queen Drone (AI Command)

**Role**: Central intelligence and command authority

**Behavior**:
- Maintains stationary position at (0, 0, -20) altitude
- Continuously monitors camera feeds from all warrior drones
- Runs YOLOv8n neural network for object detection
- Identifies threats: vehicles, persons, weapons
- Requests human authorization for strike deployment
- Coordinates kamikaze launch on approval

**Key Code** (`queen.py`):
```python
def detect_threats(self):
    # Fetch warrior camera feed
    responses = self.client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.Scene)
    ], vehicle_name="Warrior1")
    
    # Run AI detection
    results = self.model(img, conf=0.35)
    
    # Analyze detections
    for detection in results[0].boxes:
        if detection.conf > 0.5:  # High confidence threat
            threat_data = {
                'class': detection.class_name,
                'confidence': detection.conf,
                'position': calculate_world_position(detection)
            }
            swarm.add_threat(threat_data)
```

### 2. Warrior Drones (Patrol Units)

**Role**: Reconnaissance and surveillance

**Behavior**:
- Execute circular patrol patterns at defined altitudes
- Warrior1: 300m radius at -20m altitude
- Warrior2: 400m radius at -30m altitude
- Hover for 3 seconds at each waypoint for detailed scanning
- Stream live camera feeds to command center
- Automatically relay position data to Queen

**Patrol Pattern**:
```python
# Multi-altitude circular patrol
patrol_radius = 300  # meters
num_waypoints = 12
altitude = -20  # meters

for angle in range(0, 360, 30):
    x = patrol_center[0] + radius * cos(radians(angle))
    y = patrol_center[1] + radius * sin(radians(angle))
    waypoint = (x, y, altitude)
    
    # Move to waypoint and hover
    self.move_to_position(waypoint)
    time.sleep(3)  # Scanning pause
```

### 3. Kamikaze Drones (Strike Units)

**Role**: Precision strike capability

**Behavior**:
- Remain in standby mode until deployment order
- Maintain position at designated coordinates
- Upon authorization:
  - Immediately engage target at maximum velocity
  - Execute direct path to threat coordinates
  - Simulate strike on arrival
  - Log mission completion

**Strike Sequence**:
```python
def execute_strike(self, target_pos):
    logger.info(f"🎯 Strike authorized! Target: {target_pos}")
    
    # Fast approach
    self.client.moveToPositionAsync(
        target_pos[0], target_pos[1], target_pos[2],
        velocity=30,  # Maximum speed
        vehicle_name=self.vehicle_name
    ).join()
    
    logger.info("💥 TARGET ELIMINATED!")
```

### 4. Shared State Management

**Purpose**: Thread-safe communication between all drones

**Features**:
- Threat queue with priority handling
- Mission status flags
- Authorization state tracking
- Kamikaze deployment coordination

**Key Methods**:
```python
class SwarmState:
    def add_threat(self, threat_data):
        with self.lock:
            self.threats.append(threat_data)
            self.pending_authorization = True
    
    def authorize_strike(self):
        with self.lock:
            self.strike_authorized = True
            return self.active_threat
```

---

## 🛠️ Configuration & Customization

### Adjusting Patrol Parameters

Edit `warrior_multi.py`:

```python
# Warrior patrol configuration
PATROL_CONFIGS = [
    {
        'name': 'Warrior1',
        'center': (0, 0),
        'radius': 300,      # Patrol radius in meters
        'altitude': -20,    # Flight altitude (negative = up)
        'waypoints': 12,    # Number of patrol points
        'hover_time': 3     # Seconds to hover at each point
    },
    # Add more warriors...
]
```

### AI Detection Sensitivity

Edit `queen.py`:

```python
# YOLOv8 detection parameters
results = self.model(
    img,
    conf=0.35,        # Confidence threshold (0.0-1.0)
    iou=0.45,         # IoU threshold for NMS
    classes=[0,2,3]   # Filter classes: 0=person, 2=car, 3=motorcycle
)
```

### Adding More Drones

1. **Update `settings.json`**:
```json
"Warrior3": {
  "VehicleType": "SimpleFlight",
  "X": 0, "Y": 50, "Z": 0,
  "Cameras": { ... }
}
```

2. **Add to `warrior_multi.py` config list**:
```python
PATROL_CONFIGS.append({
    'name': 'Warrior3',
    'center': (0, 0),
    'radius': 500,
    'altitude': -40
})
```

3. **Restart system**

---

## 🎥 Camera Configuration

### High-Quality Feeds

For better video quality, edit `settings.json`:

```json
"Cameras": {
  "front_center": {
    "CaptureSettings": [{
      "ImageType": 0,
      "Width": 1920,        # Increase resolution
      "Height": 1080,
      "FOV_Degrees": 90,
      "AutoExposureSpeed": 100,
      "MotionBlurAmount": 0
    }]
  }
}
```

**Note**: Higher resolutions require more GPU/RAM

### Multiple Camera Angles

```json
"Cameras": {
  "front_center": {...},
  "bottom_center": {
    "X": 0, "Y": 0, "Z": 0,
    "Pitch": -90,  # Points downward
    "Roll": 0, "Yaw": 0,
    "CaptureSettings": [...]
  }
}
```

---

## 🐛 Troubleshooting

### Issue: Drones Not Appearing in Simulation

**Symptoms**: Python connects but no drones visible

**Solutions**:
1. Check `settings.json` is in correct location: `C:\Users\[You]\Documents\AirSim\`
2. Restart simulation completely
3. Verify spawn coordinates aren't underground (Z should be 0 or slightly above)
4. Press **F1** in simulator to cycle camera views

---

### Issue: "Connection Timeout" Error

**Symptoms**: 
```
TimeoutError: Timeout waiting for response from AirSim
```

**Solutions**:
1. Ensure AirSim is fully loaded (wait 20-30 seconds after launch)
2. Check firewall isn't blocking localhost connections
3. Verify no other Python scripts are using AirSim API
4. Restart both simulation and Python script

---

### Issue: Warriors Not Moving

**Symptoms**: Warriors stay at spawn location

**Solutions**:
1. Check logs for "moveToPositionAsync" errors
2. Verify patrol radius isn't too large for environment
3. Ensure altitude values are negative (up) not positive
4. Add debug logging:
```python
logger.info(f"Moving to: {waypoint}")
future = self.client.moveToPositionAsync(...)
logger.info("Move command sent")
```

---

### Issue: No AI Detections

**Symptoms**: Queen never detects threats

**Solutions**:
1. Verify YOLOv8 model downloaded: `~/.ultralytics/yolov8n.pt` exists
2. Check environment has detectable objects (cars, people)
3. Lower confidence threshold in `queen.py`:
```python
results = self.model(img, conf=0.25)  # Lower from 0.35
```
4. Add simulated threat for testing:
```python
def detect_threats(self):
    if time.time() - self.start_time > 30:  # After 30 seconds
        return {
            'class': 'person',
            'confidence': 0.85,
            'world_pos': (25, 25)
        }
```

---

### Issue: Web Interface Shows Black Feeds

**Symptoms**: Camera feeds display as black screens

**Solutions**:
1. Verify drones have taken off (not on ground)
2. Check camera configuration in `settings.json`
3. Ensure proper lighting in simulation environment
4. Test with AirSim's test script:
```python
import airsim
client = airsim.MultirotorClient()
client.confirmConnection()
responses = client.simGetImages([
    airsim.ImageRequest("front_center", airsim.ImageType.Scene)
], vehicle_name="Queen")
print(f"Image size: {len(responses[0].image_data_uint8)}")
```

---

### Issue: Status API Stuck/Slow

**Symptoms**: `/status` endpoint takes forever or times out

**Solutions**:
1. Reduce status update frequency in JavaScript
2. Use cached positions instead of live queries
3. Add timeout to API calls:
```python
@app.route('/status')
def status():
    try:
        # Use cached data from swarm state
        return jsonify({
            'threats': len(swarm.threats),
            'queen_pos': swarm.last_queen_pos,
            'authorization': swarm.pending_authorization
        })
    except Exception as e:
        logger.error(f"Status error: {e}")
        return jsonify({'error': 'unavailable'}), 500
```

---

### Issue: "ImportError: No module named 'airsim'"

**Solution**:
```bash
pip install airsim
# If that fails:
pip install --upgrade airsim
```

---

### Issue: Tornado Encoding Error on Windows

**Symptoms**:
```
UnicodeEncodeError: 'charmap' codec can't encode character
```

**Solution**:
```bash
pip uninstall tornado -y
pip install tornado==6.1
```

---

## 📊 Performance Optimization

### For Slower Systems

1. **Reduce Resolution**:
```json
"Width": 640,
"Height": 480
```

2. **Limit Warrior Count**: Use 1-2 warriors instead of 3+

3. **Increase Patrol Intervals**:
```python
hover_time = 5  # More time at each waypoint = fewer moves
```

4. **Disable AI During Testing**:
```python
# In queen.py
def detect_threats(self):
    return None  # Skip AI processing temporarily
```

### For High-Performance Systems

1. **4K Camera Feeds**: Set resolution to 3840x2160
2. **Add More Drones**: Scale to 5-10 warriors
3. **Complex AI Models**: Upgrade to YOLOv8m or YOLOv8l
4. **Multi-threaded Processing**: Process multiple feeds simultaneously

---

## 🔐 Safety & Ethics

### Human-in-the-Loop Design

This system demonstrates responsible AI development:

- **Authorization Required**: No autonomous strikes without human approval
- **Clear Visual Indicators**: Threat alerts clearly displayed
- **Manual Override**: System can be stopped at any time
- **Audit Trail**: All actions logged with timestamps

### Simulation-Only Notice

**This is a research/educational project designed for simulation environments only.**

Real-world deployment would require:
- Comprehensive safety testing
- Regulatory compliance (FAA Part 107 in US)
- Failsafe mechanisms
- Insurance and liability coverage
- Ethical review board approval

---

## 🚀 Future Enhancements

### Planned Features

- [ ] **Voice Commands**: Natural language control via speech recognition
- [ ] **3D Mission Mapping**: Real-time visualization of drone positions
- [ ] **Multi-Threat Handling**: Priority queue for simultaneous threats
- [ ] **Dynamic Terrain Adaptation**: Auto-adjust altitudes for obstacles
- [ ] **Swarm Formations**: Coordinated flight patterns (V-formation, line abreast)
- [ ] **Emergency RTH**: Return-to-home on low battery/connection loss
- [ ] **Weather Simulation**: Wind, rain effects on flight dynamics
- [ ] **Night Vision Mode**: Thermal camera integration

### Research Extensions

- **Machine Learning**: Train custom detection models on domain-specific data
- **Reinforcement Learning**: Optimize patrol patterns via RL agents
- **Multi-Agent RL**: Cooperative behavior emergence in swarms
- **Edge AI Deployment**: Run detection models on drone hardware
- **ROS Integration**: Connect to real drone platforms via Robot Operating System

---

## 📚 Technical Documentation

### API Endpoints

**Command Center (Flask - Port 5000)**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard |
| `/status` | GET | Current mission status (JSON) |
| `/authorize_strike` | POST | Approve kamikaze deployment |
| `/queen_feed` | GET | Queen camera stream (MJPEG) |
| `/warrior/<id>_feed` | GET | Warrior camera stream |
| `/kamikaze/<id>_feed` | GET | Kamikaze camera stream |

### Logging System

All drones log to console with timestamps:

```
[HH:MM:SS] [SWARM] [DRONE_NAME] Message
[10:23:45] [SWARM] [QUEEN] Threat detected: person 85%
[10:23:46] [SWARM] [WARRIOR1] Patrol waypoint reached
[10:23:50] [SWARM] [KAMIKAZE1] Strike authorized
```

### Thread Architecture

```
Main Thread
├─> Queen Thread (AI detection loop)
├─> Warrior1 Thread (patrol loop)
├─> Warrior2 Thread (patrol loop)
├─> Kamikaze1 Thread (standby loop)
└─> Flask Thread (web server)
```

All threads communicate via `SwarmState` with mutex locks.

---

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-capability`
3. Make changes and test thoroughly
4. Commit with descriptive messages
5. Push and create pull request

### Code Style

- Follow PEP 8 for Python code
- Use descriptive variable names
- Add docstrings to all functions
- Include type hints where applicable

```python
def calculate_waypoint(center: tuple, radius: float, angle: float) -> tuple:
    """
    Calculate patrol waypoint coordinates.
    
    Args:
        center: (x, y) center of patrol circle
        radius: Patrol radius in meters
        angle: Angle in degrees (0-360)
    
    Returns:
        (x, y, z) waypoint coordinates
    """
    # Implementation...
```

---

## 📄 License

MIT License - See LICENSE file for details

**Academic Use**: Citation appreciated but not required  
**Commercial Use**: Permitted with attribution  
**Modification**: Permitted  
**Distribution**: Permitted

---

## 🙏 Acknowledgments

- **Microsoft AirSim Team**: For the incredible simulation platform
- **Ultralytics**: YOLOv8 object detection framework
- **Flask Project**: Web framework
- **OpenCV**: Computer vision library
- **Hackathon Organizers**: For the inspiration and opportunity

---

## 📞 Contact & Support

**Project Maintainer**: Manav Adwani
**Email**: [manavadwani86@gmail.com]  
**GitHub**: [github.com/manav108-hub/drone-swarm-system]

### Getting Help

1. Check this README thoroughly
2. Search existing GitHub Issues
3. Join discussions in project Discussions tab
4. Create new Issue with "Question" label

### Reporting Bugs

Please include:
- Python version
- AirSim version
- Operating system
- Full error traceback
- Steps to reproduce


## ⭐ Star History

If this project helped you, consider giving it a star on GitHub! ⭐

---

**Built with 💙 for autonomous systems research and education**