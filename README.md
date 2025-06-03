# Blind Navigation System

This project implements a lightweight AI-based navigation system to assist visually impaired individuals. It fuses camera input and pseudo-LiDAR scan data to detect obstacles, generate a real-time BEV (Bird’s Eye View) occupancy map, plan safe walking paths using A* algorithm, and provide voice feedback alerts.

---

## 🗂️ Project Structure

blind_nav_system/
├── main.py # Main control loop
├── fusion/
│ ├── data_align.py # Synchronize camera and LiDAR inputs
│ ├── simple_fusion.py # Convert scan data to BEV grid
│ └── bev_map.py # Visualize BEV occupancy grid
├── perception/
│ └── yolov5.py # YOLOv5 detector (Torch Hub API)
├── navigation/
│ └── simple_planner.py # A* pathfinding and voice feedback
├── sensors/
│ ├── camera_module.py # Webcam frame capture
│ └── pseudo_lidar2d.py # Generate simulated LiDAR scan
├── assets/ # (Optional) Model weights or test media
└── README.md # Project documentation


---

## 🚀 Getting Started

### 1. Install Dependencies

Make sure you have Python 3.8+ installed. Then run:

```bash
pip install -r requirements.txt
Required packages include:

torch

torchvision

opencv-python

numpy

pyttsx3

2. Run the Main Program

python main.py
This will:

Capture camera input and simulate LiDAR

Run YOLOv5 object detection

Generate BEV map and plan safe paths

Provide voice alerts like “Obstacle 1.2 meters to your left”

Press ESC to exit.

🧪 Sample Code Snippet
python

from fusion.data_align import InputAligner
from perception.yolov5 import YoloV5Detector
from navigation.simple_planner import astar_path_planning, simulate_and_speak_alerts

aligner = InputAligner()
detector = YoloV5Detector(model_size="yolov5s")
frame, scan = aligner.get_aligned_inputs()
detections = detector.detect(frame)
📌 Notes
You can change the YOLO model size in main.py (yolov5s, yolov5m, etc.)

If using GPU, make sure CUDA is available

Voice engine uses pyttsx3 (offline); configure English voice in simple_planner.py

📷 Example Screenshots
BEV Occupancy Grid with Path

YOLO Detection Overlay

Real-time Voice Alert Feedback

(Add your own screenshots here in markdown format if needed.)
