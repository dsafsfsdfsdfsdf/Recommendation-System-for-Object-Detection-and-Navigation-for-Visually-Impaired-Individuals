This project implements a real-time assistive navigation system designed for visually impaired individuals. It combines RGB camera input with simulated 2D LiDAR (pseudo-LiDAR) to detect obstacles, plan safe paths, and provide voice feedback using deep learning and sensor fusion techniques.

🧠 Features
Real-time object detection using YOLOv5 (person, car, static obstacles)

Pseudo-LiDAR simulation from RGB image scanlines

BEV occupancy grid generation for environmental modeling

A path planning* to avoid obstacles dynamically

Text-to-speech voice alerts (e.g. "Obstacle ahead at 1.5 meters")

Modular architecture with clear separation of perception, planning, and control

Debugging visualization of detection results and BEV map with planned path

🗂️ Project Structure

blind_nav_system/
├── main.py                       # Main control loop
├── fusion/
│   ├── data_align.py            # Synchronize camera and LiDAR inputs
│   ├── simple_fusion.py         # Convert scan data to BEV grid
│   └── bev_map.py               # Visualize BEV occupancy grid
├── perception/
│   └── yolov5.py                # YOLOv5 detector (Torch Hub API)
├── navigation/
│   └── simple_planner.py        # A* pathfinding and voice feedback
├── sensors/
│   ├── camera_module.py         # Webcam frame capture
│   └── pseudo_lidar2d.py        # Generate simulated LiDAR scan
├── assets/                      # (Optional) Model weights or test media
└── README.md                    # Project documentation
🚀 Getting Started
Prerequisites
Python ≥ 3.8

opencv-python

numpy

torch and torchvision

pyttsx3 (for offline voice output)

Install dependencies:

pip install -r requirements.txt
If you use GPU, ensure CUDA and PyTorch are correctly installed.

Run the System

python main.py
Press ESC to stop the system.

📷 System Overview
Camera captures real-time video frames.

Pseudo-LiDAR simulates a 2D scan using a horizontal slice of the image.

YOLOv5 detects objects using a pre-trained model from Torch Hub.

Scan-to-BEV conversion creates a binary grid map of free and occupied space.

A* path planning computes a collision-free path from current location.

Voice module issues feedback like:
“Obstacle 2.1 meters to your left”

🧪 Example Outputs
YOLOv5 Detection	BEV Occupancy Map	Path Planning

🧩 Future Improvements
Integrate real 2D/3D LiDAR sensor (e.g., RPLiDAR)

Add dynamic goal selection via voice or gesture

Improve object classification for semantic-level navigation

Mobile or embedded deployment (e.g. Jetson Nano)

📄 License
This project is for academic and research purposes. Contact the author for collaboration or deployment use cases.
