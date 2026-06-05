🎯 Multithreaded High-Speed SOT & Trajectory Predictor

📺 System Demo

🏛️ System Architecture

Multithreaded Architecture Details:

Thread 0 (Tkinter GUI): Handles real-time telemetry updates (FPS, tracking confidence, target speeds, Re-ID latency), processes mouse canvas drag events for manual ROI initialization, and schedules display render ticks from the display queue.

Thread 1 (Video Acquisition): A dedicated, zero-lag frame reader. It manages resolution configurations and feeds the queue at consistent hardware intervals.

Thread 2 (Core SOT Pipeline): Updates the active tracker (ViT, CSRT, or KCF) and calculates kinematics. When tracking confidence degrades below critical thresholds, it halts local tracking, queries the Trajectory Predictor for a localized "Search ROI", and commands the Re-ID system to recover the target.

Thread 3 (Detection Engine): An isolated worker thread that wakes up on signal events to execute full-frame YOLO/RT-DETR inference. This is used for semantic auto-initialization.

Note: Thread-safety is fully guaranteed via a dedicated model re-entrance lock (self.inference_lock), preventing simultaneous prediction queries from Thread 2 and Thread 3.

📈 Key Algorithms & Visualizations

1. Kinematic Trajectory Prediction

2. Deep Visual Re-Identification (Re-ID) Target Recovery

📂 Project Structure

├── demo/
│   └── demo1.mp4     # Rendered multiview demonstration video.
├── images/
│   ├── re-id-arch.png
│   ├── system-arch.png
│   └── trajectory-arch.png
├── detector.py       # Wrapper for YOLO11/RT-DETR models with safety threading locks.
├── main.py           # Tkinter GUI, Threading Managers, and Queue Coordination.
├── predictor.py      # WLS & RANSAC kinematics, trajectory and Dynamic ROI generation.
├── reid.py           # PyTorch deep feature extraction (MobileNetV2, ResNet18).
├── trackers.py       # Interface for OpenCV's TrackerVit (ONNX), CSRT, and KCF.
└── README.md         # Documentation.


⚡ Performance Optimization

🚀 Getting Started

Prerequisites

pip install torch torchvision numpy opencv-contrib-python ultralytics scikit-learn pillow


Vision Transformer (ViT) Tracker Setup

Running the System

Start the main multithreaded application:

python main.py


Select your target Video Input Source (Webcam index or path to a local video).

Choose your Primary SOT Engine (ViT, CSRT, or KCF).

Initialize the target:

Manual: Click "Manual ROI", then click and drag a bounding box directly on the video canvas.

Automatic: Select a target class (e.g., Car, Person) and click "YOLO Semantic Auto-Init".

Monitor tracking, speed vectors, and auto-recovery processes live on the visual dashboard!


