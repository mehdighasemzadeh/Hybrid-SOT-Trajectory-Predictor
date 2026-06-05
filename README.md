<div align="center">

# 🎯 Multithreaded High-Speed SOT & Trajectory Predictor

</div>

An advanced, real-time object tracking system built in Python. This framework integrates GPU-accelerated deep-learning detection backbones (**YOLO11, RT-DETR**) with classic, high-frequency Single Object Trackers (**ViT, CSRT, KCF**).

The platform implements robust **Weighted Least Squares (WLS)/RANSAC kinematic path prediction** to estimate trajectory vectors and features a visual **Deep Re-Identification (Re-ID)** engine running MobileNetV2/ResNet18 to recover lost targets under full occlusion.

---

## 📺 System Demo

<div align="center">
  <em>(Double-click to edit this link and embed your own recording or repository GIF!)</em>
  <br>
  <a href="https://github.com/user-attachments/assets/your-demo-video-placeholder">Watch the System Demo</a>
</div>

---

## 🏛️ System Architecture

The core runtime is designed around a **Decoupled Producer-Consumer Pipeline** spread across 4 dedicated threads. This design isolates heavy deep-learning model evaluation from UI interactions, maintaining interactive framerates even under maximum CPU/GPU compute loads.

### Multithreaded Architecture Details:

* **Thread 0 (Tkinter GUI):** Handles real-time telemetry updates (FPS, tracking confidence, target speeds, Re-ID latency), processes mouse canvas drag events for manual ROI initialization, and schedules display render ticks from the display queue.
* **Thread 1 (Video Acquisition):** A dedicated, zero-lag frame reader. It manages resolution configurations and feeds the queue at consistent hardware intervals.
* **Thread 2 (Core SOT Pipeline):** Updates the active tracker (ViT, CSRT, or KCF) and calculates kinematics. When tracking confidence degrades below critical thresholds, it halts local tracking, queries the **Trajectory Predictor** for a localized "Search ROI", and commands the Re-ID system to recover the target.
* **Thread 3 (Detection Engine):** An isolated worker thread that wakes up on signal events to execute full-frame YOLO/RT-DETR inference. This is used for semantic auto-initialization.

> **Note:** Thread-safety is fully guaranteed via a dedicated model re-entrance lock (`self.inference_lock`), preventing simultaneous prediction queries from Thread 2 and Thread 3.

---

## 📈 Key Algorithms & Visualizations[cite: 3]

### 1. Kinematic Trajectory Prediction[cite: 3]

The system maintains a frame-coordinate history queue[cite: 3]. When a target's trajectory vector indicates it is escaping through the boundaries of the frame, the **Predictor** dynamically overrides local crops to establish a search region encompassing the frame's outer borders where the target is likely to re-appear[cite: 3].

* **Weighted Least Squares (WLS):** Uses a time-decaying weight distribution favoring recent coordinate measurements to estimate velocities with minimal lag[cite: 3].
* **RANSAC Regression:** Mathematically isolates tracking noise, coordinate jitter, and outlier frame points during rapid, erratic target movements[cite: 3].

### 2. Deep Visual Re-Identification (Re-ID) Target Recovery[cite: 3]

If SOT tracking fails completely, the engine initiates a localized search sweep[cite: 3]. Rather than running slow, full-frame deep-learning evaluations, the model utilizes the target's trajectory prediction to sweep only the most probable spatial areas[cite: 3].

1. **Candidate Proposal:** The YOLO detector identifies class-specific bounding boxes within the predicted **Search ROI**[cite: 3].
2. **Batched FP16 Inference:** Crops of all candidates are fed in parallel directly to the deep Re-ID model (MobileNetV2 or ResNet18) to extract multi-dimensional feature vectors[cite: 3].
3. **Similarity Assessment:** Cosine similarity metrics are evaluated against the initial target gallery[cite: 3]. If a candidate scores higher than the user-defined threshold (default: 0.75), the tracker is dynamically re-initialized at those coordinates[cite: 3].

---

## 📂 Project Structure[cite: 3]

```text
├── detector.py       # Wrapper for YOLO11/RT-DETR models with safety threading locks.
├── main.py           # Tkinter GUI, Threading Managers, and Queue Coordination.
├── predictor.py      # WLS & RANSAC kinematics, trajectory and Dynamic ROI generation.
├── reid.py           # PyTorch deep feature extraction (MobileNetV2, ResNet18).
├── trackers.py       # Interface for OpenCV's TrackerVit (ONNX), CSRT, and KCF.
└── README.md         # Documentation.
