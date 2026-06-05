# **🎯 Multithreaded High-Speed SOT & Trajectory Predictor**

An advanced, real-time object tracking system built in Python. This framework integrates GPU-accelerated deep-learning detection backbones (**YOLO11, RT-DETR**) with classic, high-frequency Single Object Trackers (**ViT, CSRT, KCF**).

The platform implements robust **Weighted Least Squares (WLS)/RANSAC kinematic path prediction** to estimate trajectory vectors and features a visual **Deep Re-Identification (Re-ID)** engine running MobileNetV2/ResNet18 to recover lost targets under full occlusion.

## **📺 System Demo**



## **🏛️ System Architecture**

The core runtime is designed around a **Decoupled Producer-Consumer Pipeline** spread across 4 dedicated threads. This design isolates heavy deep-learning model evaluation from UI interactions, maintaining interactive framerates even under maximum CPU/GPU compute loads.

### **Multithreaded Architecture Details:**

* **Thread 0 (Tkinter GUI):** Handles real-time telemetry updates (FPS, tracking confidence, target speeds, Re-ID latency), processes mouse canvas drag events for manual ROI initialization, and schedules display render ticks from the display queue.  
* **Thread 1 (Video Acquisition):** A dedicated, zero-lag frame reader. It manages resolution configurations and feeds the queue at consistent hardware intervals.  
* **Thread 2 (Core SOT Pipeline):** Updates the active tracker (ViT, CSRT, or KCF) and calculates kinematics. When tracking confidence degrades below critical thresholds, it halts local tracking, queries the **Trajectory Predictor** for a localized "Search ROI", and commands the Re-ID system to recover the target.  
* **Thread 3 (Detection Engine):** An isolated worker thread that wakes up on signal events to execute full-frame YOLO/RT-DETR inference. This is used for semantic auto-initialization.

*Note: Thread-safety is fully guaranteed via a dedicated model re-entrance lock (self.inference\_lock), preventing simultaneous prediction queries from Thread 2 and Thread 3\.*

## **📈 Key Algorithms & Visualizations**

### **1\. Kinematic Trajectory Prediction**

The system maintains a frame-coordinate history queue. When a target's trajectory vector indicates it is escaping through the boundaries of the frame, the **Predictor** dynamically overrides local crops to establish a search region encompassing the frame's outer borders where the target is likely to re-appear.

* **Weighted Least Squares (WLS):** Uses a time-decaying weight distribution (![][image1]) favoring recent coordinate measurements to estimate velocities with minimal lag.  
* **RANSAC Regression:** Mathematically isolates tracking noise, coordinate jitter, and outlier frame points during rapid, erratic target movements.

### **2\. Deep Visual Re-Identification (Re-ID) Target Recovery**

If SOT tracking fails completely, the engine initiates a localized search sweep. Rather than running slow, full-frame deep-learning evaluations, the model utilizes the target's trajectory prediction to sweep only the most probable spatial areas.

1. **Candidate Proposal:** The YOLO detector identifies class-specific bounding boxes within the predicted **Search ROI**.  
2. **Batched FP16 Inference:** Crops of all candidates are fed in parallel directly to the deep Re-ID model (MobileNetV2 or ResNet18) to extract ![][image2]\-dimensional feature vectors.  
3. **Similarity Assessment:** Consine similarity metrics are evaluated against the initial target gallery. If a candidate scores higher than the user-defined threshold (default: 0.75), the tracker is dynamically re-initialized at those coordinates.

## **📂 Project Structure**

├── detector.py       \# Wrapper for YOLO11/RT-DETR models with safety threading locks.  
├── main.py           \# Tkinter GUI, Threading Managers, and Queue Coordination.  
├── predictor.py      \# WLS & RANSAC kinematics, trajectory and Dynamic ROI generation.  
├── reid.py           \# PyTorch deep feature extraction (MobileNetV2, ResNet18).  
├── trackers.py       \# Interface for OpenCV's TrackerVit (ONNX), CSRT, and KCF.  
└── README.md         \# Documentation.

## **⚡ Performance Optimization**

* **Inference Speedups:** The detection and Re-ID modules utilize half-precision float configurations (FP16) when CUDA is available, yielding up to a ![][image3] speedup on NVIDIA architecture.  
* **OpenCV Multithreading:** The frame preprocessing engine explicitly leverages OpenCV parallel processing loops:  
  cv.setNumThreads(4)

* **Batched Re-ID:** Features extraction queries are stacked as contiguous tensors and evaluated in a single forward pass, reducing overhead latency to \< 15ms for up to 10 candidates.

## **🚀 Getting Started**

### **Prerequisites**

pip install torch torchvision numpy opencv-contrib-python ultralytics scikit-learn pillow

### **Vision Transformer (ViT) Tracker Setup**

To run the lightweight **ViT** transformer tracker, you must download the ONNX model graph structure from the official OpenCV Model Zoo:

1. Download [object\_tracking\_vittrack\_2023sep.onnx](https://github.com/opencv/opencv_zoo/tree/master/models/object_tracking_vittrack).  
2. Place the file directly in your project root directory.

### **Running the System**

Start the main multithreaded application:

python main.py

1. Select your target **Video Input Source** (Webcam index or path to a local video).  
2. Choose your **Primary SOT Engine** (ViT, CSRT, or KCF).  
3. Initialize the target:  
   * **Manual:** Click "Manual ROI", then click and drag a bounding box directly on the video canvas.  
   * **Automatic:** Select a target class (e.g., Car, Person) and click "YOLO Semantic Auto-Init".  
4. Monitor tracking, speed vectors, and auto-recovery processes live on the visual dashboard\!

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAaCAYAAABVX2cEAAABTklEQVR4Xu2TPyhFYRjG7w2bUjjK+f+nMKrDZDBZ7kRWkww2AylZGC3KbDZIiQwGisFgu9dd3TIabQoDvzfn1Xc/Ga7BoPPUr773fZ/7fOf7zj2VSqlSf6Q0TfuiKLq1+6o4jqeZ1/GF9uybMG9gXrL7KmZr8B6G4YI9a1Oe5z0Yz1lW7Zmq8LTYdNaetYndZmDd6p3CitU7ITDRmvUxzJkeOeIOximtPc/z5UiwaNi6qK+NWsLmsywbMnsSdkhYrnUQBBMYn13XHdQedQ3fttY/CuMBrBZllfU+PPHjsWI+DHXf9/ulpj8up6F39BWiKt7kK1xAC+7gAV7gCh7xbKlf7jJJklHxGjGf4o4GGJzBG1wWd1aDe4GgZWzd6uf4Af1N2DVifi82aHK3kxJszzqSvCyeqsEdevZfqmPxSY0QdsPT7TmO02vPS/0XfQBnuEh7uvzc9AAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAZCAYAAAA8CX6UAAABGklEQVR4Xu2SsUrDUBSGK3QRqls6hCQ3IYGMgmspopPoi/QFXHwEV3cXwQeQTkqpRQsdXATBwVFwFHFrl/Y7enKthzrrkA9+mtzvz03u7W00av4G59w2uSZvZE6e9F5yQz6SJBnxu2ufXQnFC5koDMN4eTwIghbjfTKN47i77FZC8YU82HEhz/O2fBm5te4HWZZt6bJOrKvADaUTRdG6dR4KR1JK03Tfugr8nXRYXm6dhwmuKM1kP6xTmrq0eVEUm1Z+ops5JQPrKnjRji793joP8lBLx9ZV4E6lw4Q96zwUnsk7l03ryrLcwI3Jo/xz1nsoZPqmS6PWGN9zX4fz/Ne946R2KEzITJf16r5Ps0QmOCMH9tmamn/LApNkTCSvG0AgAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABoAAAAZCAYAAAAv3j5gAAABtUlEQVR4Xu2TO0sDURCFN8FXo4KgaJLdPCESFNEg2IhPsBIsFUsLCxUsrPQ/iAiKFqJYCVa2BnyAhQTBTkGFgK2ihSBYxW/CLoyXJM1a5sBh75w5d+bu3F3LqsEvYrHYCMxHo9Ev97lgenzDcZwsfKLBaDgcjrA+oVERrpleX6BgjiZTXpxKpRrRCvCbdbv2+kFQxgWfE4lEqyfyVnvuW/3bCOso9iZFI5FIyhOJN0Wj4Yo2E0+m0+lmrWmwp58DO6ZeQjwe78MwrjVGee42mjD0IbSzbDZbr3WBbdvD5K71ZKqCBnH4A28IA2aeZjMUPNQ5tE78t7BLWasD8zGFXuULNHMe8CzDXZYB+WBYXzD6XtNXEZxsjk0fjKHHzJnAuwr38V/K3Zj5ipD/iQ0FmgyauXKQMcEHeEoYNPNlEQqFbDa8yIV6GvGY3If2eXDv5A4OsF7kkEdWmfv8A4xNMC9j0zpFNigwrTVBMpnsIHevc8RLcMeq1kxOg+kT5uAVfITvsEjzbu3lwtukCZzXugBtHW6begl8MS1SsBIzmUyD9nOoLZrPak2DPQfyU5t6DTXU4B+/KMhph+fhHjYAAAAASUVORK5CYII=>
