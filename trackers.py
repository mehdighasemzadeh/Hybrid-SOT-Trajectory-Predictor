import cv2 as cv
import numpy as np

class HybridTracker:
    def __init__(self, tracker_type="ViT", model_path="object_tracking_vittrack_2023sep.onnx"):
        """
        Unified tracker interface implementing CUDA-accelerated ViTTrack, classical CSRT, and high-speed KCF.
        """
        self.tracker_type = tracker_type
        self.model_path = model_path
        self.tracker = None
        self.initialized = False
        
        # Check if CUDA backends are compiled in OpenCV
        self.has_cuda = cv.cuda.getCudaEnabledDeviceCount() > 0

    def init_tracker(self, frame, bbox):
        """
        Initializes the selected tracker architecture.
        """
        x, y, w, h = bbox
        if w <= 0 or h <= 0 or x < 0 or y < 0:
            return False

        if self.tracker_type == "ViT":
            try:
                params = cv.TrackerVit_Params()
                params.net = self.model_path
                
                # Check for OpenCV CUDA capability and set acceleration backends
                if self.has_cuda:
                    params.backend = cv.dnn.DNN_BACKEND_CUDA
                    params.target = cv.dnn.DNN_TARGET_CUDA
                    print("ViTTrack GPU/CUDA execution enabled successfully.")
                else:
                    params.backend = cv.dnn.DNN_BACKEND_OPENCV
                    params.target = cv.dnn.DNN_TARGET_CPU
                    print("ViTTrack executing on CPU. (Ensure CUDA builds of OpenCV are present)")
                
                self.tracker = cv.TrackerVit_create(params)
            except Exception as e:
                print(f"Error initializing CUDA ViTTrack: {e}. Falling back to CSRT.")
                self.tracker_type = "CSRT"

        if self.tracker_type == "CSRT":
            # CSRT is CPU bound but highly accurate (Low FPS).
            self.tracker = cv.TrackerCSRT_create()
            
        if self.tracker_type == "KCF":
            # KCF is extremely lightweight and fast (~100+ FPS) but handles occlusion poorly compared to ViT.
            self.tracker = cv.TrackerKCF_create()

        try:
            self.tracker.init(frame, (int(x), int(y), int(w), int(h)))
            self.initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize {self.tracker_type} tracker: {e}")
            self.initialized = False
            return False

    def update(self, frame):
        """
        Updates the tracker coordinates on a new frame.
        """
        if not self.initialized or self.tracker is None:
            return False, (0, 0, 0, 0), 0.0

        success, bbox = self.tracker.update(frame)
        score = 1.0
        
        if success and self.tracker_type == "ViT":
            score = self.tracker.getTrackingScore()

        return success, bbox, score
