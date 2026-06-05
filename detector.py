import numpy as np
import cv2 as cv
import torch
from ultralytics import YOLO, RTDETR
import threading

class DetectionEngine:
    def __init__(self, model_key="yolo11n", confidence_threshold=0.4):
        """
        Engine wrapping YOLO and RT-DETR models optimized for high-speed CUDA inference and targeting.
        """
        self.confidence_threshold = confidence_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_key = model_key
        self.inference_lock = threading.Lock()

        # Core model mappings
        self.model_files = {
            "yolo11n": "yolo11n.pt",
            "yolo11s": "yolo11s.pt",
            "yolo11m": "yolo11m.pt",
            "yolo11l": "yolo11l.pt",
            "rtdetr-l": "rtdetr-l.pt"
        }
        
        # Load the initial default model
        self.model = None
        self._load_model(self.model_key)
        
        # Initialize background subtractor 
        self.bg_subtractor = cv.createBackgroundSubtractorMOG2(
            history=500, 
            varThreshold=16, 
            detectShadows=True
        )
        
        # Standard COCO class mappings for reference
        self.coco_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck',
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 
            24: 'backpack', 25: 'umbrella', 26: 'handbag', 28: 'suitcase', 39: 'bottle'
        }

    def _load_model(self, model_key):
        """
        Loads the target model architecture into RAM/VRAM.
        Handles both YOLO variants and RT-DETR explicitly.
        """
        model_file = self.model_files.get(model_key, "yolo11n.pt")
        print(f"[DetectionEngine] Attempting to load model weights: {model_file} on {self.device}")
        
        try:
            if "rtdetr" in model_key:
                self.model = RTDETR(model_file)
            else:
                self.model = YOLO(model_file)
            print(f"[DetectionEngine] Successfully loaded model: {model_key.upper()}")
        except Exception as e:
            print(f"[DetectionEngine] Critical error loading {model_key} model: {e}. Falling back to YOLO11n.")
            self.model = YOLO("yolo11n.pt")
            self.model_key = "yolo11n"

    def change_model(self, new_model_key):
        """
        Safely swaps the active detector model and flushes PyTorch CUDA memory.
        """
        if new_model_key == self.model_key:
            return True
            
        print(f"[DetectionEngine] Swapping backbone model from {self.model_key} to {new_model_key}...")
        
        # Release references to older model
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.model_key = new_model_key
        self._load_model(new_model_key)
        return True
        
    def get_agnostic_proposals(self, frame):
        """
        Uses background subtraction to find moving objects (Agnostic to class).
        Returns a list of (x, y, w, h) bounding boxes.
        """
        if frame is None or frame.size == 0:
            return []
            
        fg_mask = self.bg_subtractor.apply(frame)
        _, thresh = cv.threshold(fg_mask, 200, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        proposals = []
        for cnt in contours:
            if cv.contourArea(cnt) > 500:  # Filter small background noise
                x, y, w, h = cv.boundingRect(cnt)
                proposals.append((x, y, w, h))
        return proposals

    def detect(self, frame, target_class_id=2, selection_strategy='largest'):
        """
        Runs model detection using GPU-accelerated pipelines and returns a single bounding box (x, y, w, h).
        """
        if frame is None or frame.size == 0 or self.model is None:
            return None

        # Run inference using CUDA, half-precision (FP16), and stream mode to reduce latency
        with self.inference_lock:
            results = self.model.predict(
                frame, 
                conf=self.confidence_threshold, 
                device=self.device,
                half=(self.device == "cuda"),  # FP16 acceleration on CUDA GPUs
                verbose=False,
                imgsz=320  # Optimized fast inference size
            )
        
        if not results or len(results) == 0:
            return None

        boxes_obj = results[0].boxes
        if boxes_obj is None or len(boxes_obj) == 0:
            return None

        cls_data = boxes_obj.cls.cpu().numpy()
        boxes_xyxy = boxes_obj.xyxy.cpu().numpy()
        conf_data = boxes_obj.conf.cpu().numpy()

        valid_detections = []
        frame_h, frame_w, _ = frame.shape
        center_x_frame = frame_w / 2.0
        center_y_frame = frame_h / 2.0

        for i in range(len(cls_data)):
            if int(cls_data[i]) == target_class_id:
                x1, y1, x2, y2 = boxes_xyxy[i]
                w = x2 - x1
                h = y2 - y1
                conf = conf_data[i]
                
                area = w * h
                cx = x1 + (w / 2.0)
                cy = y1 + (h / 2.0)
                dist_to_center = np.hypot(cx - center_x_frame, cy - center_y_frame)
                
                valid_detections.append({
                    'bbox': (int(x1), int(y1), int(w), int(h)),
                    'area': area,
                    'dist': dist_to_center,
                    'conf': conf
                })

        if not valid_detections:
            return None

        # Sort based on specified resolution strategy
        if selection_strategy == 'largest':
            valid_detections.sort(key=lambda x: x['area'], reverse=True)
        elif selection_strategy == 'centered':
            valid_detections.sort(key=lambda x: x['dist'])
        elif selection_strategy == 'highest_conf':
            valid_detections.sort(key=lambda x: x['conf'], reverse=True)

        return valid_detections[0]['bbox']

    def detect_candidates(self, frame, target_class_id=2):
        """
        Runs detector and returns ALL bounding boxes matching the target class.
        Used for Re-ID scan sweeps when a target is lost.
        """
        if frame is None or frame.size == 0 or self.model is None:
            return []
        with self.inference_lock:

            results = self.model.predict(
                frame, 
                conf=self.confidence_threshold, 
                device=self.device,
                half=(self.device == "cuda"),
                verbose=False,
                imgsz=320
            )
        
        if not results or len(results) == 0:
            return []

        boxes_obj = results[0].boxes
        if boxes_obj is None or len(boxes_obj) == 0:
            return []

        cls_data = boxes_obj.cls.cpu().numpy()
        boxes_xyxy = boxes_obj.xyxy.cpu().numpy()

        candidates = []
        for i in range(len(cls_data)):
            if int(cls_data[i]) == target_class_id:
                x1, y1, x2, y2 = boxes_xyxy[i]
                w = x2 - x1
                h = y2 - y1
                candidates.append((int(x1), int(y1), int(w), int(h)))
                
        return candidates