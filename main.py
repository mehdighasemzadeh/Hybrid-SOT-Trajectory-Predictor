import sys
import time
import queue
import threading
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import cv2 as cv
import numpy as np
from PIL import Image, ImageTk
import torch

# Import decoupled optimized pipelines
from detector import DetectionEngine
from trackers import HybridTracker
from predictor import TrajectoryPredictor
from reid import ReIDEngine

class DashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multithreaded High-Speed SOT & Trajectory Predictor")
        self.root.geometry("1280x800")
        
        cv.setNumThreads(4)
        print(f"OpenCV processing thread count explicitly set to: {cv.getNumThreads()}")
        
        self.frame_queue = queue.Queue(maxsize=2)
        self.display_queue = queue.Queue(maxsize=2)
        self.detection_trigger = threading.Event()
        self.detection_result_queue = queue.Queue()
        
        self.state_lock = threading.Lock()
        self.active_tracker_type = "ViT"
        self.target_class_id = 2
        self.selection_strategy = "largest"
        self.active_detection_model = "yolo11n"
        self.is_tracking = False
        self.is_recovering = False
        self.fps_rate = 0.0
        self.current_score = 0.0
        self.speed_vector = (0.0, 0.0)
        
        self.reid_enabled = True
        self.reid_threshold = 0.75
        self.last_reid_latency = 0.0
        self.active_reid_model = "mobilenet_v2" # Default Fast model
        
        self.target_resolution = (640, 480)
        self.native_resolution_active = False
        
        self.video_source_path = "0"
        self.source_changed_event = threading.Event()
        
        self.is_recording = False
        self.video_writer = None
        self.video_writer_initialized = False
        
        # Canvas dimensional tracking for ROI
        self.canvas_size = (850, 580)
        self.selection_mode = False
        self.roi_start = None
        self.canvas_rect = None
        self.latest_raw_frame = None # Thread safe shared frame resource
        self.canvas_scale_info = (1.0, 0, 0)
        
        self.current_bbox = None
        self.future_path = []
        self.last_known_bbox = None
        self.app_running = True
        
        self.coco_classes = {
            2: 'Car', 0: 'Person', 3: 'Motorcycle',
            5: 'Bus', 7: 'Truck', 15: 'Cat', 16: 'Dog'
        }

        # Detection backend naming maps
        self.detection_model_map = {
            "YOLO11 Nano (Fast)": "yolo11n",
            "YOLO11 Small": "yolo11s",
            "YOLO11 Medium": "yolo11m",
            "YOLO11 Large": "yolo11l",
            "RT-DETR Large (Accurate)": "rtdetr-l"
        }

        self.detector = DetectionEngine(model_key=self.active_detection_model)
        self.tracker_engine = HybridTracker(tracker_type="ViT")
        self.reid_engine = ReIDEngine(backbone_type=self.active_reid_model)
        self.predictor = TrajectoryPredictor(history_len=15, prediction_steps=15, method="wls")

        # Keeping a persistent variable to prevent Tkinter PhotoImage Garbage Collection cleanup
        self.img_tk = None

        self.setup_ui_dashboard()
        self.spawn_execution_threads()
        self.bind_closing_events()

    def setup_ui_dashboard(self):
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        video_frame = ttk.LabelFrame(main_paned, text=" Real-Time Video Stream with Predicted Paths ")
        self.video_canvas = tk.Canvas(video_frame, bg="black", width=850, height=580)
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Interactive Canvas Bindings
        self.video_canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.video_canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.video_canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.video_canvas.bind("<Configure>", self.on_canvas_resize)

        main_paned.add(video_frame, weight=3)
        
        control_frame = ttk.LabelFrame(main_paned, text=" Telemetry Dashboard & Controls ")
        control_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        main_paned.add(control_frame, weight=1)
        
        source_lf = ttk.LabelFrame(control_frame, text=" Video Input Source & Processing Resolution ")
        source_lf.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(source_lf, text="Source Type:").pack(anchor=tk.W, padx=10, pady=2)
        self.source_type_cb = ttk.Combobox(source_lf, values=["Video File", "Webcam 0", "Webcam 1"], state="readonly")
        self.source_type_cb.set("Webcam 0")
        self.source_type_cb.pack(fill=tk.X, padx=10, pady=4)
        self.source_type_cb.bind("<<ComboboxSelected>>", self.on_source_type_changed)
        
        self.file_frame = ttk.Frame(source_lf)
        self.file_frame.pack(fill=tk.X, padx=10, pady=4)
        
        self.file_path_var = tk.StringVar(value="")
        self.file_entry = ttk.Entry(self.file_frame, textvariable=self.file_path_var, state="disabled")
        self.file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.browse_btn = ttk.Button(self.file_frame, text="Browse", command=self.browse_video_file, state="disabled")
        self.browse_btn.pack(side=tk.RIGHT)
        
        ttk.Label(source_lf, text="Target Capture/Frame Size:").pack(anchor=tk.W, padx=10, pady=2)
        self.resolution_cb = ttk.Combobox(source_lf, values=["320x240", "640x480 (Default)", "1280x720 (HD)", "1920x1080 (FHD)", "Native Resolution"], state="readonly")
        self.resolution_cb.set("640x480 (Default)")
        self.resolution_cb.pack(fill=tk.X, padx=10, pady=4)
        self.resolution_cb.bind("<<ComboboxSelected>>", self.on_resolution_changed)

        self.apply_source_btn = ttk.Button(source_lf, text="🔄 Apply & Restart Source", command=self.apply_video_source)
        self.apply_source_btn.pack(fill=tk.X, padx=10, pady=5)
        
        telemetry_lf = ttk.LabelFrame(control_frame, text=" Live Telemetry ")
        telemetry_lf.pack(fill=tk.X, padx=5, pady=5)
        
        self.fps_lbl = ttk.Label(telemetry_lf, text="System Loop: 0.00 FPS", font=("Segoe UI", 11, "bold"))
        self.fps_lbl.pack(anchor=tk.W, padx=10, pady=5)
        
        self.score_lbl = ttk.Label(telemetry_lf, text="Tracker Score: 0.00", font=("Segoe UI", 11))
        self.score_lbl.pack(anchor=tk.W, padx=10, pady=5)
        
        self.speed_lbl = ttk.Label(telemetry_lf, text="Speed: 0.00 px/frame", font=("Segoe UI", 11))
        self.speed_lbl.pack(anchor=tk.W, padx=10, pady=5)

        self.reid_time_lbl = ttk.Label(telemetry_lf, text="Re-ID Latency: 0.00 ms", font=("Segoe UI", 11))
        self.reid_time_lbl.pack(anchor=tk.W, padx=10, pady=5)
        
        self.status_lbl = ttk.Label(telemetry_lf, text="Status: IDLE", font=("Segoe UI", 11, "italic"), foreground="orange")
        self.status_lbl.pack(anchor=tk.W, padx=10, pady=5)
        
        settings_lf = ttk.LabelFrame(control_frame, text=" SOT Configuration ")
        settings_lf.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(settings_lf, text="Primary SOT Engine:").pack(anchor=tk.W, padx=10, pady=2)
        
        self.tracker_sel_cb = ttk.Combobox(settings_lf, values=["ViT", "CSRT", "KCF"], state="readonly")
        self.tracker_sel_cb.set("ViT")
        self.tracker_sel_cb.pack(fill=tk.X, padx=10, pady=4)
        self.tracker_sel_cb.bind("<<ComboboxSelected>>", self.on_tracker_config_changed)

        ttk.Label(settings_lf, text="Trajectory Modeling Algorithm:").pack(anchor=tk.W, padx=10, pady=2)
        self.pred_method_cb = ttk.Combobox(settings_lf, values=["Weighted LS (Fast)", "RANSAC (Robust)"], state="readonly")
        self.pred_method_cb.set("Weighted LS (Fast)")
        self.pred_method_cb.pack(fill=tk.X, padx=10, pady=4)
        self.pred_method_cb.bind("<<ComboboxSelected>>", self.on_pred_method_changed)

        reid_lf = ttk.LabelFrame(control_frame, text=" SOT Visual Re-ID Recovery ")
        reid_lf.pack(fill=tk.X, padx=5, pady=5)

        self.reid_enabled_var = tk.BooleanVar(value=True)
        self.reid_chk = ttk.Checkbutton(reid_lf, text="Enable Re-ID Auto-Recovery", variable=self.reid_enabled_var, command=self.on_reid_toggle)
        self.reid_chk.pack(anchor=tk.W, padx=10, pady=4)

        ttk.Label(reid_lf, text="Similarity Threshold:").pack(anchor=tk.W, padx=10, pady=2)
        self.reid_thresh_cb = ttk.Combobox(reid_lf, values=["0.50", "0.60", "0.70", "0.75 (Default)", "0.80", "0.85", "0.90"], state="readonly")
        self.reid_thresh_cb.set("0.75 (Default)")
        self.reid_thresh_cb.pack(fill=tk.X, padx=10, pady=4)
        self.reid_thresh_cb.bind("<<ComboboxSelected>>", self.on_reid_threshold_changed)
        
        ttk.Label(reid_lf, text="Re-ID Backbone Model:").pack(anchor=tk.W, padx=10, pady=2)
        self.reid_model_cb = ttk.Combobox(reid_lf, values=["mobilenet_v2 (Fast)", "resnet18 (Accurate)"], state="readonly")
        self.reid_model_cb.set("mobilenet_v2 (Fast)")
        self.reid_model_cb.pack(fill=tk.X, padx=10, pady=4)
        self.reid_model_cb.bind("<<ComboboxSelected>>", self.on_reid_model_changed)
        
        detection_lf = ttk.LabelFrame(control_frame, text=" Auto Detection Settings ")
        detection_lf.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(detection_lf, text="Inference Backbone Model:").pack(anchor=tk.W, padx=10, pady=2)
        self.det_model_cb = ttk.Combobox(detection_lf, values=list(self.detection_model_map.keys()), state="readonly")
        self.det_model_cb.set("YOLO11 Nano (Fast)")
        self.det_model_cb.pack(fill=tk.X, padx=10, pady=4)
        self.det_model_cb.bind("<<ComboboxSelected>>", self.on_detection_model_changed)
        
        ttk.Label(detection_lf, text="Target Semantic Object:").pack(anchor=tk.W, padx=10, pady=2)
        self.class_sel_cb = ttk.Combobox(detection_lf, values=list(self.coco_classes.values()), state="readonly")
        self.class_sel_cb.set("Car")
        self.class_sel_cb.pack(fill=tk.X, padx=10, pady=4)
        self.class_sel_cb.bind("<<ComboboxSelected>>", self.on_class_config_changed)
        
        ttk.Label(detection_lf, text="Multi-Target Conflict Resolution:").pack(anchor=tk.W, padx=10, pady=2)
        self.strategy_cb = ttk.Combobox(detection_lf, values=["largest", "centered", "highest_conf"], state="readonly")
        self.strategy_cb.set("largest")
        self.strategy_cb.pack(fill=tk.X, padx=10, pady=4)
        self.strategy_cb.bind("<<ComboboxSelected>>", self.on_strategy_changed)
        
        btn_lf = ttk.LabelFrame(control_frame, text=" Tracker Actions & Recording ")
        btn_lf.pack(fill=tk.X, padx=5, pady=5)
        
        self.auto_btn = ttk.Button(btn_lf, text="YOLO Semantic Auto-Init", command=self.trigger_yolo_auto_initialization)
        self.auto_btn.pack(fill=tk.X, padx=10, pady=5)
        
        self.manual_btn = ttk.Button(btn_lf, text="Manual ROI (Draw on Video)", command=self.toggle_manual_roi_mode)
        self.manual_btn.pack(fill=tk.X, padx=10, pady=5)
        
        self.reset_btn = ttk.Button(btn_lf, text="Reset Engine State", command=self.reset_pipeline_states)
        self.reset_btn.pack(fill=tk.X, padx=10, pady=5)

        rec_separator = ttk.Separator(btn_lf, orient='horizontal')
        rec_separator.pack(fill=tk.X, padx=10, pady=10)

        self.record_btn = ttk.Button(btn_lf, text="🔴 Start Recording", command=self.toggle_recording)
        self.record_btn.pack(fill=tk.X, padx=10, pady=5)

        self.rec_status_lbl = ttk.Label(btn_lf, text="Recording: STOPPED", font=("Segoe UI", 9, "bold"), foreground="gray")
        self.rec_status_lbl.pack(anchor=tk.W, padx=15, pady=2)

    def spawn_execution_threads(self):
        self.acq_thread = threading.Thread(target=self.video_acquisition_worker, daemon=True)
        self.acq_thread.start()

        self.track_thread = threading.Thread(target=self.tracking_worker, daemon=True)
        self.track_thread.start()

        self.det_thread = threading.Thread(target=self.yolo_detection_worker, daemon=True)
        self.det_thread.start()

        self.root.after(10, self.update_dashboard_frame_callback)

    def video_acquisition_worker(self):
        cap = None
        current_src = None

        while self.app_running:
            if cap is None or self.source_changed_event.is_set():
                if cap is not None:
                    cap.release()
                
                with self.state_lock:
                    current_src = self.video_source_path
                    res_w, res_h = self.target_resolution
                    is_native = self.native_resolution_active
                
                if current_src == "Webcam 0":
                    open_target = 0
                elif current_src == "Webcam 1":
                    open_target = 1
                else:
                    open_target = current_src
                
                cap = cv.VideoCapture(open_target)
                
                if not is_native and isinstance(open_target, int):
                    cap.set(cv.CAP_PROP_FRAME_WIDTH, res_w)
                    cap.set(cv.CAP_PROP_FRAME_HEIGHT, res_h)
                
                while not self.frame_queue.empty():
                    try: self.frame_queue.get_nowait()
                    except queue.Empty: break
                while not self.display_queue.empty():
                    try: self.display_queue.get_nowait()
                    except queue.Empty: break
                
                self.reset_pipeline_states()
                self.source_changed_event.clear()

            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    is_webcam = (current_src == "Webcam 0" or current_src == "Webcam 1")
                    if not is_webcam:
                        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                    else:
                        time.sleep(0.05)
                    continue
                
                with self.state_lock:
                    res_w, res_h = self.target_resolution
                    is_native = self.native_resolution_active
                
                if not is_native:
                    fh, fw, _ = frame.shape
                    if fw != res_w or fh != res_h:
                        frame = cv.resize(frame, (res_w, res_h), interpolation=cv.INTER_AREA)

                if self.frame_queue.full():
                    try: self.frame_queue.get_nowait()
                    except queue.Empty: pass
                
                self.frame_queue.put(frame)
                time.sleep(0.01)
            else:
                time.sleep(0.1)
            
        if cap is not None:
            cap.release()

    def tracking_worker(self):
        fps_meter = cv.TickMeter()
        fps_meter.start()
        
        while self.app_running:
            try:
                frame = self.frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            raw_frame = frame.copy()
            annotated_frame = frame.copy()

            # Safely expose latest frame to the detection worker to prevent frame stealing
            with self.state_lock:
                self.latest_raw_frame = raw_frame.copy()

            current_bbox_local = None
            future_path_local = []
            vel_magnitude = 0.0

            with self.state_lock:
                tracking_active = self.is_tracking
                recovering_active = self.is_recovering

            if tracking_active:
                success, bbox, score = self.tracker_engine.update(raw_frame)
                
                h_f, w_f, _ = frame.shape
                is_drifted = False
                if success:
                    x, y, w, h = bbox
                    if w > 0.8 * w_f or h > 0.8 * h_f or w < 5 or h < 5:
                        is_drifted = True

                if success and score >= 0.25 and not is_drifted:
                    self.predictor.add_point(bbox)
                    vel_vec, path, _ = self.predictor.predict()
                    
                    current_bbox_local = bbox
                    future_path_local = path
                    vel_magnitude = np.hypot(vel_vec[0], vel_vec[1])
                    
                    with self.state_lock:
                        self.current_score = score
                        self.speed_vector = vel_vec
                        self.is_recovering = False
                else:
                    with self.state_lock:
                        self.is_recovering = True
                        self.current_score = 0.0
                        reid_active = self.reid_enabled
                        reid_thresh_val = self.reid_threshold
                        target_cls = self.target_class_id
                        strat = self.selection_strategy

                    search_roi = self.predictor.generate_search_roi(frame.shape)
                    rx, ry, rw, rh = search_roi
                    search_crop = raw_frame[ry:ry+rh, rx:rx+rw]
                    
                    if search_crop.size > 0:
                        if reid_active:
                            crop_candidates = self.detector.detect_candidates(search_crop, target_class_id=target_cls)
                            
                            global_candidates = []
                            for cb in crop_candidates:
                                bx, by, bw, bh = cb
                                global_candidates.append((bx + rx, by + ry, bw, bh))
                            
                            best_bbox, score_val, elapsed_time = self.reid_engine.find_best_match(
                                raw_frame,
                                global_candidates,
                                target_id="active_target",
                                similarity_threshold=reid_thresh_val,
                                return_time=True
                            )
                            
                            with self.state_lock:
                                self.last_reid_latency = elapsed_time
                            
                            if best_bbox:
                                self.tracker_engine.init_tracker(raw_frame, best_bbox)
                                self.predictor.clear()
                                self.predictor.add_point(best_bbox)
                                with self.state_lock:
                                    self.is_recovering = False
                        else:
                            recovery_bbox = self.detector.detect(search_crop, target_class_id=target_cls, selection_strategy=strat)
                            if recovery_bbox:
                                bx, by, bw, bh = recovery_bbox
                                corrected_bbox = (bx + rx, by + ry, bw, bh)
                                self.tracker_engine.init_tracker(raw_frame, corrected_bbox)
                                self.predictor.clear()
                                self.predictor.add_point(corrected_bbox)
                                with self.state_lock:
                                    self.is_recovering = False

            if current_bbox_local:
                x, y, w, h = map(int, current_bbox_local)
                cv.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(annotated_frame, f"Track: {self.current_score:.2f}", (x, y - 10), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                for i in range(1, len(self.predictor.x_history)):
                    pt1 = (int(self.predictor.x_history[i-1]), int(self.predictor.y_history[i-1]))
                    pt2 = (int(self.predictor.x_history[i]), int(self.predictor.y_history[i]))
                    cv.line(annotated_frame, pt1, pt2, (255, 255, 0), 2)

            if future_path_local:
                for pt in future_path_local:
                    cv.circle(annotated_frame, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)

            if recovering_active:
                search_roi = self.predictor.generate_search_roi(frame.shape)
                rx, ry, rw, rh = search_roi
                cv.rectangle(annotated_frame, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)

            fps_meter.stop()
            with self.state_lock:
                self.fps_rate = fps_meter.getFPS()
                recording_active = self.is_recording
                canvas_w, canvas_h = self.canvas_size 
            fps_meter.reset()
            fps_meter.start()

            if recording_active:
                if not self.video_writer_initialized:
                    h, w, _ = annotated_frame.shape
                    fourcc = cv.VideoWriter_fourcc(*'XVID')
                    self.video_writer = cv.VideoWriter('hybrid_track_output.avi', fourcc, 20.0, (w, h))
                    self.video_writer_initialized = True
                
                if self.video_writer is not None:
                    self.video_writer.write(annotated_frame)
                    
            # Offloading display preprocessing workloads
            frame_rgb = cv.cvtColor(annotated_frame, cv.COLOR_BGR2RGB)
            h, w, _ = frame_rgb.shape
            
            scale_ratio = 1.0
            dx, dy = 0, 0
            img_pil = None
            
            if canvas_w > 10 and canvas_h > 10:
                scale_ratio = min(canvas_w / w, canvas_h / h)
                new_w = int(w * scale_ratio)
                new_h = int(h * scale_ratio)
                img_resized = cv.resize(frame_rgb, (new_w, new_h), interpolation=cv.INTER_NEAREST)
                img_pil = Image.fromarray(img_resized)
                dx = (canvas_w - new_w) / 2
                dy = (canvas_h - new_h) / 2

            scale_info = (scale_ratio, dx, dy)

            if self.display_queue.full():
                try: self.display_queue.get_nowait()
                except queue.Empty: pass
                
            self.display_queue.put((raw_frame, img_pil, scale_info))

    def yolo_detection_worker(self):
        """
        Runs model processing in an isolated pipeline.
        Optimized to copy the target frame OUTSIDE the critical state lock context.
        """
        while self.app_running:
            if self.detection_trigger.wait(timeout=0.1):
                self.detection_trigger.clear()
                
                # Fetch frame pointer quickly using state lock, copy outside of lock
                raw_frame_ref = None
                with self.state_lock:
                    if self.latest_raw_frame is not None:
                        raw_frame_ref = self.latest_raw_frame
                
                if raw_frame_ref is None:
                    self.detection_result_queue.put(None)
                    continue

                frame = raw_frame_ref.copy()

                with self.state_lock:
                    target_cls = self.target_class_id
                    strat = self.selection_strategy

                bbox = self.detector.detect(frame, target_class_id=target_cls, selection_strategy=strat)
                self.detection_result_queue.put((frame, bbox))

    def on_canvas_resize(self, event):
        with self.state_lock:
            self.canvas_size = (event.width, event.height)

    def update_dashboard_frame_callback(self):
        try:
            raw_frame, img_pil, scale_info = self.display_queue.get_nowait()
            self.canvas_scale_info = scale_info
            
            if img_pil:
                # Store PhotoImage in persistent attribute to avoid Garbage Collection erasure
                self.img_tk = ImageTk.PhotoImage(image=img_pil)
                self.video_canvas.delete("video_image")
                img_id = self.video_canvas.create_image(scale_info[1], scale_info[2], anchor=tk.NW, image=self.img_tk, tags="video_image")
                self.video_canvas.tag_lower(img_id) 
        except queue.Empty:
            pass

        with self.state_lock:
            fps_val = self.fps_rate
            score_val = self.current_score
            speed_val = np.hypot(self.speed_vector[0], self.speed_vector[1])
            reid_time_val = self.last_reid_latency
            is_recovering_val = self.is_recovering
            is_tracking_val = self.is_tracking

        self.fps_lbl.config(text=f"System Loop: {fps_val:.2f} FPS")
        self.score_lbl.config(text=f"Tracker Score: {score_val:.2f}")
        self.speed_lbl.config(text=f"Speed: {speed_val:.2f} px/frame")
        self.reid_time_lbl.config(text=f"Re-ID Latency: {reid_time_val:.2f} ms")

        if is_recovering_val:
            self.status_lbl.config(text="Status: RECOVERING LOST TARGET", foreground="red")
        elif is_tracking_val:
            self.status_lbl.config(text="Status: TRACKING ACTIVE", foreground="green")
        else:
            self.status_lbl.config(text="Status: IDLE", foreground="orange")

        if self.app_running:
            self.root.after(15, self.update_dashboard_frame_callback)

    # In-App Canvas ROI Event Handlers
    def toggle_manual_roi_mode(self):
        self.selection_mode = not self.selection_mode
        if self.selection_mode:
            self.manual_btn.config(text="Cancel Manual Selection")
            self.video_canvas.config(cursor="cross")
        else:
            self.manual_btn.config(text="Manual ROI (Draw on Video)")
            self.video_canvas.config(cursor="")
            if self.canvas_rect:
                self.video_canvas.delete(self.canvas_rect)
                self.canvas_rect = None

    def on_canvas_press(self, event):
        if not self.selection_mode or self.latest_raw_frame is None: return
        self.roi_start = (event.x, event.y)
        if self.canvas_rect:
            self.video_canvas.delete(self.canvas_rect)
        self.canvas_rect = self.video_canvas.create_rectangle(
            event.x, event.y, event.x, event.y, 
            outline="magenta", width=3, dash=(4, 4), tags="roi_rect"
        )

    def on_canvas_drag(self, event):
        if not self.selection_mode or not self.roi_start or not self.canvas_rect: return
        self.video_canvas.coords(self.canvas_rect, self.roi_start[0], self.roi_start[1], event.x, event.y)

    def on_canvas_release(self, event):
        if not self.selection_mode or not self.roi_start or self.latest_raw_frame is None: return
        x0, y0 = self.roi_start
        x1, y1 = event.x, event.y
        
        scale_ratio, dx, dy = self.canvas_scale_info
        
        h_f, w_f, _ = self.latest_raw_frame.shape
        real_x0 = int((min(x0, x1) - dx) / scale_ratio)
        real_y0 = int((min(y0, y1) - dy) / scale_ratio)
        real_x1 = int((max(x0, x1) - dx) / scale_ratio)
        real_y1 = int((max(y0, y1) - dy) / scale_ratio)
        
        real_x0 = max(0, min(w_f - 1, real_x0))
        real_y0 = max(0, min(h_f - 1, real_y0))
        real_x1 = max(0, min(w_f - 1, real_x1))
        real_y1 = max(0, min(h_f - 1, real_y1))
        
        real_w = real_x1 - real_x0
        real_h = real_y1 - real_y0
        
        # Cleanup drawn rectangle bounding box shape references from Tkinter Canvas
        if self.canvas_rect:
            self.video_canvas.delete(self.canvas_rect)
            self.canvas_rect = None
            
        self.toggle_manual_roi_mode() 
        
        if real_w > 5 and real_h > 5:
            roi = (real_x0, real_y0, real_w, real_h)
            self.initialize_tracker_with_roi(self.latest_raw_frame.copy(), roi)

    def initialize_tracker_with_roi(self, frame, roi):
        with self.state_lock:
            t_type = self.active_tracker_type
        
        self.tracker_engine.tracker_type = t_type
        init_ok = self.tracker_engine.init_tracker(frame, roi)
        if init_ok:
            self.predictor.clear()
            self.predictor.add_point(roi)
            self.reid_engine.register_target("active_target", frame, roi)
            
            with self.state_lock:
                self.is_tracking = True

    # Native Dashboard Actions & Events
    def on_source_type_changed(self, event):
        source_type = self.source_type_cb.get()
        if source_type == "Video File":
            self.file_entry.config(state="normal")
            self.browse_btn.config(state="normal")
        else:
            self.file_entry.config(state="disabled")
            self.browse_btn.config(state="disabled")

    def browse_video_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov *.mpeg *.ts"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)

    def apply_video_source(self):
        source_type = self.source_type_cb.get()
        with self.state_lock:
            if source_type == "Video File":
                self.video_source_path = self.file_path_var.get()
            else:
                self.video_source_path = source_type
        self.source_changed_event.set()

    def on_resolution_changed(self, event):
        res_str = self.resolution_cb.get()
        with self.state_lock:
            if "320x240" in res_str:
                self.target_resolution = (320, 240)
                self.native_resolution_active = False
            elif "640x480" in res_str:
                self.target_resolution = (640, 480)
                self.native_resolution_active = False
            elif "1280x720" in res_str:
                self.target_resolution = (1280, 720)
                self.native_resolution_active = False
            elif "1920x1080" in res_str:
                self.target_resolution = (1920, 1080)
                self.native_resolution_active = False
            elif "Native" in res_str:
                self.native_resolution_active = True

    def on_reid_toggle(self):
        val = self.reid_enabled_var.get()
        with self.state_lock:
            self.reid_enabled = val

    def on_reid_threshold_changed(self, event):
        val_str = self.reid_thresh_cb.get()
        numeric_part = val_str.split()[0]
        try:
            val_float = float(numeric_part)
            with self.state_lock:
                self.reid_threshold = val_float
        except ValueError:
            pass

    def on_reid_model_changed(self, event):
        """
        Dynamically swaps the Re-ID backend models and empties CUDA memory.
        """
        model_str = self.reid_model_cb.get()
        model_name = "resnet18" if "resnet18" in model_str else "mobilenet_v2"
        
        with self.state_lock:
            self.active_reid_model = model_name
            # Recreate engine with selected backbone
            self.reid_engine = ReIDEngine(backbone_type=model_name)
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA memory cache cleared after changing Re-ID model backbone.")

    def on_detection_model_changed(self, event):
        """
        Dynamically transitions the detection model backbone via GUI combobox.
        """
        selected_gui_name = self.det_model_cb.get()
        model_key = self.detection_model_map.get(selected_gui_name, "yolo11n")
        
        # Guard changes using self.state_lock
        with self.state_lock:
            self.active_detection_model = model_key
            self.detector.change_model(model_key)
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"CUDA memory cache explicitly swept after switching to detection backbone: {model_key}")

    def toggle_recording(self):
        with self.state_lock:
            if not self.is_recording:
                self.is_recording = True
                self.video_writer_initialized = False
                self.record_btn.config(text="⏹️ Stop Recording")
                self.rec_status_lbl.config(text="Recording: ACTIVE", foreground="red")
            else:
                self.is_recording = False
                if self.video_writer is not None:
                    self.video_writer.release()
                    self.video_writer = None
                self.video_writer_initialized = False
                self.record_btn.config(text="🔴 Start Recording")
                self.rec_status_lbl.config(text="Recording: STOPPED", foreground="gray")

    def trigger_yolo_auto_initialization(self):
        self.detection_trigger.set()
        def check_result_poller():
            try:
                result = self.detection_result_queue.get_nowait()
                if result:
                    frame, bbox = result
                    if bbox:
                        self.initialize_tracker_with_roi(frame, bbox)
            except queue.Empty:
                if self.app_running:
                    self.root.after(50, check_result_poller)
        self.root.after(50, check_result_poller)

    def reset_pipeline_states(self):
        """
        Resets tracking flags, parameters, and frees PyTorch memory allocations.
        """
        with self.state_lock:
            self.is_tracking = False
            self.is_recovering = False
            self.current_score = 0.0
            self.speed_vector = (0.0, 0.0)
            self.last_reid_latency = 0.0
            self.latest_raw_frame = None
        self.predictor.clear()
        self.tracker_engine.initialized = False
        
        # Free CUDA cache dynamically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("PyTorch CUDA memory cache explicitly cleared.")

    def on_tracker_config_changed(self, event):
        val = self.tracker_sel_cb.get()
        with self.state_lock:
            self.active_tracker_type = val

    def on_pred_method_changed(self, event):
        val = self.pred_method_cb.get()
        method_key = "ransac" if "RANSAC" in val else "wls"
        self.predictor.method = method_key

    def on_class_config_changed(self, event):
        selected_name = self.class_sel_cb.get()
        for idx, val in self.coco_classes.items():
            if val == selected_name:
                with self.state_lock:
                    self.target_class_id = idx
                break

    def on_strategy_changed(self, event):
        val = self.strategy_cb.get()
        with self.state_lock:
            self.selection_strategy = val

    def bind_closing_events(self):
        def on_app_closed_handler():
            self.app_running = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.root.destroy()
            sys.exit(0)
        self.root.protocol("WM_DELETE_WINDOW", on_app_closed_handler)

if __name__ == "__main__":
    tk_root = tk.Tk()
    app = DashboardApp(tk_root)
    tk_root.mainloop()