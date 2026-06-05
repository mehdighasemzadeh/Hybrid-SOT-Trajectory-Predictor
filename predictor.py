import numpy as np
import warnings

# Safe import of scikit-learn with graceful CPU fallback
try:
    from sklearn.linear_model import RANSACRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class TrajectoryPredictor:
    def __init__(self, history_len=15, prediction_steps=20, method="wls"):
        """
        Calculates target trajectory vectors. Supports both ultra-fast NumPy 
        Weighted Least Squares (WLS) and robust RANSAC Regression.
        
        method: "wls" or "ransac"
        """
        self.history_len = history_len
        self.prediction_steps = prediction_steps
        self.method = method if (method != "ransac" or SKLEARN_AVAILABLE) else "wls"
        
        # Internal coordinate storage arrays
        self.t_history = []
        self.x_history = []
        self.y_history = []
        self.w_history = []
        self.h_history = []
        
        self.current_frame_id = 0

        if not SKLEARN_AVAILABLE and method == "ransac":
            warnings.warn("scikit-learn not installed. Defaulting trajectory method to WLS.")

    def clear(self):
        self.t_history.clear()
        self.x_history.clear()
        self.y_history.clear()
        self.w_history.clear()
        self.h_history.clear()
        self.current_frame_id = 0

    def add_point(self, bbox):
        """
        Appends a tracked bounding box (x, y, w, h) to estimation history.
        """
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0
        
        self.t_history.append(self.current_frame_id)
        self.x_history.append(cx)
        self.y_history.append(cy)
        self.w_history.append(w)
        self.h_history.append(h)
        
        self.current_frame_id += 1

        if len(self.t_history) > self.history_len:
            self.t_history.pop(0)
            self.x_history.pop(0)
            self.y_history.pop(0)
            self.w_history.pop(0)
            self.h_history.pop(0)

    def predict(self):
        """
        Applies either NumPy-optimized Weighted Linear Regression or robust RANSAC.
        """
        n_points = len(self.t_history)
        if n_points < 4:
            return (0.0, 0.0), [], (0.0, 0.0)

        T = np.array(self.t_history).reshape(-1, 1)
        X = np.array(self.x_history)
        Y = np.array(self.y_history)

        # Method 1: Robust RANSAC
        if self.method == "ransac" and SKLEARN_AVAILABLE:
            try:
                ransac_x = RANSACRegressor(min_samples=min(3, n_points - 1), max_trials=50, residual_threshold=10.0, random_state=42)
                ransac_y = RANSACRegressor(min_samples=min(3, n_points - 1), max_trials=50, residual_threshold=10.0, random_state=42)
                
                ransac_x.fit(T, X)
                ransac_y.fit(T, Y)
                
                vx = ransac_x.estimator_.coef_[0] if hasattr(ransac_x.estimator_, 'coef_') else 0.0
                vy = ransac_y.estimator_.coef_[0] if hasattr(ransac_y.estimator_, 'coef_') else 0.0
                
                last_t = self.t_history[-1]
                future_t = np.arange(last_t + 1, last_t + self.prediction_steps + 1).reshape(-1, 1)
                
                pred_x = ransac_x.predict(future_t)
                pred_y = ransac_y.predict(future_t)
                
                predicted_path = list(zip(pred_x, pred_y))
                avg_w = np.mean(self.w_history)
                avg_h = np.mean(self.h_history)
                
                return (vx, vy), predicted_path, (avg_w, avg_h)

            except Exception:
                pass

        # Method 2: Weighted Least Squares (Default high-speed NumPy path)
        try:
            T_flat = T.flatten()
            weights = np.linspace(0.5, 1.0, n_points)
            
            poly_x = np.polyfit(T_flat, X, 1, w=weights)
            poly_y = np.polyfit(T_flat, Y, 1, w=weights)
            
            vx = poly_x[0]
            vy = poly_y[0]
            
            last_t = T_flat[-1]
            future_steps = np.arange(last_t + 1, last_t + self.prediction_steps + 1)
            
            pred_x = poly_x[0] * future_steps + poly_x[1]
            pred_y = poly_y[0] * future_steps + poly_y[1]
            
            predicted_path = list(zip(pred_x, pred_y))
            avg_w = np.mean(self.w_history)
            avg_h = np.mean(self.h_history)
            
            return (vx, vy), predicted_path, (avg_w, avg_h)

        except Exception:
            if n_points >= 2:
                vx = self.x_history[-1] - self.x_history[-2]
                vy = self.y_history[-1] - self.y_history[-2]
                predicted_path = [
                    (self.x_history[-1] + vx * i, self.y_history[-1] + vy * i)
                    for i in range(1, self.prediction_steps + 1)
                ]
                return (vx, vy), predicted_path, (self.w_history[-1], self.h_history[-1])
            
            return (0.0, 0.0), [], (0.0, 0.0)

    def generate_search_roi(self, frame_shape):
        """
        Creates a dynamic recovery bounding box based on predicted trajectory.
        
        SMART BOUNDARY ESCAPE UPGRADE:
        If the target's trajectory vector indicates it is escaping through the edges of the frame,
        this method overrides the local crop and sets the search area to cover the corresponding
        entire outer band (top, bottom, left, or right) where the object is likely to linger or re-appear.
        """
        h_f, w_f, _ = frame_shape
        if len(self.t_history) < 2:
            return (0, 0, w_f, h_f)

        (vx, vy), path, (target_w, target_h) = self.predict()
        
        # Pull last known coordinates and dimensions
        last_cx = self.x_history[-1]
        last_cy = self.y_history[-1]
        last_w = self.w_history[-1]
        last_h = self.h_history[-1]

        # Estimate the next expected positions
        next_cx = last_cx + vx * 2.0
        next_cy = last_cy + vy * 2.0

        # Define dynamic thresholds (buffer margins near frame borders)
        margin_x = max(last_w * 0.8, 50.0)
        margin_y = max(last_h * 0.8, 50.0)

        # Escape detection logic
        is_escaping_top = (next_cy - last_h/2.0 < margin_y) or (vy < -1.5 and last_cy < margin_y * 2.0)
        is_escaping_bottom = (next_cy + last_h/2.0 > h_f - margin_y) or (vy > 1.5 and last_cy > h_f - margin_y * 2.0)
        is_escaping_left = (next_cx - last_w/2.0 < margin_x) or (vx < -1.5 and last_cx < margin_x * 2.0)
        is_escaping_right = (next_cx + last_w/2.0 > w_f - margin_x) or (vx > 1.5 and last_cx > w_f - margin_x * 2.0)

        # Define outer-band sweep thicknesses
        band_thickness_y = int(max(last_h * 2.5, 120.0))
        band_thickness_x = int(max(last_w * 2.5, 120.0))

        # Apply specific full-border sweep regions based on exit vectors
        if is_escaping_top:
            print("[Predictor] Target escaping through TOP. Sweeping complete top band.")
            return (0, 0, w_f, min(h_f, band_thickness_y))
            
        elif is_escaping_bottom:
            print("[Predictor] Target escaping through BOTTOM. Sweeping complete bottom band.")
            y_start = max(0, h_f - band_thickness_y)
            return (0, y_start, w_f, h_f - y_start)
            
        elif is_escaping_left:
            print("[Predictor] Target escaping through LEFT. Sweeping complete left band.")
            return (0, 0, min(w_f, band_thickness_x), h_f)
            
        elif is_escaping_right:
            print("[Predictor] Target escaping through RIGHT. Sweeping complete right band.")
            x_start = max(0, w_f - band_thickness_x)
            return (x_start, 0, w_f - x_start, h_f)

        # Standard Fallback: Localized dynamic target tracking search region
        if not path:
            return (0, 0, w_f, h_f)

        target_cx, target_cy = path[3] if len(path) > 4 else path[-1]
        speed = np.hypot(vx, vy)
        scale_factor = 2.0 + min(1.5, speed * 0.1)
        
        search_w = int(target_w * scale_factor)
        search_h = int(target_h * scale_factor)
        
        search_x = int(target_cx - search_w / 2.0)
        search_y = int(target_cy - search_h / 2.0)
        
        x1 = max(0, search_x)
        y1 = max(0, search_y)
        x2 = min(w_f, search_x + search_w)
        y2 = min(h_f, search_y + search_h)
        
        # Ensure minimum valid bounds
        if x2 - x1 < 10:
            x1 = max(0, x1 - 20)
            x2 = min(w_f, x2 + 20)
        if y2 - y1 < 10:
            y1 = max(0, y1 - 20)
            y2 = min(h_f, y2 + 20)

        return (x1, y1, x2 - x1, y2 - y1)