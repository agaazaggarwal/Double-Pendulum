import cv2
import numpy as np
from utils import smooth_trajectory, unwrap_angle

class DoublePendulumTracker:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.pivot = None
        self.blue_range = None   
        self.red_range = None    

    def _click_point(self, frame, window_name, instruction):
        print(f"\n👉 {instruction}")
        point = []
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                point.append((x, y))
                print(f"Selected at: ({x}, {y})")
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)
        while True:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF != 255 or len(point) > 0:
                break
        cv2.destroyAllWindows()
        if not point:
            raise RuntimeError(f"No point selected for {window_name}")
        return np.array(point[0], dtype=np.float32)

    def _get_hsv_range_from_click(self, frame, marker_name):
        center = self._click_point(frame, f"Click on {marker_name}", 
                                   f"Click exactly on the {marker_name} marker")
        x, y = int(center[0]), int(center[1])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[y, x]
        h_min = max(0, h - 15)
        h_max = min(179, h + 15)
        s_min = max(0, s - 60)
        s_max = min(255, s + 60)
        v_min = max(0, v - 60)
        v_max = min(255, v + 60)
        lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
        print(f"HSV range for {marker_name}: {lower} -> {upper}")
        return lower, upper

    def track(self):
        ret, first_frame = self.cap.read()
        if not ret:
            raise RuntimeError("Could not read video")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.pivot = self._click_point(first_frame, "Click on PIVOT", 
                                       "Click exactly on the FIXED PIVOT point (where the top rod hangs from the support)")
        self.blue_range = self._get_hsv_range_from_click(first_frame, "TOP (blue) marker")
        self.red_range = self._get_hsv_range_from_click(first_frame, "BOTTOM (red) marker")

        positions_top = []
        positions_bot = []
        timestamps = []
        frame_idx = 0
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"\nProcessing {total_frames} frames...")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx}/{total_frames} frames")

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            
            mask_blue = cv2.inRange(hsv, self.blue_range[0], self.blue_range[1])
            mask_blue = cv2.erode(mask_blue, None, iterations=2)
            mask_blue = cv2.dilate(mask_blue, None, iterations=2)
            contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            top_center = None
            if contours_blue:
                largest = max(contours_blue, key=cv2.contourArea)
                if cv2.contourArea(largest) > 50:
                    M = cv2.moments(largest)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        top_center = np.array([cx, cy])

            
            mask_red = cv2.inRange(hsv, self.red_range[0], self.red_range[1])
            mask_red = cv2.erode(mask_red, None, iterations=2)
            mask_red = cv2.dilate(mask_red, None, iterations=2)
            contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bot_center = None
            if contours_red:
                largest = max(contours_red, key=cv2.contourArea)
                if cv2.contourArea(largest) > 50:
                    M = cv2.moments(largest)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        bot_center = np.array([cx, cy])

            if top_center is not None and bot_center is not None:
                
                dist_top = np.linalg.norm(top_center - self.pivot)
                dist_bot = np.linalg.norm(bot_center - self.pivot)
                if dist_top > dist_bot:
                    top_center, bot_center = bot_center, top_center
                positions_top.append(top_center)
                positions_bot.append(bot_center)
                timestamps.append(frame_idx / self.fps)

        self.cap.release()

        if len(positions_top) < 10:
            print(f"WARNING: Only detected {len(positions_top)} frames with both markers.")
            if len(positions_top) == 0:
                raise RuntimeError("No frames with both markers detected. Check video and lighting.")

        print(f"Tracking complete. Found {len(positions_top)} frames with both markers.")

        
        positions_top = np.array(positions_top)
        positions_bot = np.array(positions_bot)
        timestamps = np.array(timestamps)

        
        unique_idx = np.unique(timestamps, return_index=True)[1]
        positions_top = positions_top[unique_idx]
        positions_bot = positions_bot[unique_idx]
        timestamps = timestamps[unique_idx]

        
        if len(positions_top) >= 5:
            positions_top = smooth_trajectory(positions_top)
            positions_bot = smooth_trajectory(positions_bot)

        
        theta1 = []
        theta2 = []
        for (x1, y1), (x2, y2) in zip(positions_top, positions_bot):
            
            v1 = (x1 - self.pivot[0], y1 - self.pivot[1])
            
            v2 = (x2 - x1, y2 - y1)
            
            ang1 = np.arctan2(v1[0], v1[1])   
            ang2 = np.arctan2(v2[0], v2[1])
            theta1.append(ang1)
            theta2.append(ang2)

        theta1 = unwrap_angle(np.array(theta1))
        theta2 = unwrap_angle(np.array(theta2))

        
        dt = np.diff(timestamps)
        dt = np.maximum(dt, 1e-6)
        omega1 = np.zeros_like(theta1)
        omega2 = np.zeros_like(theta2)
        omega1[1:-1] = (theta1[2:] - theta1[:-2]) / (timestamps[2:] - timestamps[:-2])
        omega2[1:-1] = (theta2[2:] - theta2[:-2]) / (timestamps[2:] - timestamps[:-2])
        omega1[0] = (theta1[1] - theta1[0]) / dt[0]
        omega2[0] = (theta2[1] - theta2[0]) / dt[0]
        omega1[-1] = (theta1[-1] - theta1[-2]) / dt[-1]
        omega2[-1] = (theta2[-1] - theta2[-2]) / dt[-1]
        omega1 = np.nan_to_num(omega1, nan=0.0, posinf=0.0, neginf=0.0)
        omega2 = np.nan_to_num(omega2, nan=0.0, posinf=0.0, neginf=0.0)

        return timestamps, theta1, theta2, omega1, omega2, positions_top, positions_bot