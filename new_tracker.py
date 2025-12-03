import cv2
import pyrealsense2 as rs
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import time
import pandas as pd
from datetime import datetime

class HybridFocusTracker:
    def __init__(self, yolo_model_path='yolo11n-pose.pt'):
        """
        Initialize RealSense, MediaPipe, and YOLO.
        """
        # 1. Setup Intel RealSense D455
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure streams at 30 FPS (Matches Thesis Req)
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        
        # Start pipeline and get intrinsics for 3D mapping
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color) # Align depth to color
        self.intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        # 2. Setup MediaPipe Pose (CPU - Upper Body)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,      # 1 is balanced for real-time
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 3. Setup YOLO11 (GPU - Hands)
        # Load your custom hand-pose model here
        self.yolo_model = YOLO(yolo_model_path) 
        
        # Data Logging
        self.session_data = []
        self.start_time = None

    def get_3d_point(self, u, v, depth_frame):
        """
        Convert 2D pixel (u, v) + Depth to 3D Metric Coordinates (x, y, z).
        Returns (x, y, z) in meters.
        """
        # Ensure coordinates are within frame bounds
        if u < 0 or u >= 848 or v < 0 or v >= 480:
            return None
        
        # Get distance (Z) from depth map
        dist = depth_frame.get_distance(u, v)
        if dist == 0:
            return None # Invalid depth

        # Deproject pixel to 3D point
        point_3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], dist)
        return point_3d

    def run_session(self, duration_sec=30):
        print(f"Starting Hybrid Tracking Session for {duration_sec} seconds...")
        self.start_time = time.time() * 1000 # Milliseconds
        
        try:
            while True:
                # -- 1. Data Acquisition --
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue

                # Current timestamp (ms)
                current_ts = time.time() * 1000
                elapsed = (current_ts - self.start_time) / 1000
                
                # Convert images for processing
                color_image = np.asanyarray(color_frame.get_data())
                
                # -- 2. MediaPipe Processing (Upper Body) --
                # MP requires RGB
                color_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                mp_results = self.pose.process(color_rgb)
                
                body_data = {}
                
                if mp_results.pose_landmarks:
                    # Draw Skeleton (Visual Debug)
                    mp.solutions.drawing_utils.draw_landmarks(
                        color_image, mp_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    
                    # Extract Shoulders (11=Left, 12=Right) and Nose (0)
                    landmarks = mp_results.pose_landmarks.landmark
                    h, w, c = color_image.shape
                    
                    for idx, name in [(11, 'L_Shoulder'), (12, 'R_Shoulder'), (0, 'Nose')]:
                        lm = landmarks[idx]
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        
                        # Get 3D position
                        point_3d = self.get_3d_point(cx, cy, depth_frame)
                        
                        if point_3d:
                            body_data[f"{name}_X"] = point_3d[0]
                            body_data[f"{name}_Y"] = point_3d[1]
                            body_data[f"{name}_Z"] = point_3d[2] # Depth

                # -- 3. YOLO Processing (Hands) --
                # YOLO runs on the BGR image directly
                yolo_results = self.yolo_model(color_image, verbose=False)
                
                hand_data = {}
                
                # Process YOLO detections
                for r in yolo_results:
                    # If using a Pose model, 'keypoints' contains skeleton data
                    if r.keypoints is not None:
                        # Logic depends on your specific Hand-Pose model keypoint indices
                        # Here we visualize the bounding boxes for context
                        boxes = r.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = box.conf[0]
                            
                            # Draw Box (Visual Debug)
                            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Calculate Center of Hand for Depth
                            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                            hand_3d = self.get_3d_point(center_x, center_y, depth_frame)
                            
                            if hand_3d:
                                # Log simplified hand center for now
                                # (Refine this loop for Left/Right separation later)
                                hand_data[f"Hand_Box_{len(hand_data)//3}_X"] = hand_3d[0]
                                hand_data[f"Hand_Box_{len(hand_data)//3}_Y"] = hand_3d[1]
                                hand_data[f"Hand_Box_{len(hand_data)//3}_Z"] = hand_3d[2]

                # -- 4. Data Logging --
                # Combine all data into one row
                row = {'Timestamp_ms': current_ts}
                row.update(body_data)
                row.update(hand_data)
                self.session_data.append(row)

                # -- 5. Visualization --
                cv2.imshow('Hybrid Focus Tracker', color_image)
                
                # Exit conditions
                if cv2.waitKey(1) & 0xFF == ord('q') or elapsed > duration_sec:
                    break

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            self.save_data()

    def save_data(self):
        """Save captured data to CSV"""
        df = pd.DataFrame(self.session_data)
        filename = f"session_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        print(f"Session Saved: {filename}")

if __name__ == "__main__":
    # Ensure you point this to your actual model file
    tracker = HybridFocusTracker(yolo_model_path='yolo11n-pose.pt')
    tracker.run_session(duration_sec=15)