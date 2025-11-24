import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import csv
import time
import os

# ================================================
# GLOBAL: baseline shoulder width for rotation detection
# ================================================
baseline_shoulder_width = None


# -----------------------------
# 1. Depth helper function
# -----------------------------
def get_depth_distance(x_pixel, y_pixel, depth_frame):
    try:
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        x = int(x_pixel * width)
        y = int(y_pixel * height)
        if 0 <= x < width and 0 <= y < height:
            return depth_frame.get_distance(x, y)
        return 0
    except:
        return 0


# -----------------------------
# 2. Logging system (CSV + Video)
# -----------------------------
class SessionLogger:
    def __init__(self, session_name="session"):
        self.timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        self.folder = f"recordings/{session_name}_{self.timestamp_str}"
        os.makedirs(self.folder, exist_ok=True)

        self.csv_upper_path = f"{self.folder}/upper_body.csv"
        self.csv_hands_path = f"{self.folder}/hands_data.csv"
        self.video_path = f"{self.folder}/output.avi"

        self.fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.out = cv2.VideoWriter(self.video_path, self.fourcc, 30.0, (1280, 720))

        self.setup_csvs()
        print(f"[INFO] Recording started: {self.folder}")

    def setup_csvs(self):
        # Upper Body CSV
        with open(self.csv_upper_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp_ms",
                "Nose_X", "Nose_Y", "Nose_Dist_M",
                "L_Shoulder_X", "L_Shoulder_Y", "L_Shoulder_Dist_M",
                "R_Shoulder_X", "R_Shoulder_Y", "R_Shoulder_Dist_M"
            ])

        # Hands CSV
        with open(self.csv_hands_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp_ms",
                "Fingers_Detected",
                "Hand_Count"
            ])

    def log_frame(self, frame, timestamp_ms, pose_landmarks, hand_landmarks, depth_frame):
        # Save video frame
        self.out.write(frame)

        # ---- Upper Body ----
        if pose_landmarks:
            lm = pose_landmarks.landmark

            def get_data(idx):
                x = lm[idx].x
                y = lm[idx].y
                dist = get_depth_distance(x, y, depth_frame)
                return [x, y, dist]

            row = [timestamp_ms] + get_data(0) + get_data(11) + get_data(12)

            with open(self.csv_upper_path, "a", newline="") as f:
                csv.writer(f).writerow(row)

        # ---- Hands ----
        fingers_detected = "No"
        hand_count = 0

        if hand_landmarks:
            fingers_detected = "Yes"
            hand_count = len(hand_landmarks)

        with open(self.csv_hands_path, "a", newline="") as f:
            csv.writer(f).writerow([timestamp_ms, fingers_detected, hand_count])

    def close(self):
        self.out.release()
        print(f"[INFO] Recording saved in {self.folder}")


# -----------------------------
# 3. Main tracking function
# -----------------------------
def start_thesis_tracking():
    global baseline_shoulder_width

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    # Pose model
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )

    # Hands model
    hands = mp_hands.Hands(
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        max_num_hands=2,
        model_complexity=1
    )

    # RealSense setup
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    print("[INFO] System Ready — Press 'r' to Record, 'q' to Quit.")
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    logger = None
    is_recording = False

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            timestamp_ms = frames.get_timestamp()

            image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            pose_results = pose.process(image_rgb)
            hands_results = hands.process(image_rgb)

            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            h, w, _ = image_bgr.shape

            # ---------------------------------------------------------
            # POSE (UPPER BODY + ENGAGED + BODY TURN UNFOCUSED)
            # ---------------------------------------------------------
            if pose_results.pose_landmarks:
                lm = pose_results.pose_landmarks.landmark

                left_shoulder = lm[11]
                right_shoulder = lm[12]
                nose = lm[0]

                ls_px = (int(left_shoulder.x * w), int(left_shoulder.y * h))
                rs_px = (int(right_shoulder.x * w), int(right_shoulder.y * h))

                # Shoulder line (for visualization)
                cv2.line(image_bgr, ls_px, rs_px, (255, 0, 0), 3)

                # ---------- 1) Width ratio (screen-space) ----------
                current_width = abs(left_shoulder.x - right_shoulder.x)

                if baseline_shoulder_width is None and current_width > 0:
                    baseline_shoulder_width = current_width

                width_ratio = (
                    current_width / baseline_shoulder_width
                    if baseline_shoulder_width
                    else 1.0
                )

                # ---------- 2) Depth difference (3D) ----------
                depth_left = get_depth_distance(left_shoulder.x, left_shoulder.y, depth_frame)
                depth_right = get_depth_distance(right_shoulder.x, right_shoulder.y, depth_frame)
                depth_diff = abs(depth_left - depth_right) if depth_left > 0 and depth_right > 0 else 0.0

                # ---------- 3) Height difference (projection) ----------
                height_diff = abs(left_shoulder.y - right_shoulder.y)

                # ---------- Combined rotation score ----------
                rotation_score = 0

                # Width shrunk significantly (turned sideways)
                if baseline_shoulder_width and width_ratio < 0.75:
                    rotation_score += 1

                # One shoulder clearly closer (rotating chest)
                if depth_diff > 0.10:  # meters, tweak if needed
                    rotation_score += 1

                # One shoulder visibly higher than the other
                if height_diff > 0.03:
                    rotation_score += 1

                # ---------- Decide focus status ----------
                nose_dist = get_depth_distance(nose.x, nose.y, depth_frame)

                if rotation_score >= 2:
                    # Strong evidence of rotation
                    cv2.putText(
                        image_bgr,
                        "UNFOCUSED (BODY TURN)",
                        (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        3,
                    )
                else:
                    # Use distance-based engagement
                    if 0 < nose_dist < 0.80:
                        cv2.putText(
                            image_bgr,
                            "ENGAGED",
                            (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )
                    else:
                        cv2.putText(
                            image_bgr,
                            "UNFOCUSED (DISTANCE)",
                            (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 255),
                            3,
                        )

                # ---- Upper-body-only drawing ----
                upper_points = [
                    0, 1, 2, 3, 4,
                    11, 12, 13, 14, 15, 16,
                    23, 24
                ]

                for i in upper_points:
                    x = int(lm[i].x * w)
                    y = int(lm[i].y * h)
                    cv2.circle(image_bgr, (x, y), 5, (0, 255, 255), -1)

                connections = [
                    (11, 12),
                    (11, 13), (13, 15),
                    (12, 14), (14, 16),
                    (11, 23), (12, 24),
                    (23, 24)
                ]

                for (a, b) in connections:
                    ax, ay = int(lm[a].x * w), int(lm[a].y * h)
                    bx, by = int(lm[b].x * w), int(lm[b].y * h)
                    cv2.line(image_bgr, (ax, ay), (bx, by), (255, 255, 255), 2)

            # ---------------------------------------------------------
            # HANDS — EXACTLY HOW YOU WANT THEM
            # ---------------------------------------------------------
            if hands_results and hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    wx, wy = int(wrist.x * w), int(wrist.y * h)
                    cv2.circle(image_bgr, (wx, wy), 6, (0, 255, 255), -1)

                    fingertip_ids = [
                        mp_hands.HandLandmark.THUMB_TIP,
                        mp_hands.HandLandmark.INDEX_FINGER_TIP,
                        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        mp_hands.HandLandmark.RING_FINGER_TIP,
                        mp_hands.HandLandmark.PINKY_TIP
                    ]

                    for tip in fingertip_ids:
                        pt = hand_landmarks.landmark[tip]
                        tx, ty = int(pt.x * w), int(pt.y * h)
                        cv2.line(image_bgr, (wx, wy), (tx, ty), (0, 255, 255), 2)
                        cv2.circle(image_bgr, (tx, ty), 4, (0, 255, 255), -1)

            # ---------------------------------------------------------
            # RECORDING
            # ---------------------------------------------------------
            if is_recording:
                cv2.circle(image_bgr, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(image_bgr, "REC", (50, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                logger.log_frame(
                    image_bgr,
                    timestamp_ms,
                    pose_results.pose_landmarks if pose_results else None,
                    hands_results.multi_hand_landmarks if (hands_results and hands_results.multi_hand_landmarks) else None,
                    depth_frame
                )

            cv2.imshow("Thesis Tracker: Clean Upper Body + Hands", image_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                if not is_recording:
                    logger = SessionLogger()
                    is_recording = True
                else:
                    logger.close()
                    logger = None
                    is_recording = False

    finally:
        if logger:
            logger.close()
        pipeline.stop()
        pose.close()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    start_thesis_tracking()
