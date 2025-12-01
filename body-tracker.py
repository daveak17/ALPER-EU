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


# ----------------------------
# 1. Depth helper function
# ----------------------------
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


def smooth(prev, new, alpha=0.4):
    """Simple exponential smoothing helper.

    Works with scalars or numpy arrays. If `prev` is None, returns `new`.
    """
    if prev is None:
        return new
    return prev * (1 - alpha) + new * alpha


# Hand smoothing storage (per-hand, per landmark)
smooth_hands = {}

# Upper body smoothing storage
upper_body_ids = [0, 11, 12, 13, 14, 15, 16, 23, 24]  # nose + full upper body
smooth_upper = {i: {"x": None, "y": None} for i in upper_body_ids}


def smooth_hand_landmarks(hand_id, hand_landmarks, alpha=0.45):
    """Smooths an entire hand's 21 landmarks.

    Returns a list of (x,y) tuples for the smoothed landmarks.
    """
    if hand_id not in smooth_hands:
        smooth_hands[hand_id] = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
        return smooth_hands[hand_id]

    new_list = []
    for i, lm in enumerate(hand_landmarks.landmark):
        prev_x, prev_y = smooth_hands[hand_id][i]
        new_x = smooth(prev_x, lm.x, alpha)
        new_y = smooth(prev_y, lm.y, alpha)
        new_list.append((new_x, new_y))

    smooth_hands[hand_id] = new_list
    return new_list


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
    # smoothing state for key landmarks (shoulders + nose)
    smooth_lm = {
        "ls": {"x": None, "y": None, "d": None},
        "rs": {"x": None, "y": None, "d": None},
        "nose": {"x": None, "y": None, "d": None}
    }

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
                visibility_threshold = 0.5

                # SKIP frame if shoulders are not visible enough
                if lm[11].visibility < 0.5 or lm[12].visibility < 0.5:
                    cv2.putText(image_bgr, "LOW CONFIDENCE", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    continue

                # Extract raw landmarks
                raw_ls = lm[11]
                raw_rs = lm[12]
                raw_nose = lm[0]

                # --------------------- SMOOTH LANDMARKS ---------------------
                smooth_lm["ls"]["x"] = smooth(smooth_lm["ls"]["x"], raw_ls.x)
                smooth_lm["ls"]["y"] = smooth(smooth_lm["ls"]["y"], raw_ls.y)

                smooth_lm["rs"]["x"] = smooth(smooth_lm["rs"]["x"], raw_rs.x)
                smooth_lm["rs"]["y"] = smooth(smooth_lm["rs"]["y"], raw_rs.y)

                smooth_lm["nose"]["x"] = smooth(smooth_lm["nose"]["x"], raw_nose.x)
                smooth_lm["nose"]["y"] = smooth(smooth_lm["nose"]["y"], raw_nose.y)

                # --------------------- DEPTH SMOOTHING ----------------------
                ls_depth_raw = get_depth_distance(raw_ls.x, raw_ls.y, depth_frame)
                rs_depth_raw = get_depth_distance(raw_rs.x, raw_rs.y, depth_frame)
                nose_depth_raw = get_depth_distance(raw_nose.x, raw_nose.y, depth_frame)

                smooth_lm["ls"]["d"] = smooth(smooth_lm["ls"]["d"], ls_depth_raw)
                smooth_lm["rs"]["d"] = smooth(smooth_lm["rs"]["d"], rs_depth_raw)
                smooth_lm["nose"]["d"] = smooth(smooth_lm["nose"]["d"], nose_depth_raw)

                # --------------------- USE SMOOTHED VALUES ---------------------
                ls_x = smooth_lm["ls"]["x"]
                ls_y = smooth_lm["ls"]["y"]
                rs_x = smooth_lm["rs"]["x"]
                rs_y = smooth_lm["rs"]["y"]
                nose_x = smooth_lm["nose"]["x"]
                nose_y = smooth_lm["nose"]["y"]

                depth_left = smooth_lm["ls"]["d"]
                depth_right = smooth_lm["rs"]["d"]
                nose_dist = smooth_lm["nose"]["d"]

                # Pixel version for drawing
                ls_px = (int(ls_x * w), int(ls_y * h))
                rs_px = (int(rs_x * w), int(rs_y * h))

                cv2.line(image_bgr, ls_px, rs_px, (255, 0, 0), 3)

                # ---------- 1) Width ratio ----------
                current_width = abs(ls_x - rs_x)

                if baseline_shoulder_width is None and current_width > 0:
                    baseline_shoulder_width = current_width

                width_ratio = (
                    current_width / baseline_shoulder_width
                    if baseline_shoulder_width
                    else 1.0
                )

                # ---------- 2) Depth difference ----------
                depth_diff = abs(depth_left - depth_right) if depth_left and depth_right else 0.0

                # ---------- 3) Height difference ----------
                height_diff = abs(ls_y - rs_y)

                # ---------- Combined rotation score ----------
                rotation_score = 0

                if baseline_shoulder_width and width_ratio < 0.75:
                    rotation_score += 1

                if depth_diff > 0.10:
                    rotation_score += 1

                if height_diff > 0.03:
                    rotation_score += 1

                # ---------- Decide focus status ----------
                if rotation_score >= 2:
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

                # ---------- Smooth & draw upper-body points ----------
                for i in upper_body_ids:
                    raw = lm[i]
                    prev = smooth_upper[i]

                    sx = smooth(prev["x"], raw.x, 0.4)
                    sy = smooth(prev["y"], raw.y, 0.4)

                    smooth_upper[i]["x"] = sx
                    smooth_upper[i]["y"] = sy

                    x = int(np.clip(sx * w, 0, w - 1))
                    y = int(np.clip(sy * h, 0, h - 1))
                    cv2.circle(image_bgr, (x, y), 5, (0, 255, 255), -1)

            # ---------------------------------------------------------
            # HANDS — EXACTLY HOW YOU WANT THEM
            # ---------------------------------------------------------
            if hands_results and hands_results.multi_hand_landmarks:
                # Use handedness/ordering to assign a stable hand_id per hand
                for hid, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                    # Smooth the whole hand and draw using smoothed coords
                    smoothed = smooth_hand_landmarks(hid, hand_landmarks, alpha=0.45)

                    # Wrist (landmark 0)
                    wx_f, wy_f = smoothed[0]
                    wx, wy = int(wx_f * w), int(wy_f * h)
                    cv2.circle(image_bgr, (wx, wy), 6, (0, 255, 255), -1)

                    fingertip_ids = [
                        mp_hands.HandLandmark.THUMB_TIP,
                        mp_hands.HandLandmark.INDEX_FINGER_TIP,
                        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        mp_hands.HandLandmark.RING_FINGER_TIP,
                        mp_hands.HandLandmark.PINKY_TIP,
                    ]

                    for tip in fingertip_ids:
                        sx, sy = smoothed[tip]
                        tx, ty = int(sx * w), int(sy * h)
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
