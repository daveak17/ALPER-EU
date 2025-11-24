import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
import csv
import time
from datetime import datetime

# -----------------------------
# Config / Constants
# -----------------------------
COLOR_WIDTH = 640
COLOR_HEIGHT = 480
FPS = 30

# Output file names (you can change these or pass as parameters later)
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
UPPER_CSV_PATH = f"body_upper_{timestamp_str}.csv"
LOWER_CSV_PATH = f"body_lower_{timestamp_str}.csv"
VIDEO_PATH = f"body_video_{timestamp_str}.mp4"

# MediaPipe landmark indices
# Based on BlazePose full body (33 landmarks)
UPPER_JOINTS = [
    0,   # NOSE
    1,2,3,4,5,6,   # EYES
    7,8,           # EARS
    9,10,          # MOUTH
    11,12,         # SHOULDERS
    13,14,         # ELBOWS
    15,16,         # WRISTS
    17,18,19,20,   # FINGERS (approx upper)
    21,22          # THUMBS
]

LOWER_JOINTS = [
    23,24,         # HIPS
    25,26,         # KNEES
    27,28,         # ANKLES
    29,30,         # HEELS
    31,32          # FOOT INDEX
]


# -----------------------------
# Helper: landmark id -> name
# -----------------------------
def get_landmark_name(idx: int) -> str:
    try:
        return mp.solutions.pose.PoseLandmark(idx).name
    except ValueError:
        return f"LANDMARK_{idx}"


# -----------------------------
# Helper: extract joints for one frame
# Returns list of dicts: [{...}, ...]
# -----------------------------
def extract_joint_data(landmarks, depth_image, depth_scale, frame_w, frame_h,
                       frame_id, timestamp_ms):
    joint_rows = []

    for idx, lm in enumerate(landmarks):
        # Normalized coordinates (0..1)
        x_norm = lm.x
        y_norm = lm.y
        z_norm = lm.z
        visibility = lm.visibility

        # Pixel coordinates
        x_px = int(x_norm * frame_w)
        y_px = int(y_norm * frame_h)

        # Depth
        if 0 <= x_px < frame_w and 0 <= y_px < frame_h:
            depth_raw = depth_image[y_px, x_px]
            depth_m = depth_raw * depth_scale
        else:
            depth_m = np.nan

        joint_rows.append({
            "timestamp_ms": timestamp_ms,
            "frame_id": frame_id,
            "joint_id": idx,
            "joint_name": get_landmark_name(idx),
            "x_norm": x_norm,
            "y_norm": y_norm,
            "z_norm": z_norm,
            "visibility": visibility,
            "x_px": x_px,
            "y_px": y_px,
            "depth_m": depth_m
        })

    return joint_rows


# -----------------------------
# Main runner
# -----------------------------
def run_body_tracker(recording=True, recording_duration_sec=None):
    """
    recording=True  -> saves CSV + video
    recording_duration_sec=None -> run until ESC
    """

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # -------------------------
    # RealSense setup
    # -------------------------
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, COLOR_WIDTH, COLOR_HEIGHT, rs.format.z16, FPS)

    profile = pipeline.start(config)

    # Align depth to color stream
    align = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"[INFO] Depth scale = {depth_scale:.5f} meters/unit")

    # -------------------------
    # Video writer (RGB)
    # -------------------------
    video_writer = None
    if recording:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            VIDEO_PATH,
            fourcc,
            FPS,
            (COLOR_WIDTH, COLOR_HEIGHT)
        )
        print(f"[INFO] Video recording to: {VIDEO_PATH}")

    # -------------------------
    # CSV setup
    # -------------------------
    upper_file = open(UPPER_CSV_PATH, mode="w", newline="", encoding="utf-8")
    lower_file = open(LOWER_CSV_PATH, mode="w", newline="", encoding="utf-8")

    upper_writer = csv.writer(upper_file)
    lower_writer = csv.writer(lower_file)

    header = [
        "timestamp_ms",
        "frame_id",
        "joint_id",
        "joint_name",
        "x_norm",
        "y_norm",
        "z_norm",
        "visibility",
        "x_px",
        "y_px",
        "depth_m"
    ]
    upper_writer.writerow(header)
    lower_writer.writerow(header)

    print(f"[INFO] Upper-body CSV: {UPPER_CSV_PATH}")
    print(f"[INFO] Lower-body CSV: {LOWER_CSV_PATH}")

    # -------------------------
    # MediaPipe Pose
    # -------------------------
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print("[INFO] Starting body tracking... Press ESC to exit.")

    frame_id = 0
    start_time = time.time()

    try:
        while True:
            # Stop after duration if set
            if recording_duration_sec is not None:
                if time.time() - start_time >= recording_duration_sec:
                    print("[INFO] Recording duration reached. Stopping.")
                    break

            # Read frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asarray(color_frame.get_data())
            depth_image = np.asarray(depth_frame.get_data())

            frame_h, frame_w, _ = color_image.shape

            # Timestamp + frame counter
            timestamp_ms = int(time.time() * 1000)
            frame_id += 1

            # MediaPipe expects RGB
            rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            output = color_image.copy()

            if results.pose_landmarks:
                # Draw pose on output frame
                mp_drawing.draw_landmarks(
                    output,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                # Extract joints
                joint_rows = extract_joint_data(
                    results.pose_landmarks.landmark,
                    depth_image,
                    depth_scale,
                    frame_w,
                    frame_h,
                    frame_id,
                    timestamp_ms
                )

                if recording:
                    # Split into upper / lower CSVs
                    for row in joint_rows:
                        if row["joint_id"] in UPPER_JOINTS:
                            upper_writer.writerow([
                                row["timestamp_ms"],
                                row["frame_id"],
                                row["joint_id"],
                                row["joint_name"],
                                row["x_norm"],
                                row["y_norm"],
                                row["z_norm"],
                                row["visibility"],
                                row["x_px"],
                                row["y_px"],
                                row["depth_m"]
                            ])
                        elif row["joint_id"] in LOWER_JOINTS:
                            lower_writer.writerow([
                                row["timestamp_ms"],
                                row["frame_id"],
                                row["joint_id"],
                                row["joint_name"],
                                row["x_norm"],
                                row["y_norm"],
                                row["z_norm"],
                                row["visibility"],
                                row["x_px"],
                                row["y_px"],
                                row["depth_m"]
                            ])

            # Write video frame
            if recording and video_writer is not None:
                video_writer.write(output)

            # Show preview
            cv2.imshow("Body Tracker (MediaPipe + RealSense)", output)

            # ESC to exit
            if cv2.waitKey(1) & 0xFF == 27:
                print("[INFO] ESC pressed. Exiting.")
                break

    finally:
        # Cleanup
        pipeline.stop()
        pose.close()
        cv2.destroyAllWindows()

        upper_file.close()
        lower_file.close()

        if video_writer is not None:
            video_writer.release()

        print("[INFO] Resources released. Bye.")


# -----------------------------
# Run as script
# -----------------------------
if __name__ == "__main__":
    # recording_duration_sec=None -> manual stop (ESC)
    # You can set e.g. 60 for 1 minute recording
    run_body_tracker(recording=True, recording_duration_sec=None)
