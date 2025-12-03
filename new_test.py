import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO
import csv, time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt


model = YOLO("yolov8n-pose.pt")

pipeline = rs.pipeline()
config = rs.config()


config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 90)

profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"body_tracking_3D_{timestamp_str}.csv"

joint_names = {
    0: "Nose", 1: "Left Eye", 2: "Right Eye", 3: "Left Ear", 4: "Right Ear",
    5: "Left Shoulder", 6: "Right Shoulder", 7: "Left Elbow", 8: "Right Elbow",
    9: "Left Wrist", 10: "Right Wrist", 11: "Left Hip", 12: "Right Hip",
    13: "Left Knee", 14: "Right Knee", 15: "Left Ankle", 16: "Right Ankle"
}

# Define skeleton connections (from YOLO pose format)
skeleton_pairs = [
    (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 6), (5, 11), (6, 12),         # shoulders to hips
    (11, 13), (13, 15), (12, 14), (14, 16)  # legs
]

CONF_THRESHOLD = 0.5
start_time = time.time()
prev_time = start_time


with open(csv_filename, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow([
        'timestamp_ms', 'person_id', 'joint_index', 'joint_name',
        'x', 'y', 'depth_m', 'X', 'Y', 'Z', 'confidence', 'visible'
    ])

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

            results = model(color_image, verbose=False)
            left_depth, right_depth = np.nan, np.nan

            for result in results:
                keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints else None
                confs = result.keypoints.conf.cpu().numpy() if result.keypoints else None
                annotated = color_image.copy()

                if keypoints is not None:
                    for pid, (person_kps, conf) in enumerate(zip(keypoints, confs)):
                        for (a, b) in skeleton_pairs:
                            if conf[a] > CONF_THRESHOLD and conf[b] > CONF_THRESHOLD:
                                pt1 = tuple(map(int, person_kps[a]))
                                pt2 = tuple(map(int, person_kps[b]))
                                cv2.line(annotated, pt1, pt2, (0, 255, 255), 2)

                        for j, (x, y) in enumerate(person_kps):
                            joint_name = joint_names.get(j, f"Joint_{j}")
                            c = float(conf[j])

                            if c < CONF_THRESHOLD:
                                writer.writerow([
                                    int((time.time() - start_time) * 1000),
                                    pid, j, joint_name,
                                    np.nan, np.nan, np.nan,
                                    np.nan, np.nan, np.nan,
                                    c, "Not visible"
                                ])
                                continue

                            x_int, y_int = int(x), int(y)
                            if 0 <= x_int < depth_image.shape[1] and 0 <= y_int < depth_image.shape[0]:
                                r = 5 
                                x1, x2 = max(0, x_int - r), min(depth_image.shape[1], x_int + r)
                                y1, y2 = max(0, y_int - r), min(depth_image.shape[0], y_int + r)
                                region = depth_image[y1:y2, x1:x2]
                                valid_depth = region[region > 0]
                                depth = np.nanmedian(valid_depth) * depth_scale if valid_depth.size > 0 else np.nan
                            else:
                                depth = np.nan

                            if not np.isnan(depth) and depth > 0:
                                X, Y, Z = rs.rs2_deproject_pixel_to_point(depth_intrin, [x_int, y_int], depth)
                                vis = "Visible"
                                cv2.circle(annotated, (x_int, y_int), 5, (0, 255, 0), -1)
                            else:
                                X, Y, Z = np.nan, np.nan, np.nan
                                vis = "Not visible"

                            if joint_name == "Left Shoulder": left_depth = Z
                            if joint_name == "Right Shoulder": right_depth = Z

                            writer.writerow([
                                int((time.time() - start_time) * 1000),
                                pid, j, joint_name, float(x), float(y),
                                float(depth), X, Y, Z, c, vis
                            ])

                color_image = annotated

            
            if not np.isnan(left_depth) and not np.isnan(right_depth):
                diff = abs(right_depth - left_depth)
                status_text = f"L:{left_depth:.2f}m  R:{right_depth:.2f}m  Δ:{diff*100:.1f}cm"
            elif not np.isnan(left_depth):
                status_text = f"Left:{left_depth:.2f}m  Right:Hidden"
            elif not np.isnan(right_depth):
                status_text = f"Left:Hidden  Right:{right_depth:.2f}m"
            else:
                status_text = "No shoulders visible"

            cv2.putText(color_image, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(color_image, f"FPS: {fps:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('YOLOv8 3D Body Tracking', color_image)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

print(f"[INFO] Saved 3D body data to {csv_filename}")



print("[INFO] Loading and analyzing results...")
df = pd.read_csv(csv_filename)
df = df[df["visible"] == "Visible"]

left = df[df["joint_name"] == "Left Shoulder"]
right = df[df["joint_name"] == "Right Shoulder"]

mean_L, mean_R = left["Z"].mean(), right["Z"].mean()
left_vis, right_vis = len(left) > 0, len(right) > 0

if left_vis: print(f"Left Shoulder: {mean_L:.3f} m")
else: print("Left Shoulder: Not visible")
if right_vis: print(f"Right Shoulder: {mean_R:.3f} m")
else: print("Right Shoulder: Not visible")
if left_vis and right_vis:
    print(f"Difference (R-L): {mean_R - mean_L:.3f} m")

plt.figure(figsize=(6, 5))
if left_vis: plt.scatter(left["X"], left["Z"], label="Left Shoulder", color='blue', alpha=0.7)
if right_vis: plt.scatter(right["X"], right["Z"], label="Right Shoulder", color='red', alpha=0.7)
plt.xlabel("X (m)")
plt.ylabel("Z (m)")
plt.title("3D Shoulder Positions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("[INFO] Analysis complete ✅")
