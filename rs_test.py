import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO
import csv
import time
from datetime import datetime

model = YOLO("yolov8n-pose.pt")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"[INFO] Depth scale: {depth_scale:.5f} meters per unit")

timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"body_tracking_3D_{timestamp_str}.csv"
csv_file = open(csv_filename, 'w', newline='')
writer = csv.writer(csv_file)
writer.writerow(['timestamp_ms', 'person_id', 'joint_index', 'x', 'y', 'depth_m', 'confidence'])

print("[INFO] Running 3D body tracking... Press ESC to stop.")
start_time = time.time()

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

        results = model(color_image, verbose=False)

        for result in results:
            keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints else None
            confs = result.keypoints.conf.cpu().numpy() if result.keypoints else None

            if keypoints is not None:
                annotated = result.plot()
                for pid, (person_kps, conf) in enumerate(zip(keypoints, confs)):
                    for j, (x, y) in enumerate(person_kps):
                        x_int, y_int = int(x), int(y)

                        if 0 <= x_int < depth_image.shape[1] and 0 <= y_int < depth_image.shape[0]:
                            depth_value = depth_image[y_int, x_int] * depth_scale  # in meters
                        else:
                            depth_value = np.nan

                        writer.writerow([
                            int((time.time() - start_time) * 1000),
                            pid, j, float(x), float(y), float(depth_value), float(conf[j])
                        ])

                cv2.imshow('YOLOv8 3D Body Tracking', annotated)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    csv_file.close()
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"[INFO] Saved 3D body data to {csv_filename}")
