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
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"body_tracking_3D_{timestamp_str}.csv"
csv_file = open(csv_filename, 'w', newline='')
writer = csv.writer(csv_file)
writer.writerow(['timestamp_ms','person_id','joint_index','joint_name','x','y','depth_m','X','Y','Z','confidence','visible'])

start_time = time.time()
joint_names = {
    0:"Nose",1:"Left Eye",2:"Right Eye",3:"Left Ear",4:"Right Ear",
    5:"Left Shoulder",6:"Right Shoulder",7:"Left Elbow",8:"Right Elbow",
    9:"Left Wrist",10:"Right Wrist",11:"Left Hip",12:"Right Hip",
    13:"Left Knee",14:"Right Knee",15:"Left Ankle",16:"Right Ankle"
}
CONF_THRESHOLD = 0.5

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
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        left_depth, right_depth = np.nan, np.nan

        for result in results:
            keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints else None
            confs = result.keypoints.conf.cpu().numpy() if result.keypoints else None
            if keypoints is not None:
                annotated = result.plot()
                for pid, (person_kps, conf) in enumerate(zip(keypoints, confs)):
                    for j,(x,y) in enumerate(person_kps):
                        joint_name = joint_names[j]
                        c = float(conf[j])
                        if c < CONF_THRESHOLD:
                            writer.writerow([int((time.time()-start_time)*1000),pid,j,joint_name,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,c,"Not visible"])
                            continue
                        x_int, y_int = int(x), int(y)
                        if 0<=x_int<depth_image.shape[1] and 0<=y_int<depth_image.shape[0]:
                            r=3
                            x1,x2=max(0,x_int-r),min(depth_image.shape[1],x_int+r)
                            y1,y2=max(0,y_int-r),min(depth_image.shape[0],y_int+r)
                            region=depth_image[y1:y2,x1:x2]
                            depth=np.nanmedian(region)*depth_scale
                        else:
                            depth=np.nan
                        if not np.isnan(depth) and depth>0:
                            X,Y,Z=rs.rs2_deproject_pixel_to_point(depth_intrin,[x_int,y_int],depth)
                            vis="Visible"
                        else:
                            X,Y,Z=np.nan,np.nan,np.nan
                            vis="Not visible"
                        if joint_name=="Left Shoulder": left_depth=Z
                        if joint_name=="Right Shoulder": right_depth=Z
                        writer.writerow([int((time.time()-start_time)*1000),pid,j,joint_name,float(x),float(y),float(depth),X,Y,Z,c,vis])
                color_image=annotated

        status_text=""
        if not np.isnan(left_depth) and not np.isnan(right_depth):
            diff=abs(right_depth-left_depth)
            status_text=f"L:{left_depth:.2f}m  R:{right_depth:.2f}m  D:{diff*10:.1f}cm"
        elif not np.isnan(left_depth):
            status_text=f"Left:{left_depth:.2f}m  Right:Hidden"
        elif not np.isnan(right_depth):
            status_text=f"Left:Hidden  Right:{right_depth:.2f}m"
        else:
            status_text="No shoulders visible"

        cv2.putText(color_image,status_text,(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        cv2.imshow('YOLOv8 3D Body Tracking',color_image)
        if cv2.waitKey(1)&0xFF==27: break

finally:
    csv_file.close()
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"[INFO] Saved 3D body data to {csv_filename}")

print("[INFO] Loading and analyzing results...")
df=pd.read_csv(csv_filename)
df=df[df["visible"]=="Visible"]
left=df[df["joint_name"]=="Left Shoulder"]
right=df[df["joint_name"]=="Right Shoulder"]
mean_L,left_vis=left["Z"].mean(),len(left)>0
mean_R,right_vis=right["Z"].mean(),len(right)>0
if left_vis: print(f"Left Shoulder: {mean_L:.3f} m")
else: print("Left Shoulder: Not visible")
if right_vis: print(f"Right Shoulder: {mean_R:.3f} m")
else: print("Right Shoulder: Not visible")
if left_vis and right_vis: print(f"Difference (R-L): {mean_R-mean_L:.3f} m")

plt.figure(figsize=(6,5))
if left_vis: plt.scatter(left["X"],left["Z"],label="Left Shoulder",color='blue',alpha=0.7)
if right_vis: plt.scatter(right["X"],right["Z"],label="Right Shoulder",color='red',alpha=0.7)
plt.xlabel("X (m)")
plt.ylabel("Z (m)")
plt.title("3D Shoulder Positions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("[INFO] Analysis complete ✅")
