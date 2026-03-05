import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import csv
import time
import os
import threading


def epoch_ms() -> int:
    """Return current time in milliseconds since epoch."""
    return time.time_ns() // 1_000_000


# ----------------------------
# Depth helper
# ----------------------------
def get_depth_distance(x_pixel, y_pixel, depth_frame):
    """Return depth in metres at normalised (x, y) coordinates.

    Returns 0 if the pixel is out of bounds or the query fails.
    """
    try:
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        x = int(x_pixel * width)
        y = int(y_pixel * height)
        if 0 <= x < width and 0 <= y < height:
            return depth_frame.get_distance(x, y)
        return 0
    except Exception as e:
        print(f"[WARN] get_depth_distance: {e}")
        return 0


def smooth(prev, new, alpha=0.4):
    """Exponential smoothing.  Returns `new` unchanged when `prev` is None."""
    if prev is None:
        return new
    return prev * (1 - alpha) + new * alpha


def smooth_hand_landmarks(smooth_hands: dict, hand_id: str,
                          hand_landmarks, alpha: float = 0.45):
    """Smooth all 21 landmarks for one hand.

    `smooth_hands` is the per-instance dict owned by BodyTrackerEngine so there
    is no module-level shared state.

    Returns a list of (x, y) tuples.
    """
    if hand_id not in smooth_hands:
        smooth_hands[hand_id] = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
        return smooth_hands[hand_id]

    new_list = []
    for i, lm in enumerate(hand_landmarks.landmark):
        prev_x, prev_y = smooth_hands[hand_id][i]
        new_list.append((smooth(prev_x, lm.x, alpha),
                         smooth(prev_y, lm.y, alpha)))

    smooth_hands[hand_id] = new_list
    return new_list


# Landmark indices that make up the upper body
UPPER_BODY_IDS = [0, 11, 12, 13, 14, 15, 16, 23, 24]


# ----------------------------
# Logging system (CSV + Video)
# ----------------------------
class SessionLogger:
    """Writes per-frame body tracking data to CSV files and a video file."""

    def __init__(self, folder: str):
        self.folder = folder
        os.makedirs(self.folder, exist_ok=True)

        self.csv_upper_path = os.path.join(folder, "upper_body.csv")
        self.csv_hands_path = os.path.join(folder, "hands_data.csv")
        self.video_path = os.path.join(folder, "realsense_color.avi")

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.out = cv2.VideoWriter(self.video_path, fourcc, 30.0, (1280, 720))

        # Lock to prevent race condition between log_frame() (preview thread)
        # and close() (main thread) accessing the VideoWriter simultaneously.
        self._write_lock = threading.Lock()

        self._setup_csvs()

        # Keep file handles open across frames — avoids repeated open/close overhead
        self.upper_f = open(self.csv_upper_path, "a", newline="")
        self.hands_f = open(self.csv_hands_path, "a", newline="")
        self.upper_writer = csv.writer(self.upper_f)
        self.hands_writer = csv.writer(self.hands_f)

        print(f"[INFO] SessionLogger started: {self.folder}")

    def _setup_csvs(self):
        with open(self.csv_upper_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch_ms", "device_ms",
                "Nose_X", "Nose_Y", "Nose_Dist_M",
                "L_Shoulder_X", "L_Shoulder_Y", "L_Shoulder_Dist_M",
                "R_Shoulder_X", "R_Shoulder_Y", "R_Shoulder_Dist_M",
                "Distance_OK", "Facing_Forward", "Body_Engaged",
            ])

        with open(self.csv_hands_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch_ms", "device_ms",
                "Fingers_Detected", "Hand_Count",
            ])

    def log_frame(self, image_bgr: np.ndarray, epoch_ms_val: int,
                  device_ms: float,
                  pose_landmarks, hand_landmarks,
                  depth_frame,
                  distance_ok=None, facing_forward=None, body_engaged=None):
        """Write one frame of tracking data.

        Args:
            image_bgr:      Annotated BGR numpy array (1280×720).
            epoch_ms_val:   Wall-clock epoch time in ms (primary timestamp).
            device_ms:      Hardware device timestamp from RealSense in ms.
            pose_landmarks: MediaPipe pose landmark object (or None).
            hand_landmarks: List of MediaPipe hand landmark objects (or None).
            depth_frame:    RealSense depth frame for distance queries.
            distance_ok:    Bool — student within distance threshold.
            facing_forward: Bool — student facing the screen.
            body_engaged:   Bool — distance_ok AND facing_forward simultaneously.
        """
        # ---- Video ----
        # Acquire lock before writing — prevents race with close() on main thread.
        with self._write_lock:
            if self.out.isOpened():
                self.out.write(image_bgr)

        # ---- Upper Body CSV ----
        if pose_landmarks:
            lm = pose_landmarks.landmark

            def get_data(idx):
                return [lm[idx].x, lm[idx].y,
                        get_depth_distance(lm[idx].x, lm[idx].y, depth_frame)]

            # FIX 3: device_ms is passed in explicitly — no more always-0 values.
            # Write booleans as explicit "True"/"False" strings for clarity in CSV
            row = ([epoch_ms_val, device_ms]
                   + get_data(0)    # Nose
                   + get_data(11)   # Left shoulder
                   + get_data(12)   # Right shoulder
                   + [str(distance_ok), str(facing_forward), str(body_engaged)])
            try:
                self.upper_writer.writerow(row)
                self.upper_f.flush()
            except Exception as e:
                print(f"[WARN] upper_body CSV write failed: {e}")

        # ---- Hands CSV ----
        fingers_detected = "Yes" if hand_landmarks else "No"
        hand_count = len(hand_landmarks) if hand_landmarks else 0
        try:
            self.hands_writer.writerow(
                [epoch_ms_val, device_ms, fingers_detected, hand_count])
            self.hands_f.flush()
        except Exception as e:
            print(f"[WARN] hands_data CSV write failed: {e}")

    def close(self):
        # Acquire lock before releasing — ensures no write() is in progress.
        with self._write_lock:
            try:
                self.out.release()
            except Exception:
                pass
        for fh in (getattr(self, 'upper_f', None), getattr(self, 'hands_f', None)):
            try:
                if fh:
                    fh.close()
            except Exception:
                pass
        print(f"[INFO] SessionLogger closed: {self.folder}")


# ----------------------------
# BodyTrackerEngine
# ----------------------------
class BodyTrackerEngine:
    """RealSense + MediaPipe tracking engine.

    Provides a clean start/stop API consumed by MainController.
    All previously module-level state (baseline, smoothing dicts, thresholds)
    is now per-instance so multiple sessions never contaminate each other.
    """

    # Default adjustable thresholds
    DEFAULT_DISTANCE_M = 0.70
    DEFAULT_TURN_SCORE = 2

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._preview_running = False
        self._is_recording = False
        self.logger: SessionLogger | None = None
        self.go_epoch_ms: int | None = None

        # FIX 4: Per-instance state — no module-level globals
        self._baseline_shoulder_width: float | None = None
        self._distance_threshold: float = self.DEFAULT_DISTANCE_M
        self._turned_threshold: int = self.DEFAULT_TURN_SCORE
        self._smooth_hands: dict = {}
        self._smooth_upper: dict = {i: {"x": None, "y": None}
                                    for i in UPPER_BODY_IDS}

        # Thread-safe live state — read by MainController to update engagement display.
        # Written by _preview_loop (background thread), read by main thread via polling.
        # Only simple bool/None values — no lock needed for CPython GIL-protected reads.
        self.live_state: dict = {
            "distance_ok":    None,   # bool | None
            "facing_forward": None,   # bool | None
            "body_engaged":   None,   # bool | None  (distance_ok AND facing_forward)
        }

    def _reset_session_state(self):
        """Reset per-session calibration, smoothing state and live engagement state."""
        self._baseline_shoulder_width = None
        self._smooth_hands = {}
        self._smooth_upper = {i: {"x": None, "y": None} for i in UPPER_BODY_IDS}
        self.live_state["distance_ok"]    = None
        self.live_state["facing_forward"] = None
        self.live_state["body_engaged"]   = None

    def start_preview(self, block: bool = False):
        if self._preview_running:
            return
        self._reset_session_state()
        self._stop_event.clear()
        if block:
            self._preview_loop()
        else:
            self._thread = threading.Thread(target=self._preview_loop, daemon=True)
            self._thread.start()

    def _preview_loop(self):  # noqa: C901  (long but intentionally self-contained)
        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands

        pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0,   # 0 = fastest; sufficient for shoulder/engagement detection
        )
        hands_model = mp_hands.Hands(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            max_num_hands=2,
            model_complexity=0,   # 0 = fastest; fingertip detection unaffected
        )

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        pipeline.start(config)
        align = rs.align(rs.stream.color)

        self._preview_running = True

        # Per-loop smoothing state (shoulders + nose only; others in self._smooth_upper)
        smooth_lm = {
            "ls":   {"x": None, "y": None, "d": None},
            "rs":   {"x": None, "y": None, "d": None},
            "nose": {"x": None, "y": None, "d": None},
        }

        # Hands throttle: run inference every HANDS_EVERY frames, hold result in between.
        # At 30 FPS this means hands updates ~10 times/sec — sufficient for engagement analysis.
        HANDS_EVERY = 3
        frame_counter = 0
        hands_results = None   # holds last valid result across skipped frames

        window_name = "Body Tracker: Upper Body + Hands"
        cv2.namedWindow(window_name)
        cv2.createTrackbar("Distance (cm)", window_name,
                           int(self._distance_threshold * 100), 150, lambda x: None)
        cv2.createTrackbar("Turn Score Thr", window_name,
                           int(self._turned_threshold), 3, lambda x: None)

        try:
            while not self._stop_event.is_set():
                frames = pipeline.wait_for_frames()
                aligned = align.process(frames)

                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                # Primary wall-clock timestamp (ms) — used for CSV and sync gate
                primary_epoch_ms = epoch_ms()

                # FIX 3: Capture device timestamp BEFORE converting to numpy
                device_ms_val = frames.get_timestamp()

                # Read threshold sliders
                dist_cm = max(40, cv2.getTrackbarPos("Distance (cm)", window_name))
                self._distance_threshold = dist_cm / 100.0

                turn_thr = max(1, cv2.getTrackbarPos("Turn Score Thr", window_name))
                self._turned_threshold = turn_thr

                color_image = np.asanyarray(color_frame.get_data())

                image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False

                # Pose runs every frame — drives engagement detection
                pose_results = pose.process(image_rgb)

                # Hands runs every HANDS_EVERY frames to stay within 33ms budget.
                # Last result is reused on skipped frames; smoothing covers the gap.
                frame_counter += 1
                if frame_counter % HANDS_EVERY == 0:
                    hands_results = hands_model.process(image_rgb)

                image_rgb.flags.writeable = True
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                h, w, _ = image_bgr.shape
                distance_ok = None
                facing_forward = None

                # ---- Pose ----
                if pose_results and pose_results.pose_landmarks:
                    lm = pose_results.pose_landmarks.landmark

                    if (lm[11].visibility < 0.5 or lm[12].visibility < 0.5):
                        cv2.putText(image_bgr, "LOW CONFIDENCE", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    else:
                        # Smooth shoulders and nose
                        for key, idx in (("ls", 11), ("rs", 12), ("nose", 0)):
                            smooth_lm[key]["x"] = smooth(smooth_lm[key]["x"], lm[idx].x)
                            smooth_lm[key]["y"] = smooth(smooth_lm[key]["y"], lm[idx].y)
                            raw_d = get_depth_distance(lm[idx].x, lm[idx].y, depth_frame)
                            smooth_lm[key]["d"] = smooth(smooth_lm[key]["d"], raw_d)

                        ls_x, ls_y = smooth_lm["ls"]["x"], smooth_lm["ls"]["y"]
                        rs_x, rs_y = smooth_lm["rs"]["x"], smooth_lm["rs"]["y"]
                        depth_left  = smooth_lm["ls"]["d"]
                        depth_right = smooth_lm["rs"]["d"]
                        nose_dist   = smooth_lm["nose"]["d"]

                        cv2.line(image_bgr,
                                 (int(ls_x * w), int(ls_y * h)),
                                 (int(rs_x * w), int(rs_y * h)),
                                 (255, 0, 0), 3)

                        # Rotation score (3 independent cues)
                        current_width = abs(ls_x - rs_x)
                        if self._baseline_shoulder_width is None and current_width > 0:
                            self._baseline_shoulder_width = current_width

                        width_ratio = (current_width / self._baseline_shoulder_width
                                       if self._baseline_shoulder_width else 1.0)

                        valid_left  = depth_left  is not None and depth_left  > 0
                        valid_right = depth_right is not None and depth_right > 0
                        depth_diff  = (abs(depth_left - depth_right)
                                       if (valid_left and valid_right) else 0.0)
                        height_diff = abs(smooth_lm["ls"]["y"] - smooth_lm["rs"]["y"])

                        rotation_score = 0
                        if self._baseline_shoulder_width and width_ratio < 0.75:
                            rotation_score += 1
                        if depth_diff > 0.10:
                            rotation_score += 1
                        if height_diff > 0.03:
                            rotation_score += 1

                        distance_ok    = (nose_dist is not None
                                          and 0 < nose_dist <= self._distance_threshold)
                        facing_forward = rotation_score < self._turned_threshold
                        body_engaged   = bool(distance_ok and facing_forward)

                        # Update thread-safe live state — read by MainController
                        self.live_state["distance_ok"]    = distance_ok
                        self.live_state["facing_forward"] = facing_forward
                        self.live_state["body_engaged"]   = body_engaged

                        # Body engagement label on preview window
                        if body_engaged:
                            cv2.putText(image_bgr, "BODY: ENGAGED", (50, 140),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        else:
                            reasons = []
                            if not distance_ok:
                                reasons.append("DISTANCE")
                            if not facing_forward:
                                reasons.append("TURN")
                            label = "BODY: UNFOCUSED (" + " & ".join(reasons) + ")"
                            cv2.putText(image_bgr, label, (50, 150),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

                        # Draw upper-body landmarks
                        for i in UPPER_BODY_IDS:
                            prev = self._smooth_upper[i]
                            sx = smooth(prev["x"], lm[i].x, 0.4)
                            sy = smooth(prev["y"], lm[i].y, 0.4)
                            self._smooth_upper[i] = {"x": sx, "y": sy}
                            cx = int(np.clip(sx * w, 0, w - 1))
                            cy = int(np.clip(sy * h, 0, h - 1))
                            cv2.circle(image_bgr, (cx, cy), 5, (0, 255, 255), -1)

                # ---- Hands ----
                if hands_results and hands_results.multi_hand_landmarks:
                    for hand_lm, hand_hw in zip(hands_results.multi_hand_landmarks,
                                                hands_results.multi_handedness):
                        label = hand_hw.classification[0].label
                        if hand_hw.classification[0].score < 0.35:
                            continue

                        smoothed = smooth_hand_landmarks(
                            self._smooth_hands, label, hand_lm, alpha=0.45)

                        wx_f, wy_f = smoothed[0]
                        if wx_f is None or wy_f is None:
                            continue
                        wx = int(np.clip(wx_f * w, 0, w - 1))
                        wy = int(np.clip(wy_f * h, 0, h - 1))
                        cv2.circle(image_bgr, (wx, wy), 6, (0, 255, 255), -1)
                        cv2.putText(image_bgr, label[0], (wx + 6, wy + 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

                        tip_ids = [
                            mp_hands.HandLandmark.THUMB_TIP,
                            mp_hands.HandLandmark.INDEX_FINGER_TIP,
                            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                            mp_hands.HandLandmark.RING_FINGER_TIP,
                            mp_hands.HandLandmark.PINKY_TIP,
                        ]
                        for tip in tip_ids:
                            sx, sy = smoothed[int(tip)]
                            if sx is None or sy is None:
                                continue
                            tx = int(np.clip(sx * w, 0, w - 1))
                            ty = int(np.clip(sy * h, 0, h - 1))
                            cv2.line(image_bgr, (wx, wy), (tx, ty), (0, 255, 255), 2)
                            cv2.circle(image_bgr, (tx, ty), 4, (0, 255, 255), -1)

                # ---- Overlay ----
                cv2.putText(image_bgr,
                            f"Dist_Thr: {self._distance_threshold:.2f}m  "
                            f"Turn_Thr: {self._turned_threshold}",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if distance_ok is not None:
                    c = (0, 255, 0) if distance_ok else (0, 0, 255)
                    cv2.putText(image_bgr, f"Distance_OK: {distance_ok}",
                                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)

                if facing_forward is not None:
                    c = (0, 255, 0) if facing_forward else (0, 0, 255)
                    cv2.putText(image_bgr, f"Facing_Forward: {facing_forward}",
                                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)

                # ---- Recording ----
                if (self._is_recording
                        and (self.go_epoch_ms is None
                             or primary_epoch_ms >= self.go_epoch_ms)):
                    cv2.circle(image_bgr, (30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(image_bgr, "REC", (50, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    try:
                        # FIX 2: Pass image_bgr (numpy array), not color_frame (RS object)
                        # FIX 3: Pass device_ms_val explicitly so CSV gets real hardware ts
                        self.logger.log_frame(
                            image_bgr,
                            primary_epoch_ms,
                            device_ms_val,
                            pose_results.pose_landmarks if pose_results else None,
                            (hands_results.multi_hand_landmarks
                             if hands_results and hands_results.multi_hand_landmarks
                             else None),
                            depth_frame,
                            distance_ok=distance_ok,
                            facing_forward=facing_forward,
                            body_engaged=body_engaged,
                        )
                    except Exception as e:
                        print(f"[WARN] logger.log_frame failed: {e}")

                cv2.imshow(window_name, image_bgr)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        finally:
            if self.logger:
                try:
                    self.logger.close()
                except Exception:
                    pass
            for resource, name in ((pipeline, "pipeline"), (pose, "pose"),
                                   (hands_model, "hands")):
                try:
                    resource.stop() if name == "pipeline" else resource.close()
                except Exception:
                    pass
            cv2.destroyAllWindows()
            self._preview_running = False

    def start_recording(self, output_folder: str, go_epoch_ms: int | None = None):
        """Begin writing data to output_folder.

        Preview must already be running.  go_epoch_ms is the shared epoch gate
        after which both sensors start writing (set by MainController before the
        countdown begins).
        """
        if not self._preview_running:
            raise RuntimeError("Call start_preview() before start_recording()")
        if self._is_recording:
            return
        os.makedirs(output_folder, exist_ok=True)
        self.logger = SessionLogger(folder=output_folder)
        self.go_epoch_ms = go_epoch_ms
        self._is_recording = True

    def stop_recording(self):
        if not self._is_recording:
            return
        try:
            if self.logger:
                self.logger.close()
        except Exception:
            pass
        self.logger = None
        self._is_recording = False
        self.go_epoch_ms = None

    def stop_preview(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def is_running_preview(self) -> bool:
        return self._preview_running

    def is_recording(self) -> bool:
        return self._is_recording


# ----------------------------
# Standalone entry point
# FIX 7: Removed 250-line duplicate. Engine handles both modes.
# ----------------------------
if __name__ == "__main__":
    engine = BodyTrackerEngine()
    print("[INFO] System Ready — press ESC in the tracker window to quit.")
    try:
        engine.start_preview(block=True)
    except KeyboardInterrupt:
        engine.stop_preview()