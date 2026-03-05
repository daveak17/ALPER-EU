import os
import time
import csv
import json
from dataclasses import dataclass

import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs


# ============================================================
# CONFIG / CONSTANTS
# ============================================================

WINDOW_NAME = "Thesis Tracker: Clean Upper Body + Hands"
TRACKBAR_DIST = "Dist tolerance (cm)"

# Distance slider bounds (in cm) - tolerance window around focused posture
MIN_DIST_CM = 5
MAX_DIST_CM = 40

# Auto-calibrated body turn threshold parameters
BASELINE_MARGIN_DEG = 20.0   # deviation from focused posture
MIN_TURN_ANGLE_DEG = 35.0    # absolute lower bound
MAX_TURN_ANGLE_DEG = 75.0    # absolute upper bound

# Hysteresis thresholds for body turn detection
TURN_ENTER_FRAMES = 6      # frames required to enter BODY TURN
TURN_EXIT_FRAMES = 2       # frames required to return to FACING FORWARD
MAX_TURN_COUNTER = 12      # safety cap

# Distance hysteresis thresholds
DIST_ENTER_MARGIN = 1.0    # must exceed tolerance × this to become UNFOCUSED
DIST_EXIT_MARGIN = 0.7     # must return inside tolerance × this to become ENGAGED


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class TrackingThresholds:
    """Runtime-adjustable thresholds (overridden by OpenCV trackbars)."""
    distance_m: float = 0.80   # meters
    turn_angle_deg: float = 35.0  # shoulder angle threshold in degrees


@dataclass
class ShoulderState:
    """Stores baseline shoulder width for rotation detection with calibration."""
    baseline_width: float | None = None
    calibrated: bool = False
    calibration_buffer: list[float] = None  # Collects shoulder widths during calibration
    turn_counter: int = 0  # Tracks sustained rotation for hysteresis
    facing_forward: bool | None = None  # Tracks previous facing state for asymmetric hysteresis
    distance_ok: bool | None = None  # Tracks distance engagement state for hysteresis
    
    def __post_init__(self):
        if self.calibration_buffer is None:
            self.calibration_buffer = []


@dataclass
class FocusBaseline:
    """
    Subject-specific baseline captured during initial focused posture.
    All engagement decisions are computed relative to this baseline.
    """
    nose_dist: float | None = None
    shoulder_width: float | None = None
    depth_diff: float | None = None
    height_diff: float | None = None
    shoulder_angle_deg: float | None = None  # Auto-calibrated baseline shoulder angle
    calibrated: bool = False
    buffer: list[dict] = None

    def __post_init__(self):
        if self.buffer is None:
            self.buffer = []


@dataclass
class EngagementState:
    """Represents user engagement based on multiple tracking signals."""
    distance_ok: bool | None
    facing_forward: bool | None
    hands_present: bool
    rotation_score: int | None
    shoulder_angle_deg: float | None = None
    deviation_cues: int = 0  # Number of active posture deviation indicators

    @property
    def engaged(self) -> bool | None:
        """
        Returns True if user is engaged (distance OK + facing forward).
        Returns None if insufficient data to determine engagement.
        """
        if self.distance_ok is None or self.facing_forward is None:
            return None
        return self.distance_ok and self.facing_forward


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_depth_distance(x_norm: float, y_norm: float, depth_frame) -> float:
    """
    Return depth distance (meters) for normalized [0–1] coords.
    If out of bounds or error, returns 0.0.
    """
    try:
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        x = int(x_norm * width)
        y = int(y_norm * height)
        if 0 <= x < width and 0 <= y < height:
            return depth_frame.get_distance(x, y)
        return 0.0
    except Exception:
        return 0.0


def smooth(prev, new, alpha: float = 0.4):
    """
    Simple exponential smoothing helper.

    Works with scalars or numpy arrays.
    If `prev` is None, returns `new`.
    """
    if prev is None:
        return new
    return prev * (1 - alpha) + new * alpha


def draw_status_line(img, text, y, color=(255, 255, 255), scale=0.6):
    """Helper function for clean overlay text layout."""
    cv2.putText(
        img,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
    )


class SmoothedPoint2D:
    """Helper class for smoothing 2D normalized coordinates."""

    def __init__(self, alpha: float = 0.4):
        self.x: float | None = None
        self.y: float | None = None
        self.alpha = alpha

    def update(self, x: float, y: float) -> tuple[float | None, float | None]:
        self.x = smooth(self.x, x, self.alpha)
        self.y = smooth(self.y, y, self.alpha)
        return self.x, self.y


class SmoothedScalar:
    """Helper class for smoothing a single scalar value (e.g., depth)."""

    def __init__(self, alpha: float = 0.4):
        self.v: float | None = None
        self.alpha = alpha

    def update(self, v: float) -> float | None:
        self.v = smooth(self.v, v, self.alpha)
        return self.v


# ============================================================
# SMOOTHING CLASSES (Replaces global state)
# ============================================================

class HandSmoother:
    """Manages smoothing for multiple hands (by hand ID)."""
    
    def __init__(self, alpha: float = 0.45):
        self.alpha = alpha
        self.hands: dict[str, list[tuple[float, float]]] = {}
    
    def smooth_landmarks(self, hand_id: str, hand_landmarks, alpha: float | None = None):
        """
        Smooths an entire hand's 21 landmarks.
        
        Args:
            hand_id: Unique identifier for the hand (e.g., "Left", "Right")
            hand_landmarks: MediaPipe hand landmarks
            alpha: Optional override for smoothing factor
            
        Returns:
            List of (x,y) tuples for the smoothed landmarks
        """
        alpha = alpha or self.alpha
        
        if hand_id not in self.hands:
            self.hands[hand_id] = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            return self.hands[hand_id]
        
        new_list: list[tuple[float, float]] = []
        for i, lm in enumerate(hand_landmarks.landmark):
            prev_x, prev_y = self.hands[hand_id][i]
            new_x = smooth(prev_x, lm.x, alpha)
            new_y = smooth(prev_y, lm.y, alpha)
            new_list.append((new_x, new_y))
        
        self.hands[hand_id] = new_list
        return new_list


class UpperBodySmoother:
    """Manages smoothing for upper body pose landmarks."""
    
    # MediaPipe Pose indices for upper body
    UPPER_BODY_IDS = [0, 11, 12, 13, 14, 15, 16, 23, 24]
    
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self.points = {i: {"x": None, "y": None} for i in self.UPPER_BODY_IDS}
    
    def smooth_landmarks(self, pose_landmarks):
        """
        Smooths upper body landmarks from pose results.
        
        Args:
            pose_landmarks: MediaPipe pose landmarks
            
        Returns:
            Dict mapping landmark index to {"x": float, "y": float}
        """
        lm = pose_landmarks.landmark
        for i in self.UPPER_BODY_IDS:
            raw = lm[i]
            prev = self.points[i]
            
            sx = smooth(prev["x"], raw.x, self.alpha)
            sy = smooth(prev["y"], raw.y, self.alpha)
            
            self.points[i]["x"] = sx
            self.points[i]["y"] = sy
        
        return self.points


def compute_rotation_score(
    ls_x: float,
    ls_y: float,
    rs_x: float,
    rs_y: float,
    depth_left: float | None,
    depth_right: float | None,
    baseline_width: float | None,
) -> tuple[int, float, float, float]:
    """
    Compute rotation score based on shoulder width, depth difference, and height difference.
    
    Algorithm: Uses weighted normalized components to determine body rotation.
    
    Weights:
        - Width ratio: 1.0 (primary indicator)
        - Depth difference: 1.0 (secondary indicator)
        - Height difference: 0.8 (tertiary indicator)

    Returns:
        (rotation_score, width_ratio, depth_diff, height_diff)
    """
    current_width = abs(ls_x - rs_x)
    width_ratio = current_width / baseline_width if baseline_width else 1.0
    depth_diff = abs(depth_left - depth_right) if depth_left and depth_right else 0.0
    height_diff = abs(ls_y - rs_y)

    # Normalized components (0-1 range where 1 = turned)
    width_c = max(0.0, 1.0 - width_ratio / 0.75) if baseline_width and width_ratio < 0.75 else 0.0
    depth_c = min(1.0, depth_diff / 0.10) if depth_diff > 0 else 0.0
    # Height difference is a weak cue; allow more tolerance
    height_c = min(1.0, height_diff / 0.06) if height_diff > 0 else 0.0
    
    # Count how many rotation cues are active
    active_cues = sum([
        width_c > 0.4,
        depth_c > 0.4,
        height_c > 0.4,
    ])
    
    # Weighted score (0-3 scale, matching original behavior)
    w_width = 1.0
    w_depth = 1.0
    # Height is a secondary cue; reduce its influence
    w_height = 0.3
    
    weighted_score = (w_width * width_c) + (w_depth * depth_c) + (w_height * height_c)
    
    # Require at least two active cues to consider a body turn
    if active_cues < 2:
        score = 0
    else:
        score = int(round(min(3.0, weighted_score)))

    return score, width_ratio, depth_diff, height_diff


# ============================================================
# LOGGING SYSTEM (CSV + VIDEO + CONFIG)
# ============================================================

class SessionLogger:
    """
    Handles CSV logging, video recording, and configuration snapshot for a session.
    
    Features:
        - Dual timestamping (host + RealSense)
        - Enhanced CSV schema with validity flags
        - FPS-decoupled recording
        - Configuration snapshot (config.json)
    """

    def __init__(self, session_name: str = "session", config_dict: dict = None):
        self.timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        self.folder = os.path.join("recordings", f"{session_name}_{self.timestamp_str}")
        os.makedirs(self.folder, exist_ok=True)

        self.csv_upper_path = os.path.join(self.folder, "upper_body.csv")
        self.csv_hands_path = os.path.join(self.folder, "hands_data.csv")
        self.video_path = os.path.join(self.folder, "output.avi")
        self.config_path = os.path.join(self.folder, "config.json")

        self.fourcc = cv2.VideoWriter_fourcc(*"XVID")
        # Default size matches color stream (1280x720)
        self.out = cv2.VideoWriter(self.video_path, self.fourcc, 30.0, (1280, 720))

        # Open CSV files once and reuse writers
        self.upper_file = open(self.csv_upper_path, "w", newline="")
        self.hands_file = open(self.csv_hands_path, "w", newline="")
        self.upper_writer = csv.writer(self.upper_file)
        self.hands_writer = csv.writer(self.hands_file)

        self._setup_csvs()
        
        # Save configuration snapshot
        if config_dict:
            self._save_config(config_dict)

        print(f"[INFO] Recording started: {self.folder}")

    def _setup_csvs(self):
        """Initialize CSV headers with enhanced schema."""
        # Upper Body CSV header (with new columns)
        self.upper_writer.writerow([
            "Timestamp_Host_ms", "Timestamp_RS_ms",
            "Nose_X", "Nose_Y", "Nose_Dist_M",
            "L_Shoulder_X", "L_Shoulder_Y", "L_Shoulder_Dist_M",
            "R_Shoulder_X", "R_Shoulder_Y", "R_Shoulder_Dist_M",
            "Pose_Valid", "Shoulders_Visible", "Nose_Visible",
            "Distance_OK", "Facing_Forward", "Engagement_State",
        ])

        # Hands CSV header (with new columns)
        self.hands_writer.writerow([
            "Timestamp_Host_ms", "Timestamp_RS_ms",
            "Fingers_Detected",
            "Hand_Count",
            "Hands_Present",
        ])
    
    def _save_config(self, config: dict):
        """Save session configuration snapshot to config.json."""
        try:
            with open(self.config_path, "w") as f:
                json.dump(config, f, indent=2)
            print(f"[INFO] Configuration saved: {self.config_path}")
        except Exception as e:
            print(f"[WARN] Failed to save config: {e}")

    def log_frame(
        self,
        frame,
        timestamp_host_ms: float,
        timestamp_rs_ms: float,
        pose_landmarks,
        hand_landmarks,
        depth_frame,
        engagement: 'EngagementState' = None,
    ):
        """
        Log one frame: video + upper-body CSV + hands CSV.
        
        Args:
            frame: BGR image
            timestamp_host_ms: Host system timestamp (ms)
            timestamp_rs_ms: RealSense timestamp (ms)
            pose_landmarks: MediaPipe pose landmarks (or None)
            hand_landmarks: MediaPipe hand landmarks (or None)
            depth_frame: RealSense depth frame
            engagement: EngagementState instance (or None)
        """
        # Save video frame
        self.out.write(frame)

        # ---- Upper Body ----
        pose_valid = pose_landmarks is not None
        shoulders_visible = False
        nose_visible = False
        
        if pose_valid:
            lm = pose_landmarks.landmark
            
            # Check visibility
            shoulders_visible = (lm[11].visibility > 0.5 and lm[12].visibility > 0.5)
            nose_visible = lm[0].visibility > 0.5

            def get_data(idx: int):
                x = lm[idx].x
                y = lm[idx].y
                dist = get_depth_distance(x, y, depth_frame)
                return [x, y, dist]

            # Determine engagement state string
            if engagement:
                if engagement.engaged is True:
                    eng_state = "ENGAGED"
                elif engagement.engaged is False:
                    eng_state = "UNFOCUSED"
                else:
                    eng_state = "UNKNOWN"
            else:
                eng_state = "UNKNOWN"

            row = (
                [timestamp_host_ms, timestamp_rs_ms]
                + get_data(0)   # Nose
                + get_data(11)  # L shoulder
                + get_data(12)  # R shoulder
                + [pose_valid, shoulders_visible, nose_visible]
                + [engagement.distance_ok if engagement else None,
                   engagement.facing_forward if engagement else None,
                   eng_state]
            )
            self.upper_writer.writerow(row)
        else:
            # Write row with nulls when pose is invalid
            row = [timestamp_host_ms, timestamp_rs_ms] + [None] * 15
            self.upper_writer.writerow(row)

        # ---- Hands ----
        fingers_detected = "No"
        hand_count = 0
        hands_present = False

        if hand_landmarks:
            fingers_detected = "Yes"
            hand_count = len(hand_landmarks)
            hands_present = hand_count > 0

        self.hands_writer.writerow([
            timestamp_host_ms, timestamp_rs_ms,
            fingers_detected, hand_count, hands_present
        ])

    def close(self):
        """Release resources."""
        self.out.release()
        self.upper_file.close()
        self.hands_file.close()
        print(f"[INFO] Recording saved in {self.folder}")

    # Allow usage with 'with' if desired.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# ============================================================
# POSE PROCESSING / DRAWING
# ============================================================

def process_pose(
    pose_results,
    depth_frame,
    thresholds: TrackingThresholds,
    shoulders: ShoulderState,
    focus_baseline: FocusBaseline,
    smoothed_pose_points: dict[str, SmoothedPoint2D],
    smoothed_depth: dict[str, SmoothedScalar],
    upper_body_smoother: UpperBodySmoother,
    image_shape: tuple[int, int, int],
    hands_present: bool = False,
) -> EngagementState:
    """
    Process pose landmarks to compute engagement status and update smoothing.
    
    Calibration Phase:
        Collects shoulder widths for ~30 frames, then computes median baseline.
        During calibration, rotation_score is None and facing_forward is None.
    
    Tracking Phase:
        Computes rotation score based on calibrated baseline.
    
    Returns:
        EngagementState instance with distance_ok, facing_forward, hands_present, rotation_score
    """
    h, w, _ = image_shape

    distance_ok = None
    facing_forward = None
    rotation_score = None
    shoulder_angle_deg = None
    body_turned = False  # Combined turn signal
    deviation_cues = 0  # Count of active posture deviation indicators

    if not pose_results.pose_landmarks:
        return EngagementState(
            distance_ok=distance_ok,
            facing_forward=facing_forward,
            hands_present=hands_present,
            rotation_score=rotation_score,
            shoulder_angle_deg=shoulder_angle_deg,
            deviation_cues=deviation_cues,
        )

    lm = pose_results.pose_landmarks.landmark
    visibility_threshold = 0.5

    # Shoulder visibility check
    if lm[11].visibility < visibility_threshold or lm[12].visibility < visibility_threshold:
        # Shoulders low confidence: return incomplete state
        return EngagementState(
            distance_ok=distance_ok,
            facing_forward=facing_forward,
            hands_present=hands_present,
            rotation_score=rotation_score,
            shoulder_angle_deg=shoulder_angle_deg,
            deviation_cues=deviation_cues,
        )

    # --- Raw landmarks ---
    raw_ls = lm[11]  # left shoulder
    raw_rs = lm[12]  # right shoulder
    raw_nose = lm[0]  # nose

    # --- Smooth shoulder & nose positions ---
    ls_x, ls_y = smoothed_pose_points["ls"].update(raw_ls.x, raw_ls.y)
    rs_x, rs_y = smoothed_pose_points["rs"].update(raw_rs.x, raw_rs.y)
    nose_x, nose_y = smoothed_pose_points["nose"].update(raw_nose.x, raw_nose.y)

    # Absolute shoulder orientation check (guardrail)
    # If shoulders are nearly vertical on screen, user is clearly turned
    dx = abs(ls_x - rs_x)
    dy = abs(ls_y - rs_y)
    
    # Ignore shoulder angle when shoulders visually overlap (prevents false positives)
    if dx < 0.05:
        shoulder_angle_deg = None
    else:
        shoulder_angle_deg = np.degrees(np.arctan2(dy, dx + 1e-6))

    # --- Smooth depths ---
    ls_depth_raw = get_depth_distance(raw_ls.x, raw_ls.y, depth_frame)
    rs_depth_raw = get_depth_distance(raw_rs.x, raw_rs.y, depth_frame)
    nose_depth_raw = get_depth_distance(raw_nose.x, raw_nose.y, depth_frame)

    depth_left = smoothed_depth["ls"].update(ls_depth_raw)
    depth_right = smoothed_depth["rs"].update(rs_depth_raw)
    nose_dist = smoothed_depth["nose"].update(nose_depth_raw)

    # --- Calibration Phase: Collect shoulder baseline + focused posture ---
    if ls_x is not None and rs_x is not None:
        current_width = abs(ls_x - rs_x)
        
        if not shoulders.calibrated:
            # Calibration: collect valid shoulder widths and focused posture features
            # Only accept frontal posture samples (shoulder_angle_deg < 20°)
            if current_width > 0 and len(shoulders.calibration_buffer) < 30:
                if shoulder_angle_deg is not None and shoulder_angle_deg < 20.0:
                    shoulders.calibration_buffer.append(current_width)
                
                    # Collect focused posture baseline
                    if (
                        depth_left is not None and depth_right is not None and
                        nose_dist is not None
                    ):
                        focus_baseline.buffer.append({
                            "nose_dist": nose_dist,
                            "shoulder_width": current_width,
                            "depth_diff": abs(depth_left - depth_right),
                            "height_diff": abs(ls_y - rs_y),
                            "shoulder_angle_deg": shoulder_angle_deg,
                        })
                    
            elif len(shoulders.calibration_buffer) >= 30:
                # Compute baseline as median
                shoulders.baseline_width = np.median(shoulders.calibration_buffer)
                shoulders.calibrated = True
                print(f"[INFO] Calibration complete! Baseline width: {shoulders.baseline_width:.4f}")
                
    # Compute focus baseline once shoulder calibration completes
    if shoulders.calibrated and not focus_baseline.calibrated and len(focus_baseline.buffer) > 0:
        focus_baseline.nose_dist = float(np.median([b["nose_dist"] for b in focus_baseline.buffer]))
        focus_baseline.shoulder_width = float(np.median([b["shoulder_width"] for b in focus_baseline.buffer]))
        focus_baseline.depth_diff = float(np.median([b["depth_diff"] for b in focus_baseline.buffer]))
        focus_baseline.height_diff = float(np.median([b["height_diff"] for b in focus_baseline.buffer]))
        focus_baseline.shoulder_angle_deg = float(np.median([b["shoulder_angle_deg"] for b in focus_baseline.buffer]))
        focus_baseline.calibrated = True
        
        # Auto-calibrate turn angle threshold from focused posture
        auto_turn_angle = focus_baseline.shoulder_angle_deg + BASELINE_MARGIN_DEG
        thresholds.turn_angle_deg = float(
            np.clip(auto_turn_angle, MIN_TURN_ANGLE_DEG, MAX_TURN_ANGLE_DEG)
        )
        
        print(f"[INFO] Focus baseline captured (subject-relative)")
        print(f"[INFO] Baseline shoulder angle: {focus_baseline.shoulder_angle_deg:.1f}°")
        print(
            f"[INFO] Auto turn threshold set to "
            f"{thresholds.turn_angle_deg:.1f}° "
            f"(baseline {focus_baseline.shoulder_angle_deg:.1f}° + {BASELINE_MARGIN_DEG:.0f}°)"
        )
    
    # --- Tracking Phase: Compute rotation score with relative deviation gate ---
    if shoulders.calibrated and focus_baseline.calibrated:
        # Only compute rotation score if deviation from focused posture is significant
        delta_width = abs(abs(ls_x - rs_x) - focus_baseline.shoulder_width) / focus_baseline.shoulder_width
        delta_depth = abs(abs(depth_left - depth_right) - focus_baseline.depth_diff) if depth_left and depth_right else 0.0
        delta_height = abs(abs(ls_y - rs_y) - focus_baseline.height_diff)

        # Count how many relative posture cues deviate from focused baseline
        deviation_cues = sum([
            delta_width > 0.10,
            delta_depth > 0.06,
            delta_height > 0.03,
        ])

        # Require at least TWO cues to consider a meaningful posture deviation
        significant_deviation = deviation_cues >= 2
        
        # Absolute turn check: shoulder angle threshold (only if valid angle)
        absolute_turn = (
            shoulder_angle_deg is not None and
            shoulder_angle_deg >= thresholds.turn_angle_deg
        )
        
        # Combined body turn signal (absolute OR relative)
        body_turned = absolute_turn or significant_deviation
        
        # rotation_score is a relative posture indicator, not a turn magnitude (debugging only)
        if body_turned:
            rotation_score, _, _, _ = compute_rotation_score(
                ls_x, ls_y, rs_x, rs_y, depth_left, depth_right, shoulders.baseline_width
            )
        else:
            rotation_score = 0

    # Distance check (relative to focused posture) with hysteresis
    if focus_baseline.calibrated and nose_dist is not None:
        distance_delta = abs(nose_dist - focus_baseline.nose_dist)
        tolerance = thresholds.distance_m
        
        # Hysteresis to prevent flickering at boundary
        if shoulders.distance_ok is None:
            # Initial state determination
            shoulders.distance_ok = distance_delta <= tolerance
        elif shoulders.distance_ok:
            # Currently engaged → allow some overshoot before marking unfocused
            if distance_delta > tolerance * DIST_ENTER_MARGIN:
                shoulders.distance_ok = False
        else:
            # Currently unfocused → require strong return before marking engaged
            if distance_delta < tolerance * DIST_EXIT_MARGIN:
                shoulders.distance_ok = True
        
        distance_ok = shoulders.distance_ok
    else:
        distance_ok = None
    
    # Asymmetric hysteresis for stable body-turn detection
    if shoulders.calibrated and rotation_score is not None:

        # Use body_turned signal (absolute angle OR relative deviation)
        # This replaces the old rotation_score >= threshold check
        if body_turned:
            # Sustained body turn
            shoulders.turn_counter = min(
                MAX_TURN_COUNTER,
                shoulders.turn_counter + 1
            )
        else:
            # Returned to frontal
            shoulders.turn_counter = max(
                0,
                shoulders.turn_counter - 2
            )

        # Facing-forward decision with hysteresis
        if shoulders.turn_counter >= TURN_ENTER_FRAMES:
            facing_forward = False
        elif shoulders.turn_counter <= TURN_EXIT_FRAMES:
            facing_forward = True
        else:
            # Preserve previous state when in hysteresis zone
            facing_forward = shoulders.facing_forward
        
        # Store state for next frame
        shoulders.facing_forward = facing_forward

    # Smooth & store upper-body (for drawing)
    upper_body_smoother.smooth_landmarks(pose_results.pose_landmarks)

    return EngagementState(
        distance_ok=distance_ok,
        facing_forward=facing_forward,
        hands_present=hands_present,
        rotation_score=rotation_score,
        shoulder_angle_deg=shoulder_angle_deg,
        deviation_cues=deviation_cues,
    )


def draw_pose_overlay(
    image_bgr,
    pose_results,
    smoothed_pose_points: dict[str, SmoothedPoint2D],
    upper_body_smoother: UpperBodySmoother,
    engagement: EngagementState,
    shoulders: ShoulderState,
    focus_baseline: FocusBaseline,
):
    """
    Draw upper-body lines, smoothed keypoints, and engagement status.
    
    Shows calibration status if not yet calibrated.
    """
    h, w, _ = image_bgr.shape

    if pose_results and pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark

        visibility_threshold = 0.5
        if lm[11].visibility < visibility_threshold or lm[12].visibility < visibility_threshold:
            cv2.putText(
                image_bgr,
                "LOW CONFIDENCE",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )
        else:
            # Draw shoulder line using smoothed coords if available
            ls = smoothed_pose_points["ls"]
            rs = smoothed_pose_points["rs"]

            if ls.x is not None and rs.x is not None:
                ls_px = (int(ls.x * w), int(ls.y * h))
                rs_px = (int(rs.x * w), int(rs.y * h))
                cv2.line(image_bgr, ls_px, rs_px, (255, 0, 0), 3)

            # Draw smoothed upper-body points
            smooth_upper = upper_body_smoother.points
            for i in upper_body_smoother.UPPER_BODY_IDS:
                sx = smooth_upper[i]["x"]
                sy = smooth_upper[i]["y"]
                if sx is None or sy is None:
                    continue
                x = int(np.clip(sx * w, 0, w - 1))
                y = int(np.clip(sy * h, 0, h - 1))
                cv2.circle(image_bgr, (x, y), 5, (0, 255, 255), -1)

            # --- Show calibration or engagement status ---
            if not shoulders.calibrated:
                # Calibration phase
                buffer_size = len(shoulders.calibration_buffer)
                cv2.putText(
                    image_bgr,
                    f"CALIBRATING SHOULDERS... ({buffer_size}/30)",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                )
            elif not focus_baseline.calibrated:
                # Computing focus baseline
                cv2.putText(
                    image_bgr,
                    "CAPTURING FOCUSED BASELINE...",
                    (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                )
            else:
                # Tracking phase - show baseline set and engagement
                cv2.putText(
                    image_bgr,
                    "BASELINE SET",
                    (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 200, 0),
                    2,
                )
                if engagement.engaged is True:
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
                    if engagement.distance_ok is False:
                        cv2.putText(
                            image_bgr,
                            "UNFOCUSED (DISTANCE)",
                            (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 255),
                            3,
                        )
                    elif engagement.facing_forward is False:
                        cv2.putText(
                            image_bgr,
                            "UNFOCUSED (BODY TURN)",
                            (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 255),
                            3,
                        )


# ============================================================
# HANDS PROCESSING / DRAWING
# ============================================================

def process_and_draw_hands(
    image_bgr,
    hands_results,
    mp_hands_mod,
    hand_smoother: HandSmoother,
):
    """
    Process hands and draw smoothed wrists + fingertips.

    Returns:
        (hand_count: int, fingers_detected: bool)
    """
    if not (hands_results and hands_results.multi_hand_landmarks):
        return 0, False

    h, w, _ = image_bgr.shape
    hand_count = 0

    fingertip_ids = [
        mp_hands_mod.HandLandmark.THUMB_TIP,
        mp_hands_mod.HandLandmark.INDEX_FINGER_TIP,
        mp_hands_mod.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands_mod.HandLandmark.RING_FINGER_TIP,
        mp_hands_mod.HandLandmark.PINKY_TIP,
    ]

    for hand_landmarks, hand_handedness in zip(
        hands_results.multi_hand_landmarks,
        hands_results.multi_handedness,
    ):
        label = hand_handedness.classification[0].label  # "Left"/"Right"
        score = hand_handedness.classification[0].score

        # Skip very low-confidence handedness results
        if score < 0.35:
            continue

        hand_id = label  # Stable key: 'Left' / 'Right'
        smoothed = hand_smoother.smooth_landmarks(hand_id, hand_landmarks, alpha=0.45)

        # Wrist is landmark 0
        wx_f, wy_f = smoothed[0]
        if wx_f is None or wy_f is None:
            continue

        wx = int(np.clip(wx_f * w, 0, w - 1))
        wy = int(np.clip(wy_f * h, 0, h - 1))

        cv2.circle(image_bgr, (wx, wy), 6, (0, 255, 255), -1)
        # Swap L/R since MediaPipe labels are mirrored (camera perspective)
        display_label = "R" if label == "Left" else "L"
        cv2.putText(
            image_bgr,
            display_label,
            (wx + 6, wy + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 0),
            2,
        )

        for tip in fingertip_ids:
            sx, sy = smoothed[tip]
            if sx is None or sy is None:
                continue
            tx = int(np.clip(sx * w, 0, w - 1))
            ty = int(np.clip(sy * h, 0, h - 1))
            cv2.line(image_bgr, (wx, wy), (tx, ty), (0, 255, 255), 2)
            cv2.circle(image_bgr, (tx, ty), 4, (0, 255, 255), -1)

        hand_count += 1

    return hand_count, hand_count > 0


# ============================================================
# MAIN TRACKING FUNCTION
# ============================================================

def start_thesis_tracking():
    """
    Main tracking loop with calibration, dual timestamping, and FPS-decoupled recording.
    """
    thresholds = TrackingThresholds()
    shoulders = ShoulderState()
    focus_baseline = FocusBaseline()

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    # Pose model
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    )

    # Hands model
    hands = mp_hands.Hands(
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        max_num_hands=2,
        model_complexity=1,
    )

    # RealSense setup
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    print("[INFO] System Ready — Press 'r' to Record, 'esc' to Quit.")
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    logger: SessionLogger | None = None
    is_recording = False

    # Smoothing state for key landmarks
    smoothed_pose_points = {
        "ls": SmoothedPoint2D(alpha=0.4),
        "rs": SmoothedPoint2D(alpha=0.4),
        "nose": SmoothedPoint2D(alpha=0.4),
    }
    smoothed_depth = {
        "ls": SmoothedScalar(alpha=0.4),
        "rs": SmoothedScalar(alpha=0.4),
        "nose": SmoothedScalar(alpha=0.4),
    }
    
    # Instantiate smoothers (replaces global state)
    hand_smoother = HandSmoother(alpha=0.45)
    upper_body_smoother = UpperBodySmoother(alpha=0.4)
    
    # Recording FPS decoupling
    target_record_fps = 30
    record_interval = 1.0 / target_record_fps
    last_record_time = 0.0

    # -----------------------
    # OpenCV "GUI" Trackbars
    # -----------------------
    cv2.namedWindow(WINDOW_NAME)

    def _noop(_):
        pass

    cv2.createTrackbar(TRACKBAR_DIST, WINDOW_NAME, 20, MAX_DIST_CM, _noop)
    # Turn angle threshold is now auto-calibrated, no slider needed

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames()
                aligned = align.process(frames)

                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                # Read thresholds from sliders
                dist_cm = cv2.getTrackbarPos(TRACKBAR_DIST, WINDOW_NAME)
                dist_cm = max(MIN_DIST_CM, min(dist_cm, MAX_DIST_CM))
                thresholds.distance_m = dist_cm / 100.0

                # Turn angle threshold is auto-calibrated from baseline (no slider)

                color_image = np.asanyarray(color_frame.get_data())
                
                # Dual timestamping (host + RealSense)
                timestamp_host_ms = time.perf_counter_ns() / 1e6
                timestamp_rs_ms = frames.get_timestamp()

                image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False

                pose_results = pose.process(image_rgb)
                hands_results = hands.process(image_rgb)

                image_rgb.flags.writeable = True
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                h, w, _ = image_bgr.shape

                # ---------------------------------------------------------
                # HANDS PROCESSING
                # ---------------------------------------------------------
                hand_count, fingers_detected = process_and_draw_hands(
                    image_bgr, hands_results, mp_hands, hand_smoother
                )
                hands_present = hand_count > 0
                
                # ---------------------------------------------------------
                # POSE PROCESSING
                # ---------------------------------------------------------
                engagement = process_pose(
                    pose_results,
                    depth_frame,
                    thresholds,
                    shoulders,
                    focus_baseline,
                    smoothed_pose_points,
                    smoothed_depth,
                    upper_body_smoother,
                    image_bgr.shape,
                    hands_present=hands_present,
                )
                
                # ---------------------------------------------------------
                # POSE DRAWING
                # ---------------------------------------------------------
                draw_pose_overlay(
                    image_bgr, pose_results, smoothed_pose_points,
                    upper_body_smoother, engagement, shoulders, focus_baseline
                )

                # ---------------------------------------------------------
                # OVERLAY LIVE STATUS (Clean Layout)
                # ---------------------------------------------------------
                status_y = 30
                line_h = 28

                draw_status_line(
                    image_bgr,
                    f"Dist_Thr: {thresholds.distance_m:.2f}m | Turn_Angle_Thr: {int(thresholds.turn_angle_deg)} deg (auto)",
                    status_y
                )

                status_y += line_h
                draw_status_line(
                    image_bgr,
                    f"Status: {'TRACKING' if shoulders.calibrated else 'CALIBRATING'}",
                    status_y,
                    (0, 255, 0) if shoulders.calibrated else (0, 255, 255)
                )

                status_y += line_h
                draw_status_line(
                    image_bgr,
                    f"Baseline: {'SET' if focus_baseline.calibrated else 'CAPTURING'}",
                    status_y,
                    (0, 200, 0) if focus_baseline.calibrated else (0, 255, 255)
                )

                status_y += line_h
                if engagement.distance_ok is not None:
                    draw_status_line(
                        image_bgr,
                        f"Distance_OK: {engagement.distance_ok}",
                        status_y,
                        (0, 255, 0) if engagement.distance_ok else (0, 0, 255)
                    )

                status_y += line_h
                if engagement.facing_forward is not None:
                    draw_status_line(
                        image_bgr,
                        f"Facing_Forward: {engagement.facing_forward}",
                        status_y,
                        (0, 255, 0) if engagement.facing_forward else (0, 0, 255)
                    )
                
                status_y += line_h
                if engagement.rotation_score is not None:
                    draw_status_line(
                        image_bgr,
                        f"Posture_Deviation_Score: {engagement.rotation_score}",
                        status_y
                    )
                    # Debug: show turn counter for tuning
                    status_y += line_h
                    draw_status_line(
                        image_bgr,
                        f"TurnCounter: {shoulders.turn_counter}",
                        status_y,
                        (200, 200, 200),
                        scale=0.5
                    )
                    # Debug: show deviation cues count
                    status_y += line_h
                    draw_status_line(
                        image_bgr,
                        f"Deviation_Cues: {engagement.deviation_cues}",
                        status_y,
                        (180, 180, 180),
                        scale=0.5
                    )
                
                status_y += line_h
                if engagement.shoulder_angle_deg is not None:
                    draw_status_line(
                        image_bgr,
                        f"Shoulder_Angle: {engagement.shoulder_angle_deg:.1f} deg",
                        status_y,
                        (180, 180, 180),
                        scale=0.5
                    )
                
                status_y += line_h
                if focus_baseline.shoulder_angle_deg is not None:
                    draw_status_line(
                        image_bgr,
                        f"Baseline_Angle: {focus_baseline.shoulder_angle_deg:.1f} deg",
                        status_y,
                        (180, 180, 180),
                        scale=0.5
                    )

                # ---------------------------------------------------------
                # RECORDING (FPS-Decoupled)
                # ---------------------------------------------------------
                if is_recording and logger is not None:
                    cv2.circle(image_bgr, (30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(
                        image_bgr,
                        "REC",
                        (50, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                    # FPS-decoupled logging: only log at target_record_fps
                    current_time = time.perf_counter()
                    if current_time - last_record_time >= record_interval:
                        logger.log_frame(
                            image_bgr,
                            timestamp_host_ms,
                            timestamp_rs_ms,
                            pose_results.pose_landmarks if pose_results else None,
                            hands_results.multi_hand_landmarks
                            if (hands_results and hands_results.multi_hand_landmarks)
                            else None,
                            depth_frame,
                            engagement=engagement,
                        )
                        last_record_time = current_time

                cv2.imshow(WINDOW_NAME, image_bgr)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord("r"):
                    if not is_recording:
                        # Create configuration snapshot
                        config_dict = {
                            "session_start": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "thresholds": {
                                "distance_m": thresholds.distance_m,
                                "turn_angle_deg": thresholds.turn_angle_deg,
                            },
                            "focus_baseline": {
                                "nose_dist": focus_baseline.nose_dist,
                                "shoulder_width": focus_baseline.shoulder_width,
                                "depth_diff": focus_baseline.depth_diff,
                                "height_diff": focus_baseline.height_diff,
                                "shoulder_angle_deg": focus_baseline.shoulder_angle_deg,
                            },
                            "camera": {
                                "depth_resolution": "848x480",
                                "color_resolution": "1280x720",
                                "fps": 30,
                            },
                            "models": {
                                "pose": "MediaPipe Pose (complexity=1)",
                                "hands": "MediaPipe Hands (complexity=1, max_hands=2)",
                            },
                            "recording": {
                                "target_fps": target_record_fps,
                            }
                        }
                        logger = SessionLogger(config_dict=config_dict)
                        is_recording = True
                        last_record_time = time.perf_counter()
                    else:
                        if logger:
                            logger.close()
                        logger = None
                        is_recording = False

            except Exception as e:
                # Don't kill the whole app on a single-frame error
                print(f"[WARN] Error in main loop: {e}")
                continue

    finally:
        if logger:
            logger.close()
        pipeline.stop()
        pose.close()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    start_thesis_tracking()