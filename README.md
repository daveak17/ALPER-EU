# ALPER-EU - Multimodal Engagement Analysis System

> **Thesis project** - Electronic Engineering MEng  
> Part of the **ALPER EU** (Agile Learning of Programming with Educational Robotics) research project.

A Python desktop application that monitors student attention and behaviour during educational robotics learning sessions. It captures synchronised eye-gaze and body-skeleton data from two sensors, records structured datasets, and computes engagement metrics for post-session analysis.

---

## Hardware Requirements

| Device | Purpose |
|-|-|
| **Tobii Eye Tracker 4C** | Gaze coordinates, eye position, attention direction |
| **Intel RealSense D455** | Depth video, body skeleton tracking, 3D joint coordinates |

---

## System Architecture

```
ALPER-EU/
- MainApp.py                        - Single entry point - unified GUI
- Gaze APP Python/
-   - GazeAppAlpha.py               - Tobii gaze tracking + analysis tab
- RealSenseBodyTracker/
-   - body-tracker.py               - RealSense body tracking engine
- analysis/
-   - engagement_analysis.py        - Post-session engagement analysis
-   - multi_session_comparison.py   - Multi-session comparison plots
-   - session_exporter.py           - Excel report generator
- recordings/
    - both_YYYYMMDD_HHMMSS/         - One folder per session
        - tobii_gaze.csv
        - upper_body.csv
        - hands_data.csv
        - realsense_color.avi
        - session_metadata.json
        - analysis/
            - session_summary.json
            - engagement_merged.csv
            - engagement_score.csv
            - disengagement_events.csv
            - engagement_timeline.png
            - gaze_heatmap.png
            - signal_summary.png
            - engagement_score.png
            - session_report.xlsx
```

---

## Installation

### 1. Python

Python 3.11.9 is required. Download from https://www.python.org/downloads/release/python-3119/

During installation, check **Add Python 3.11 to PATH** before clicking Install Now.

Note: If the machine already has other Python versions installed, use the Windows Python Launcher
(py) to invoke Python 3.11 explicitly throughout these steps. Verify with:

    py -3.11 --version

### 2. PowerShell execution policy (Windows)

Run once in PowerShell before any other steps:

    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

### 3. Install Git

Download and install Git from https://git-scm.com/download/win

Run the installer with default options. After installation, close and reopen PowerShell, then verify:

    git --version

### 4. Clone the repository

    git clone https://github.com/daveak17/ALPER-EU.git
    cd ALPER-EU

### 5. Python dependencies

    py -3.11 -m pip install -r requirements.txt

Note: pyrealsense2 bundles the full RealSense runtime. No separate Intel RealSense SDK installation is required.
Note: numpy and opencv are resolved automatically by mediapipe. Do not pin them separately.

### 6. Tobii Eye Tracking Core Software

Download version 2.16.8 from https://gaming.tobii.com/getstarted/

Select: Tobii -> Tobii Eye Tracker 4C -> Tobii Eye Tracking Core Software

The installer filename is Tobii_Eye_Tracking_Core_v2.16.8.214_x86.exe

This installer includes the Tobii Eye Tracking application, device drivers, and the Stream Engine service. No additional downloads are required.

Note: The Tobii download page states this software is not officially supported on Windows 11. It was successfully tested on Windows 11 Pro Version 23H2 and operates correctly despite this warning.

After installation, plug in the Tobii Eye Tracker 4C and complete the following mandatory steps:
1. Open the Tobii Eye Tracking application from the Windows system tray
2. Create a user profile
3. Complete the eye tracking calibration

TobiiStream will not produce valid gaze data until a calibrated user profile exists.

### 7. TobiiStream

TobiiStream is included in this repository at Required/TobiiStream/TobiiStream/. No installation is required. Before launching the application, double-click TobiiStream.exe to start the gaze data bridge. Keep the console window open during recording sessions.

### 8. Run the application

    py -3.11 MainApp.py

If Python 3.11 is the only version installed, python MainApp.py will also work.

## Running the Application

    py -3.11 MainApp.py

If Python 3.11 is the only version installed, python MainApp.py will also work.

This opens a single unified window with three tabs.

---

## GUI Tabs

### - Session Control

The primary recording interface.

- **Start BOTH** - starts eye and body sensors simultaneously. A session metadata dialog appears first (Student ID, Task Name, Facilitator, Notes). Use **Guest / Skip** for demos and tests.
- **Start Eye Only / Start Body Only** - run a single sensor independently.
- **Countdown** - configurable delay (0-60 s). Both sensors enter preview mode immediately; recording begins after the countdown so the student is settled.
- **Live Engagement Monitor** - four real-time indicators update every 200 ms:
  - - Gaze on screen
  - - Within distance
  - - Facing forward
  - - Session time elapsed
  - - Overall engagement verdict (disengaged only when all three signals fail simultaneously)
- **Verification panel** - shown after recording stops; confirms row counts, estimated FPS, and video file size for all four output files.

### - Live Tracking

The Tobii gaze tracking interface. Shows a live gaze dot on a miniature screen canvas, current gaze coordinates, on/off-screen status, cumulative on/off-screen time percentages, and a configurable session timer with optional countdown.

Sampling rate is selectable from 30 to 90 FPS. When running in synchronised mode with the body tracker, the system locks to **30 FPS** to match the RealSense frame rate.

### - Data Analysis

Post-session analysis tools. Select a session folder and click **Run Full Analysis** to generate all outputs. Individual plots can be displayed inline. An Excel report can be downloaded for teachers.

Multi-session comparison: add two or more analysed sessions to produce side-by-side score bars, signal breakdowns, and overlaid engagement score curves.

---

## Data Collection

### Synchronisation

Both sensors share a common `epoch_ms` timestamp (milliseconds since Unix epoch) computed from `time.time_ns()`. A shared `go_epoch_ms` gate is calculated before the countdown begins and passed to both recording engines. Neither sensor writes to its CSV until wall-clock time reaches this gate value, ensuring both datasets start at the same absolute moment.

Gaze and body data are merged during analysis using `pandas.merge_asof` with a 50 ms tolerance.

### Output files per session

| File | Sensor | Rate | Contents |
|-|-|-|-|
| `tobii_gaze.csv` | Tobii 4C | 30 FPS (sync) / up to 90 FPS (standalone) | `epoch_ms`, `x`, `y`, `on_screen` |
| `upper_body.csv` | RealSense | 30 FPS | `epoch_ms`, `device_ms`, Nose X/Y/dist, L/R Shoulder X/Y/dist, `Distance_OK`, `Facing_Forward`, `Body_Engaged` |
| `hands_data.csv` | RealSense | 30 FPS | `epoch_ms`, `device_ms`, `Fingers_Detected`, `Hand_Count` |
| `realsense_color.avi` | RealSense | 30 FPS | Annotated RGB video (1280 - 720, XVID) |
| `session_metadata.json` | GUI | - | Student ID, task name, facilitator, notes, timestamp |

---

## Engagement Model

### Three-signal rule

A student is considered **disengaged** only when all three conditions fail simultaneously:

- Gaze is **off-screen**
- Distance is **outside threshold** (default 0.70 m, adjustable via slider)
- Student is **not facing forward** (shoulder rotation score >= threshold)

In all other cases the student is considered **engaged**. This conservative rule avoids false disengagement when a student glances at a robot, types on a keyboard, or briefly looks away.

### Engagement score (0-100)

A weighted score is computed per frame and smoothed over a 3-second rolling window:

```
score = (gaze_on_screen * 0.50 + facing_forward * 0.30 + distance_ok * 0.20) * 100
```

| Zone | Score | Interpretation |
|-|-|-|
| High | >= 80 | Strongly engaged |
| Moderate | 50-79 | Partially engaged |
| Low | < 50 | At risk of disengagement |

### Body orientation detection

Facing-forward status is determined by a three-cue rotation score:

1. **Shoulder width ratio** - current width vs calibrated baseline (score +1 if < 75 %)
2. **Depth asymmetry** - difference in depth between left and right shoulder (score +1 if > 10 cm)
3. **Height asymmetry** - vertical position difference between shoulders (score +1 if > 3 %)

If the total score reaches the turn threshold (default 2), the student is classified as turned away.

## Analysis Pipeline

Run manually from the terminal or via the GUI **Run Full Analysis** button.

```bash
python analysis/engagement_analysis.py recordings/both_YYYYMMDD_HHMMSS
```

Outputs saved to `recordings/both_YYYYMMDD_HHMMSS/analysis/`:

| Output | Description |
|-|-|
| `session_summary.json` | All metrics in machine-readable format |
| `engagement_merged.csv` | Per-frame merged gaze + body data with engagement verdict |
| `engagement_score.csv` | Per-second score, smoothed score, and individual signal means |
| `disengagement_events.csv` | Each disengagement episode with start, end, and duration |
| `engagement_timeline.png` | Four-row colour band timeline (gaze / distance / facing / verdict) |
| `gaze_heatmap.png` | Gaussian-smoothed gaze density map on a 1920 - 1200 canvas |
| `signal_summary.png` | Bar chart of % time each condition was true |
| `engagement_score.png` | Smoothed score over time with zone shading and signal traces |

### Multi-session comparison

```bash
python analysis/multi_session_comparison.py recordings/both_* -output recordings/comparison
```

Produces `comparison_scores.png`, `comparison_signals.png`, `comparison_curves.png`, and `comparison_table.csv` in the output directory.

### Excel teacher report

```bash
python analysis/session_exporter.py recordings/both_YYYYMMDD_HHMMSS
```

Generates `session_report.xlsx` with five sheets: Session Report, Engagement Summary, Gaze Data, Body Data, Disengagement Events.

---

## Dependencies

| Package | Purpose |
|-|-|
| `pyzmq` | ZeroMQ socket for Tobii stream |
| `pyrealsense2` | Intel RealSense SDK Python wrapper |
| `mediapipe` | Body pose and hand landmark detection |
| `opencv-python` | Camera frames, video recording, preview window |
| `pandas` | CSV loading and timestamp merging |
| `numpy` | Numerical operations |
| `matplotlib` | All analysis plots |
| `scipy` | Gaussian filter for heatmaps |
| `Pillow` | Inline image display in the GUI |
| `openpyxl` | Excel report generation |
| `tkinter` | GUI (included with Python standard library) |

---

## Key Design Decisions

**Why ZeroMQ for Tobii?** The Tobii 4C publishes gaze data over a local TCP socket via the TobiiStream service. ZeroMQ provides a clean non-blocking SUB socket interface without polling, with a 500 ms silence threshold to detect device disconnection.

**Why MediaPipe Complexity 0?** The lightest pose model (`model_complexity=0`) is used to keep body tracking within the 33 ms frame budget at 30 FPS. Shoulder and nose visibility is sufficient for the distance and orientation signals needed by the engagement model.

**Why XVID for video?** XVID (AVI container) is widely supported on Windows without additional codec installation, produces manageable file sizes at 30 FPS / 1280 - 720, and does not require FFmpeg.

**Why conservative disengagement rule?** Educational robotics tasks naturally cause students to look away from the screen (at the robot, at a worksheet, at a peer). Triggering disengagement on a single off-screen gaze would produce many false positives. The all-three-false rule produces a meaningful signal that correlates with genuine task abandonment.
