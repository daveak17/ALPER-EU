<<<<<<< HEAD
# 🧠 ALPER-EU

Research project integrating **eye-tracking** and **body-tracking** technologies to capture and analyze human gaze and movement data.  
Combines **Tobii** and **Intel RealSense** systems with **Python** for real-time data collection, synchronization, and visualization.

---

# 👁️ Eye Tracking Analysis Dashboard

A modern Python desktop application for real-time gaze tracking, visualization, and data analysis using **Tobii eye-tracking sensors**.

---

## 🧩 Overview

This app connects to a local Tobii Stream service and provides a live visualization of gaze points on screen, as well as multiple post-tracking analysis tools including heatmaps, space maps, attention metrics, and presence analysis.

It is built using **Python**, **Tkinter**, **ZeroMQ**, **Pandas**, **NumPy**, and **Matplotlib**.

---

## ⚙️ Features

### 🎯 Live Tracking
- Connects automatically to Tobii stream (`tcp://127.0.0.1:5556`)
- Displays gaze coordinates in real time
- Saves all captured samples to a CSV file
- Smooth visualization at ~30 FPS

### 🔥 Data Analysis Tools
- **Heatmap** — visualize gaze density across the screen  
- **Space Map** — scatter plot of all gaze points  
- **Attention Analysis** — detect reading, scanning, and idle intervals  
- **Presence Analysis** — calculate total on-screen presence percentage  

### 💾 Data Handling
- Automatic timestamped CSV export  
- Simple CSV import for post-session analysis  
- Handles missing or invalid data gracefully  

---

## 🖥️ Interface

The application contains two main tabs:

| Tab | Description |
|------|--------------|
| **Live Tracking** | Start/stop Tobii data capture and visualize gaze points in real time. |
| **Data Analysis** | Load existing CSV data and perform analytical visualizations (heatmap, attention timeline, etc.). |

---

## 📦 Installation

### 1️⃣ Requirements
Make sure you have:
- Python 3.10 or newer  
- Official **Tobii drivers**  
- `tobiistream` running locally (port `5556`)  

### 2️⃣ Install dependencies
```bash
pip install pyzmq pandas numpy matplotlib
```

### 3️⃣ Run the application
python GazeAnalysisApp.py

### 📁 Data Format
CSV columns used by the app:
timestamp, x, y
- timestamp — milliseconds since stream start
- x, y — gaze coordinates within screen bounds (0–1920 × 0–1080)
=======
Tkinter desktop app that subscribes to TobiiStream (ZeroMq), records gaze data, and provides heatmaps and attention/presence analysis.


1) Ensure TobiiStream on tcp://127.0.0.1:5556 is running
2) `python GazeAppAlpha.py`
3) Use the app to track, save CSV, and analyze
>>>>>>> bad7ed5 (Initial upload: GazeAppAlpha.py and repo scaffolding)
