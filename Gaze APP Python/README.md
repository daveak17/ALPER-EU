# Gaze APP Python

This folder contains `GazeAppAlpha.py`, the gaze tracking module for the ALPER-EU Multimodal Engagement Assessment System.

## Purpose

`GazeAppAlpha.py` implements the Tobii gaze capture pipeline. It is not run directly - it is launched and controlled by `MainApp.py` in the project root.

It connects to the TobiiStream local service over ZeroMQ (`tcp://127.0.0.1:5556`), receives gaze coordinates in real time, and writes timestamped gaze data to `tobii_gaze.csv` in the current session folder.

## Key behaviours

- Uses a two-loop background thread architecture that separates message reception from timed sample production
- Always uses the most recent gaze position, draining the ZeroMQ queue before each sample
- Detects Tobii signal loss via a 500 ms silence threshold
- Locks to 30 FPS when running in synchronized mode with the RealSense body tracker
- Supports standalone operation at up to 90 FPS

## Dependencies

All dependencies are shared with the main application. See `requirements.txt` in the project root.

## TobiiStream

Before running any gaze capture, TobiiStream must be running. The executable is located at:

`Required/TobiiStream/TobiiStream/TobiiStream.exe`

Double-click to launch it before starting `MainApp.py`.
