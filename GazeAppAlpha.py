import zmq
import time
import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
import threading
import sys
import queue  # Vital for thread safety
import os     # Moved to top
import shutil # Moved to top

# Note: Ensure Tobii drivers are installed and Tobiistream is running.

class GazeVisualization(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(bg='black', width=320, height=180)
        self.dot = self.create_oval(0, 0, 10, 10, fill='red', outline='white')
        self.screen_width = 1920
        self.screen_height = 1200
        self.last_update = 0
        # Default visualization update interval (90 FPS)
        self.update_interval = 1 / 90

    def update_position(self, x, y):
        current_time = time.time()
        # Simple throttle for visual smoothness
        if current_time - self.last_update < self.update_interval:
            return

        # Calculate relative position on the canvas
        canvas_x = int((x / self.screen_width) * self.winfo_width())
        canvas_y = int((y / self.screen_height) * self.winfo_height())

        self.coords(self.dot,
                    canvas_x - 5, canvas_y - 5,
                    canvas_x + 5, canvas_y + 5)

        self.last_update = current_time


class ModernButton(ttk.Button):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)

    def on_enter(self, e):
        self['style'] = 'Accent.TButton'

    def on_leave(self, e):
        self['style'] = 'TButton'


class GazeAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gaze Analysis Application")
        self.root.geometry("1000x850")
        self.root.configure(bg='#f0f0f0')

        self.setup_styles()

        # Tracking State
        self.is_tracking = False
        self.tracking_thread = None
        self.sample_count = 0
        self.current_file = None

        # THREAD SAFETY: Queue to pass data from thread to GUI
        self.data_queue = queue.Queue()

        # Adjustable software sampling rate (downsampling)
        self.target_fps = 90           # default sampling FPS
        self._last_save_time = 0.0     # last time we saved/processed a sample (perf_counter)
        self._flush_counter = 0        # counter for CSV flush operations

        # Heatmap smoothing parameter (sigma for gaussian_filter)
        self.heatmap_sigma_var = tk.DoubleVar(value=2.0)

        # Heatmap color scale preference (logarithmic by default)
        self.heatmap_use_log_var = tk.BooleanVar(value=True)

        # Timer state
        self.timer_running = False
        self.timer_start_time = None
        self.elapsed_time = 0.0
        self.timer_after_id = None
        self.auto_start_timer = tk.BooleanVar(value=True)
        
        # Countdown state
        self.use_countdown = tk.BooleanVar(value=False)
        self.countdown_stop_tracking = tk.BooleanVar(value=True)
        self.countdown_hours_var = tk.IntVar(value=0)
        self.countdown_minutes_var = tk.IntVar(value=1)
        self.countdown_seconds_var = tk.IntVar(value=0)
        self.countdown_end_time = None

        # Gaze Duration Tracking State
        self.total_on_screen_time = 0  # milliseconds
        self.total_off_screen_time = 0  # milliseconds
        self.last_timestamp = None  # milliseconds
        self.last_gaze_state = None  # True (on-screen), False (off-screen), or None

        # Network
        self.context = zmq.Context()
        self.socket = None

        self.setup_gui()
        # Start the GUI → queue polling loop
        # Create a persistent bound-method to avoid ephemeral Tcl command names
        self._process_queue_callback = self.process_queue
        self._process_queue_after_id = None
        try:
            if self.root.winfo_exists():
                # schedule the first call; process_queue will reschedule itself
                self._process_queue_after_id = self.root.after(15, self._process_queue_callback)
        except Exception:
            # If scheduling fails, fall back to direct call once
            try:
                self.process_queue()
            except Exception:
                pass

    def setup_styles(self):
        style = ttk.Style()
        style.configure('TNotebook', background='#f0f0f0')
        style.configure('TNotebook.Tab', padding=[20, 10], font=('Helvetica', 10))
        style.configure('TLabelframe', background='#f0f0f0')
        style.configure('TLabelframe.Label', font=('Helvetica', 11, 'bold'))
        style.configure('TButton', padding=[15, 8], font=('Helvetica', 10))
        style.configure('Accent.TButton', background='#007bff')
        style.configure('Status.TLabel', font=('Helvetica', 12))
        style.configure('Title.TLabel', font=('Helvetica', 14, 'bold'))
        style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'))

    def setup_gui(self):
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill='x', padx=20, pady=10)
        ttk.Label(title_frame, text="Eye Tracking Analysis Dashboard",
                  style='Title.TLabel').pack()

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=20, pady=10)

        self.tracking_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.tracking_frame, text="Live Tracking")

        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="Data Analysis")

        self.setup_tracking_tab()
        self.setup_analysis_tab()

    def setup_tracking_tab(self):
        control_frame = ttk.LabelFrame(self.tracking_frame, text="Tracking Control")
        control_frame.pack(fill='x', padx=20, pady=10)

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(padx=20, pady=15)

        self.start_button = ModernButton(button_frame, text="▶ Start Tracking",
                                         command=self.start_tracking)
        self.start_button.pack(side='left', padx=10)

        self.stop_button = ModernButton(button_frame, text="⏹ Stop Tracking",
                                        command=self.stop_tracking, state='disabled')
        self.stop_button.pack(side='left', padx=10)

        # ---------- Sampling Rate Dropdown (30–90 FPS) ----------
        fps_frame = ttk.Frame(control_frame)
        fps_frame.pack(padx=20, pady=(0, 10), anchor='w')

        ttk.Label(fps_frame, text="Sampling Rate:").pack(side='left')
        self.fps_combo = ttk.Combobox(
            fps_frame,
            values=['30', '45', '60', '75', '90'],
            width=5,
            state='readonly'
        )
        self.fps_combo.set(str(self.target_fps))
        self.fps_combo.pack(side='left', padx=(8, 0))
        self.fps_combo.bind('<<ComboboxSelected>>', self.on_fps_change)
        # -------------------------------------------------------

        viz_frame = ttk.LabelFrame(self.tracking_frame, text="Live Gaze Visualization")
        viz_frame.pack(fill='x', padx=20, pady=10)

        self.gaze_viz = GazeVisualization(viz_frame)
        self.gaze_viz.pack(padx=20, pady=15)

        status_frame = ttk.LabelFrame(self.tracking_frame, text="Live Status")
        status_frame.pack(fill='x', padx=20, pady=10)

        status_content = ttk.Frame(status_frame)
        status_content.pack(padx=20, pady=15)

        ttk.Label(status_content, text="Current Status:",
                  style='Header.TLabel').pack()
        self.status_label = ttk.Label(status_content, text="Not tracking",
                                      style='Status.TLabel')
        self.status_label.pack(pady=5)

        ttk.Label(status_content, text="Gaze Position:",
                  style='Header.TLabel').pack(pady=(10, 0))
        self.coordinates_label = ttk.Label(status_content, text="Coordinates: --",
                                           style='Status.TLabel')
        self.coordinates_label.pack(pady=5)

        # LIVE ON-SCREEN / OFF-SCREEN INDICATOR
        self.screen_status_label = ttk.Label(
            status_content,
            text="STATUS: NOT TRACKING",
            font=("Helvetica", 14, "bold"),
            foreground="gray"
        )
        self.screen_status_label.pack(pady=(10, 0))

        # Gaze Duration Tracking UI
        gaze_duration_frame = ttk.LabelFrame(self.tracking_frame, text="Gaze Duration Analysis")
        gaze_duration_frame.pack(fill='x', padx=20, pady=10)

        duration_content = ttk.Frame(gaze_duration_frame)
        duration_content.pack(padx=20, pady=15)

        # On-screen time label
        self.gaze_on_screen_label = ttk.Label(
            duration_content,
            text="On-screen time: 00:00.0",
            font=("Helvetica", 11)
        )
        self.gaze_on_screen_label.pack(pady=5)

        # Off-screen time label
        self.gaze_off_screen_label = ttk.Label(
            duration_content,
            text="Off-screen time: 00:00.0",
            font=("Helvetica", 11)
        )
        self.gaze_off_screen_label.pack(pady=5)

        # Percentage label
        self.gaze_percentage_label = ttk.Label(
            duration_content,
            text="On-screen 0.0% | Off-screen 0.0%",
            font=("Helvetica", 11, "bold"),
            foreground="#007bff"
        )
        self.gaze_percentage_label.pack(pady=5)

        # Timer controls
        timer_frame = ttk.LabelFrame(self.tracking_frame, text="Timer")
        timer_frame.pack(fill='x', padx=20, pady=10)

        timer_content = ttk.Frame(timer_frame)
        timer_content.pack(padx=20, pady=10)

        self.timer_label = ttk.Label(timer_content, text="Timer: 00:00:00", style='Status.TLabel', font=(None, 12, 'bold'))
        self.timer_label.pack(side='left', padx=(0, 20))

        ModernButton(timer_content, text="▶ Start Timer", command=self.start_timer).pack(side='left', padx=5)
        ModernButton(timer_content, text="⏸ Stop Timer", command=self.stop_timer).pack(side='left', padx=5)
        ModernButton(timer_content, text="↺ Reset", command=self.reset_timer).pack(side='left', padx=5)

        ttk.Checkbutton(timer_content, text='Auto-start with tracking', variable=self.auto_start_timer).pack(side='left', padx=20)

        # Countdown configuration
        countdown_frame = ttk.Frame(timer_frame)
        countdown_frame.pack(fill='x', padx=20, pady=(8, 0))

        ttk.Checkbutton(countdown_frame, text='Use countdown', variable=self.use_countdown).pack(side='left')
        ttk.Label(countdown_frame, text='Duration:').pack(side='left', padx=(12, 4))
        ttk.Spinbox(countdown_frame, from_=0, to=23, width=3, textvariable=self.countdown_hours_var).pack(side='left')
        ttk.Label(countdown_frame, text='h').pack(side='left')
        ttk.Spinbox(countdown_frame, from_=0, to=59, width=3, textvariable=self.countdown_minutes_var).pack(side='left', padx=(6,0))
        ttk.Label(countdown_frame, text='m').pack(side='left')
        ttk.Spinbox(countdown_frame, from_=0, to=59, width=3, textvariable=self.countdown_seconds_var).pack(side='left', padx=(6,0))
        ttk.Label(countdown_frame, text='s').pack(side='left', padx=(4,12))

        ttk.Checkbutton(countdown_frame, text='Stop tracking when finished', variable=self.countdown_stop_tracking).pack(side='left')

        # Countdown remaining label
        self.countdown_label = ttk.Label(timer_content, text="Countdown: --:--:--", style='Status.TLabel')
        self.countdown_label.pack(side='left', padx=(20,0))

    def on_fps_change(self, event=None):
        """Handle changes to the sampling FPS from the dropdown."""
        try:
            fps = int(self.fps_combo.get())
            self.target_fps = fps

            # Also adjust visualization throttle to roughly match
            try:
                if hasattr(self, 'gaze_viz') and fps > 0:
                    self.gaze_viz.update_interval = 1.0 / float(fps)
            except Exception:
                pass

            if self.is_tracking:
                self.status_label.config(text=f"✅ Tracking active - Sampling: {fps} FPS")
        except Exception:
            # If anything weird happens, just ignore and keep previous FPS
            pass

    def setup_analysis_tab(self):
        file_frame = ttk.LabelFrame(self.analysis_frame, text="Data Source")
        file_frame.pack(fill='x', padx=20, pady=10)

        file_content = ttk.Frame(file_frame)
        file_content.pack(padx=20, pady=15)

        ModernButton(file_content, text="📂 Load CSV File",
                     command=self.load_csv).pack(side='left', padx=10)
        self.file_label = ttk.Label(file_content, text="No file loaded",
                                    style='Status.TLabel')
        self.file_label.pack(side='left', padx=10)

        analysis_frame = ttk.LabelFrame(self.analysis_frame, text="Analysis Tools")
        analysis_frame.pack(fill='x', padx=20, pady=10)

        analysis_content = ttk.Frame(analysis_frame)
        analysis_content.pack(padx=20, pady=15)

        ModernButton(analysis_content, text="🔥 Generate Heatmap",
                     command=self.generate_heatmap).pack(side='left', padx=10)
        ModernButton(analysis_content, text="👁 Attention Analysis",
                     command=self.analyze_attention).pack(side='left', padx=10)
        ModernButton(analysis_content, text="📊 Presence Analysis",
                     command=self.analyze_presence).pack(side='left', padx=10)
        ModernButton(analysis_content, text="📍 Space Map",
                     command=self.generate_SpaceMap).pack(side='left', padx=10)

        # Heatmap smoothing controls
        heatmap_settings = ttk.Frame(analysis_frame)
        heatmap_settings.pack(fill='x', padx=20, pady=(6, 12))

        ttk.Label(heatmap_settings, text="Heatmap sigma:").pack(side='left')
        try:
            sigma_spin = ttk.Spinbox(
                heatmap_settings,
                from_=0.0,
                to=10.0,
                increment=0.5,
                textvariable=self.heatmap_sigma_var,
                width=5
            )
        except Exception:
            # Fallback to tk.Spinbox if ttk.Spinbox not available
            sigma_spin = tk.Spinbox(
                heatmap_settings,
                from_=0.0,
                to=10.0,
                increment=0.5,
                textvariable=self.heatmap_sigma_var,
                width=5
            )
        sigma_spin.pack(side='left', padx=(6, 10))

        # Wire auto-regeneration callbacks for heatmap controls
        def on_sigma_change(*args):
            if hasattr(self, 'data'):
                self.generate_heatmap()
        
        self.heatmap_sigma_var.trace('w', on_sigma_change)

        self.results_frame = ttk.LabelFrame(self.analysis_frame, text="Analysis Results")
        self.results_frame.pack(fill='both', expand=True, padx=20, pady=10)

        welcome_text = """
        Eye Tracking Analysis Dashboard!
        
        To begin:
        1. Click "Load CSV File"
        2. Choose a gaze CSV
        3. View analysis results here
        """
        ttk.Label(self.results_frame, text=welcome_text,
                  style='Status.TLabel').pack(padx=20, pady=20)

    def start_tracking(self):
        if not self.is_tracking:
            try:
                # Clear queue before starting
                with self.data_queue.mutex:
                    self.data_queue.queue.clear()

                self.socket = self.context.socket(zmq.SUB)
                self.socket.connect("tcp://127.0.0.1:5556")
                self.socket.setsockopt_string(zmq.SUBSCRIBE, "TobiiStream")

                self.temp_file = f"temp_gaze_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                self.csv_file = open(self.temp_file, 'w', newline='')
                self.writer = csv.writer(self.csv_file)
                self.writer.writerow(['timestamp', 'x', 'y', 'on_screen'])

                self.is_tracking = True
                self.sample_count = 0
                # reset downsample timer so first sample is saved immediately
                self._last_save_time = 0.0

                # Reset gaze duration timers for new session
                self.reset_gaze_timers()

                self.tracking_thread = threading.Thread(target=self.tracking_loop)
                self.tracking_thread.daemon = True
                self.tracking_thread.start()

                # Auto-start timer
                try:
                    if self.auto_start_timer.get():
                        self.start_timer()
                except Exception:
                    pass

                self.start_button.config(state='disabled')
                self.stop_button.config(state='normal')
                self.status_label.config(text="✅ Tracking active")
                self.screen_status_label.config(text="STATUS: WAITING DATA", foreground="orange")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to start tracking: {str(e)}")
                self.stop_tracking()

    def stop_tracking(self):
        if self.is_tracking:
            self.is_tracking = False
            if self.tracking_thread:
                self.tracking_thread.join(timeout=1.0)
            if self.socket:
                self.socket.close()
            if hasattr(self, 'csv_file'):
                self.csv_file.close()

            try:
                if self.auto_start_timer.get():
                    self.stop_timer()
            except Exception:
                pass

            default_name = f"gazetrack_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            save_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                initialfile=default_name,
                filetypes=[("CSV files", "*.csv")],
                title="Save Gaze Tracking Data"
            )

            if save_path:
                try:
                    shutil.copy2(self.temp_file, save_path)
                    os.remove(self.temp_file)
                    
                    # Export session summary after successful CSV save
                    summary_path = self._export_session_summary(save_path, self.sample_count)
                    
                    if summary_path:
                        messagebox.showinfo("Success", 
                            f"Data saved to {save_path}\n\nSession summary: {os.path.basename(summary_path)}")
                    else:
                        messagebox.showinfo("Success", f"Data saved to {save_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save file: {str(e)}")
            else:
                # Clean up temp file if user cancelled save
                try:
                    os.remove(self.temp_file)
                except:
                    pass

            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')
            self.status_label.config(text="📁 Tracking stopped")
            self.coordinates_label.config(text="Coordinates: --")
            self.screen_status_label.config(text="STATUS: NOT TRACKING", foreground="gray")
            self.gaze_viz.update_position(0, 0)

    def shutdown(self):
        """Cleanly stop background work so the process can exit and be restarted.

        This stops tracking, joins threads (best-effort), closes sockets and the
        ZMQ context, cancels pending after callbacks, stops timers, and closes
        any open files.
        """
        # Stop tracking thread
        try:
            if self.is_tracking:
                self.is_tracking = False
        except Exception:
            pass

        try:
            if self.tracking_thread and self.tracking_thread.is_alive():
                self.tracking_thread.join(timeout=1.0)
        except Exception:
            pass

        # Close socket and terminate ZMQ context
        try:
            if self.socket:
                try:
                    self.socket.close()
                except Exception:
                    pass
                self.socket = None
        except Exception:
            pass

        try:
            if self.context:
                try:
                    self.context.term()
                except Exception:
                    pass
                self.context = None
        except Exception:
            pass

        # Close CSV file if open
        try:
            if hasattr(self, 'csv_file'):
                try:
                    self.csv_file.close()
                except Exception:
                    pass
                delattr(self, 'csv_file')
        except Exception:
            pass

        # Stop timer and cancel its after callback
        try:
            self.stop_timer()
        except Exception:
            pass

        # Cancel process_queue after callback if present
        try:
            if hasattr(self, '_process_queue_after_id') and self._process_queue_after_id is not None:
                try:
                    if self.root.winfo_exists():
                        self.root.after_cancel(self._process_queue_after_id)
                except Exception:
                    pass
                self._process_queue_after_id = None
        except Exception:
            pass

    def tracking_loop(self):
        # Runs in BACKGROUND THREAD
        # Produces samples at target_fps regardless of whether Tobii data arrives
        
        # High-resolution timer for precise sampling
        next_sample_time = time.perf_counter()
        sample_interval = 1.0 / float(getattr(self, 'target_fps', 90))
        
        # Last received gaze data (fallback if Tobii stops sending)
        last_raw_x = -1
        last_raw_y = -1
        last_validity = 1  # invalid by default
        last_tobii_timestamp = 0.0
        
        while self.is_tracking:
            now = time.perf_counter()
            
            # Check for new Tobii data (non-blocking)
            if self.socket.poll(0):
                try:
                    message = self.socket.recv_string()
                    parts = message.split()

                    if len(parts) >= 4 and parts[0] == "TobiiStream":
                        # Update last received gaze data
                        # Use REAL TIME (perf_counter) when packet was received, not Tobii's timestamp
                        last_tobii_timestamp = time.perf_counter()
                        last_raw_x = float(parts[2])
                        last_raw_y = float(parts[3])
                        
                        if len(parts) >= 5:
                            try:
                                last_validity = int(parts[4])
                            except ValueError:
                                last_validity = 1
                        else:
                            last_validity = 0
                except zmq.ZMQError:
                    pass
            
            # Produce sample at fixed intervals (regardless of Tobii data)
            if now >= next_sample_time:
                # Normalize gaze data (use last received, or fallback to -1, -1)
                norm_x, norm_y, on_screen = self._normalize_gaze_data(
                    last_raw_x, last_raw_y, last_validity, last_tobii_timestamp
                )
                
                # Clamp valid gaze to canvas bounds for visualization
                vis_x = max(0, min(last_raw_x, 1920)) if on_screen else 0
                vis_y = max(0, min(last_raw_y, 1200)) if on_screen else 0
                
                # Use system timestamp (in milliseconds) for this sample
                sample_timestamp_ms = now * 1000
                
                # ALWAYS write CSV row
                try:
                    self.writer.writerow([sample_timestamp_ms, norm_x, norm_y, on_screen])
                    
                    # Flush only every 30 samples to avoid blocking loop
                    self._flush_counter += 1
                    if self._flush_counter >= 30:
                        self.csv_file.flush()
                        self._flush_counter = 0
                except Exception:
                    pass
                
                self.sample_count += 1
                
                # ALWAYS send data to GUI queue
                data_packet = {
                    'x': vis_x,
                    'y': vis_y,
                    'raw_x': norm_x,
                    'raw_y': norm_y,
                    'on_screen': on_screen,
                    'timestamp_ms': int(sample_timestamp_ms),
                    'count': self.sample_count
                }
                self.data_queue.put(data_packet)
                
                # Schedule next sample
                next_sample_time += sample_interval
                
                # Prevent accumulation if loop falls behind (e.g., after pause/resume)
                if now > next_sample_time + sample_interval:
                    next_sample_time = now + sample_interval

    def process_queue(self):
    # Runs in MAIN THREAD
        try:
            while True:
                # Get data without blocking
                data = self.data_queue.get_nowait()
                
                # Update gaze duration tracking with every sample
                self.update_gaze_durations(data['timestamp_ms'], data['on_screen'])
                
                # Visual update (only when on-screen; keep previous position otherwise)
                if data['on_screen']:
                    self.gaze_viz.update_position(data['x'], data['y'])
                
                # Text updates (Throttle to every 10th sample to reduce CPU load)
                if data['count'] % 10 == 0:
                    if data['on_screen']:
                        self.coordinates_label.config(
                            text=f"👀 Position: x={data['raw_x']:.2f}, y={data['raw_y']:.2f}"
                        )
                        self.screen_status_label.config(
                            text="STATUS: ON-SCREEN", foreground="green"
                        )
                    else:
                        self.coordinates_label.config(text="👀 Position: --")
                        self.screen_status_label.config(
                            text="STATUS: OFF-SCREEN / WAITING", foreground="red"
                        )

                    self.status_label.config(
                        text=f"✅ Active - Samples: {data['count']}"
                    )
                    self.refresh_gaze_time_labels()

        except queue.Empty:
            pass

        # Reschedule next poll (~60fps)
        if self.root.winfo_exists():
            self.root.after(15, self.process_queue)

# -------------------- Fixed Timer Methods --------------------
    def start_timer(self):
        # 1. Prevent double-starting
        if self.timer_running:
            return

        self.timer_running = True
        
        # 2. Calculate start time based on how much time previously elapsed
        # If elapsed is 0 (fresh start), start_time is Now.
        # If elapsed is 10s (resume), start_time is Now - 10s.
        self.timer_start_time = time.time() - self.elapsed_time

        # 3. Handle Countdown Logic safely
        if self.use_countdown.get():
            try:
                # Safe retrieval of values (handles empty boxes defaulting to 0)
                h = self.countdown_hours_var.get()
                m = self.countdown_minutes_var.get()
                s = self.countdown_seconds_var.get()
                total_seconds = (h * 3600) + (m * 60) + s
            except Exception:
                total_seconds = 0
            
            if total_seconds <= 0:
                total_seconds = 0 # infinite if 0, or handle as finish immediately
            
            # Calculate when the timer should end
            self.countdown_end_time = time.time() + total_seconds - self.elapsed_time
        else:
            self.countdown_end_time = None

        # 4. Start the update loop
        self.update_timer()

    def stop_timer(self):
        if self.timer_running:
            self.timer_running = False
            
            # 1. Kill the background loop immediately
            if self.timer_after_id:
                try:
                    self.root.after_cancel(self.timer_after_id)
                except Exception:
                    pass
                self.timer_after_id = None
            
            # 2. Save the exact amount of time passed so we can resume later
            if self.timer_start_time is not None:
                self.elapsed_time = time.time() - self.timer_start_time

    def reset_timer(self):
        # 1. Stop everything first
        self.stop_timer()
        
        # 2. Zero out variables
        self.elapsed_time = 0.0
        self.timer_start_time = None
        self.countdown_end_time = None
        
        # 3. Reset Labels visually
        try:
            self.timer_label.config(text="Timer: 00:00:00")
            self.countdown_label.config(text="Countdown: --:--:--")
        except Exception:
            pass

    def update_timer(self):
        if not self.timer_running:
            return

        # 1. Calculate current duration
        current_time = time.time()
        if self.timer_start_time is None:
            self.timer_start_time = current_time - self.elapsed_time
            
        self.elapsed_time = current_time - self.timer_start_time

        # 2. Format and display (Up-counting Timer)
        total = int(self.elapsed_time)
        hrs = total // 3600
        mins = (total % 3600) // 60
        secs = total % 60
        
        self.timer_label.config(text=f"Timer: {hrs:02d}:{mins:02d}:{secs:02d}")

        # 3. Handle Countdown Logic
        if self.countdown_end_time is not None:
            remaining = self.countdown_end_time - current_time
            
            if remaining <= 0:
                # --- TIME IS UP: RESET LOGIC START ---
                
                # 1. Execute the "Stop Tracking" action if enabled
                if self.countdown_stop_tracking.get() and self.is_tracking:
                    self.root.after(10, self.stop_tracking)
                
                # 2. CRITICAL FIX: Reset variables immediately
                # This clears 'elapsed_time' back to 0 so the next start is fresh.
                self.reset_timer() 
                
                # 3. Visual feedback (Optional: show 00s for a moment)
                self.countdown_label.config(text="Countdown: 00:00:00")
                
                return # Exit the loop
                # --- TIME IS UP: RESET LOGIC END ---
            
            # Format Countdown Display
            r_total = int(remaining + 0.99)
            r_h = r_total // 3600
            r_m = (r_total % 3600) // 60
            r_s = r_total % 60
            self.countdown_label.config(text=f"Countdown: {r_h:02d}:{r_m:02d}:{r_s:02d}")

        # 4. Schedule next update
        self.timer_after_id = self.root.after(100, self.update_timer)

    # -------------------- Gaze Duration Tracking Methods --------------------
    def _format_ms(self, ms):
        """
        Convert milliseconds to MM:SS.t format.
        Args:
            ms: milliseconds (int or float)
        Returns:
            String in format "MM:SS.t"
        """
        total_seconds = ms / 1000.0
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        tenths = int((total_seconds % 1) * 10)
        return f"{minutes:02d}:{seconds:02d}.{tenths}"

    def _normalize_gaze_data(self, raw_x, raw_y, validity, last_tobii_timestamp):
        """
        Normalize gaze data to handle invalid or out-of-bounds values.
        Includes tracking loss detection to ensure OFF-SCREEN on sensor loss.
        
        Args:
            raw_x: raw X coordinate from Tobii
            raw_y: raw Y coordinate from Tobii
            validity: validity flag (0 = valid, non-zero = invalid)
            last_tobii_timestamp: timestamp (in seconds) of last Tobii sample
        
        Returns:
            tuple: (normalized_x, normalized_y, on_screen_bool)
                - If tracking lost (no sample for >50ms): (-1, -1, False)
                - If valid and in bounds: (raw_x, raw_y, True)
                - If invalid or out-of-bounds: (-1, -1, False)
        """
        # Detect tracking loss: no new Tobii sample for >50 ms
        now = time.perf_counter()
        tracking_lost = (now - last_tobii_timestamp) > 0.050
        
        if tracking_lost:
            return -1, -1, False
        
        # Check validity and bounds
        is_valid = (validity == 0)
        within_bounds = (0 <= raw_x <= 1920) and (0 <= raw_y <= 1200)
        on_screen = bool(is_valid and within_bounds)
        
        if on_screen:
            return raw_x, raw_y, True
        else:
            return -1, -1, False

    def update_gaze_durations(self, timestamp_ms, gaze_on_screen):
        """
        Track gaze on-screen and off-screen durations.
        Called for each gaze sample from the queue.
        
        Args:
            timestamp_ms: timestamp in milliseconds (from gaze sample)
            gaze_on_screen: bool indicating if gaze is on-screen
        """
        # Initialize on first call
        if self.last_timestamp is None:
            self.last_timestamp = timestamp_ms
            self.last_gaze_state = gaze_on_screen
            return

        # Calculate time delta
        dt = timestamp_ms - self.last_timestamp

        # Ignore invalid dt values (negative or impossibly large jumps)
        # At 30-90 FPS, dt should be 11-33 ms. Reject anything outside reasonable bounds.
        if dt < 0 or dt > 2000:
            # Reset and skip this sample
            self.last_timestamp = timestamp_ms
            self.last_gaze_state = gaze_on_screen
            return

        # Add time to appropriate counter (use previous state)
        if self.last_gaze_state is True:
            self.total_on_screen_time += dt
        elif self.last_gaze_state is False:
            self.total_off_screen_time += dt

        # Update state for next iteration
        self.last_timestamp = timestamp_ms
        self.last_gaze_state = gaze_on_screen

    def refresh_gaze_time_labels(self):
        """
        Update all gaze duration UI labels with current values.
        """
        # Calculate percentages
        total_time = self.total_on_screen_time + self.total_off_screen_time
        if total_time > 0:
            on_pct = (self.total_on_screen_time / total_time) * 100
            off_pct = (self.total_off_screen_time / total_time) * 100
        else:
            on_pct = 0
            off_pct = 0

        # Format time strings
        on_time_str = self._format_ms(self.total_on_screen_time)
        off_time_str = self._format_ms(self.total_off_screen_time)

        # Update labels
        try:
            self.gaze_on_screen_label.config(text=f"On-screen time: {on_time_str}")
            self.gaze_off_screen_label.config(text=f"Off-screen time: {off_time_str}")
            self.gaze_percentage_label.config(
                text=f"On-screen {on_pct:.1f}% | Off-screen {off_pct:.1f}%"
            )
        except Exception:
            pass

    def reset_gaze_timers(self):
        """
        Reset all gaze duration counters and state variables.
        Call this when a new recording session starts.
        """
        self.total_on_screen_time = 0
        self.total_off_screen_time = 0
        self.last_timestamp = None
        self.last_gaze_state = None
        self.refresh_gaze_time_labels()

    def _compute_gaze_durations_from_csv(self, data):
        """
        Compute on-screen and off-screen durations using the same algorithm as live tracking.
        
        Args:
            data: pandas DataFrame with 'timestamp' (ms) and 'on_screen' (bool) columns
        
        Returns:
            tuple: (total_on_time_ms, total_off_time_ms, total_time_ms)
        """
        if len(data) < 2:
            return 0, 0, 0
        
        total_on = 0
        total_off = 0
        
        timestamps = data['timestamp'].values
        on_screen = data['on_screen'].values
        
        for i in range(1, len(data)):
            dt = timestamps[i] - timestamps[i - 1]
            
            # Ignore invalid dt values (same as live logic)
            if dt < 0 or dt > 2000:
                continue
            
            # Accumulate based on previous state
            if on_screen[i - 1]:
                total_on += dt
            else:
                total_off += dt
        
        total_time = total_on + total_off
        return total_on, total_off, total_time

    def _export_session_summary(self, csv_filepath, sample_count):
        """
        Export a JSON summary of the recording session with timing metrics.
        
        Args:
            csv_filepath: Path to the saved gaze CSV file
            sample_count: Total number of gaze samples recorded
        """
        try:
            import json
            
            # Compute average FPS
            total_duration_sec = (self.total_on_screen_time + self.total_off_screen_time) / 1000.0
            avg_fps = sample_count / total_duration_sec if total_duration_sec > 0 else 0
            
            # Create summary dictionary
            summary = {
                "session_metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "gaze_csv_path": csv_filepath,
                    "sample_count": sample_count
                },
                "gaze_duration_analysis": {
                    "total_on_screen_time_ms": self.total_on_screen_time,
                    "total_off_screen_time_ms": self.total_off_screen_time,
                    "total_session_duration_ms": self.total_on_screen_time + self.total_off_screen_time,
                    "on_screen_percentage": (self.total_on_screen_time / (self.total_on_screen_time + self.total_off_screen_time) * 100) if (self.total_on_screen_time + self.total_off_screen_time) > 0 else 0,
                    "off_screen_percentage": (self.total_off_screen_time / (self.total_on_screen_time + self.total_off_screen_time) * 100) if (self.total_on_screen_time + self.total_off_screen_time) > 0 else 0
                },
                "timing_metrics": {
                    "total_duration_seconds": total_duration_sec,
                    "total_on_screen_seconds": self.total_on_screen_time / 1000.0,
                    "total_off_screen_seconds": self.total_off_screen_time / 1000.0,
                    "average_fps": round(avg_fps, 2),
                    "formatted_total_duration": self._format_ms(self.total_on_screen_time + self.total_off_screen_time),
                    "formatted_on_screen_time": self._format_ms(self.total_on_screen_time),
                    "formatted_off_screen_time": self._format_ms(self.total_off_screen_time)
                }
            }
            
            # Generate summary filename in same directory as CSV
            summary_dir = os.path.dirname(csv_filepath)
            summary_filename = f"session_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            summary_filepath = os.path.join(summary_dir, summary_filename)
            
            # Write JSON file
            with open(summary_filepath, 'w') as f:
                json.dump(summary, f, indent=2)
            
            return summary_filepath
        except Exception as e:
            print(f"Warning: Failed to export session summary: {str(e)}")
            return None

    def load_csv(self):
        filename = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv")],
            title="Select a gaze data CSV file"
        )
        if filename:
            try:
                self.data = pd.read_csv(filename)

                if 'on_screen' in self.data.columns:
                    self.data['on_screen'] = self.data['on_screen'].astype(str).str.lower() == 'true'
                else:
                    self.data['on_screen'] = (
                        (self.data['x'] >= 0) &
                        (self.data['y'] >= 0) &
                        (self.data['x'] <= 1920) &
                        (self.data['y'] <= 1200)
                    )

                self.file_label.config(text=f"📊 Loaded: {os.path.basename(filename)}")
                messagebox.showinfo("Success", "File loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")

    def generate_heatmap(self):
        if not hasattr(self, 'data'):
            messagebox.showwarning("Warning", "Please load a CSV file first")
            return

        for widget in self.results_frame.winfo_children():
            widget.destroy()
            
        # CLOSE PREVIOUS FIGURES TO PREVENT MEMORY LEAK
        plt.close('all')

        fig, ax = plt.subplots(figsize=(10, 8))
        plt.style.use('seaborn-v0_8')

        # Exclude invalid/off-screen samples so they do not get plotted at (0,0)
        valid = self.data[self.data['on_screen'] == True]
        x_data = valid['x']
        y_data = valid['y']

        # Compute raw 2D histogram, then apply Gaussian smoothing for better visualization
        heatmap_data, xedges, yedges = np.histogram2d(
            x_data, y_data,
            bins=50,
            range=[[0, 1920], [0, 1200]]
        )

        # Smooth the histogram with a Gaussian filter (sigma controls smoothing)
        try:
            sigma = float(self.heatmap_sigma_var.get())
        except Exception:
            sigma = 2.0
        smoothed = gaussian_filter(heatmap_data, sigma=sigma)

        # Display the smoothed heatmap. Transpose so x/y align with histogram2d axes
        # Always use logarithmic normalization with proper vmin based on minimum positive value
        min_positive = smoothed[smoothed > 0].min() if np.any(smoothed > 0) else 1e-3
        vmin = max(min_positive, 1e-3)
        vmax = smoothed.max() if smoothed.size > 0 else 1
        
        norm = LogNorm(vmin=vmin, vmax=vmax)

        img = ax.imshow(
            smoothed.T,
            origin='lower',
            cmap='turbo',
            extent=[0, 1920, 0, 1200],
            aspect='auto',
            norm=norm
        )

        # Colorbar for the smoothed heatmap
        fig.colorbar(img, ax=ax, label='Gaze Density')
        ax.set_title("Gaze Heatmap Analysis", pad=20)
        ax.set_xlabel("Screen X Coordinate")
        ax.set_ylabel("Screen Y Coordinate")

        ax.set_xlim(0, 1920)
        ax.set_ylim(1200, 0)  
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.results_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

    def generate_SpaceMap(self):
        if not hasattr(self, 'data'):
            messagebox.showwarning("Warning", "Please load a CSV file first")
            return

        for widget in self.results_frame.winfo_children():
            widget.destroy()
            
        plt.close('all')

        fig, ax = plt.subplots(figsize=(10, 8))
        plt.style.use('seaborn-v0_8-whitegrid')

        x_data = np.clip(self.data['x'], 0, 1920)
        y_data = np.clip(self.data['y'], 0, 1200)

        ax.scatter(
            x_data, y_data,
            alpha=0.5,
            s=20,
            c='red',
            label='Gaze Points'
        )

        ax.set_xlim(0, 1920)
        ax.set_ylim(1200, 0) 

        ax.set_xlabel("Screen X Coordinate")
        ax.set_ylabel("Screen Y Coordinate")
        ax.set_title("Gaze Space Map", pad=20)

        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        ax.text(
            0.02, 0.98,
            f'Total Samples: {len(x_data)}',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.results_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

    def analyze_attention(self):
        if not hasattr(self, 'data'):
            messagebox.showwarning("Warning", "Please load a CSV file first")
            return

        for widget in self.results_frame.winfo_children():
            widget.destroy()
            
        plt.close('all')

        # Compute movement for each sample
        self.data['movement'] = np.sqrt(
            self.data['x'].diff() ** 2 + self.data['y'].diff() ** 2
        )
        
        # Compute gaze durations using live tracking algorithm
        total_on_ms, total_off_ms, total_time_ms = self._compute_gaze_durations_from_csv(self.data)
        total_sec = total_time_ms / 1000.0
        
        # Classify durations based on movement thresholds (30-90 FPS baseline: ~11-33ms per sample)
        movement_threshold = 50
        reading_threshold_min = 5
        
        total_no_movement = 0
        total_reading = 0
        total_scanning = 0
        
        timestamps = self.data['timestamp'].values
        on_screen = self.data['on_screen'].values
        movement = self.data['movement'].values
        
        for i in range(1, len(self.data)):
            dt = timestamps[i] - timestamps[i - 1]
            
            # Ignore invalid dt values
            if dt < 0 or dt > 2000:
                continue
            
            # Only count time when on-screen
            if on_screen[i - 1]:
                mov = movement[i]
                
                if mov < reading_threshold_min:
                    total_no_movement += dt
                elif mov < movement_threshold:
                    total_reading += dt
                else:
                    total_scanning += dt
        
        # Convert to seconds
        no_movement_sec = total_no_movement / 1000.0
        reading_sec = total_reading / 1000.0
        scanning_sec = total_scanning / 1000.0
        on_screen_sec = total_on_ms / 1000.0
        
        # Compute percentages
        no_movement_pct = (no_movement_sec / total_sec * 100) if total_sec > 0 else 0
        reading_pct = (reading_sec / total_sec * 100) if total_sec > 0 else 0
        scanning_pct = (scanning_sec / total_sec * 100) if total_sec > 0 else 0
        
        # Format time strings
        total_time_str = self._format_ms(total_time_ms)
        no_movement_str = self._format_ms(total_no_movement)
        reading_str = self._format_ms(total_reading)
        scanning_str = self._format_ms(total_scanning)
        
        # Plot eye movement timeline
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.style.use('seaborn-v0_8')
        
        time_seconds = (self.data['timestamp'] - self.data['timestamp'].min()) / 1000
        
        ax.plot(time_seconds, self.data['movement'], 'b-', alpha=0.5, label='Eye Movement')
        ax.axhline(y=movement_threshold, color='r', linestyle='--', label='High Movement Threshold')
        ax.axhline(y=reading_threshold_min, color='g', linestyle='--', label='Reading Movement Threshold')
        
        ax.fill_between(time_seconds, 0, reading_threshold_min,
                        where=self.data['movement'] < reading_threshold_min,
                        color='red', alpha=0.2, label='No Movement')
        ax.fill_between(time_seconds, reading_threshold_min, movement_threshold,
                        where=self.data['movement'].between(reading_threshold_min, movement_threshold),
                        color='green', alpha=0.2, label='Reading Movement')
        ax.fill_between(time_seconds, movement_threshold, self.data['movement'].max(),
                        where=self.data['movement'] >= movement_threshold,
                        color='blue', alpha=0.2, label='Scanning Movement')
        
        ax.set_title("Eye Movement Timeline", pad=20)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Movement (pixels)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        stats_text = f"""
        📊 Attention Analysis (dt-based)
        ════════════════════════════════
        
        🕒 Total Session Duration: {total_time_str}
        👁 Total On-Screen Time: {self._format_ms(total_on_ms)} ({on_screen_sec/total_sec*100:.1f}%)
        
        Movement Classification:
        ━━━━━━━━━━━━━━━━━━━━━━
        🛑 No Movement (<{reading_threshold_min} px): {no_movement_str} ({no_movement_pct:.1f}%)
        📖 Reading Movement ({reading_threshold_min}-{movement_threshold} px): {reading_str} ({reading_pct:.1f}%)
        🔍 Scanning Movement (≥{movement_threshold} px): {scanning_str} ({scanning_pct:.1f}%)
        
        Analysis Parameters:
        • High movement threshold: {movement_threshold} pixels
        • Reading movement range: {reading_threshold_min}-{movement_threshold} pixels
        • Total samples analyzed: {len(self.data)}
        • Computation: Same dt-based algorithm as real-time tracking
        """
        
        stats_frame = ttk.Frame(self.results_frame)
        stats_frame.pack(fill='x', padx=20, pady=10)
        
        stats_label = ttk.Label(stats_frame, text=stats_text,
                                style='Status.TLabel', justify='left')
        stats_label.pack()
        
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.results_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

    def analyze_presence(self):
        if not hasattr(self, 'data'):
            messagebox.showwarning("Warning", "Please load a CSV file first")
            return

        # Compute gaze durations using live tracking algorithm
        total_on_ms, total_off_ms, total_time_ms = self._compute_gaze_durations_from_csv(self.data)
        
        # Convert to seconds
        total_on_sec = total_on_ms / 1000.0
        total_off_sec = total_off_ms / 1000.0
        total_sec = total_time_ms / 1000.0
        
        # Compute percentage
        presence_percentage = (total_on_sec / total_sec * 100) if total_sec > 0 else 0
        
        # Format time strings using the same format as live durations
        on_time_str = self._format_ms(total_on_ms)
        off_time_str = self._format_ms(total_off_ms)
        total_time_str = self._format_ms(total_time_ms)

        result_text = f"""
        👤 Presence Analysis Results
        ═══════════════════════════
        
        🕒 Total Session Duration: {total_time_str}
        ✓ Time Present at Screen: {on_time_str}
        ✗ Time Off-Screen: {off_time_str}
        📊 Presence Percentage: {presence_percentage:.1f}%
        
        Analysis Parameters:
        • Screen bounds: 1920x1200
        • Total samples analyzed: {len(self.data)}
        • Computation: Same dt-based algorithm as real-time tracking
        """

        for widget in self.results_frame.winfo_children():
            widget.destroy()

        result_label = ttk.Label(self.results_frame, text=result_text,
                                 style='Status.TLabel', justify='left')
        result_label.pack(padx=20, pady=20)


if __name__ == "__main__":
    root = tk.Tk()
    app = GazeAnalysisApp(root)
    def on_closing():
        # Perform full app shutdown then destroy and exit so the process ends
        try:
            app.shutdown()
        except Exception:
            pass
        try:
            if root.winfo_exists():
                # Cancel any remaining after callbacks (including ephemeral Tcl commands)
                try:
                    info = root.tk.call('after', 'info')
                    # `info` may be a Tcl list or empty string
                    if info:
                        # Normalize to Python list
                        if isinstance(info, str):
                            ids = info.split()
                        else:
                            ids = list(info)
                        for aid in ids:
                            try:
                                root.after_cancel(aid)
                            except Exception:
                                pass
                except Exception:
                    pass
                root.destroy()
        except Exception:
            pass
        try:
            sys.exit(0)
        except Exception:
            pass
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
