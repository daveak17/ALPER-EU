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
import threading
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
        self.screen_height = 1080
        self.last_update = 0
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

        # Network
        self.context = zmq.Context()
        self.socket = None

        self.setup_gui()
        
        # Start the GUI update loop
        self.process_queue()

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

    def tracking_loop(self):
        # Runs in BACKGROUND THREAD
        # We do NOT update GUI here. We put data in queue.
        while self.is_tracking:
            # Use poll to avoid blocking indefinitely
            if self.socket.poll(50):
                try:
                    message = self.socket.recv_string()
                    parts = message.split()

                    if len(parts) >= 4 and parts[0] == "TobiiStream":
                        timestamp = float(parts[1])
                        raw_x = float(parts[2])
                        raw_y = float(parts[3])

                        x = max(0, min(raw_x, 1920))
                        y = max(0, min(raw_y, 1080))

                        if len(parts) >= 5:
                            try:
                                validity = int(parts[4])
                            except ValueError:
                                validity = 1
                        else:
                            validity = 0 

                        is_valid = (validity == 0)
                        within_bounds = (0 <= raw_x <= 1920) and (0 <= raw_y <= 1080)
                        on_screen = bool(is_valid and within_bounds)

                        self.writer.writerow([timestamp, raw_x, raw_y, on_screen])
                        self.sample_count += 1

                        # Send data to GUI Thread
                        data_packet = {
                            'x': x,
                            'y': y,
                            'raw_x': raw_x,
                            'raw_y': raw_y,
                            'on_screen': on_screen,
                            'count': self.sample_count
                        }
                        self.data_queue.put(data_packet)
                except zmq.ZMQError:
                    continue
            else:
                # No data received, sleep briefly to not hog CPU
                time.sleep(0.001)

    def process_queue(self):
        # Runs in MAIN THREAD
        try:
            while True:
                # Get data without blocking
                data = self.data_queue.get_nowait()
                
                # Visual update
                self.gaze_viz.update_position(data['x'], data['y'])
                
                # Text updates (Throttle to every 10th sample to reduce CPU load)
                if data['count'] % 10 == 0:
                    self.coordinates_label.config(
                        text=f"👀 Position: x={data['raw_x']:.2f}, y={data['raw_y']:.2f}"
                    )
                    self.status_label.config(
                        text=f"✅ Active - Samples: {data['count']}"
                    )
                    
                    if data['on_screen']:
                        self.screen_status_label.config(text="STATUS: ON-SCREEN", foreground="green")
                    else:
                        self.screen_status_label.config(text="STATUS: OFF-SCREEN", foreground="red")
                        
        except queue.Empty:
            pass
        
        # Schedule next check in 15ms (~60fps check rate)
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

    # -------------------- Data Analysis Methods --------------------
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
                        (self.data['y'] <= 1080)
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

        x_data = np.clip(self.data['x'], 0, 1920)
        y_data = np.clip(self.data['y'], 0, 1080)

        heatmap = ax.hist2d(
            x_data, y_data,
            bins=50,
            cmap='hot',
            range=[[0, 1920], [0, 1080]]
        )

        plt.colorbar(heatmap[3], ax=ax, label='Gaze Density')
        ax.set_title("Gaze Heatmap Analysis", pad=20)
        ax.set_xlabel("Screen X Coordinate")
        ax.set_ylabel("Screen Y Coordinate")

        ax.set_xlim(0, 1920)
        ax.set_ylim(1080, 0) 

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
        y_data = np.clip(self.data['y'], 0, 1080)

        ax.scatter(
            x_data, y_data,
            alpha=0.5,
            s=20,
            c='red',
            label='Gaze Points'
        )

        ax.set_xlim(0, 1920)
        ax.set_ylim(1080, 0) 

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

        fig, ax = plt.subplots(figsize=(12, 6))
        plt.style.use('seaborn-v0_8')

        self.data['time_diff'] = self.data['timestamp'].diff()
        movement_threshold = 50
        self.data['movement'] = np.sqrt(
            self.data['x'].diff() ** 2 + self.data['y'].diff() ** 2
        )

        presence_mask = self.data['on_screen']

        movement_mask = (
            (self.data['movement'] > movement_threshold) |
            (self.data['movement'].between(5, movement_threshold))
        )

        attention_score = presence_mask.astype(int) * movement_mask.astype(int)

        time_seconds = (self.data['timestamp'] - self.data['timestamp'].min()) / 1000

        ax.plot(time_seconds, self.data['movement'], 'b-', alpha=0.5, label='Eye Movement')
        ax.axhline(y=movement_threshold, color='r', linestyle='--', label='High Movement Threshold')
        ax.axhline(y=5, color='g', linestyle='--', label='Reading Movement Threshold')
        ax.fill_between(time_seconds, 0, 5,
                        where=self.data['movement'] < 5,
                        color='red', alpha=0.2, label='No Movement')
        ax.fill_between(time_seconds, 5, movement_threshold,
                        where=self.data['movement'].between(5, movement_threshold),
                        color='green', alpha=0.2, label='Reading Movement')
        ax.fill_between(time_seconds, movement_threshold, self.data['movement'].max(),
                        where=self.data['movement'] >= movement_threshold,
                        color='blue', alpha=0.2, label='Scanning Movement')
        ax.set_title("Eye Movement Timeline", pad=20)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Movement (pixels)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        total_time = time_seconds.max()
        mean_dt = self.data['time_diff'].mean() / 1000 if self.data['time_diff'].notna().any() else 0

        attentive_time = attention_score.sum() * mean_dt if mean_dt > 0 else 0
        attentive_percentage = (attentive_time / total_time * 100) if total_time > 0 else 0

        reading_time = (
                (self.data['movement'].between(5, movement_threshold)) &
                presence_mask
        ).sum() * mean_dt if mean_dt > 0 else 0

        scanning_time = (
                (self.data['movement'] >= movement_threshold) &
                presence_mask
        ).sum() * mean_dt if mean_dt > 0 else 0

        total_time_str = f"{int(total_time // 60)}m {int(total_time % 60)}s"
        attentive_time_str = f"{int(attentive_time // 60)}m {int(attentive_time % 60)}s"
        reading_time_str = f"{int(reading_time // 60)}m {int(reading_time % 60)}s"
        scanning_time_str = f"{int(scanning_time // 60)}m {int(scanning_time % 60)}s"

        stats_text = f"""
        📊 Analysis Statistics
        ═══════════════════
        
        🕒 Total Session Duration: {total_time_str}
        👁 Total Attentive Time: {attentive_time_str} ({attentive_percentage:.1f}%)
        📖 Reading Time: {reading_time_str} ({(reading_time / total_time * 100) if total_time > 0 else 0:.1f}%)
        🔍 Scanning Time: {scanning_time_str} ({(scanning_time / total_time * 100) if total_time > 0 else 0:.1f}%)
        
        Analysis Parameters:
        • High movement threshold: {movement_threshold} pixels
        • Reading movement range: 5-{movement_threshold} pixels
        • Total samples analyzed: {len(self.data)}
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

        presence_mask = self.data['on_screen']

        self.data['time_diff'] = self.data['timestamp'].diff()

        total_time = (self.data['timestamp'].max() - self.data['timestamp'].min()) / 1000
        presence_time = self.data[presence_mask]['time_diff'].sum() / 1000

        total_time_str = f"{int(total_time // 60)}m {int(total_time % 60)}s"
        presence_time_str = f"{int(presence_time // 60)}m {int(presence_time % 60)}s"
        presence_percentage = (presence_time / total_time * 100) if total_time > 0 else 0

        result_text = f"""
        👤 Presence Analysis Results
        ═══════════════════════════
        
        🕒 Total Session Duration: {total_time_str}
        ✓ Time Present at Screen: {presence_time_str}
        📊 Presence Percentage: {presence_percentage:.1f}%
        
        Analysis Parameters:
        • Screen bounds: 1920x1080
        • Total samples analyzed: {len(self.data)}
        """

        for widget in self.results_frame.winfo_children():
            widget.destroy()

        result_label = ttk.Label(self.results_frame, text=result_text,
                                 style='Status.TLabel', justify='left')
        result_label.pack(padx=20, pady=20)


if __name__ == "__main__":
    root = tk.Tk()
    app = GazeAnalysisApp(root)
    root.mainloop()