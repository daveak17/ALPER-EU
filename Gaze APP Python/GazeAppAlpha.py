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
import queue
import os
import shutil

# Note: Ensure Tobii drivers are installed and TobiiStream is running on tcp://127.0.0.1:5556


def epoch_ms() -> int:
    """Return current time in milliseconds since epoch."""
    return time.time_ns() // 1_000_000


# ---------------------------------------------------------------------------
# GazeVisualization — lightweight canvas dot showing live gaze position
# ---------------------------------------------------------------------------
class GazeVisualization(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(bg='black', width=320, height=180)
        self.dot = self.create_oval(0, 0, 10, 10, fill='red', outline='white')
        self.screen_width = 1920
        self.screen_height = 1200
        self.last_update = 0
        self.update_interval = 1 / 30  # throttle; updated when FPS changes

    def update_position(self, x, y):
        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return
        canvas_x = int((x / self.screen_width) * self.winfo_width())
        canvas_y = int((y / self.screen_height) * self.winfo_height())
        self.coords(self.dot,
                    canvas_x - 5, canvas_y - 5,
                    canvas_x + 5, canvas_y + 5)
        self.last_update = current_time


# ---------------------------------------------------------------------------
# ModernButton — hover highlight helper
# ---------------------------------------------------------------------------
class ModernButton(ttk.Button):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)

    def on_enter(self, e):
        self['style'] = 'Accent.TButton'

    def on_leave(self, e):
        self['style'] = 'TButton'


# ---------------------------------------------------------------------------
# GazeAnalysisApp
# ---------------------------------------------------------------------------
class GazeAnalysisApp:
    def __init__(self, root, tracking_frame=None, analysis_frame=None):
        """Initialise the gaze analysis backend.

        Embedded mode: pass tracking_frame and analysis_frame (pre-created by
        MainController).  UI is built into those frames — no Toplevel created.
        Standalone mode: root is a Tk or Toplevel; setup_gui() builds its own
        notebook structure as before.
        """
        self.root = root
        self._embedded = (tracking_frame is not None and analysis_frame is not None)

        if not self._embedded:
            self.root.title("Gaze Analysis Application")
            self.root.geometry("1000x850")
            self.root.configure(bg='#f0f0f0')

        self.setup_styles()

        # ---- Tracking state ----
        self.is_tracking = False
        self.tracking_thread = None
        self.sample_count = 0
        self.current_file = None

        # Thread-safe queue: background thread → GUI
        self.data_queue = queue.Queue()

        # Sampling rate (software downsampling)
        # Default is 30 FPS to match the RealSense body tracker rate.
        # Can be changed via the dropdown when running eye-only mode.
        self.target_fps = 30
        self._last_save_time = 0.0
        self._flush_counter = 0

        # Heatmap settings
        self.heatmap_sigma_var = tk.DoubleVar(value=2.0)
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

        # Gaze duration accumulators
        self.total_on_screen_time = 0   # ms
        self.total_off_screen_time = 0  # ms
        self.last_timestamp = None      # ms
        self.last_gaze_state = None

        # Live state — current gaze on-screen status.
        # Written by tracking_loop (background thread), read by MainController.
        # Single bool assignment is GIL-safe in CPython.
        self.gaze_on_screen: bool | None = None

        # ZMQ — FIX 1: context starts as a live instance (not None)
        # It will be recreated lazily in start_streaming_preview if terminated.
        self.context = zmq.Context()
        self.socket = None

        # External-control recording state
        self._preview_running = False
        self._recording = False
        self._record_file = None
        self._record_writer = None
        self._record_flush_counter = 0
        self._record_go_epoch_ms = None

        # Flag to stop process_queue rescheduling after shutdown  (FIX 4)
        self._shutdown = False

        if self._embedded:
            # Frames are pre-created by MainController — populate them directly
            self.tracking_frame = tracking_frame
            self.analysis_frame = analysis_frame
            self.setup_tracking_tab()
            self.setup_analysis_tab()
        else:
            self.setup_gui()

        # Start GUI ↔ queue polling loop
        self._process_queue_after_id = None
        try:
            if self.root.winfo_exists():
                self._process_queue_after_id = self.root.after(
                    15, self.process_queue)
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Styles
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # GUI setup
    # -----------------------------------------------------------------------
    def setup_gui(self):
        """Build standalone window (only used when not embedded in MainController)."""
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
        # Wrap entire tracking tab in a scrollable canvas (same pattern as analysis tab)
        _canvas = tk.Canvas(self.tracking_frame, borderwidth=0, highlightthickness=0)
        _scrollbar = ttk.Scrollbar(self.tracking_frame, orient='vertical',
                                    command=_canvas.yview)
        _canvas.configure(yscrollcommand=_scrollbar.set)
        _scrollbar.pack(side='right', fill='y')
        _canvas.pack(side='left', fill='both', expand=True)

        _inner = ttk.Frame(_canvas)
        _canvas_window = _canvas.create_window((0, 0), window=_inner, anchor='nw')

        def _on_frame_configure(event):
            _canvas.configure(scrollregion=_canvas.bbox('all'))

        def _on_canvas_configure(event):
            _canvas.itemconfig(_canvas_window, width=event.width)

        _inner.bind('<Configure>', _on_frame_configure)
        _canvas.bind('<Configure>', _on_canvas_configure)

        def _on_mousewheel(event):
            _canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')

        _canvas.bind('<Enter>',
                     lambda e: _canvas.bind_all('<MouseWheel>', _on_mousewheel))
        _canvas.bind('<Leave>',
                     lambda e: _canvas.unbind_all('<MouseWheel>'))

        # All content packs into _inner instead of self.tracking_frame
        _tf = _inner   # alias — every pack() call below uses _tf

        control_frame = ttk.LabelFrame(_tf, text="Tracking Control")
        control_frame.pack(fill='x', padx=20, pady=10)

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(padx=20, pady=15)

        self.start_button = ModernButton(button_frame, text="▶ Start Tracking",
                                        command=self.start_tracking)
        self.start_button.pack(side='left', padx=10)

        self.stop_button = ModernButton(button_frame, text="⏹ Stop Tracking",
                                       command=self.stop_tracking, state='disabled')
        self.stop_button.pack(side='left', padx=10)

        # Sampling rate dropdown (30–90 FPS)
        fps_frame = ttk.Frame(control_frame)
        fps_frame.pack(padx=20, pady=(0, 10), anchor='w')
        ttk.Label(fps_frame, text="Sampling Rate:").pack(side='left')
        self.fps_combo = ttk.Combobox(fps_frame,
                                      values=['30', '45', '60', '75', '90'],
                                      width=5, state='readonly')
        self.fps_combo.set(str(self.target_fps))
        self.fps_combo.pack(side='left', padx=(8, 0))
        self.fps_combo.bind('<<ComboboxSelected>>', self.on_fps_change)

        viz_frame = ttk.LabelFrame(_tf, text="Live Gaze Visualization")
        viz_frame.pack(fill='x', padx=20, pady=10)
        self.gaze_viz = GazeVisualization(viz_frame)
        self.gaze_viz.pack(padx=20, pady=15)

        status_frame = ttk.LabelFrame(_tf, text="Live Status")
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

        self.screen_status_label = ttk.Label(
            status_content, text="STATUS: NOT TRACKING",
            font=("Helvetica", 14, "bold"), foreground="gray")
        self.screen_status_label.pack(pady=(10, 0))

        # Gaze duration display
        gaze_duration_frame = ttk.LabelFrame(_tf,
                                             text="Gaze Duration Analysis")
        gaze_duration_frame.pack(fill='x', padx=20, pady=10)
        duration_content = ttk.Frame(gaze_duration_frame)
        duration_content.pack(padx=20, pady=15)

        self.gaze_on_screen_label = ttk.Label(duration_content,
                                              text="On-screen time: 00:00.0",
                                              font=("Helvetica", 11))
        self.gaze_on_screen_label.pack(pady=5)

        self.gaze_off_screen_label = ttk.Label(duration_content,
                                               text="Off-screen time: 00:00.0",
                                               font=("Helvetica", 11))
        self.gaze_off_screen_label.pack(pady=5)

        self.gaze_percentage_label = ttk.Label(
            duration_content,
            text="On-screen 0.0% | Off-screen 0.0%",
            font=("Helvetica", 11, "bold"), foreground="#007bff")
        self.gaze_percentage_label.pack(pady=5)

        # Timer controls
        timer_frame = ttk.LabelFrame(_tf, text="Timer")
        timer_frame.pack(fill='x', padx=20, pady=10)
        timer_content = ttk.Frame(timer_frame)
        timer_content.pack(padx=20, pady=10)

        self.timer_label = ttk.Label(timer_content, text="Timer: 00:00:00",
                                     style='Status.TLabel',
                                     font=(None, 12, 'bold'))
        self.timer_label.pack(side='left', padx=(0, 20))

        ModernButton(timer_content, text="▶ Start Timer",
                     command=self.start_timer).pack(side='left', padx=5)
        ModernButton(timer_content, text="⏸ Stop Timer",
                     command=self.stop_timer).pack(side='left', padx=5)
        ModernButton(timer_content, text="↺ Reset",
                     command=self.reset_timer).pack(side='left', padx=5)
        ttk.Checkbutton(timer_content, text='Auto-start with tracking',
                        variable=self.auto_start_timer).pack(side='left', padx=20)

        self.countdown_label = ttk.Label(timer_content, text="Countdown: --:--:--",
                                         style='Status.TLabel')
        self.countdown_label.pack(side='left', padx=(20, 0))

        # Countdown configuration
        countdown_frame = ttk.Frame(timer_frame)
        countdown_frame.pack(fill='x', padx=20, pady=(8, 0))
        ttk.Checkbutton(countdown_frame, text='Use countdown',
                        variable=self.use_countdown).pack(side='left')
        ttk.Label(countdown_frame, text='Duration:').pack(side='left', padx=(12, 4))
        ttk.Spinbox(countdown_frame, from_=0, to=23, width=3,
                    textvariable=self.countdown_hours_var).pack(side='left')
        ttk.Label(countdown_frame, text='h').pack(side='left')
        ttk.Spinbox(countdown_frame, from_=0, to=59, width=3,
                    textvariable=self.countdown_minutes_var).pack(side='left', padx=(6, 0))
        ttk.Label(countdown_frame, text='m').pack(side='left')
        ttk.Spinbox(countdown_frame, from_=0, to=59, width=3,
                    textvariable=self.countdown_seconds_var).pack(side='left', padx=(6, 0))
        ttk.Label(countdown_frame, text='s').pack(side='left', padx=(4, 12))
        ttk.Checkbutton(countdown_frame, text='Stop tracking when finished',
                        variable=self.countdown_stop_tracking).pack(side='left')

    def on_fps_change(self, event=None):
        """Handle FPS dropdown change."""
        try:
            fps = int(self.fps_combo.get())
            self.target_fps = fps
            # FIX 3: target_fps is read dynamically by tracking_loop each iteration,
            # so no extra action needed here — the loop picks it up automatically.
            try:
                if hasattr(self, 'gaze_viz') and fps > 0:
                    self.gaze_viz.update_interval = 1.0 / float(fps)
            except Exception:
                pass
            if self.is_tracking:
                self.status_label.config(
                    text=f"✅ Tracking active - Sampling: {fps} FPS")
        except Exception:
            pass

    def setup_analysis_tab(self):
        # Wrap entire analysis tab in a scrollable canvas so nothing gets clipped
        canvas = tk.Canvas(self.analysis_frame, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.analysis_frame, orient='vertical',
                                  command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)

        # Inner frame that holds all the actual content
        inner = ttk.Frame(canvas)
        inner_id = canvas.create_window((0, 0), window=inner, anchor='nw')

        def _on_inner_configure(event):
            canvas.configure(scrollregion=canvas.bbox('all'))

        def _on_canvas_configure(event):
            # Make inner frame match canvas width so content fills horizontally
            canvas.itemconfig(inner_id, width=event.width)

        inner.bind('<Configure>', _on_inner_configure)
        canvas.bind('<Configure>', _on_canvas_configure)

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        canvas.bind_all('<MouseWheel>', _on_mousewheel)

        # Redirect self.analysis_frame references inside this method to inner
        _orig_analysis_frame = self.analysis_frame
        self.analysis_frame = inner

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
            sigma_spin = ttk.Spinbox(heatmap_settings, from_=0.0, to=10.0,
                                     increment=0.5,
                                     textvariable=self.heatmap_sigma_var, width=5)
        except Exception:
            sigma_spin = tk.Spinbox(heatmap_settings, from_=0.0, to=10.0,
                                    increment=0.5,
                                    textvariable=self.heatmap_sigma_var, width=5)
        sigma_spin.pack(side='left', padx=(6, 10))

        def on_sigma_change(*args):
            if hasattr(self, 'data'):
                self.generate_heatmap()

        self.heatmap_sigma_var.trace('w', on_sigma_change)

        # ---- Session Analysis section ----
        session_frame = ttk.LabelFrame(self.analysis_frame,
                                       text="Session Engagement Analysis")
        session_frame.pack(fill='x', padx=20, pady=10)

        sf_top = ttk.Frame(session_frame)
        sf_top.pack(fill='x', padx=16, pady=(10, 4))

        ttk.Label(sf_top, text="Session folder:",
                  font=('Helvetica', 10, 'bold')).pack(side='left')
        self._session_folder_var = tk.StringVar(value="")
        ttk.Entry(sf_top, textvariable=self._session_folder_var,
                  width=42).pack(side='left', padx=(8, 6))
        ttk.Button(sf_top, text="📂 Browse",
                   command=self._browse_session_folder).pack(side='left', padx=4)
        ttk.Button(sf_top, text="📂 Use Last Session",
                   command=self._use_last_session).pack(side='left', padx=4)

        sf_btn = ttk.Frame(session_frame)
        sf_btn.pack(fill='x', padx=16, pady=(4, 10))
        ttk.Button(sf_btn, text="▶  Run Full Analysis",
                   command=self._run_session_analysis).pack(side='left', padx=4)
        ttk.Button(sf_btn, text="📈  Timeline",
                   command=lambda: self._show_analysis_image(
                       'engagement_timeline.png')).pack(side='left', padx=4)
        ttk.Button(sf_btn, text="🔥  Heatmap",
                   command=lambda: self._show_analysis_image(
                       'gaze_heatmap.png')).pack(side='left', padx=4)
        ttk.Button(sf_btn, text="📊  Signal Summary",
                   command=lambda: self._show_analysis_image(
                       'signal_summary.png')).pack(side='left', padx=4)
        ttk.Button(sf_btn, text="📉  Engagement Score",
                   command=lambda: self._show_analysis_image(
                       'engagement_score.png')).pack(side='left', padx=4)

        # Download buttons row
        sf_dl = ttk.Frame(session_frame)
        sf_dl.pack(fill='x', padx=16, pady=(0, 4))
        ttk.Label(sf_dl, text="Download:",
                  font=('Helvetica', 9, 'bold')).pack(side='left', padx=(0, 6))
        ttk.Button(sf_dl, text="⬇  Session Report (.xlsx)",
                   command=self._download_session_xlsx).pack(side='left', padx=4)

        self._session_status_label = ttk.Label(
            session_frame,
            text="Select a session folder and click Run Full Analysis.",
            foreground='gray')
        self._session_status_label.pack(anchor='w', padx=16, pady=(0, 8))

        # ---- Multi-session comparison section ----
        compare_frame = ttk.LabelFrame(self.analysis_frame,
                                       text="Multi-Session Comparison")
        compare_frame.pack(fill='x', padx=20, pady=(0, 10))

        cf_top = ttk.Frame(compare_frame)
        cf_top.pack(fill='x', padx=16, pady=(10, 4))

        ttk.Button(cf_top, text="➕ Add Session",
                   command=self._add_comparison_session).pack(side='left', padx=4)
        ttk.Button(cf_top, text="🗑 Clear All",
                   command=self._clear_comparison_sessions).pack(side='left', padx=4)

        # Session list display
        self._comparison_listbox = tk.Listbox(
            compare_frame, height=4, selectmode='single',
            font=('Courier', 9))
        self._comparison_listbox.pack(fill='x', padx=16, pady=(4, 4))

        cf_btn = ttk.Frame(compare_frame)
        cf_btn.pack(fill='x', padx=16, pady=(0, 10))
        ttk.Button(cf_btn, text="▶  Run Comparison",
                   command=self._run_comparison).pack(side='left', padx=4)
        ttk.Button(cf_btn, text="📊  Score Bars",
                   command=lambda: self._show_comparison_image(
                       'comparison_scores.png')).pack(side='left', padx=4)
        ttk.Button(cf_btn, text="📈  Score Curves",
                   command=lambda: self._show_comparison_image(
                       'comparison_curves.png')).pack(side='left', padx=4)
        ttk.Button(cf_btn, text="📉  Signal Breakdown",
                   command=lambda: self._show_comparison_image(
                       'comparison_signals.png')).pack(side='left', padx=4)

        self._comparison_status_label = ttk.Label(
            compare_frame,
            text="Add at least 2 analysed sessions to compare.",
            foreground='gray')
        self._comparison_status_label.pack(anchor='w', padx=16, pady=(0, 8))

        # Internal state
        self._comparison_folders = []   # list of absolute folder paths
        self._comparison_output_dir = None

        # ---- Results area ----
        self.results_frame = ttk.LabelFrame(self.analysis_frame,
                                            text="Analysis Results")
        self.results_frame.pack(fill='both', expand=True, padx=20, pady=10)

        welcome_text = (
            "\n    Eye Tracking Analysis Dashboard!\n\n"
            "    To begin:\n"
            "    1. Click \"Load CSV File\"\n"
            "    2. Choose a gaze CSV\n"
            "    3. View analysis results here\n"
        )
        ttk.Label(self.results_frame, text=welcome_text,
                  style='Status.TLabel').pack(padx=20, pady=20)

        # Restore analysis_frame to the original tab frame
        self.analysis_frame = _orig_analysis_frame

    # -----------------------------------------------------------------------
    # GUI-driven tracking (Start / Stop buttons)
    # -----------------------------------------------------------------------
    def start_tracking(self):
        """GUI button handler.  FIX 6: exits immediately if external API is active."""
        if self._recording or self._preview_running:
            # Being controlled externally by MainController — don't interfere
            return

        if not self.is_tracking:
            try:
                self.start_streaming_preview()

                # Open a temporary CSV for the GUI-driven session
                self.temp_file = (
                    f"temp_gaze_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                self._record_file = open(self.temp_file, 'w', newline='')
                self._record_writer = csv.writer(self._record_file)
                self._record_writer.writerow(['epoch_ms', 'x', 'y', 'on_screen'])
                self._record_go_epoch_ms = epoch_ms()
                self._recording = True

                self.is_tracking = True
                self.sample_count = 0
                self._last_save_time = 0.0
                self.reset_gaze_timers()

                if not (self.tracking_thread and self.tracking_thread.is_alive()):
                    self.tracking_thread = threading.Thread(
                        target=self.tracking_loop, daemon=True)
                    self.tracking_thread.start()

                try:
                    if self.auto_start_timer.get():
                        self.start_timer()
                except Exception:
                    pass

                self.start_button.config(state='disabled')
                self.stop_button.config(state='normal')
                self.status_label.config(text="✅ Tracking active")
                self.screen_status_label.config(
                    text="STATUS: WAITING DATA", foreground="orange")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to start tracking: {str(e)}")
                self.stop_tracking()

    def stop_tracking(self):
        """GUI button handler — stops recording and offers a save dialog."""
        if self.is_tracking:
            self.is_tracking = False
            try:
                self.stop_recording()
            except Exception:
                pass
            try:
                self.stop_streaming_preview()
            except Exception:
                pass
            try:
                if self.auto_start_timer.get():
                    self.stop_timer()
            except Exception:
                pass

            default_name = (
                f"gazetrack_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            save_path = filedialog.asksaveasfilename(
                defaultextension=".csv", initialfile=default_name,
                filetypes=[("CSV files", "*.csv")],
                title="Save Gaze Tracking Data")

            if save_path and hasattr(self, 'temp_file'):
                try:
                    shutil.copy2(self.temp_file, save_path)
                    os.remove(self.temp_file)
                    summary_path = self._export_session_summary(
                        save_path, self.sample_count)
                    if summary_path:
                        messagebox.showinfo(
                            "Success",
                            f"Data saved to {save_path}\n\n"
                            f"Session summary: {os.path.basename(summary_path)}")
                    else:
                        messagebox.showinfo("Success", f"Data saved to {save_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save file: {str(e)}")
            else:
                try:
                    if hasattr(self, 'temp_file'):
                        os.remove(self.temp_file)
                except Exception:
                    pass

            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')
            self.status_label.config(text="📁 Tracking stopped")
            self.coordinates_label.config(text="Coordinates: --")
            self.screen_status_label.config(
                text="STATUS: NOT TRACKING", foreground="gray")
            self.gaze_viz.update_position(0, 0)

    # -----------------------------------------------------------------------
    # External control API  (used by MainController)
    # -----------------------------------------------------------------------
    def start_streaming_preview(self):
        """Start ZMQ subscription and background sampling without opening a CSV."""
        if self._preview_running:
            return

        # FIX 1: Recreate ZMQ context if it was terminated by a previous shutdown()
        if self.context is None:
            self.context = zmq.Context()

        with self.data_queue.mutex:
            self.data_queue.queue.clear()

        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect("tcp://127.0.0.1:5556")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "TobiiStream")

        self.is_tracking = True
        self._preview_running = True
        self.sample_count = 0
        self._last_save_time = 0.0

        self.tracking_thread = threading.Thread(
            target=self.tracking_loop, daemon=True)
        self.tracking_thread.start()

    def stop_streaming_preview(self):
        """Stop background sampling.  No file dialogs."""
        if not self._preview_running:
            return
        self.is_tracking = False
        self._preview_running = False
        try:
            if self.tracking_thread and self.tracking_thread.is_alive():
                self.tracking_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self.socket:
                self.socket.close()
                self.socket = None
        except Exception:
            pass

    def start_recording(self, csv_path: str,
                        go_epoch_ms: int | None = None,
                        target_fps: int = 30):
        """Open csv_path and start writing samples after go_epoch_ms.

        FIX 2: If a recording is already open, close it cleanly before opening
        the new one — prevents orphaned file handles.
        """
        if not self._preview_running and not self.is_tracking:
            self.start_streaming_preview()

        # FIX 2: close any previously open recording instead of silently returning
        if self._recording:
            self.stop_recording()

        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

        self._record_file = open(csv_path, 'w', newline='')
        self._record_writer = csv.writer(self._record_file)
        self._record_writer.writerow(['epoch_ms', 'x', 'y', 'on_screen'])
        self._record_go_epoch_ms = go_epoch_ms
        self.target_fps = target_fps   # FIX 3: tracking_loop reads this each iteration
        self._record_flush_counter = 0
        self._recording = True

    def stop_recording(self):
        """Flush and close the current CSV recording."""
        if not self._recording:
            return
        try:
            if self._record_file:
                self._record_file.flush()
                self._record_file.close()
        except Exception:
            pass
        self._record_file = None
        self._record_writer = None
        self._recording = False
        self._record_go_epoch_ms = None

    def is_running_preview(self) -> bool:
        return self._preview_running

    def is_recording(self) -> bool:
        return self._recording

    @staticmethod
    def create_ui(parent):
        """Create the gaze UI in a standalone Toplevel window."""
        top = tk.Toplevel(parent)
        app = GazeAnalysisApp(top)
        return app

    @staticmethod
    def attach_to_tabs(root, tracking_frame, analysis_frame):
        """Embed the gaze UI into pre-existing frames — no new window created.

        Called by MainController to achieve a single unified window.
        root            -- the master Tk() window (for after() scheduling)
        tracking_frame  -- ttk.Frame for the Live Tracking tab
        analysis_frame  -- ttk.Frame for the Data Analysis tab
        """
        return GazeAnalysisApp(
            root,
            tracking_frame=tracking_frame,
            analysis_frame=analysis_frame,
        )

    def shutdown(self):
        """Cleanly stop all background work so the process can exit."""
        # FIX 4: Set shutdown flag so process_queue stops rescheduling itself
        self._shutdown = True

        try:
            self.is_tracking = False
        except Exception:
            pass

        try:
            if self.tracking_thread and self.tracking_thread.is_alive():
                self.tracking_thread.join(timeout=1.0)
        except Exception:
            pass

        try:
            if self.socket:
                self.socket.close()
                self.socket = None
        except Exception:
            pass

        # FIX 1: set context to None after terminating so start_streaming_preview
        # can recreate it cleanly if called again.
        try:
            if self.context:
                self.context.term()
                self.context = None
        except Exception:
            pass

        try:
            self.stop_recording()
        except Exception:
            pass

        try:
            self.stop_timer()
        except Exception:
            pass

        try:
            if self._process_queue_after_id is not None:
                if self.root.winfo_exists():
                    self.root.after_cancel(self._process_queue_after_id)
                self._process_queue_after_id = None
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Background tracking loop  (runs in a daemon thread)
    # -----------------------------------------------------------------------
    def tracking_loop(self):
        """Two-loop design:
        - Receiver: non-blocking ZMQ poll, updates latest raw gaze immediately.
        - Sampler: fires at exactly target_fps intervals, writes to CSV + queue.
        """
        next_sample_time = time.perf_counter()

        last_raw_x = -1
        last_raw_y = -1
        last_validity = 1
        # Initialise 1s in the past — treated as "not yet seen" until first
        # real message arrives.  500ms is the silence threshold so this gives
        # a clean startup without needing special-case logic.
        last_tobii_timestamp = time.perf_counter() - 1.0

        while getattr(self, 'is_tracking', False):
            now = time.perf_counter()

            # ---- Receiver: drain ALL available ZMQ messages without blocking ----
            # Loop until the socket has no more pending messages so we always
            # have the most recent gaze position before the sampler fires.
            # This prevents the "stale off-screen value" problem on recovery.
            while self.socket and self.socket.poll(0):
                try:
                    message = self.socket.recv_string(zmq.NOBLOCK)
                    parts = message.split()
                    if len(parts) >= 4 and parts[0] == "TobiiStream":
                        last_tobii_timestamp = time.perf_counter()
                        last_raw_x = float(parts[2])
                        last_raw_y = float(parts[3])
                        last_validity = int(parts[4]) if len(parts) >= 5 else 0
                except (zmq.ZMQError, ValueError):
                    break

            # ---- Sampler: produce one sample at fixed interval ----
            if now >= next_sample_time:
                # FIX 3: Read target_fps dynamically each iteration so changes
                # made by start_recording() or the FPS dropdown take effect
                # immediately without restarting the thread.
                sample_interval = 1.0 / float(self.target_fps)

                norm_x, norm_y, on_screen = self._normalize_gaze_data(
                    last_raw_x, last_raw_y, last_validity, last_tobii_timestamp)

                vis_x = max(0, min(last_raw_x, 1920)) if on_screen else 0
                vis_y = max(0, min(last_raw_y, 1200)) if on_screen else 0

                sample_epoch_ms = epoch_ms()

                # Write to CSV if recording and gate time reached
                if self._recording and self._record_writer:
                    try:
                        if (self._record_go_epoch_ms is None
                                or sample_epoch_ms >= self._record_go_epoch_ms):
                            self._record_writer.writerow(
                                [sample_epoch_ms, norm_x, norm_y, on_screen])
                            self._record_flush_counter += 1
                            if self._record_flush_counter >= 30:
                                try:
                                    self._record_file.flush()
                                except Exception:
                                    pass
                                self._record_flush_counter = 0
                    except Exception:
                        pass

                self.sample_count += 1

                # Update live state for polling by MainController
                self.gaze_on_screen = on_screen

                # Push to GUI queue (always, regardless of file I/O)
                self.data_queue.put({
                    'x': vis_x,
                    'y': vis_y,
                    'raw_x': norm_x,
                    'raw_y': norm_y,
                    'on_screen': on_screen,
                    'timestamp_ms': sample_epoch_ms,
                    'count': self.sample_count,
                })

                next_sample_time += sample_interval

                # Prevent drift accumulation if loop fell behind
                if now > next_sample_time + sample_interval:
                    next_sample_time = now + sample_interval

            # Yield the thread — sleep until next sample or 5ms max.
            # Without this, the tight spin loop starves itself on Windows
            # (OS deprioritises threads that never yield) causing delayed
            # recovery when gaze returns on-screen.
            sleep_s = min(next_sample_time - time.perf_counter(), 0.005)
            if sleep_s > 0:
                time.sleep(sleep_s)

    # -----------------------------------------------------------------------
    # GUI queue consumer  (runs on the Tkinter main thread via after())
    # -----------------------------------------------------------------------
    def process_queue(self):
        # Runs in MAIN THREAD — called periodically via root.after()
        last_on_screen = None   # track state within this batch to detect transitions

        try:
            while True:
                data = self.data_queue.get_nowait()
                on_screen = data['on_screen']

                self.update_gaze_durations(data['timestamp_ms'], on_screen)

                if on_screen:
                    self.gaze_viz.update_position(data['x'], data['y'])

                # Update status label immediately on any on/off-screen state change.
                # This is what makes the transition feel instant — no waiting for the
                # 10-sample throttle to fire.
                if on_screen != last_on_screen:
                    if on_screen:
                        self.screen_status_label.config(
                            text="STATUS: ON-SCREEN", foreground="green")
                    else:
                        self.screen_status_label.config(
                            text="STATUS: OFF-SCREEN / WAITING", foreground="red")
                    last_on_screen = on_screen

                # Throttle the coordinate text and sample counter — these are noisy
                # and don't need to update faster than every 10 samples
                if data['count'] % 10 == 0:
                    if on_screen:
                        self.coordinates_label.config(
                            text=f"👀 Position: x={data['raw_x']:.2f}, "
                                 f"y={data['raw_y']:.2f}")
                    else:
                        self.coordinates_label.config(text="👀 Position: --")

                    self.status_label.config(
                        text=f"✅ Active - Samples: {data['count']}")
                    self.refresh_gaze_time_labels()

        except queue.Empty:
            pass

        # FIX 4: Stop rescheduling after shutdown() has been called
        try:
            if not self._shutdown and self.root.winfo_exists():
                self._process_queue_after_id = self.root.after(
                    15, self.process_queue)
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Timer
    # -----------------------------------------------------------------------
    def start_timer(self):
        if self.timer_running:
            return
        self.timer_running = True
        self.timer_start_time = time.time() - self.elapsed_time

        if self.use_countdown.get():
            try:
                h = self.countdown_hours_var.get()
                m = self.countdown_minutes_var.get()
                s = self.countdown_seconds_var.get()
                total_seconds = h * 3600 + m * 60 + s
            except Exception:
                total_seconds = 0
            self.countdown_end_time = (time.time() + total_seconds
                                       - self.elapsed_time)
        else:
            self.countdown_end_time = None

        self.update_timer()

    def stop_timer(self):
        if self.timer_running:
            self.timer_running = False
            if self.timer_after_id:
                try:
                    self.root.after_cancel(self.timer_after_id)
                except Exception:
                    pass
                self.timer_after_id = None
            if self.timer_start_time is not None:
                self.elapsed_time = time.time() - self.timer_start_time

    def reset_timer(self):
        self.stop_timer()
        self.elapsed_time = 0.0
        self.timer_start_time = None
        self.countdown_end_time = None
        try:
            self.timer_label.config(text="Timer: 00:00:00")
            self.countdown_label.config(text="Countdown: --:--:--")
        except Exception:
            pass

    def update_timer(self):
        if not self.timer_running:
            return

        current_time = time.time()
        if self.timer_start_time is None:
            self.timer_start_time = current_time - self.elapsed_time
        self.elapsed_time = current_time - self.timer_start_time

        total = int(self.elapsed_time)
        self.timer_label.config(
            text=f"Timer: {total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}")

        if self.countdown_end_time is not None:
            remaining = self.countdown_end_time - current_time
            if remaining <= 0:
                if self.countdown_stop_tracking.get() and self.is_tracking:
                    self.root.after(10, self.stop_tracking)
                self.reset_timer()
                self.countdown_label.config(text="Countdown: 00:00:00")
                return
            r = int(remaining + 0.99)
            self.countdown_label.config(
                text=f"Countdown: {r // 3600:02d}:{(r % 3600) // 60:02d}:{r % 60:02d}")

        self.timer_after_id = self.root.after(100, self.update_timer)

    # -----------------------------------------------------------------------
    # Gaze duration tracking
    # -----------------------------------------------------------------------
    def _format_ms(self, ms: float) -> str:
        """Convert milliseconds to MM:SS.t string."""
        total_seconds = ms / 1000.0
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        tenths = int((total_seconds % 1) * 10)
        return f"{minutes:02d}:{seconds:02d}.{tenths}"

    def _normalize_gaze_data(self, raw_x, raw_y, validity, last_tobii_timestamp):
        """Normalise raw Tobii data; return (x, y, on_screen).

        on_screen logic:
          - If the device has been completely silent for 500ms: tracking lost,
            return off-screen.  This catches cable unplugged / driver crash.
          - Otherwise: on_screen = coordinates are within screen bounds.
            We do NOT gate on validity alone because the Tobii 4C can send
            validity=1 (one eye tracked) or validity=2 (partial) with perfectly
            good coordinates.  Trusting the coordinates directly is more robust
            and matches what the Tobii SDK actually guarantees: it only emits
            a gaze point when it has a usable estimate.
        """
        now = time.perf_counter()
        # 500ms of complete silence = device has gone away entirely
        tracking_lost = (now - last_tobii_timestamp) > 0.5

        if tracking_lost:
            return -1, -1, False

        # Trust coordinates: if x,y are within screen bounds the gaze is on screen
        within_bounds = (0 <= raw_x <= 1920) and (0 <= raw_y <= 1200)

        return (raw_x, raw_y, True) if within_bounds else (-1, -1, False)

    def update_gaze_durations(self, timestamp_ms: int, gaze_on_screen: bool):
        """Accumulate on/off-screen time using delta-time method."""
        if self.last_timestamp is None:
            self.last_timestamp = timestamp_ms
            self.last_gaze_state = gaze_on_screen
            return

        dt = timestamp_ms - self.last_timestamp
        if dt < 0 or dt > 2000:
            self.last_timestamp = timestamp_ms
            self.last_gaze_state = gaze_on_screen
            return

        if self.last_gaze_state is True:
            self.total_on_screen_time += dt
        elif self.last_gaze_state is False:
            self.total_off_screen_time += dt

        self.last_timestamp = timestamp_ms
        self.last_gaze_state = gaze_on_screen

    def refresh_gaze_time_labels(self):
        total_time = self.total_on_screen_time + self.total_off_screen_time
        if total_time > 0:
            on_pct = self.total_on_screen_time / total_time * 100
            off_pct = self.total_off_screen_time / total_time * 100
        else:
            on_pct = off_pct = 0.0
        try:
            self.gaze_on_screen_label.config(
                text=f"On-screen time: {self._format_ms(self.total_on_screen_time)}")
            self.gaze_off_screen_label.config(
                text=f"Off-screen time: {self._format_ms(self.total_off_screen_time)}")
            self.gaze_percentage_label.config(
                text=f"On-screen {on_pct:.1f}% | Off-screen {off_pct:.1f}%")
        except Exception:
            pass

    def reset_gaze_timers(self):
        self.total_on_screen_time = 0
        self.total_off_screen_time = 0
        self.last_timestamp = None
        self.last_gaze_state = None
        self.gaze_on_screen = None
        self.refresh_gaze_time_labels()

    def _compute_gaze_durations_from_csv(self, data: pd.DataFrame):
        """Reproduce live dt-based duration computation on saved CSV data."""
        if len(data) < 2:
            return 0, 0, 0
        total_on = total_off = 0
        timestamps = data['epoch_ms'].values
        on_screen = data['on_screen'].values
        for i in range(1, len(data)):
            dt = timestamps[i] - timestamps[i - 1]
            if dt < 0 or dt > 2000:
                continue
            if on_screen[i - 1]:
                total_on += dt
            else:
                total_off += dt
        return total_on, total_off, total_on + total_off

    def _export_session_summary(self, csv_filepath: str,
                                sample_count: int) -> str | None:
        """Write a JSON session summary alongside the CSV."""
        try:
            import json
            total_ms = self.total_on_screen_time + self.total_off_screen_time
            total_sec = total_ms / 1000.0
            avg_fps = sample_count / total_sec if total_sec > 0 else 0

            summary = {
                "session_metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "gaze_csv_path": csv_filepath,
                    "sample_count": sample_count,
                },
                "gaze_duration_analysis": {
                    "total_on_screen_time_ms": self.total_on_screen_time,
                    "total_off_screen_time_ms": self.total_off_screen_time,
                    "total_session_duration_ms": total_ms,
                    "on_screen_percentage": (
                        self.total_on_screen_time / total_ms * 100
                        if total_ms > 0 else 0),
                    "off_screen_percentage": (
                        self.total_off_screen_time / total_ms * 100
                        if total_ms > 0 else 0),
                },
                "timing_metrics": {
                    "total_duration_seconds": total_sec,
                    "average_fps": round(avg_fps, 2),
                    "formatted_total_duration": self._format_ms(total_ms),
                    "formatted_on_screen_time": self._format_ms(
                        self.total_on_screen_time),
                    "formatted_off_screen_time": self._format_ms(
                        self.total_off_screen_time),
                },
            }

            summary_path = os.path.join(
                os.path.dirname(csv_filepath),
                f"session_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            return summary_path
        except Exception as e:
            print(f"[WARN] Failed to export session summary: {e}")
            return None

    # -----------------------------------------------------------------------
    # Analysis tab — session folder analysis
    # -----------------------------------------------------------------------
    def _browse_session_folder(self):
        """Let the user pick a session recording folder."""
        # Default to the recordings/ folder so the user sees session folders
        # directly, not the analysis subfolders inside them
        here = os.path.dirname(os.path.abspath(__file__))
        recordings_dir = os.path.normpath(os.path.join(here, '..', 'recordings'))
        if not os.path.isdir(recordings_dir):
            recordings_dir = here

        folder = filedialog.askdirectory(
            title="Select Session Folder (e.g. both_20260306_110143)",
            initialdir=recordings_dir)
        if folder:
            # Guard against accidentally selecting the analysis subfolder
            if os.path.basename(folder) == 'analysis':
                folder = os.path.dirname(folder)
            self._session_folder_var.set(folder)
            self._session_status_label.config(
                text=f"Folder selected: {os.path.basename(folder)}",
                foreground='black')

    def _use_last_session(self):
        """Auto-fill the most recent session folder.

        Priority:
          1. _last_session_folder set by MainController during this app run
          2. Most recently modified folder inside recordings/
        """
        # Priority 1: session recorded in this app run
        last = getattr(self.root, '_last_session_folder', None)
        if last and os.path.isdir(last):
            self._session_folder_var.set(last)
            self._session_status_label.config(
                text=f"Using last session: {os.path.basename(last)}",
                foreground='black')
            return

        # Priority 2: scan recordings/ folder for the most recent session
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            recordings_dir = os.path.normpath(
                os.path.join(here, '..', 'recordings'))

            if os.path.isdir(recordings_dir):
                # Get all session subfolders (must contain tobii_gaze.csv)
                sessions = [
                    os.path.join(recordings_dir, d)
                    for d in os.listdir(recordings_dir)
                    if os.path.isdir(os.path.join(recordings_dir, d))
                    and os.path.exists(
                        os.path.join(recordings_dir, d, 'tobii_gaze.csv'))
                ]
                if sessions:
                    # Sort by modification time, pick the newest
                    latest = max(sessions, key=os.path.getmtime)
                    self._session_folder_var.set(latest)
                    self._session_status_label.config(
                        text=f"Latest session: {os.path.basename(latest)}",
                        foreground='black')
                    return

            self._session_status_label.config(
                text="No sessions found in recordings/ — use Browse instead.",
                foreground='gray')
        except Exception as e:
            self._session_status_label.config(
                text=f"Could not find sessions: {e}",
                foreground='gray')

    def _run_session_analysis(self):
        """Run the full engagement analysis in a background thread."""
        folder = self._session_folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("Warning",
                                   "Please select a valid session folder first.")
            return

        # Check required files exist
        required = ['tobii_gaze.csv', 'upper_body.csv']
        missing = [f for f in required
                   if not os.path.exists(os.path.join(folder, f))]
        if missing:
            messagebox.showerror(
                "Missing Files",
                f"Required files not found in folder:\n" + "\n".join(missing))
            return

        self._session_status_label.config(
            text="⏳ Running analysis… please wait.",
            foreground='#e67e22')
        self.root.update_idletasks()

        # Always resolve to absolute path before passing to background thread
        abs_folder = os.path.abspath(folder)

        def _worker():
            try:
                import importlib.util, traceback as _tb

                # GazeAppAlpha.py is in <project>/Gaze APP Python/
                # engagement_analysis.py is in <project>/analysis/
                here = os.path.dirname(os.path.abspath(__file__))
                script = os.path.normpath(
                    os.path.join(here, '..', 'analysis',
                                 'engagement_analysis.py'))

                if not os.path.exists(script):
                    raise FileNotFoundError(
                        f"engagement_analysis.py not found at:\n{script}\n\n"
                        "Make sure analysis/engagement_analysis.py exists in "
                        "your project root.")

                # Use unique module name to avoid importlib caching conflicts
                spec = importlib.util.spec_from_file_location(
                    "engagement_analysis_mod", script)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)

                summary, merged = mod.run_analysis(abs_folder)

                self.root.after(0, lambda: self._on_analysis_done(
                    abs_folder, summary, merged))

            except Exception:
                err = __import__('traceback').format_exc()
                self.root.after(0, lambda: self._on_analysis_error(err))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_analysis_done(self, folder: str, summary: dict, merged):
        """Called on main thread when analysis finishes."""
        self._session_status_label.config(
            text="✅ Analysis complete — outputs saved to analysis/ subfolder.",
            foreground='#27ae60')

        # Show summary stats in the results frame
        self._clear_results()

        # Title
        ttk.Label(self.results_frame,
                  text=f"Session: {os.path.basename(folder)}",
                  font=('Helvetica', 12, 'bold')).pack(pady=(10, 2))
        ttk.Label(self.results_frame,
                  text=f"Duration: {summary['duration_s']:.1f}s  |  "
                       f"{summary['total_frames']} frames  |  "
                       f"{summary['effective_fps']} FPS",
                  foreground='gray').pack(pady=(0, 10))

        # Stats grid
        stats_frame = ttk.Frame(self.results_frame)
        stats_frame.pack(padx=20, pady=4)

        rows = [
            ("👁  Gaze on-screen",    f"{summary['gaze_on_screen_pct']:.1f}%"),
            ("📏  Distance OK",        f"{summary['distance_ok_pct']:.1f}%"),
            ("🔄  Facing forward",     f"{summary['facing_forward_pct']:.1f}%"),
            ("🤸  Body engaged",       f"{summary['body_engaged_pct']:.1f}%"),
            ("",                       ""),
            ("🎯  Overall ENGAGED",   f"{summary['engaged_pct']:.1f}%"),
            ("⚠   Overall DISENGAGED",f"{summary['disengaged_pct']:.1f}%"),
            ("",                       ""),
            ("Disengagement events",  str(summary['disengagement_events'])),
            ("Avg disengagement dur", f"{summary['avg_disengagement_dur_s']:.2f}s"),
            ("",                       ""),
            ("📉 Mean engagement score",
             f"{summary.get('mean_engagement_score', '—')}"),
            ("   Min score",
             f"{summary.get('min_engagement_score', '—')}"),
            ("   Max score",
             f"{summary.get('max_engagement_score', '—')}"),
        ]

        for i, (label, value) in enumerate(rows):
            if not label:
                ttk.Separator(stats_frame, orient='horizontal').grid(
                    row=i, column=0, columnspan=2, sticky='ew', pady=4)
                continue
            colour = ('#27ae60' if 'ENGAGED' in label and 'DIS' not in label
                      else '#c0392b' if 'DISENGAGED' in label else 'black')
            ttk.Label(stats_frame, text=label,
                      font=('Helvetica', 10, 'bold')).grid(
                row=i, column=0, sticky='w', padx=(0, 20), pady=2)
            ttk.Label(stats_frame, text=value,
                      font=('Helvetica', 10), foreground=colour).grid(
                row=i, column=1, sticky='w', pady=2)

        # Quick-view: show engagement score inline after results
        self._show_analysis_image('engagement_score.png', clear=False)

    def _on_analysis_error(self, error: str):
        """Called on main thread when analysis raises an exception."""
        self._session_status_label.config(
            text=f"❌ Analysis failed — see console for details.",
            foreground='#c0392b')
        messagebox.showerror("Analysis Error", error)

    def _show_analysis_image(self, filename: str, clear: bool = True):
        """Display one of the saved analysis PNG files in the results frame."""
        folder = self._session_folder_var.get().strip()
        if not folder:
            messagebox.showwarning("Warning", "No session folder selected.")
            return

        img_path = os.path.join(folder, 'analysis', filename)
        if not os.path.exists(img_path):
            messagebox.showwarning(
                "Not Found",
                f"Image not found:\n{img_path}\n\nRun Full Analysis first.")
            return

        try:
            from PIL import Image, ImageTk
            if clear:
                self._clear_results()

            pil_img = Image.open(img_path)

            # Scale image to fill the available results frame width.
            # We read the actual frame width after layout; fall back to 1200
            # if the frame hasn't been rendered yet.
            self.results_frame.update_idletasks()
            frame_w = self.results_frame.winfo_width()
            if frame_w < 100:
                frame_w = 1200

            # Scale proportionally to frame width, no fixed height cap
            orig_w, orig_h = pil_img.size
            scale = frame_w / orig_w
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

            tk_img = ImageTk.PhotoImage(pil_img)
            label = ttk.Label(self.results_frame, image=tk_img)
            label.image = tk_img   # keep reference to prevent GC
            label.pack(pady=10, fill='x')

        except ImportError:
            # Pillow not installed — open in the OS default image viewer instead
            import subprocess, platform
            if platform.system() == 'Windows':
                os.startfile(img_path)
            elif platform.system() == 'Darwin':
                subprocess.call(['open', img_path])
            else:
                subprocess.call(['xdg-open', img_path])

            if clear:
                self._clear_results()
            ttk.Label(self.results_frame,
                      text=f"Image opened in external viewer:\n{img_path}\n\n"
                           "(Install Pillow for inline display: pip install Pillow)",
                      foreground='gray').pack(pady=20)

    # -----------------------------------------------------------------------
    # Analysis tab — downloads
    # -----------------------------------------------------------------------
    def _download_session_xlsx(self):
        """Export session data as a teacher-friendly Excel workbook."""
        folder = self._session_folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("Warning",
                                   "Please select a session folder first.")
            return

        # Check analysis has been run
        if not os.path.exists(os.path.join(folder, 'analysis',
                                            'session_summary.json')):
            messagebox.showwarning(
                "Not Analysed",
                "Please run Full Analysis on this session first.")
            return

        # Ask user where to save
        session_label = os.path.basename(folder)
        default_name  = f"ALPER_Report_{session_label}.xlsx"
        save_path = filedialog.asksaveasfilename(
            title="Save Session Report As",
            defaultextension=".xlsx",
            initialfile=default_name,
            filetypes=[("Excel Workbook", "*.xlsx")])

        if not save_path:
            return

        self._session_status_label.config(
            text="⏳ Generating Excel report…", foreground='#e67e22')
        self.root.update_idletasks()

        def _worker():
            try:
                import importlib.util
                here   = os.path.dirname(os.path.abspath(__file__))
                script = os.path.normpath(
                    os.path.join(here, '..', 'analysis', 'session_exporter.py'))

                if not os.path.exists(script):
                    raise FileNotFoundError(
                        "session_exporter.py not found at: " + script)

                spec = importlib.util.spec_from_file_location(
                    "session_exporter_mod", script)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)

                abs_folder = os.path.abspath(folder)
                out_path   = mod.export_session_xlsx(abs_folder, save_path)

                self.root.after(0, lambda: self._on_xlsx_done(out_path))
            except Exception:
                err = __import__('traceback').format_exc()
                self.root.after(0, lambda: self._on_xlsx_error(err))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_xlsx_done(self, path: str):
        self._session_status_label.config(
            text="✅ Excel report saved successfully.",
            foreground='#27ae60')
        if messagebox.askyesno("Download Complete",
                               "Report saved to: " + path + "\n\nOpen it now?"):
            import subprocess, platform
            if platform.system() == 'Windows':
                os.startfile(path)
            elif platform.system() == 'Darwin':
                subprocess.call(['open', path])
            else:
                subprocess.call(['xdg-open', path])

    def _on_xlsx_error(self, error: str):
        self._session_status_label.config(
            text="❌ Export failed.", foreground='#c0392b')
        messagebox.showerror("Export Error", error)

    # -----------------------------------------------------------------------
    # Analysis tab — multi-session comparison
    # -----------------------------------------------------------------------
    def _add_comparison_session(self):
        """Add a session folder to the comparison list."""
        here = os.path.dirname(os.path.abspath(__file__))
        recordings_dir = os.path.normpath(
            os.path.join(here, '..', 'recordings'))
        if not os.path.isdir(recordings_dir):
            recordings_dir = here

        folder = filedialog.askdirectory(
            title="Add Session to Comparison",
            initialdir=recordings_dir)
        if not folder:
            return

        # Guard against analysis subfolder
        if os.path.basename(folder) == 'analysis':
            folder = os.path.dirname(folder)

        folder = os.path.abspath(folder)

        # Check it has been analysed already
        summary_path = os.path.join(folder, 'analysis', 'session_summary.json')
        if not os.path.exists(summary_path):
            messagebox.showwarning(
                "Not Analysed",
                "No analysis found for: " + os.path.basename(folder) + "\n\n"
                "Run Full Analysis on this session first.")
            return

        # Prevent duplicates
        if folder in self._comparison_folders:
            messagebox.showinfo("Already Added",
                                f"{os.path.basename(folder)} is already in the list.")
            return

        self._comparison_folders.append(folder)
        self._comparison_listbox.insert('end', os.path.basename(folder))
        n = len(self._comparison_folders)
        self._comparison_status_label.config(
            text=f"{n} session(s) added.  "
                 f"{'Add at least one more to compare.' if n < 2 else 'Ready to compare.'}",
            foreground='black' if n >= 2 else 'gray')

    def _clear_comparison_sessions(self):
        """Clear the comparison session list."""
        self._comparison_folders.clear()
        self._comparison_listbox.delete(0, 'end')
        self._comparison_output_dir = None
        self._comparison_status_label.config(
            text="Add at least 2 analysed sessions to compare.",
            foreground='gray')

    def _run_comparison(self):
        """Run multi-session comparison in a background thread."""
        if len(self._comparison_folders) < 2:
            messagebox.showwarning(
                "Not Enough Sessions",
                "Please add at least 2 sessions to compare.")
            return

        self._comparison_status_label.config(
            text="⏳ Running comparison… please wait.",
            foreground='#e67e22')
        self.root.update_idletasks()

        folders = list(self._comparison_folders)

        def _worker():
            try:
                import importlib.util, traceback as _tb

                here = os.path.dirname(os.path.abspath(__file__))
                script = os.path.normpath(
                    os.path.join(here, '..', 'analysis',
                                 'multi_session_comparison.py'))

                if not os.path.exists(script):
                    raise FileNotFoundError(
                        f"multi_session_comparison.py not found at: {script}")

                spec = importlib.util.spec_from_file_location(
                    "multi_session_comparison_mod", script)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)

                # Save comparison output next to the recordings folder
                parent = os.path.dirname(folders[0])
                out_dir = os.path.join(parent, 'comparison')

                sessions, out_dir = mod.run_comparison(folders, out_dir)
                self.root.after(0, lambda: self._on_comparison_done(
                    sessions, out_dir))

            except Exception:
                err = __import__('traceback').format_exc()
                self.root.after(0, lambda: self._on_comparison_error(err))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_comparison_done(self, sessions: list, out_dir: str):
        """Called on main thread when comparison finishes."""
        self._comparison_output_dir = out_dir
        n = len(sessions)
        self._comparison_status_label.config(
            text=f"✅ Comparison complete — {n} sessions — outputs in comparison/",
            foreground='#27ae60')

        # Show comparison table in results frame
        self._clear_results()

        ttk.Label(self.results_frame,
                  text=f"Session Comparison — {n} Sessions",
                  font=('Helvetica', 12, 'bold')).pack(pady=(10, 6))

        # Stats table
        tbl = ttk.Frame(self.results_frame)
        tbl.pack(padx=20, pady=4)

        headers = ['Session', 'Score', 'Engaged%', 'Gaze%', 'Facing%',
                   'Dist%', 'Events']
        for col, h in enumerate(headers):
            ttk.Label(tbl, text=h, font=('Helvetica', 9, 'bold'),
                      width=12, anchor='center').grid(
                row=0, column=col, padx=4, pady=2)

        for row_i, s in enumerate(sessions, start=1):
            sm = s['summary']
            vals = [
                s['label'][:14],
                f"{sm.get('mean_engagement_score','—')}",
                f"{sm.get('engaged_pct','—')}%",
                f"{sm.get('gaze_on_screen_pct','—')}%",
                f"{sm.get('facing_forward_pct','—')}%",
                f"{sm.get('distance_ok_pct','—')}%",
                str(sm.get('disengagement_events','—')),
            ]
            bg = '#f0fff4' if row_i % 2 == 0 else 'white'
            for col, val in enumerate(vals):
                ttk.Label(tbl, text=val, width=12,
                          anchor='center').grid(
                    row=row_i, column=col, padx=4, pady=1)

        # Auto-show score bars plot
        self._show_comparison_image('comparison_scores.png', clear=False)

    def _on_comparison_error(self, error: str):
        self._comparison_status_label.config(
            text="❌ Comparison failed — see console for details.",
            foreground='#c0392b')
        messagebox.showerror("Comparison Error", error)

    def _show_comparison_image(self, filename: str, clear: bool = True):
        """Display a comparison plot in the results frame."""
        if not self._comparison_output_dir:
            messagebox.showwarning("Warning",
                                   "Run comparison first.")
            return
        img_path = os.path.join(self._comparison_output_dir, filename)
        if not os.path.exists(img_path):
            messagebox.showwarning("Not Found",
                                   "Image not found: " + img_path + "\n\n"
                                   "Run comparison first.")
            return
        # Reuse the same image display logic as single-session
        self._session_folder_var.set('')   # temp blank so _show_analysis_image
                                           # won't complain about no folder
        try:
            from PIL import Image, ImageTk
            if clear:
                self._clear_results()

            pil_img = Image.open(img_path)
            self.results_frame.update_idletasks()
            frame_w = self.results_frame.winfo_width()
            if frame_w < 100:
                frame_w = 1200
            orig_w, orig_h = pil_img.size
            scale = frame_w / orig_w
            pil_img = pil_img.resize(
                (int(orig_w * scale), int(orig_h * scale)), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(pil_img)
            lbl = ttk.Label(self.results_frame, image=tk_img)
            lbl.image = tk_img
            lbl.pack(pady=10, fill='x')
        except ImportError:
            import subprocess, platform
            if platform.system() == 'Windows':
                os.startfile(img_path)
            elif platform.system() == 'Darwin':
                subprocess.call(['open', img_path])
            else:
                subprocess.call(['xdg-open', img_path])

    # -----------------------------------------------------------------------
    # Analysis tab — data loading and visualisation
    # -----------------------------------------------------------------------
    def load_csv(self):
        filename = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv")],
            title="Select a gaze data CSV file")
        if filename:
            try:
                self.data = pd.read_csv(filename)
                if 'on_screen' in self.data.columns:
                    self.data['on_screen'] = (
                        self.data['on_screen'].astype(str).str.lower() == 'true')
                else:
                    self.data['on_screen'] = (
                        (self.data['x'] >= 0) & (self.data['y'] >= 0) &
                        (self.data['x'] <= 1920) & (self.data['y'] <= 1200))
                self.file_label.config(
                    text=f"📊 Loaded: {os.path.basename(filename)}")
                messagebox.showinfo("Success", "File loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")

    def _clear_results(self):
        """Remove all widgets from the results frame."""
        for widget in self.results_frame.winfo_children():
            widget.destroy()

    def generate_heatmap(self):
        if not hasattr(self, 'data'):
            messagebox.showwarning("Warning", "Please load a CSV file first")
            return

        self._clear_results()

        # FIX 7: close only the specific figure, not all matplotlib figures
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.style.use('seaborn-v0_8')

        valid = self.data[self.data['on_screen'] == True]
        heatmap_data, xedges, yedges = np.histogram2d(
            valid['x'], valid['y'], bins=50,
            range=[[0, 1920], [0, 1200]])

        try:
            sigma = float(self.heatmap_sigma_var.get())
        except Exception:
            sigma = 2.0
        smoothed = gaussian_filter(heatmap_data, sigma=sigma)

        min_positive = (smoothed[smoothed > 0].min()
                        if np.any(smoothed > 0) else 1e-3)
        vmin = max(min_positive, 1e-3)
        vmax = smoothed.max() if smoothed.size > 0 else 1

        img = ax.imshow(smoothed.T, origin='lower', cmap='turbo',
                        extent=[0, 1920, 0, 1200], aspect='auto',
                        norm=LogNorm(vmin=vmin, vmax=vmax))
        fig.colorbar(img, ax=ax, label='Gaze Density')
        ax.set_title("Gaze Heatmap Analysis", pad=20)
        ax.set_xlabel("Screen X")
        ax.set_ylabel("Screen Y")
        ax.set_xlim(0, 1920)
        ax.set_ylim(1200, 0)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.results_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

        # FIX 7: close this specific figure after embedding to free memory
        plt.close(fig)

    def generate_SpaceMap(self):
        if not hasattr(self, 'data'):
            messagebox.showwarning("Warning", "Please load a CSV file first")
            return

        self._clear_results()
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.style.use('seaborn-v0_8-whitegrid')

        ax.scatter(np.clip(self.data['x'], 0, 1920),
                   np.clip(self.data['y'], 0, 1200),
                   alpha=0.5, s=20, c='red', label='Gaze Points')
        ax.set_xlim(0, 1920)
        ax.set_ylim(1200, 0)
        ax.set_xlabel("Screen X")
        ax.set_ylabel("Screen Y")
        ax.set_title("Gaze Space Map", pad=20)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.98, f'Total Samples: {len(self.data)}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.results_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        plt.close(fig)  # FIX 7

    def analyze_attention(self):
        if not hasattr(self, 'data'):
            messagebox.showwarning("Warning", "Please load a CSV file first")
            return

        self._clear_results()

        self.data['movement'] = np.sqrt(
            self.data['x'].diff() ** 2 + self.data['y'].diff() ** 2)

        total_on_ms, total_off_ms, total_time_ms = (
            self._compute_gaze_durations_from_csv(self.data))
        total_sec = total_time_ms / 1000.0

        movement_threshold = 50
        reading_threshold_min = 5
        total_no_movement = total_reading = total_scanning = 0

        timestamps = self.data['epoch_ms'].values
        on_screen = self.data['on_screen'].values
        movement = self.data['movement'].values

        for i in range(1, len(self.data)):
            dt = timestamps[i] - timestamps[i - 1]
            if dt < 0 or dt > 2000:
                continue
            if on_screen[i - 1]:
                mov = movement[i]
                if mov < reading_threshold_min:
                    total_no_movement += dt
                elif mov < movement_threshold:
                    total_reading += dt
                else:
                    total_scanning += dt

        no_movement_sec = total_no_movement / 1000.0
        reading_sec = total_reading / 1000.0
        scanning_sec = total_scanning / 1000.0
        on_screen_sec = total_on_ms / 1000.0

        no_movement_pct = no_movement_sec / total_sec * 100 if total_sec > 0 else 0
        reading_pct = reading_sec / total_sec * 100 if total_sec > 0 else 0
        scanning_pct = scanning_sec / total_sec * 100 if total_sec > 0 else 0

        fig, ax = plt.subplots(figsize=(12, 6))
        plt.style.use('seaborn-v0_8')
        time_seconds = (self.data['epoch_ms'] - self.data['epoch_ms'].min()) / 1000

        ax.plot(time_seconds, self.data['movement'], 'b-', alpha=0.5,
                label='Eye Movement')
        ax.axhline(y=movement_threshold, color='r', linestyle='--',
                   label='High Movement Threshold')
        ax.axhline(y=reading_threshold_min, color='g', linestyle='--',
                   label='Reading Movement Threshold')
        ax.fill_between(time_seconds, 0, reading_threshold_min,
                        where=self.data['movement'] < reading_threshold_min,
                        color='red', alpha=0.2, label='No Movement')
        ax.fill_between(time_seconds, reading_threshold_min, movement_threshold,
                        where=self.data['movement'].between(
                            reading_threshold_min, movement_threshold),
                        color='green', alpha=0.2, label='Reading Movement')
        ax.fill_between(time_seconds, movement_threshold,
                        self.data['movement'].max(),
                        where=self.data['movement'] >= movement_threshold,
                        color='blue', alpha=0.2, label='Scanning Movement')
        ax.set_title("Eye Movement Timeline", pad=20)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Movement (pixels)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        stats_text = (
            f"\n        📊 Attention Analysis\n"
            f"        ════════════════════════════════\n\n"
            f"        🕒 Total Duration: {self._format_ms(total_time_ms)}\n"
            f"        👁 On-Screen: {self._format_ms(total_on_ms)} "
            f"({on_screen_sec / total_sec * 100:.1f}% of total)\n\n"
            f"        Movement Classification:\n"
            f"        🛑 No Movement (<{reading_threshold_min} px): "
            f"{self._format_ms(total_no_movement)} ({no_movement_pct:.1f}%)\n"
            f"        📖 Reading ({reading_threshold_min}–{movement_threshold} px): "
            f"{self._format_ms(total_reading)} ({reading_pct:.1f}%)\n"
            f"        🔍 Scanning (≥{movement_threshold} px): "
            f"{self._format_ms(total_scanning)} ({scanning_pct:.1f}%)\n"
            f"        Total samples: {len(self.data)}\n"
        )

        stats_label = ttk.Label(self.results_frame, text=stats_text,
                                style='Status.TLabel', justify='left')
        stats_label.pack(fill='x', padx=20, pady=10)

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.results_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        plt.close(fig)  # FIX 7

    def analyze_presence(self):
        if not hasattr(self, 'data'):
            messagebox.showwarning("Warning", "Please load a CSV file first")
            return

        total_on_ms, total_off_ms, total_time_ms = (
            self._compute_gaze_durations_from_csv(self.data))
        total_sec = total_time_ms / 1000.0
        presence_pct = (total_on_ms / total_time_ms * 100
                        if total_time_ms > 0 else 0)

        result_text = (
            f"\n        👤 Presence Analysis Results\n"
            f"        ═══════════════════════════\n\n"
            f"        🕒 Total Session Duration: {self._format_ms(total_time_ms)}\n"
            f"        ✓  Time Present at Screen: {self._format_ms(total_on_ms)}\n"
            f"        ✗  Time Off-Screen: {self._format_ms(total_off_ms)}\n"
            f"        📊 Presence Percentage: {presence_pct:.1f}%\n\n"
            f"        Analysis Parameters:\n"
            f"        • Screen bounds: 1920×1200\n"
            f"        • Total samples: {len(self.data)}\n"
        )

        self._clear_results()
        ttk.Label(self.results_frame, text=result_text,
                  style='Status.TLabel', justify='left').pack(padx=20, pady=20)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = GazeAnalysisApp(root)

    def on_closing():
        try:
            app.shutdown()
        except Exception:
            pass
        try:
            if root.winfo_exists():
                # Cancel any remaining Tcl after callbacks
                try:
                    info = root.tk.call('after', 'info')
                    ids = info.split() if isinstance(info, str) else list(info)
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
        except SystemExit:
            pass

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()