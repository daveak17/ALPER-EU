import tkinter as tk
from tkinter import ttk
import time
import os
from datetime import datetime
import csv
import sys
import importlib.util

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def epoch_ms() -> int:
    """Return current time in milliseconds since epoch."""
    return time.time_ns() // 1_000_000


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

sys.path.insert(0, os.path.join(script_dir, 'Gaze APP Python'))
from GazeAppAlpha import GazeAnalysisApp

body_tracker_path = os.path.join(script_dir, 'RealSenseBodyTracker', 'body-tracker.py')
spec = importlib.util.spec_from_file_location("body_tracker", body_tracker_path)
body_tracker_module = importlib.util.module_from_spec(spec)
sys.modules["body_tracker"] = body_tracker_module
spec.loader.exec_module(body_tracker_module)
BodyTrackerEngine = body_tracker_module.BodyTrackerEngine


# ---------------------------------------------------------------------------
# MainController — single unified window
# ---------------------------------------------------------------------------
class MainController:
    """Single Tk window with three notebook tabs:
        1. Session Control  — start/stop sensors, countdown, status
        2. Live Tracking    — gaze visualisation and live metrics (from GazeAnalysisApp)
        3. Data Analysis    — heatmaps, attention, presence (from GazeAnalysisApp)

    The OpenCV body-tracker preview window is a separate OS window because
    OpenCV cannot be embedded inside Tkinter — this is unavoidable with the
    RealSense SDK on Windows.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ALPER-EU — Multimodal Engagement Recorder")
        self.root.geometry("1050x820")
        self.root.minsize(900, 700)

        self.body_engine = BodyTrackerEngine()
        self.countdown_var = tk.IntVar(value=10)
        self.session_folder: str | None = None

        # GazeAnalysisApp instance — created after notebook exists
        self.gaze_app: GazeAnalysisApp | None = None

        # Engagement poll after-ID — initialised before _build_ui so the
        # panel can reference it; actual widget is created inside _build_session_tab
        self._engagement_poll_id = None

        self._build_ui()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------
    def _build_ui(self):
        # Top-level notebook — three tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Tab 1: Session Control (built here in MainController)
        self.session_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.session_tab, text='  🎬  Session Control  ')

        # Tabs 2 & 3 are frames handed to GazeAnalysisApp
        self.tracking_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.tracking_tab, text='  👁  Live Tracking  ')

        self.analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_tab, text='  📊  Data Analysis  ')

        self._build_session_tab()

        # Attach GazeAnalysisApp into the two existing tab frames —
        # no Toplevel created, no separate window.
        self.gaze_app = GazeAnalysisApp.attach_to_tabs(
            root=self.root,
            tracking_frame=self.tracking_tab,
            analysis_frame=self.analysis_tab,
        )

    def _build_session_tab(self):
        outer = ttk.Frame(self.session_tab, padding=16)
        outer.pack(fill='both', expand=True)

        # ---- Title ----
        ttk.Label(outer, text="ALPER-EU  ·  Multimodal Engagement Recorder",
                  font=('Helvetica', 14, 'bold')).grid(
            row=0, column=0, columnspan=2, pady=(0, 16), sticky='w')

        # ---- Countdown ----
        cd_frame = ttk.LabelFrame(outer, text="Recording Start Countdown")
        cd_frame.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(0, 10))
        ttk.Label(cd_frame, text="Countdown seconds:").pack(side='left', padx=(16, 6), pady=10)
        ttk.Spinbox(cd_frame, from_=0, to=60,
                    textvariable=self.countdown_var, width=5).pack(side='left', pady=10)
        ttk.Label(cd_frame,
                  text="(sensors warm up and preview starts immediately; "
                       "recording begins after countdown)",
                  foreground='gray').pack(side='left', padx=(12, 16), pady=10)

        # ---- Start / Stop buttons ----
        btn_frame = ttk.LabelFrame(outer, text="Recording Controls")
        btn_frame.grid(row=2, column=0, columnspan=2, sticky='ew', pady=(0, 10))

        inner = ttk.Frame(btn_frame)
        inner.pack(padx=16, pady=12)

        # Both
        self.start_both_btn = ttk.Button(
            inner, text="▶  Start BOTH", width=18, command=self.start_both)
        self.start_both_btn.grid(row=0, column=0, padx=6, pady=4)
        self.stop_both_btn = ttk.Button(
            inner, text="⏹  Stop BOTH", width=18, command=self.stop_both,
            state='disabled')
        self.stop_both_btn.grid(row=0, column=1, padx=6, pady=4)

        ttk.Separator(inner, orient='vertical').grid(
            row=0, column=2, rowspan=3, padx=12, sticky='ns')

        # Eye only
        self.start_eye_btn = ttk.Button(
            inner, text="▶  Eye Only", width=16, command=self.start_eye_only)
        self.start_eye_btn.grid(row=0, column=3, padx=6, pady=4)
        self.stop_eye_btn = ttk.Button(
            inner, text="⏹  Stop Eye", width=16, command=self.stop_eye,
            state='disabled')
        self.stop_eye_btn.grid(row=0, column=4, padx=6, pady=4)

        # Body only
        self.start_body_btn = ttk.Button(
            inner, text="▶  Body Only", width=16, command=self.start_body_only)
        self.start_body_btn.grid(row=1, column=3, padx=6, pady=4)
        self.stop_body_btn = ttk.Button(
            inner, text="⏹  Stop Body", width=16, command=self.stop_body,
            state='disabled')
        self.stop_body_btn.grid(row=1, column=4, padx=6, pady=4)

        # ---- Status panel ----
        status_frame = ttk.LabelFrame(outer, text="Sensor Status")
        status_frame.grid(row=3, column=0, columnspan=2, sticky='ew', pady=(0, 10))

        sg = ttk.Frame(status_frame)
        sg.pack(padx=16, pady=10, fill='x')

        # Countdown display
        ttk.Label(sg, text="Countdown:", font=('Helvetica', 10, 'bold')).grid(
            row=0, column=0, sticky='w', padx=(0, 8))
        self.countdown_display = ttk.Label(sg, text="--",
                                           font=('Helvetica', 13, 'bold'),
                                           foreground='#c0392b')
        self.countdown_display.grid(row=0, column=1, sticky='w', pady=2)

        ttk.Label(sg, text="Session:", font=('Helvetica', 10, 'bold')).grid(
            row=1, column=0, sticky='w', padx=(0, 8))
        self.unified_status = ttk.Label(sg, text="Idle")
        self.unified_status.grid(row=1, column=1, sticky='w', pady=2)

        ttk.Label(sg, text="Eye tracker:", font=('Helvetica', 10, 'bold')).grid(
            row=2, column=0, sticky='w', padx=(0, 8))
        self.eye_status = ttk.Label(sg, text="Idle")
        self.eye_status.grid(row=2, column=1, sticky='w', pady=2)

        ttk.Label(sg, text="Body tracker:", font=('Helvetica', 10, 'bold')).grid(
            row=3, column=0, sticky='w', padx=(0, 8))
        self.body_status = ttk.Label(sg, text="Idle")
        self.body_status.grid(row=3, column=1, sticky='w', pady=2)

        # ---- Session folder ----
        folder_frame = ttk.LabelFrame(outer, text="Current Session")
        folder_frame.grid(row=4, column=0, columnspan=2, sticky='ew', pady=(0, 10))
        self.folder_label = ttk.Label(folder_frame, text="No session started yet",
                                      foreground='gray')
        self.folder_label.pack(padx=16, pady=8, anchor='w')

        # ---- Verify results (shown after stop) ----
        self.verify_frame = ttk.LabelFrame(outer, text="Last Recording — Verification")
        self.verify_frame.grid(row=5, column=0, columnspan=2, sticky='ew')
        self.verify_label = ttk.Label(self.verify_frame,
                                      text="Results will appear here after recording stops.",
                                      foreground='gray')
        self.verify_label.pack(padx=16, pady=8, anchor='w')

        # ---- Engagement Monitor ----
        eng_frame = ttk.LabelFrame(outer, text="Live Engagement Monitor")
        eng_frame.grid(row=6, column=0, columnspan=2, sticky='ew', pady=(10, 0))

        eg = ttk.Frame(eng_frame)
        eg.pack(padx=16, pady=12, fill='x')

        # Three individual signal indicators
        ttk.Label(eg, text="👁  Gaze on screen:",
                  font=('Helvetica', 10, 'bold')).grid(row=0, column=0, sticky='w', padx=(0, 10))
        self.ind_gaze = ttk.Label(eg, text="—", width=18,
                                   font=('Helvetica', 10, 'bold'), foreground='gray',
                                   relief='solid', anchor='center')
        self.ind_gaze.grid(row=0, column=1, sticky='w', pady=3)

        ttk.Label(eg, text="📏  Within distance:",
                  font=('Helvetica', 10, 'bold')).grid(row=1, column=0, sticky='w', padx=(0, 10))
        self.ind_dist = ttk.Label(eg, text="—", width=18,
                                   font=('Helvetica', 10, 'bold'), foreground='gray',
                                   relief='solid', anchor='center')
        self.ind_dist.grid(row=1, column=1, sticky='w', pady=3)

        ttk.Label(eg, text="🔄  Facing forward:",
                  font=('Helvetica', 10, 'bold')).grid(row=2, column=0, sticky='w', padx=(0, 10))
        self.ind_face = ttk.Label(eg, text="—", width=18,
                                   font=('Helvetica', 10, 'bold'), foreground='gray',
                                   relief='solid', anchor='center')
        self.ind_face.grid(row=2, column=1, sticky='w', pady=3)

        ttk.Separator(eg, orient='horizontal').grid(
            row=3, column=0, columnspan=3, sticky='ew', pady=8)

        # Combined engagement verdict — large and prominent
        ttk.Label(eg, text="🎯  Student engaged:",
                  font=('Helvetica', 12, 'bold')).grid(row=4, column=0, sticky='w', padx=(0, 10))
        self.ind_engaged = ttk.Label(eg, text="NOT MONITORING", width=22,
                                      font=('Helvetica', 12, 'bold'), foreground='gray',
                                      relief='solid', anchor='center')
        self.ind_engaged.grid(row=4, column=1, sticky='w', pady=3)

        ttk.Label(eg,
                  text="(Disengaged only if ALL THREE signals are False simultaneously)",
                  foreground='gray', font=('Helvetica', 8)).grid(
            row=5, column=0, columnspan=3, sticky='w', pady=(4, 0))

        outer.columnconfigure(0, weight=1)

        # Engagement polling is started/stopped with recording (see _start/stop_engagement_poll)

    # -----------------------------------------------------------------------
    # Button state management
    # -----------------------------------------------------------------------
    def _set_buttons_recording(self, mode: str):
        for btn in (self.start_both_btn, self.start_eye_btn, self.start_body_btn):
            btn.config(state='disabled')
        for btn in (self.stop_both_btn, self.stop_eye_btn, self.stop_body_btn):
            btn.config(state='disabled')
        {
            'both': self.stop_both_btn,
            'eye':  self.stop_eye_btn,
            'body': self.stop_body_btn,
        }[mode].config(state='normal')

    def _set_buttons_idle(self):
        for btn in (self.start_both_btn, self.start_eye_btn, self.start_body_btn):
            btn.config(state='normal')
        for btn in (self.stop_both_btn, self.stop_eye_btn, self.stop_body_btn):
            btn.config(state='disabled')

    # -----------------------------------------------------------------------
    # Countdown helper
    # -----------------------------------------------------------------------
    def _run_countdown(self, remaining: int, on_finish):
        """Tick the countdown display and call on_finish() when done."""
        if remaining <= 0:
            self.countdown_display.config(text="▶ Recording", foreground='#27ae60')
            on_finish()
            return
        self.countdown_display.config(
            text=f"{remaining}s", foreground='#c0392b')
        self.root.after(1000, lambda: self._run_countdown(remaining - 1, on_finish))

    # -----------------------------------------------------------------------
    # Session folder
    # -----------------------------------------------------------------------
    def _make_session_folder(self, mode: str) -> str:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        folder = os.path.join('recordings', f"{mode}_{ts}")
        os.makedirs(folder, exist_ok=True)
        self.session_folder = folder
        self.folder_label.config(
            text=folder, foreground='black')
        return folder

    # -----------------------------------------------------------------------
    # Start / Stop — BOTH
    # -----------------------------------------------------------------------
    def start_both(self):
        secs = int(self.countdown_var.get())
        folder = self._make_session_folder('both')
        go_epoch_ms = epoch_ms() + secs * 1000

        self._set_buttons_recording('both')
        self.unified_status.config(text='Starting up sensors…')

        # Start preview immediately so the user can see the sensors are live
        try:
            self.gaze_app.start_streaming_preview()
            self.eye_status.config(text='Previewing (waiting for countdown)')
        except Exception as e:
            print(f'[ERROR] Gaze preview: {e}')
        try:
            self.body_engine.start_preview()
            self.body_status.config(text='Previewing (waiting for countdown)')
        except Exception as e:
            print(f'[ERROR] Body preview: {e}')

        if secs > 0:
            self.unified_status.config(text=f'Countdown — recording in {secs}s')

        def go():
            gaze_csv = os.path.join(folder, 'tobii_gaze.csv')
            try:
                self.gaze_app.start_recording(gaze_csv, go_epoch_ms, target_fps=30)
                # FIX 1: Reset gaze timers HERE (after countdown) not at preview start
                self.gaze_app.reset_gaze_timers()
                self.eye_status.config(text='Recording ●')
            except Exception as e:
                print(f'[ERROR] Gaze recording: {e}')
            try:
                self.body_engine.start_recording(folder, go_epoch_ms)
                self.body_status.config(text='Recording ●')
            except Exception as e:
                print(f'[ERROR] Body recording: {e}')
            self.unified_status.config(text='Recording ● BOTH sensors active')
            self._start_engagement_poll()

        self._run_countdown(secs, go)

    def stop_both(self):
        try:
            self.gaze_app.stop_recording()
            self.gaze_app.stop_streaming_preview()
            self.eye_status.config(text='Idle')
        except Exception:
            pass
        try:
            self.body_engine.stop_recording()
            self.body_engine.stop_preview()
            self.body_status.config(text='Idle')
        except Exception:
            pass
        self.unified_status.config(text='Idle')
        self.countdown_display.config(text='--', foreground='#c0392b')
        self._stop_engagement_poll()
        self._set_buttons_idle()
        self.root.after(500, self.verify_recording)

    # -----------------------------------------------------------------------
    # Start / Stop — Eye only
    # -----------------------------------------------------------------------
    def start_eye_only(self):
        secs = int(self.countdown_var.get())
        folder = self._make_session_folder('eye')
        go_epoch_ms = epoch_ms() + secs * 1000

        self._set_buttons_recording('eye')

        try:
            self.gaze_app.start_streaming_preview()
            self.eye_status.config(text='Previewing (waiting for countdown)')
        except Exception as e:
            print(f'[ERROR] Gaze preview: {e}')

        if secs > 0:
            self.unified_status.config(text=f'Countdown — recording in {secs}s')

        def go():
            gaze_csv = os.path.join(folder, 'tobii_gaze.csv')
            try:
                self.gaze_app.start_recording(gaze_csv, go_epoch_ms, target_fps=30)
                # FIX 1: Reset gaze timers after countdown
                self.gaze_app.reset_gaze_timers()
                self.eye_status.config(text='Recording ●')
                self.unified_status.config(text='Recording ● Eye only')
                self._start_engagement_poll()
            except Exception as e:
                print(f'[ERROR] Gaze recording: {e}')

        self._run_countdown(secs, go)

    def stop_eye(self):
        try:
            self.gaze_app.stop_recording()
            self.gaze_app.stop_streaming_preview()
            self.eye_status.config(text='Idle')
            self.unified_status.config(text='Idle')
        except Exception:
            pass
        self.countdown_display.config(text='--', foreground='#c0392b')
        self._stop_engagement_poll()
        self._set_buttons_idle()

    # -----------------------------------------------------------------------
    # Start / Stop — Body only
    # -----------------------------------------------------------------------
    def start_body_only(self):
        secs = int(self.countdown_var.get())
        folder = self._make_session_folder('body')
        go_epoch_ms = epoch_ms() + secs * 1000

        self._set_buttons_recording('body')

        try:
            self.body_engine.start_preview()
            self.body_status.config(text='Previewing (waiting for countdown)')
        except Exception as e:
            print(f'[ERROR] Body preview: {e}')

        if secs > 0:
            self.unified_status.config(text=f'Countdown — recording in {secs}s')

        def go():
            try:
                self.body_engine.start_recording(folder, go_epoch_ms)
                self.body_status.config(text='Recording ●')
                self.unified_status.config(text='Recording ● Body only')
                self._start_engagement_poll()
            except Exception as e:
                print(f'[ERROR] Body recording: {e}')

        self._run_countdown(secs, go)

    def stop_body(self):
        try:
            self.body_engine.stop_recording()
            self.body_engine.stop_preview()
            self.body_status.config(text='Idle')
            self.unified_status.config(text='Idle')
        except Exception:
            pass
        self.countdown_display.config(text='--', foreground='#c0392b')
        self._stop_engagement_poll()
        self._set_buttons_idle()

    # -----------------------------------------------------------------------
    # Post-recording verification
    # -----------------------------------------------------------------------
    def verify_recording(self):
        if not self.session_folder:
            return

        files = {
            'Eye CSV (tobii_gaze.csv)':   os.path.join(self.session_folder, 'tobii_gaze.csv'),
            'Body CSV (upper_body.csv)':  os.path.join(self.session_folder, 'upper_body.csv'),
        }

        lines = []
        for label, path in files.items():
            if not os.path.exists(path):
                lines.append(f"  {label}: file not found")
                continue
            try:
                with open(path, 'r', newline='') as f:
                    reader = csv.reader(f)
                    next(reader)
                    timestamps, row_count = [], 0
                    for row in reader:
                        row_count += 1
                        if row_count <= 200:
                            try:
                                timestamps.append(int(row[0]))
                            except (ValueError, IndexError):
                                pass

                if row_count < 10:
                    lines.append(f"  ⚠ {label}: only {row_count} rows")
                    continue

                if len(timestamps) >= 2:
                    diffs = [timestamps[i+1] - timestamps[i]
                             for i in range(len(timestamps)-1)
                             if timestamps[i+1] > timestamps[i]]
                    if diffs:
                        median_dt = sorted(diffs)[len(diffs) // 2]
                        fps = 1000 / median_dt if median_dt > 0 else 0
                        lines.append(f"  ✓ {label}: {row_count} rows  ~{fps:.1f} FPS")
                    else:
                        lines.append(f"  ✓ {label}: {row_count} rows")
            except Exception as e:
                lines.append(f"  ✗ {label}: {e}")

        result_text = "\n".join(lines) if lines else "No files found."
        self.verify_label.config(text=result_text, foreground='black')

        # Also print to console as before
        print(f"\n[VERIFY] Session: {self.session_folder}")
        for line in lines:
            print(f"[VERIFY]{line}")


    # -----------------------------------------------------------------------
    # Engagement polling — reads live_state from both sensors every 200ms
    # -----------------------------------------------------------------------
    def _start_engagement_poll(self):
        """Begin polling both sensors for live engagement state."""
        if self._engagement_poll_id is not None:
            return
        self._poll_engagement()

    def _stop_engagement_poll(self):
        """Stop polling and reset all indicators to idle state."""
        if self._engagement_poll_id is not None:
            try:
                self.root.after_cancel(self._engagement_poll_id)
            except Exception:
                pass
            self._engagement_poll_id = None
        self._reset_engagement_display()

    def _reset_engagement_display(self):
        """Grey out all indicators when not monitoring."""
        for widget in (self.ind_gaze, self.ind_dist, self.ind_face, self.ind_engaged):
            widget.config(text="—", foreground="gray")

    def _poll_engagement(self):
        """Read live state from both sensors and update the engagement display.

        Runs on the Tkinter main thread every 200ms via root.after().
        Reading simple bool/None values from dicts is GIL-safe in CPython.

        Engagement rule:
            DISENGAGED  = gaze_on_screen is False
                          AND distance_ok is False
                          AND facing_forward is False
            In all other cases the student is considered ENGAGED — they may be
            typing, working on a robot, or looking at materials off-screen.
        """
        try:
            # --- Read gaze state ---
            gaze_on = None
            if self.gaze_app is not None:
                gaze_on = self.gaze_app.gaze_on_screen

            # --- Read body state ---
            dist_ok   = None
            facing_fw = None
            body_eng  = None
            if self.body_engine is not None:
                dist_ok   = self.body_engine.live_state.get("distance_ok")
                facing_fw = self.body_engine.live_state.get("facing_forward")
                body_eng  = self.body_engine.live_state.get("body_engaged")

            # --- Update individual signal indicators ---
            self._update_indicator(self.ind_gaze, gaze_on,
                                   true_text="ON-SCREEN ✓",
                                   false_text="OFF-SCREEN ✗")
            self._update_indicator(self.ind_dist, dist_ok,
                                   true_text="OK ✓",
                                   false_text="TOO FAR ✗")
            self._update_indicator(self.ind_face, facing_fw,
                                   true_text="FORWARD ✓",
                                   false_text="TURNED ✗")

            # --- Compute combined engagement verdict ---
            any_none = any(v is None for v in (gaze_on, dist_ok, facing_fw))

            if any_none:
                # Sensors not fully active yet
                self.ind_engaged.config(text="WAITING…", foreground="gray")
            else:
                # Disengaged only if ALL THREE are False
                all_false = (gaze_on is False
                             and dist_ok is False
                             and facing_fw is False)
                if all_false:
                    self.ind_engaged.config(
                        text="⚠  DISENGAGED", foreground="white",
                        background="#c0392b")
                else:
                    self.ind_engaged.config(
                        text="✓  ENGAGED", foreground="white",
                        background="#27ae60")

        except Exception as e:
            print(f"[WARN] Engagement poll error: {e}")

        # Reschedule at 200ms — fast enough for live monitoring, light enough
        # not to impact the Tkinter event loop
        self._engagement_poll_id = self.root.after(200, self._poll_engagement)

    @staticmethod
    def _update_indicator(label: tk.Label, value: bool | None,
                          true_text: str, false_text: str):
        """Update a single signal indicator label with colour coding."""
        if value is None:
            label.config(text="—", foreground="gray", background="")
        elif value:
            label.config(text=true_text,  foreground="white", background="#27ae60")
        else:
            label.config(text=false_text, foreground="white", background="#c0392b")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def on_closing(root: tk.Tk, controller: MainController):
    try:
        controller.stop_both()
    except Exception:
        pass
    try:
        if controller.gaze_app:
            controller.gaze_app.shutdown()
    except Exception:
        pass
    try:
        controller.body_engine.stop_preview()
    except Exception:
        pass
    root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app = MainController(root)
    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root, app))
    root.mainloop()