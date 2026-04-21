"""
ALPER-EU — Multimodal Engagement Analysis
==========================================
Loads a session folder containing:
    tobii_gaze.csv   — gaze x/y + on_screen  (30 FPS)
    upper_body.csv   — joints + Distance_OK + Facing_Forward (30 FPS)
    hands_data.csv   — hand detection (optional)

Produces:
    1. Session summary statistics (printed + saved as JSON)
    2. Three-signal engagement timeline plot
    3. Gaze heatmap
    4. Disengagement event table (CSV)

Usage:
    python engagement_analysis.py <session_folder>
    python engagement_analysis.py recordings/both_20260305_150700
"""

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
# Only set backend if matplotlib hasn't been initialised yet.
# When imported dynamically from a running Tkinter app, the backend
# is already set — forcing 'Agg' here would break the existing GUI.
import matplotlib.pyplot as _plt_check
_current_backend = matplotlib.get_backend()
if _current_backend.lower() in ('agg', ''):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCREEN_W = 1920
SCREEN_H = 1200
HEATMAP_SIGMA = 30       # Gaussian blur sigma in pixels
MERGE_TOLERANCE_MS = 50  # max ms gap for timestamp merge (~1.5 frames at 30fps)

# Engagement score weights (must sum to 1.0)
# Gaze is weighted highest — direct attention is the strongest engagement signal
# in an educational robotics context.
W_GAZE   = 0.50
W_FACING = 0.30
W_DIST   = 0.20
SCORE_SMOOTH_S = 3       # rolling average window in seconds


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_session(folder: str) -> dict:
    """Load all CSVs from a session folder. Returns dict of DataFrames."""
    paths = {
        'gaze':  os.path.join(folder, 'tobii_gaze.csv'),
        'body':  os.path.join(folder, 'upper_body.csv'),
        'hands': os.path.join(folder, 'hands_data.csv'),
    }

    data = {}
    for key, path in paths.items():
        if not os.path.exists(path):
            if key == 'hands':
                data[key] = None   # hands is optional
                continue
            raise FileNotFoundError(f"Required file missing: {path}")

        df = pd.read_csv(path)
        df = df.sort_values('epoch_ms').reset_index(drop=True)
        data[key] = df
        print(f"[LOAD] {key:5s}: {len(df):5d} rows  —  {path}")

    return data


# ---------------------------------------------------------------------------
# Merge & engagement computation
# ---------------------------------------------------------------------------
def build_merged(data: dict) -> pd.DataFrame:
    """Merge gaze + body by nearest timestamp, compute engagement signal."""
    gaze = data['gaze']
    body = data['body']

    # Merge: use body as the anchor (30fps) and join nearest gaze sample
    merged = pd.merge_asof(
        body,
        gaze[['epoch_ms', 'on_screen', 'x', 'y']],
        on='epoch_ms',
        direction='nearest',
        tolerance=MERGE_TOLERANCE_MS,
    )

    # Fill any unmatched gaze rows (outside tolerance) as off-screen
    merged['on_screen'] = merged['on_screen'].fillna(False)
    merged['x'] = merged['x'].fillna(-1)
    merged['y'] = merged['y'].fillna(-1)

    # Normalise timestamps to seconds from session start
    t0 = merged['epoch_ms'].iloc[0]
    merged['t'] = (merged['epoch_ms'] - t0) / 1000.0

    # Engagement rule:
    #   DISENGAGED = gaze off-screen AND distance fail AND facing fail
    #   In all other cases → ENGAGED
    #   Rationale: a student may be typing (gaze off) or looking at a robot
    #   (body turned) without being disengaged. Only all-three-false together
    #   is a reliable disengagement signal.
    merged['Engaged'] = ~(
        (~merged['on_screen']) &
        (~merged['Distance_OK']) &
        (~merged['Facing_Forward'])
    )

    # Body engaged = distance OK AND facing forward
    merged['Body_Engaged'] = merged['Distance_OK'] & merged['Facing_Forward']

    return merged


# ---------------------------------------------------------------------------
# Engagement score over time
# ---------------------------------------------------------------------------
def compute_engagement_score(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute a 0-100 engagement score per second.

    Formula (per frame):
        raw = (on_screen * W_GAZE + Facing_Forward * W_FACING + Distance_OK * W_DIST) * 100

    Steps:
        1. Compute raw score per frame
        2. Bin into 1-second intervals
        3. Smooth with a SCORE_SMOOTH_S rolling average (centred)

    Returns a DataFrame with columns:
        t_s           — second index from session start
        score         — raw mean score for that second  (0-100)
        score_smooth  — smoothed score                  (0-100)
        gaze          — mean gaze on-screen for that second
        facing        — mean facing-forward for that second
        distance      — mean distance-ok for that second
    """
    df = merged.copy()

    # Per-frame weighted score
    df['raw_score'] = (
        df['on_screen'].astype(float)       * W_GAZE   +
        df['Facing_Forward'].astype(float)  * W_FACING +
        df['Distance_OK'].astype(float)     * W_DIST
    ) * 100

    # Bin into whole seconds
    df['t_bin'] = df['t'].apply(lambda t: int(t))

    per_second = df.groupby('t_bin').agg(
        score    = ('raw_score',      'mean'),
        gaze     = ('on_screen',      'mean'),
        facing   = ('Facing_Forward', 'mean'),
        distance = ('Distance_OK',    'mean'),
    ).reset_index().rename(columns={'t_bin': 't_s'})

    # Smooth score
    per_second['score_smooth'] = (
        per_second['score']
        .rolling(window=SCORE_SMOOTH_S, center=True, min_periods=1)
        .mean()
    )

    # Round for cleaner CSV output
    per_second = per_second.round(3)

    return per_second


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def compute_summary(merged: pd.DataFrame, folder: str) -> dict:
    """Compute per-session summary statistics."""
    duration = merged['t'].iloc[-1]
    fps = len(merged) / duration if duration > 0 else 0

    # Signal percentages
    gaze_pct   = merged['on_screen'].mean()    * 100
    dist_pct   = merged['Distance_OK'].mean()  * 100
    face_pct   = merged['Facing_Forward'].mean()* 100
    body_pct   = merged['Body_Engaged'].mean() * 100
    eng_pct    = merged['Engaged'].mean()      * 100
    diseng_pct = 100 - eng_pct

    # Disengagement events: each contiguous run of Engaged==False
    diseng_mask = ~merged['Engaged']
    events = []
    in_event = False
    start_t = 0.0
    for _, row in merged.iterrows():
        if not in_event and not row['Engaged']:
            in_event = True
            start_t = row['t']
        elif in_event and row['Engaged']:
            events.append({'start_s': round(start_t, 2),
                           'end_s':   round(row['t'], 2),
                           'duration_s': round(row['t'] - start_t, 2)})
            in_event = False
    if in_event:
        events.append({'start_s': round(start_t, 2),
                       'end_s':   round(duration, 2),
                       'duration_s': round(duration - start_t, 2)})

    avg_diseng_dur = (np.mean([e['duration_s'] for e in events])
                      if events else 0.0)

    # Engagement score summary
    score_df = compute_engagement_score(merged)
    mean_score   = round(score_df['score_smooth'].mean(), 1)
    min_score    = round(score_df['score_smooth'].min(), 1)
    max_score    = round(score_df['score_smooth'].max(), 1)

    summary = {
        'session_folder':          folder,
        'duration_s':              round(duration, 2),
        'total_frames':            len(merged),
        'effective_fps':           round(fps, 1),
        'engaged_pct':             round(eng_pct, 1),
        'disengaged_pct':          round(diseng_pct, 1),
        'gaze_on_screen_pct':      round(gaze_pct, 1),
        'distance_ok_pct':         round(dist_pct, 1),
        'facing_forward_pct':      round(face_pct, 1),
        'body_engaged_pct':        round(body_pct, 1),
        'disengagement_events':    len(events),
        'avg_disengagement_dur_s': round(avg_diseng_dur, 2),
        'mean_engagement_score':   mean_score,
        'min_engagement_score':    min_score,
        'max_engagement_score':    max_score,
        'score_weights':           {'gaze': W_GAZE, 'facing': W_FACING,
                                    'distance': W_DIST},
        'events':                  events,
    }
    return summary


# ---------------------------------------------------------------------------
# Plot 1 — Three-signal engagement timeline
# ---------------------------------------------------------------------------
def plot_engagement_timeline(merged: pd.DataFrame, out_path: str):
    """
    Stacked timeline showing four rows:
        1. Gaze on-screen
        2. Distance OK
        3. Facing forward
        4. Overall engagement verdict
    Each row is a colour-filled band: green=True, red=False.
    """
    t = merged['t'].values
    signals = {
        'Gaze On-Screen':    merged['on_screen'].values,
        'Distance OK':       merged['Distance_OK'].values,
        'Facing Forward':    merged['Facing_Forward'].values,
        'ENGAGED':           merged['Engaged'].values,
    }

    fig, axes = plt.subplots(4, 1, figsize=(14, 7), sharex=True)
    fig.suptitle('Multimodal Engagement Timeline', fontsize=14, fontweight='bold', y=1.01)

    colours = {
        'Gaze On-Screen':  ('#2ecc71', '#e74c3c'),
        'Distance OK':     ('#3498db', '#e74c3c'),
        'Facing Forward':  ('#9b59b6', '#e74c3c'),
        'ENGAGED':         ('#27ae60', '#c0392b'),
    }

    for ax, (label, values) in zip(axes, signals.items()):
        col_true, col_false = colours[label]

        # Fill background red (False) then overlay green (True) spans
        ax.axhspan(0, 1, color=col_false, alpha=0.25)

        # Find contiguous True runs and shade them green
        in_run = False
        run_start = 0.0
        for i, v in enumerate(values):
            if not in_run and v:
                in_run = True
                run_start = t[i]
            elif in_run and not v:
                ax.axvspan(run_start, t[i], color=col_true, alpha=0.6)
                in_run = False
        if in_run:
            ax.axvspan(run_start, t[-1], color=col_true, alpha=0.6)

        # Style
        ax.set_ylabel(label, fontsize=9, fontweight='bold',
                      rotation=0, labelpad=120, va='center')
        ax.set_yticks([])
        ax.set_ylim(0, 1)
        ax.spines[['top', 'right', 'left']].set_visible(False)

        # Percentage annotation
        pct = values.mean() * 100
        ax.text(1.002, 0.5, f'{pct:.0f}%',
                transform=ax.transAxes, va='center', fontsize=9, color='#333')

    axes[-1].set_xlabel('Time (seconds)', fontsize=10)

    # Legend
    patch_true  = mpatches.Patch(color='#27ae60', alpha=0.6, label='True / Engaged')
    patch_false = mpatches.Patch(color='#e74c3c', alpha=0.25, label='False / Disengaged')
    fig.legend(handles=[patch_true, patch_false],
               loc='upper right', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[PLOT] Timeline saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2 — Gaze heatmap
# ---------------------------------------------------------------------------
def plot_gaze_heatmap(merged: pd.DataFrame, out_path: str):
    """Gaze density heatmap using 2D histogram with log scaling."""
    from matplotlib.colors import LogNorm
    import matplotlib.style as mplstyle
    mplstyle.use('seaborn-v0_8')

    # Only on-screen gaze points
    on = merged[merged['on_screen'] & (merged['x'] > 0) & (merged['y'] > 0)]

    if len(on) < 10:
        print("[WARN] Not enough on-screen gaze points for heatmap")
        return

    x_data = np.clip(on['x'].values, 0, SCREEN_W)
    y_data = np.clip(on['y'].values, 0, SCREEN_H)

    fig, ax = plt.subplots(figsize=(12, 7.5))

    heatmap_data, xedges, yedges = np.histogram2d(
        x_data, y_data, bins=50,
        range=[[0, SCREEN_W], [0, SCREEN_H]])

    smoothed = gaussian_filter(heatmap_data, sigma=2.0)

    min_positive = (smoothed[smoothed > 0].min()
                    if np.any(smoothed > 0) else 1e-3)
    norm = LogNorm(vmin=max(min_positive, 1e-3), vmax=smoothed.max())

    im = ax.imshow(smoothed.T, origin='lower', cmap='turbo',
                   extent=[0, SCREEN_W, 0, SCREEN_H], aspect='auto',
                   norm=norm)

    fig.colorbar(im, ax=ax, label='Gaze Density')
    ax.set_title(f'Gaze Heatmap  ({len(on)} on-screen samples)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Screen X (px)')
    ax.set_ylabel('Screen Y (px)')
    ax.set_xlim(0, SCREEN_W)
    ax.set_ylim(SCREEN_H, 0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[PLOT] Heatmap saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3 — Signal correlation bar chart
# ---------------------------------------------------------------------------
def plot_signal_summary(summary: dict, out_path: str):
    """Bar chart of all signal percentages for quick visual comparison."""
    labels = [
        'Gaze\nOn-Screen',
        'Distance\nOK',
        'Facing\nForward',
        'Body\nEngaged',
        'Overall\nEngaged',
    ]
    values = [
        summary['gaze_on_screen_pct'],
        summary['distance_ok_pct'],
        summary['facing_forward_pct'],
        summary['body_engaged_pct'],
        summary['engaged_pct'],
    ]
    colours = ['#3498db', '#9b59b6', '#e67e22', '#16a085', '#27ae60']

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colours, edgecolor='white',
                  linewidth=0.8, width=0.55)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f'{val:.1f}%', ha='center', va='bottom',
                fontweight='bold', fontsize=11)

    ax.set_ylim(0, 115)
    ax.set_ylabel('Time (%)', fontsize=11)
    ax.set_title('Signal Summary — % Time Each Condition Was True',
                 fontsize=13, fontweight='bold')
    ax.axhline(100, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))

    # Annotation
    n_events = summary['disengagement_events']
    avg_dur  = summary['avg_disengagement_dur_s']
    duration = summary['duration_s']
    ax.text(0.98, 0.96,
            f"Session: {duration:.0f}s\n"
            f"Disengagement events: {n_events}\n"
            f"Avg duration: {avg_dur:.1f}s",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, color='#555',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f8f8',
                      edgecolor='#ccc'))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[PLOT] Signal summary saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 4 — Engagement score over time
# ---------------------------------------------------------------------------
def plot_engagement_score(score_df: pd.DataFrame, summary: dict, out_path: str):
    """
    Line chart of the smoothed engagement score (0-100) over time.
    Background colour bands show engagement zones:
        80-100  green   — highly engaged
        50-80   yellow  — moderate
        0-50    red     — low engagement
    Individual signal lines (gaze/facing/distance) shown as thin traces.
    """
    t   = score_df['t_s'].values
    s   = score_df['score_smooth'].values
    raw = score_df['score'].values

    fig, ax = plt.subplots(figsize=(14, 5))

    # Zone backgrounds
    ax.axhspan(80, 100, color='#d5f5e3', alpha=0.6, zorder=0)
    ax.axhspan(50, 80,  color='#fef9e7', alpha=0.6, zorder=0)
    ax.axhspan(0,  50,  color='#fdedec', alpha=0.6, zorder=0)

    # Zone labels
    ax.text(t[-1] + 0.5, 90, 'High',     color='#27ae60', fontsize=8, va='center')
    ax.text(t[-1] + 0.5, 65, 'Moderate', color='#f39c12', fontsize=8, va='center')
    ax.text(t[-1] + 0.5, 25, 'Low',      color='#e74c3c', fontsize=8, va='center')

    # Individual signal traces (scaled to 0-100)
    ax.plot(t, score_df['gaze'].values    * 100,
            color='#3498db', alpha=0.25, linewidth=0.8, label='Gaze (×50%)')
    ax.plot(t, score_df['facing'].values  * 100,
            color='#9b59b6', alpha=0.25, linewidth=0.8, label='Facing (×30%)')
    ax.plot(t, score_df['distance'].values* 100,
            color='#e67e22', alpha=0.25, linewidth=0.8, label='Distance (×20%)')

    # Raw score
    ax.plot(t, raw, color='#aab7b8', linewidth=0.6,
            alpha=0.4, label='Raw score')

    # Smoothed score — main line
    ax.plot(t, s, color='#2c3e50', linewidth=2.5,
            label=f'Engagement score ({SCORE_SMOOTH_S}s smooth)', zorder=5)

    # Mean line
    mean_s = summary['mean_engagement_score']
    ax.axhline(mean_s, color='#7f8c8d', linestyle='--',
               linewidth=1.0, alpha=0.7,
               label=f'Session mean: {mean_s:.1f}')

    # Fill under smoothed score, colour by zone
    ax.fill_between(t, s, 0,
                    where=(s >= 80), color='#27ae60', alpha=0.15, zorder=1)
    ax.fill_between(t, s, 0,
                    where=((s >= 50) & (s < 80)),
                    color='#f39c12', alpha=0.15, zorder=1)
    ax.fill_between(t, s, 0,
                    where=(s < 50),  color='#e74c3c', alpha=0.15, zorder=1)

    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, 105)
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('Engagement Score (0-100)', fontsize=11)
    ax.set_title(
        f'Engagement Score Over Time  '
        f'(mean={mean_s:.1f}  min={summary["min_engagement_score"]:.1f}  '
        f'max={summary["max_engagement_score"]:.1f})',
        fontsize=13, fontweight='bold')

    ax.legend(loc='lower left', fontsize=8, framealpha=0.9,
              ncol=3, bbox_to_anchor=(0, -0.28))
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f'{v:.0f}'))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[PLOT] Engagement score saved: {out_path}")


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
def save_outputs(merged: pd.DataFrame, summary: dict, folder: str):
    """Save all analysis outputs to the session folder."""
    # Resolve to absolute path so output location is always unambiguous
    folder = os.path.abspath(folder)
    out_dir = os.path.join(folder, 'analysis')
    os.makedirs(out_dir, exist_ok=True)
    print(f"[SAVE] Output directory: {out_dir}")

    # 1. Summary JSON
    summary_path = os.path.join(out_dir, 'session_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVE] Summary JSON: {summary_path}")

    # 2. Merged engagement CSV (one row per body frame, all signals + verdict)
    cols = ['epoch_ms', 't',
            'on_screen', 'x', 'y',
            'Distance_OK', 'Facing_Forward', 'Body_Engaged', 'Engaged',
            'Nose_Dist_M']
    available = [c for c in cols if c in merged.columns]
    merged_path = os.path.join(out_dir, 'engagement_merged.csv')
    merged[available].to_csv(merged_path, index=False)
    print(f"[SAVE] Merged CSV:   {merged_path}")

    # 3. Disengagement events CSV
    if summary['events']:
        events_df = pd.DataFrame(summary['events'])
        events_path = os.path.join(out_dir, 'disengagement_events.csv')
        events_df.to_csv(events_path, index=False)
        print(f"[SAVE] Events CSV:   {events_path}")

    # 4. Engagement score per second CSV
    score_df = compute_engagement_score(merged)
    score_path = os.path.join(out_dir, 'engagement_score.csv')
    score_df.to_csv(score_path, index=False)
    print(f"[SAVE] Score CSV:     {score_path}")

    # 5. Plots
    plot_engagement_timeline(merged,  os.path.join(out_dir, 'engagement_timeline.png'))
    plot_gaze_heatmap(merged,         os.path.join(out_dir, 'gaze_heatmap.png'))
    plot_signal_summary(summary,      os.path.join(out_dir, 'signal_summary.png'))
    plot_engagement_score(score_df, summary,
                          os.path.join(out_dir, 'engagement_score.png'))

    return out_dir


# ---------------------------------------------------------------------------
# Print summary to console
# ---------------------------------------------------------------------------
def print_summary(summary: dict):
    print()
    print("=" * 52)
    print("  ALPER-EU SESSION ANALYSIS SUMMARY")
    print("=" * 52)
    print(f"  Session folder:        {os.path.basename(summary['session_folder'])}")
    print(f"  Duration:              {summary['duration_s']:.1f}s")
    print(f"  Total frames:          {summary['total_frames']}")
    print(f"  Effective FPS:         {summary['effective_fps']}")
    print()
    print(f"  Gaze on-screen:        {summary['gaze_on_screen_pct']:5.1f}%")
    print(f"  Distance OK:           {summary['distance_ok_pct']:5.1f}%")
    print(f"  Facing forward:        {summary['facing_forward_pct']:5.1f}%")
    print(f"  Body engaged:          {summary['body_engaged_pct']:5.1f}%")
    print()
    print(f"  ► Overall ENGAGED:     {summary['engaged_pct']:5.1f}%")
    print(f"  ► Overall DISENGAGED:  {summary['disengaged_pct']:5.1f}%")
    print()
    print(f"  Disengagement events:  {summary['disengagement_events']}")
    print(f"  Avg disengagement dur: {summary['avg_disengagement_dur_s']:.2f}s")
    print()
    print(f"  ► Mean engagement score: {summary['mean_engagement_score']}")
    print(f"    Min: {summary['min_engagement_score']}  "
          f"Max: {summary['max_engagement_score']}")
    print(f"    Weights — Gaze: {summary['score_weights']['gaze']}  "
          f"Facing: {summary['score_weights']['facing']}  "
          f"Distance: {summary['score_weights']['distance']}")
    print("=" * 52)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_analysis(folder: str):
    print(f"\n[ALPER-EU] Analysing session: {folder}\n")

    data   = load_session(folder)
    merged = build_merged(data)
    summary = compute_summary(merged, folder)
    print_summary(summary)
    out_dir = save_outputs(merged, summary, folder)

    print(f"\n[DONE] All outputs saved to: {out_dir}\n")
    return summary, merged


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ALPER-EU multimodal engagement analysis')
    parser.add_argument('folder', help='Path to session recording folder')
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"[ERROR] Folder not found: {args.folder}")
        sys.exit(1)

    run_analysis(args.folder)