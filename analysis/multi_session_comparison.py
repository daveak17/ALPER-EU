"""
ALPER-EU — Multi-Session Comparison
=====================================
Loads session_summary.json and engagement_score.csv from multiple session
folders and produces comparison plots and a summary table.

Usage (terminal):
    python analysis/multi_session_comparison.py <folder1> <folder2> ...
    python analysis/multi_session_comparison.py recordings/both_* 

Can also be called programmatically from the GUI:
    from multi_session_comparison import run_comparison
    run_comparison([folder1, folder2, ...], output_dir)
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
import matplotlib.pyplot as _plt_check
if matplotlib.get_backend().lower() in ('agg', ''):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Colour palette — one colour per session, cycles if > 10 sessions
# ---------------------------------------------------------------------------
SESSION_COLOURS = [
    '#2ecc71', '#3498db', '#e74c3c', '#9b59b6',
    '#f39c12', '#1abc9c', '#e67e22', '#34495e',
    '#e91e63', '#00bcd4',
]


def _session_label(folder: str) -> str:
    """Short display label from folder path e.g. both_20260306_101619."""
    return os.path.basename(folder.rstrip('/\\'))


# ---------------------------------------------------------------------------
# Load a single session's data
# ---------------------------------------------------------------------------
def load_session_data(folder: str) -> dict | None:
    """Load summary JSON and score CSV from a session folder.

    Returns None if required files are missing (session not yet analysed).
    """
    summary_path = os.path.join(folder, 'analysis', 'session_summary.json')
    score_path   = os.path.join(folder, 'analysis', 'engagement_score.csv')

    if not os.path.exists(summary_path):
        print(f"[SKIP] No analysis found for: {folder}  "
              f"(run engagement_analysis.py first)")
        return None

    with open(summary_path) as f:
        summary = json.load(f)

    score_df = None
    if os.path.exists(score_path):
        score_df = pd.read_csv(score_path)

    return {
        'folder':   folder,
        'label':    _session_label(folder),
        'summary':  summary,
        'score_df': score_df,
    }


# ---------------------------------------------------------------------------
# Plot 1 — Mean engagement score bar chart
# ---------------------------------------------------------------------------
def plot_score_comparison(sessions: list, out_path: str):
    """Bar chart of mean engagement score per session, sorted descending."""
    labels = [s['label'] for s in sessions]
    scores = [s['summary'].get('mean_engagement_score', 0) for s in sessions]
    colours = [SESSION_COLOURS[i % len(SESSION_COLOURS)]
               for i in range(len(sessions))]

    # Sort by score descending
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    labels  = [labels[i]  for i in order]
    scores  = [scores[i]  for i in order]
    colours = [colours[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(8, len(sessions) * 1.4), 5))
    bars = ax.bar(labels, scores, color=colours,
                  edgecolor='white', linewidth=0.8, width=0.55)

    # Value labels
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.0,
                f'{val:.1f}', ha='center', va='bottom',
                fontweight='bold', fontsize=10)

    # Zone lines
    ax.axhline(80, color='#27ae60', linestyle='--',
               linewidth=0.9, alpha=0.6, label='High (80)')
    ax.axhline(50, color='#f39c12', linestyle='--',
               linewidth=0.9, alpha=0.6, label='Moderate (50)')

    ax.set_ylim(0, 115)
    ax.set_ylabel('Mean Engagement Score (0–100)', fontsize=11)
    ax.set_title('Mean Engagement Score — Session Comparison',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f'{v:.0f}'))
    plt.xticks(rotation=20, ha='right', fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[PLOT] Score comparison saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2 — Signal comparison grouped bars
# ---------------------------------------------------------------------------
def plot_signal_comparison(sessions: list, out_path: str):
    """Grouped bar chart: gaze / facing / distance / body / overall per session."""
    labels   = [s['label'] for s in sessions]
    n        = len(sessions)
    x        = np.arange(n)
    width    = 0.15

    signals = [
        ('gaze_on_screen_pct',  'Gaze on-screen',  '#3498db'),
        ('facing_forward_pct',  'Facing forward',  '#9b59b6'),
        ('distance_ok_pct',     'Distance OK',     '#e67e22'),
        ('body_engaged_pct',    'Body engaged',    '#16a085'),
        ('engaged_pct',         'Overall engaged', '#27ae60'),
    ]

    fig, ax = plt.subplots(figsize=(max(10, n * 2.0), 6))

    for i, (key, label, colour) in enumerate(signals):
        values = [s['summary'].get(key, 0) for s in sessions]
        offset = (i - 2) * width
        bars = ax.bar(x + offset, values, width,
                      label=label, color=colour,
                      edgecolor='white', linewidth=0.5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_ylabel('% Time Condition Was True', fontsize=11)
    ax.set_title('Signal Breakdown — Session Comparison',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[PLOT] Signal comparison saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3 — Engagement score curves overlaid
# ---------------------------------------------------------------------------
def plot_score_curves(sessions: list, out_path: str):
    """All sessions' smoothed engagement score curves on one plot."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Zone backgrounds
    ax.axhspan(80, 100, color='#d5f5e3', alpha=0.4, zorder=0)
    ax.axhspan(50, 80,  color='#fef9e7', alpha=0.4, zorder=0)
    ax.axhspan(0,  50,  color='#fdedec', alpha=0.4, zorder=0)

    max_t = 0
    for i, s in enumerate(sessions):
        score_df = s.get('score_df')
        if score_df is None or 'score_smooth' not in score_df.columns:
            continue
        colour = SESSION_COLOURS[i % len(SESSION_COLOURS)]
        t = score_df['t_s'].values
        v = score_df['score_smooth'].values
        mean_v = s['summary'].get('mean_engagement_score', 0)

        ax.plot(t, v, color=colour, linewidth=2.0,
                label=f"{s['label']} (mean={mean_v:.1f})", alpha=0.85)
        ax.axhline(mean_v, color=colour, linestyle=':',
                   linewidth=0.8, alpha=0.5)
        max_t = max(max_t, t[-1])

    ax.set_xlim(0, max_t)
    ax.set_ylim(0, 105)
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('Engagement Score (0–100)', fontsize=11)
    ax.set_title('Engagement Score Over Time — All Sessions',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.spines[['top', 'right']].set_visible(False)

    # Zone labels
    ax.text(max_t + 1, 90, 'High',     color='#27ae60', fontsize=8, va='center')
    ax.text(max_t + 1, 65, 'Moderate', color='#f39c12', fontsize=8, va='center')
    ax.text(max_t + 1, 25, 'Low',      color='#e74c3c', fontsize=8, va='center')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[PLOT] Score curves saved: {out_path}")


# ---------------------------------------------------------------------------
# Summary comparison table
# ---------------------------------------------------------------------------
def build_comparison_table(sessions: list) -> pd.DataFrame:
    """Build a DataFrame with one row per session, all key metrics."""
    rows = []
    for s in sessions:
        sm = s['summary']
        rows.append({
            'session':                  s['label'],
            'duration_s':               sm.get('duration_s'),
            'total_frames':             sm.get('total_frames'),
            'effective_fps':            sm.get('effective_fps'),
            'mean_engagement_score':    sm.get('mean_engagement_score'),
            'min_engagement_score':     sm.get('min_engagement_score'),
            'max_engagement_score':     sm.get('max_engagement_score'),
            'engaged_pct':              sm.get('engaged_pct'),
            'disengaged_pct':           sm.get('disengaged_pct'),
            'gaze_on_screen_pct':       sm.get('gaze_on_screen_pct'),
            'distance_ok_pct':          sm.get('distance_ok_pct'),
            'facing_forward_pct':       sm.get('facing_forward_pct'),
            'body_engaged_pct':         sm.get('body_engaged_pct'),
            'disengagement_events':     sm.get('disengagement_events'),
            'avg_disengagement_dur_s':  sm.get('avg_disengagement_dur_s'),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Save all outputs
# ---------------------------------------------------------------------------
def save_comparison_outputs(sessions: list, output_dir: str):
    """Save all comparison plots and the summary table."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"[SAVE] Comparison output directory: {output_dir}")

    plot_score_comparison(
        sessions, os.path.join(output_dir, 'comparison_scores.png'))
    plot_signal_comparison(
        sessions, os.path.join(output_dir, 'comparison_signals.png'))
    plot_score_curves(
        sessions, os.path.join(output_dir, 'comparison_curves.png'))

    table = build_comparison_table(sessions)
    table_path = os.path.join(output_dir, 'comparison_table.csv')
    table.to_csv(table_path, index=False)
    print(f"[SAVE] Comparison table: {table_path}")

    return output_dir


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_comparison(folders: list, output_dir: str) -> tuple:
    """Load sessions, produce all outputs. Returns (sessions, output_dir)."""
    print(f"\n[ALPER-EU] Multi-session comparison: {len(folders)} folders\n")

    sessions = []
    for folder in folders:
        data = load_session_data(folder)
        if data:
            sessions.append(data)
            print(f"  ✓ {data['label']:35s}  "
                  f"score={data['summary'].get('mean_engagement_score','?'):5}  "
                  f"engaged={data['summary'].get('engaged_pct','?')}%")

    if len(sessions) < 2:
        raise ValueError(
            f"Need at least 2 analysed sessions to compare. "
            f"Found {len(sessions)} valid session(s).\n"
            f"Run engagement_analysis.py on each session folder first.")

    print(f"\n[INFO] Comparing {len(sessions)} sessions\n")
    save_comparison_outputs(sessions, output_dir)
    print(f"\n[DONE] Comparison outputs saved to: {output_dir}\n")
    return sessions, output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ALPER-EU multi-session engagement comparison')
    parser.add_argument('folders', nargs='+',
                        help='Session folders to compare')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory (default: recordings/comparison/)')
    args = parser.parse_args()

    folders = [os.path.abspath(f) for f in args.folders
               if os.path.isdir(f)]

    if not folders:
        print("[ERROR] No valid folders provided")
        sys.exit(1)

    # Default output: sibling to the first session's parent
    if args.output:
        out_dir = args.output
    else:
        parent = os.path.dirname(folders[0])
        out_dir = os.path.join(parent, 'comparison')

    run_comparison(folders, out_dir)