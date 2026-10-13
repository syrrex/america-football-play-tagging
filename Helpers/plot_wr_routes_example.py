"""
Visualizes the tracked bird's-eye routes run by the skill-position players
on one play, for the "Trajectory Visualization" Additional Experiment
(sec:trajectory_visualization). Purely qualitative -- there is no route-type
ground truth anywhere in the tb2026 metadata, so this is a single-play
illustration, not a scored classification.

Uses clip_044 (Pass, TRIO) and clip_093 (Pass, Doubles) as two separate
example plays. Reuses the GT-box-initialized SAM3 tracking cache and
per-frame GT-keypoint homography, same convention as
Play_Analysis/play_type_classification.py.

Raw per-frame tracking is noticeably jittery frame to frame (SAM3 mask
noise, not real receiver movement), so each route is smoothed with a short
moving average before plotting -- same spirit as the Play-Type trajectory
figure fitting a regression line through noisy slope points, just applied
to the 2D path instead of a 1D slope.

Usage:
  python plot_wr_routes_example.py
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MASTER_THESIS_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = MASTER_THESIS_DIR / "thesis-template" / "figures"

sys.path.insert(0, str(PROJECT_DIR / "Play_Analysis"))
sys.path.insert(0, str(PROJECT_DIR / "Pipeline"))

from classify_formation import META_CSV, CLASS_SKILL  # noqa: E402
from play_type_classification import fit_H_for_global_frame, TRACKING_CACHE  # noqa: E402
from homography import project_to_field_yards  # noqa: E402
from drawing import create_football_field  # noqa: E402

PLAY_NUMS = [44, 93]
SMOOTH_WINDOW = 5  # frames, odd so the moving average stays centered

# Distinct, colorblind-friendlier palette, one per tracked skill player
# (not the shared POSITION_COLOR_MAP, since every role here is the same
# role -- skill -- and needs per-player, not per-role, distinction).
ROUTE_COLORS = ["#E64B35", "#4DBBD5", "#00A087", "#F39B7F", "#8491B4", "#91D1C2"]


def smooth(xs: np.ndarray, ys: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Centered moving average; window shrinks near the ends instead of
    padding, so the smoothed route still starts/ends at a real position."""
    n = len(xs)
    if n < 3:
        return xs, ys
    w = min(window, n if n % 2 == 1 else n - 1)
    half = w // 2
    sm_x = np.array([xs[max(0, i - half):i + half + 1].mean() for i in range(n)])
    sm_y = np.array([ys[max(0, i - half):i + half + 1].mean() for i in range(n)])
    return sm_x, sm_y


def extract_routes(play_num: int) -> dict[int, list[tuple[float, float]]]:
    meta = pd.read_csv(META_CSV)
    row = meta[meta["play_num"] == play_num].iloc[0]
    start = int(row["start_frame"])

    with open(TRACKING_CACHE / f"clip_{play_num:03d}.pkl", "rb") as f:
        cached = pickle.load(f)
    detections_by_frame = cached["detections"]

    # tracker_id -> list of (x_yd, y_yd), one point per frame it was tracked
    routes: dict[int, list[tuple[float, float]]] = {}

    for local_idx, frame_data in sorted(detections_by_frame.items()):
        H = fit_H_for_global_frame(start + local_idx)
        if H is None:
            continue
        xyxy, cids, tids = frame_data["xyxy"], frame_data["class_id"], frame_data["tracker_id"]

        skill_idx = np.where(cids == CLASS_SKILL)[0]
        if len(skill_idx) == 0:
            continue
        # drop degenerate zero-area boxes -- same lost-track signature as play_type_classification.py
        areas = (xyxy[skill_idx, 2] - xyxy[skill_idx, 0]) * (xyxy[skill_idx, 3] - xyxy[skill_idx, 1])
        skill_idx = skill_idx[areas > 1]
        if len(skill_idx) == 0:
            continue

        positions = project_to_field_yards(xyxy[skill_idx], H)
        for tid, pos in zip(tids[skill_idx], positions):
            routes.setdefault(int(tid), []).append(pos)

    return routes


def plot_routes(routes: dict[int, list[tuple[float, float]]], out_path: Path):
    fig, ax = create_football_field(figsize=(12, 6.33))
    for i, (tid, pts) in enumerate(sorted(routes.items())):
        if len(pts) < 2:
            continue
        xs, ys = zip(*pts)
        xs, ys = smooth(np.array(xs), np.array(ys), SMOOTH_WINDOW)
        color = ROUTE_COLORS[i % len(ROUTE_COLORS)]
        ax.plot(xs, ys, "-", color=color, linewidth=2.2, alpha=0.9, zorder=4)
        ax.scatter(xs[0], ys[0], color=color, marker="o", s=70, zorder=5, edgecolors="white", linewidths=0.8)
        ax.scatter(xs[-1], ys[-1], color=color, marker="^", s=90, zorder=5, edgecolors="white", linewidths=0.8)

    import matplotlib.lines as mlines
    legend_handles = [
        mlines.Line2D([], [], color="gray", marker="o", linestyle="None", markersize=8, label="start"),
        mlines.Line2D([], [], color="gray", marker="^", linestyle="None", markersize=9, label="end"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
              ncol=2, frameon=False, fontsize=11, labelcolor="black")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    for i, play_num in enumerate(PLAY_NUMS, start=1):
        routes = extract_routes(play_num)
        out_path = FIGURES_DIR / f"wr_routes_example_{i}.png"
        plot_routes(routes, out_path)
        print(f"saved -> {out_path}  ({len(routes)} tracked skill players, clip_{play_num:03d})")


if __name__ == "__main__":
    main()
