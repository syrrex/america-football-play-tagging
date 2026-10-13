"""
Renders the Methods-section illustration for Play-Type Classification: QB
and offensive-line downfield displacement over the tracked clip, for one
clean example Run play and one clean example Pass play side by side.

Clip selection: among GT-init clips that were correctly classified
(play_type_eval.json), picks the Run example with the strongest oline
forward drive and the Pass example with the strongest combined QB-drop +
oline-retreat signal, so the two panels show the two decision-rule signals
as clearly as possible.

Usage:
  python plot_playtype_trajectories.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PLAY_ANALYSIS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLAY_ANALYSIS))

from classify_formation import META_CSV  # noqa: E402
from play_type_classification import get_attack_direction, compute_trajectory, FPS  # noqa: E402

OUT_PATH = PLAY_ANALYSIS.parent.parent / "thesis-template" / "figures" / "playtype_trajectories.png"

QB_COLOR = "#CCA300"      # darkened POSITION_COLOR_MAP["qb"] for legibility on white
OLINE_COLOR = "#FF69B4"   # POSITION_COLOR_MAP["oline"]


# Manually picked after visually comparing the top-scoring candidates below:
# the automatic score (steepest slope) favors clips with a noisy/non-monotonic
# QB path (e.g. clip_020's backfield dip-and-recover), which is a distracting
# example even though the classifier gets it right. These two give the
# cleanest-looking curves among the top candidates.
EXAMPLE_OVERRIDES = {"Run": "clip_007.mp4", "Pass": "clip_055.mp4"}


def pick_example(df: pd.DataFrame, play_type: str) -> str:
    if play_type in EXAMPLE_OVERRIDES:
        return EXAMPLE_OVERRIDES[play_type]
    sub = df[(df["true_play_type"] == play_type) & (df["pred_play_type"] == play_type)].copy()
    sub = sub.dropna(subset=["qb_slope_yd_s", "oline_slope_yd_s"])
    if play_type == "Run":
        sub["score"] = sub["oline_slope_yd_s"] - sub["qb_slope_yd_s"].abs() * 0.1
    else:
        sub["score"] = (-sub["qb_slope_yd_s"]) + (-sub["oline_slope_yd_s"])
    return sub.sort_values("score", ascending=False).iloc[0]["clip"]


def relative_traj(traj: list[tuple[int, float]], sign: float) -> tuple[np.ndarray, np.ndarray]:
    frames = np.array([f for f, _ in traj], dtype=float)
    values = np.array([v for _, v in traj], dtype=float)
    signed = sign * values
    signed -= signed[0]
    return frames / FPS, signed


def draw_panel(ax, clip_row, title):
    direction = get_attack_direction(clip_row)
    traj = compute_trajectory(clip_row)
    sign = 1.0 if direction == "inc" else -1.0

    t_qb, y_qb = relative_traj(traj["qb"], sign)
    t_ol, y_ol = relative_traj(traj["oline"], sign)

    ax.plot(t_qb, y_qb, "o-", color=QB_COLOR, label="QB", linewidth=1.8, markersize=4)
    ax.plot(t_ol, y_ol, "s-", color=OLINE_COLOR, label="Oline (median)", linewidth=1.8, markersize=4)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    slope_qb = np.polyfit(t_qb, y_qb, 1)[0] if len(t_qb) >= 2 else None
    slope_ol = np.polyfit(t_ol, y_ol, 1)[0] if len(t_ol) >= 2 else None
    if slope_qb is not None:
        ax.plot(t_qb, np.polyval(np.polyfit(t_qb, y_qb, 1), t_qb), color=QB_COLOR, linewidth=0.8, alpha=0.5)
    if slope_ol is not None:
        ax.plot(t_ol, np.polyval(np.polyfit(t_ol, y_ol, 1), t_ol), color=OLINE_COLOR, linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Time since first tracked frame (s)")
    ax.set_ylabel("Downfield displacement (yd)\n(+ = toward opponent)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)


def main():
    with open(PLAY_ANALYSIS / "play_type_eval.json") as f:
        results = json.load(f)
    df = pd.DataFrame(results["predictions"])
    meta = pd.read_csv(META_CSV).set_index("clip", drop=False)

    run_clip = pick_example(df, "Run")
    pass_clip = pick_example(df, "Pass")
    print(f"Run example:  {run_clip}")
    print(f"Pass example: {pass_clip}")

    fig, (ax_run, ax_pass) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    draw_panel(ax_run, meta.loc[run_clip], f"Run example ({run_clip})")
    draw_panel(ax_pass, meta.loc[pass_clip], f"Pass example ({pass_clip})")
    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
