"""
Draws a top-down football field with all 62 annotated keypoints.

Each keypoint group is colored differently:
  - Top sideline    (1–13)  : red
  - Bottom sideline (14–26) : orange
  - Top numbers     (27–35) : cyan
  - Bottom numbers  (36–44) : yellow
  - Top hash        (45–53) : lime
  - Bottom hash     (54–62) : magenta

Usage:
  python visualize_field_keypoints.py
  python visualize_field_keypoints.py --out field_keypoints.png --dpi 200
"""

import argparse
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from field_landmarks import ALL_LANDMARKS, FIELD_POSITIONS, LandmarkType


def create_football_field(figsize=(12, 6.33)):
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(patches.Rectangle((0, 0), 120, 53.3,
                                   linewidth=0.1, edgecolor="r",
                                   facecolor="darkgreen", zorder=0))
    for x0 in (0, 110):
        ax.add_patch(patches.Rectangle((x0, 0), 10, 53.3,
                                       linewidth=0.1, edgecolor="r",
                                       facecolor="blue", alpha=0.2, zorder=0))
    ax.plot(
        [10,10,20,20,30,30,40,40,50,50,60,60,70,70,80,80,90,90,100,100,110,110,0,0,120,120],
        [0,53.3,53.3,0,0,53.3,53.3,0,0,53.3,53.3,0,0,53.3,53.3,0,0,53.3,53.3,0,0,53.3,53.3,0,0,53.3],
        color="white", linewidth=0.8,
    )
    for x in range(20, 110, 10):
        n = x if x <= 50 else 120 - x
        ax.text(x - 0.5, 8.5,  str(n - 10), ha="center", va="center", fontsize=30, color="white", rotation=180)
        ax.text(x,       44.8, str(n - 10), ha="center", va="center", fontsize=30, color="white")
    y_hash_top, y_hash_bot = 20.0, 33.3
    for x in range(11, 110):
        for yh in (y_hash_top, y_hash_bot):
            ax.plot([x, x], [yh - 0.3, yh + 0.3], color="white", linewidth=0.6)
        ax.plot([x, x], [0.2, 0.5],   color="white", linewidth=0.6)
        ax.plot([x, x], [52.8, 53.1], color="white", linewidth=0.6)
    ax.set_xlim(0, 120)
    ax.set_ylim(58.3, -5)
    ax.axis("off")
    return fig, ax

# ── Colour palette per group ───────────────────────────────────────────────────
GROUP_COLOR = {
    LandmarkType.SIDELINE_TOP:    "#FF4444",   # red
    LandmarkType.SIDELINE_BOTTOM: "#FF9900",   # orange
    LandmarkType.NUMBER_TOP:      "#00CFFF",   # cyan
    LandmarkType.NUMBER_BOTTOM:   "#FFE600",   # yellow
    LandmarkType.HASH_TOP:        "#44FF44",   # lime
    LandmarkType.HASH_BOTTOM:     "#FF44FF",   # magenta
}

GROUP_LABEL = {
    LandmarkType.SIDELINE_TOP:    "Top sideline (1–13)",
    LandmarkType.SIDELINE_BOTTOM: "Bottom sideline (14–26)",
    LandmarkType.NUMBER_TOP:      "Top numbers (27–35)",
    LandmarkType.NUMBER_BOTTOM:   "Bottom numbers (36–44)",
    LandmarkType.HASH_TOP:        "Top hash (45–53)",
    LandmarkType.HASH_BOTTOM:     "Bottom hash (54–62)",
}

def draw(out_path: Path | None, dpi: int) -> None:
    fig, ax = create_football_field(figsize=(18, 9))

    positions = FIELD_POSITIONS

    # Plot all keypoints, colored by group
    for lm in ALL_LANDMARKS:
        x, y   = positions[lm.point_id]
        color  = GROUP_COLOR[lm.landmark_type]

        ax.scatter(x, y, c=color, s=80, zorder=7,
                   edgecolors="black", linewidths=0.4)

        # Keypoint ID label — offset slightly above the dot
        ax.text(x, y - 1.6, str(lm.point_id),
                ha="center", va="bottom", fontsize=12, color=color,
                fontweight="bold", zorder=8)

    # Legend — one proxy handle per group in display order
    legend_handles = [
        mlines.Line2D([], [], color=GROUP_COLOR[lt], marker="o", linestyle="None",
                      markersize=7, markeredgecolor="black", markeredgewidth=0.4,
                      label=GROUP_LABEL[lt])
        for lt in GROUP_LABEL
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=3,
        framealpha=0.85,
        fontsize=9,
    )

    ax.set_title("Football Field — 62 Annotated Keypoints", fontsize=13, pad=10)

    if out_path:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {out_path}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None,
                        help="Output image path (e.g. field_keypoints.png). "
                             "If omitted, displays interactively.")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    draw(args.out, args.dpi)


if __name__ == "__main__":
    main()
