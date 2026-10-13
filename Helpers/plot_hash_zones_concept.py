"""
Draws the hash-mark concept figure for the thesis Background chapter
(sec:field_position): the same top-down field renderer used for
field_keypoints.png (Field_Keypoint_Detection/utils/visualize_field_keypoints.py),
with the hash-mark tick lines recolored yellow to highlight them, plus
dimension callouts for every distance named in the surrounding text (field
length/end zones, field width, yard-number offset, hash offset). No
keypoint dots or example ball markers -- the left/middle/right positioning
is described in the accompanying text.

Usage:
  python plot_hash_zones_concept.py
"""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

MASTER_THESIS_DIR = Path(__file__).resolve().parents[2]
OUT = MASTER_THESIS_DIR / "thesis-template" / "figures" / "hash_zones_concept.png"

HASH_COLOR = "#FFE600"  # yellow, matches field_keypoints.png's NUMBER_BOTTOM group color
DIM_COLOR = "#FFFFFF"


def dim_h(ax, x0, x1, y, label, color=DIM_COLOR, fontsize=13, label_offset=-1.6):
    """Horizontal dimension line with end ticks, labeled at its midpoint."""
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="<->", color=color, linewidth=1.1), zorder=6)
    for x in (x0, x1):
        ax.plot([x, x], [y - 0.8, y + 0.8], color=color, linewidth=1.1, zorder=6)
    ax.text((x0 + x1) / 2, y + label_offset, label, ha="center", va="center",
            fontsize=fontsize, color=color, zorder=6)


def dim_v(ax, y0, y1, x, label, color=DIM_COLOR, fontsize=13, label_offset=-2.6):
    """Vertical dimension line with end ticks, labeled at its midpoint."""
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="<->", color=color, linewidth=1.1), zorder=6)
    for y in (y0, y1):
        ax.plot([x - 0.8, x + 0.8], [y, y], color=color, linewidth=1.1, zorder=6)
    ax.text(x + label_offset, (y0 + y1) / 2, label, ha="center", va="center",
            fontsize=fontsize, color=color, rotation=90, zorder=6)


def create_football_field(figsize=(18, 9)):
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(patches.Rectangle((0, 0), 120, 53.3,
                                    linewidth=0.1, edgecolor="r",
                                    facecolor="darkgreen", zorder=0))
    for x0 in (0, 110):
        ax.add_patch(patches.Rectangle((x0, 0), 10, 53.3,
                                        linewidth=0.1, edgecolor="r",
                                        facecolor="blue", alpha=0.2, zorder=0))
    ax.plot(
        [10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80, 80, 90, 90, 100, 100, 110, 110, 0, 0, 120, 120],
        [0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3],
        color="white", linewidth=0.8,
    )
    # Additional 5-yard lines, thinner than the numbered 10-yard lines above
    for x in range(15, 110, 10):
        ax.plot([x, x], [0, 53.3], color="white", linewidth=0.4)
    for x in range(20, 110, 10):
        n = x if x <= 50 else 120 - x
        ax.text(x - 0.5, 8.5, str(n - 10), ha="center", va="center", fontsize=30, color="white", rotation=180)
        ax.text(x, 44.8, str(n - 10), ha="center", va="center", fontsize=30, color="white")

    # Hash-mark ticks, highlighted in yellow instead of the usual white
    y_hash_top, y_hash_bot = 20.0, 33.3
    for x in range(11, 110):
        for yh in (y_hash_top, y_hash_bot):
            ax.plot([x, x], [yh - 0.3, yh + 0.3], color=HASH_COLOR, linewidth=1.1, zorder=2)
        ax.plot([x, x], [0.2, 0.5], color="white", linewidth=0.6)
        ax.plot([x, x], [52.8, 53.1], color="white", linewidth=0.6)

    # ── Dimension callouts, one per distance named in the surrounding text ──
    # Field length and width sit in the white margin outside the green field
    # rectangle, so they use a dark color instead of DIM_COLOR (white would be
    # invisible against that background).
    MARGIN_COLOR = "#222222"

    # Field length: 10 yd end zone + 100 yd playing field + 10 yd end zone = 120 yd
    dim_h(ax, 0, 10, -3.2, "10 yd", color=MARGIN_COLOR)
    dim_h(ax, 10, 110, -3.2, "100 yd", color=MARGIN_COLOR)
    dim_h(ax, 110, 120, -3.2, "10 yd", color=MARGIN_COLOR)
    dim_h(ax, 0, 120, -6.2, "120 yd", color=MARGIN_COLOR, label_offset=-1.8)

    # Field width: 53 1/3 yd, sideline to sideline
    dim_v(ax, 0, 53.3, -6.0, r"$53\frac{1}{3}$ yd", color=MARGIN_COLOR)

    # Hash offset: 20 yd in from each sideline (shown once per side)
    dim_v(ax, 0, y_hash_top, 15, "20 yd", fontsize=11, label_offset=-1.8)
    dim_v(ax, y_hash_bot, 53.3, 15, "20 yd", fontsize=11, label_offset=-1.8)

    # Yard-number offset: ~8 yd in from each sideline
    dim_v(ax, 0, 8.5, 33, "8 yd", fontsize=11, label_offset=-1.8)

    ax.set_xlim(-10, 120)
    ax.set_ylim(58.3, -9.5)
    ax.axis("off")
    return fig, ax


fig, ax = create_football_field(figsize=(18, 10))
plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print("saved ->", OUT)
