"""
Draws the tackle-box concept figure for the thesis Background chapter
(sec:tacklebox): a Doubles-formation top-down diagram matching the visual
style of the existing DOUBLES.png/TRIO.png formation figures, with the
tackle box overlaid to show which players count as attached (inside/on the
box) vs. detached (WRs split outside it).

Usage:
  python plot_tacklebox_concept.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Helpers/ -> Project/ -> Master_Thesis/, then into the sibling thesis-template/ dir
MASTER_THESIS_DIR = Path(__file__).resolve().parents[2]
OUT = MASTER_THESIS_DIR / "thesis-template" / "figures" / "tacklebox_concept.png"

fig, ax = plt.subplots(figsize=(10.8, 7.3))

# ── Field background, matching DOUBLES.png/TRIO.png style ──────────────────
X0, X1 = 0, 36   # yards shown laterally
Y0, Y1 = 12, 42  # yards shown downfield

ax.set_xlim(X0, X1)
ax.set_ylim(Y0, Y1)
ax.set_facecolor("white")
ax.axis("off")

# thin 1-yard lines (no field numbers)
for y in range(Y0, Y1 + 1):
    lw = 1.1 if y % 10 == 0 else 0.4
    color = "#bbbbbb" if y % 10 == 0 else "#e6e6e6"
    ax.plot([X0, X1], [y, y], color=color, linewidth=lw, zorder=0)

# hash marks (short ticks) at two lateral bands, one per yard
for y in range(Y0, Y1):
    for hx in (13.5, 22.5):
        ax.plot([hx - 0.35, hx + 0.35], [y + 0.5, y + 0.5], color="#cfcfcf", linewidth=0.8, zorder=0)

LOS = 26.0  # line of scrimmage, yards

# ── Tackle box rectangle (drawn first, behind players) ──────────────────────
BOX_LO, BOX_HI = 14.3, 21.7   # lateral edges, bounded by the tackles (LT=15.0, RT=21.0)
BOX_BACK = 22.3                # backfield extent (where RB sits)
box = mpatches.FancyBboxPatch((BOX_LO, BOX_BACK), BOX_HI - BOX_LO, LOS - BOX_BACK + 0.6,
                               boxstyle="round,pad=0,rounding_size=0.15",
                               facecolor="#fdf0d8", edgecolor="#c98a1f",
                               linewidth=1.8, linestyle="--", zorder=1)
ax.add_patch(box)
ax.text((BOX_LO + BOX_HI) / 2, BOX_BACK - 0.9, "tackle box", fontsize=12, color="#a5680f",
         ha="center", va="top", style="italic", weight="bold", zorder=1)

# ── Players ──────────────────────────────────────────────────────────────
def player(x, y, label, r=0.62, shape="circle", fc="white", ec="black", lw=2.0, fontsize=10):
    if shape == "circle":
        p = mpatches.Circle((x, y), r, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    else:
        p = mpatches.Rectangle((x - r*0.85, y - r*0.85), r*1.7, r*1.7,
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(p)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, weight="bold", zorder=4)

# Players line up behind the ball, not straddling it: their forward (downfield-
# facing) edge touches the LOS rather than the LOS cutting through their centre.
PLAYER_R = 0.62
ON_LINE_Y = LOS - PLAYER_R

# offensive line, set up just behind the LOS
ol_xs = {"LT": 15.0, "LG": 16.5, "C": 18.0, "RG": 19.5, "RT": 21.0}
for name, x in ol_xs.items():
    shape = "square" if name == "C" else "circle"
    player(x, ON_LINE_Y, name, r=PLAYER_R, shape=shape, fontsize=8.5)

# Doubles: two WR per side, detached (well outside the tackle box)
player(10.5, ON_LINE_Y, "WR", r=PLAYER_R, fontsize=9)
player(5.0, ON_LINE_Y, "WR", r=PLAYER_R, fontsize=9)
player(25.5, ON_LINE_Y, "WR", r=PLAYER_R, fontsize=9)
player(31.0, ON_LINE_Y, "WR", r=PLAYER_R, fontsize=9)

# backfield: QB and RB, close together, inside the tackle box (attached by alignment)
player(18.0, LOS - 1.8, "QB", fontsize=9)
player(18.0, LOS - 3.0, "RB", fontsize=9)

# LOS line
ax.plot([X0, X1], [LOS, LOS], color="#9a9a9a", linewidth=1.3, zorder=0.5)
ax.text(X1 - 0.6, LOS + 0.4, "LOS", fontsize=10, color="#8a8a8a", ha="right", va="bottom", style="italic")

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print("saved ->", OUT)
