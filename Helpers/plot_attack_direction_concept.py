"""
Draws the attack-direction concept figure for the thesis Methods chapter
(sec:methods_yard_hash_analysis): a top-down bird's-eye diagram showing why
comparing the Oline's and defense's downfield mean position resolves which
way the offense is facing -- the defense always lines up on the side of the
LOS the offense is attacking toward, matching the visual style of
plot_tacklebox_concept.py.

Usage:
  python plot_attack_direction_concept.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

MASTER_THESIS_DIR = Path(__file__).resolve().parents[2]
OUT = MASTER_THESIS_DIR / "thesis-template" / "figures" / "attack_direction_concept.png"

fig, ax = plt.subplots(figsize=(10.8, 7.3))

# ── Field background, matching tacklebox_concept.py style ──────────────────
X0, X1 = 0, 36    # yards shown laterally
Y0, Y1 = 12, 42   # yards shown downfield

ax.set_xlim(X0, X1)
ax.set_ylim(Y0, Y1)
ax.set_facecolor("white")
ax.axis("off")

for y in range(Y0, Y1 + 1):
    lw = 1.1 if y % 10 == 0 else 0.4
    color = "#bbbbbb" if y % 10 == 0 else "#e6e6e6"
    ax.plot([X0, X1], [y, y], color=color, linewidth=lw, zorder=0)

for y in range(Y0, Y1):
    for hx in (13.5, 22.5):
        ax.plot([hx - 0.35, hx + 0.35], [y + 0.5, y + 0.5], color="#cfcfcf", linewidth=0.8, zorder=0)

LOS = 22.0     # Oline downfield mean
DEF_Y = 28.5   # Defense downfield mean -- on the side the offense attacks toward

def player(x, y, label, r=0.62, fc="white", ec="black", lw=2.0, fontsize=8.5):
    p = mpatches.Circle((x, y), r, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(p)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, weight="bold", zorder=4)

# Oline, set up just behind the LOS
ol_xs = {"LT": 15.0, "LG": 16.5, "C": 18.0, "RG": 19.5, "RT": 21.0}
for name, x in ol_xs.items():
    player(x, LOS, name)

# Defense front, staggered laterally (not perfectly aligned, as in a real front),
# but consistently on the far side -- their downfield mean is what the method uses
def_xs = [(14.0, 28.0), (17.0, 29.2), (19.5, 27.6), (22.5, 28.8), (25.0, 27.2)]
for x, y in def_xs:
    player(x, y, "D", r=0.55, fc="#f5e6e6", ec="#8a2f2f", fontsize=9)

# ── Mean reference lines ────────────────────────────────────────────────────
LABEL_BOX = dict(facecolor="white", edgecolor="none", pad=1.5)

ax.plot([X0, X1], [LOS, LOS], color="#2f6f9a", linewidth=1.4, linestyle="--", zorder=1)
ax.text(X1 - 0.6, LOS, "Oline mean", fontsize=10, color="#2f6f9a",
        ha="right", va="center", style="italic", zorder=2, bbox=LABEL_BOX)

ax.plot([X0, X1], [DEF_Y, DEF_Y], color="#8a2f2f", linewidth=1.4, linestyle="--", zorder=1)
ax.text(X1 - 0.6, DEF_Y, "Defense mean", fontsize=10, color="#8a2f2f",
        ha="right", va="center", style="italic", zorder=2, bbox=LABEL_BOX)

# ── Attack-direction arrow, from Oline mean toward Defense mean ────────────
ARROW_X = 5.0
ax.annotate("", xy=(ARROW_X, DEF_Y), xytext=(ARROW_X, LOS),
            arrowprops=dict(arrowstyle="-|>", color="black", linewidth=2.0), zorder=5)
ax.text(ARROW_X - 1.0, (LOS + DEF_Y) / 2, "attack\ndirection", fontsize=10.5, weight="bold",
        color="black", ha="center", va="center", rotation=90)

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print("saved ->", OUT)
