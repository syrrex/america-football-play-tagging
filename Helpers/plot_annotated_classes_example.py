"""
Draws one annotated tb2026 presnap frame with GT player boxes colour-coded by
class (qb/oline/skill/defense/ref), plus a legend, for the thesis Materials
section (labeled-dataset figure next to the class description around
`6_0_Materials.tex:113`).

Usage:
  python plot_annotated_classes_example.py
"""

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Helpers/ -> Project/ -> Master_Thesis/, then into the sibling thesis-template/ dir
MASTER_THESIS_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT = MASTER_THESIS_DIR / "thesis-template" / "figures" / "annotated_classes_example.png"

FRAME = PROJECT_DIR / "Data" / "Players" / "frames" / "tb2026" / "frame_012.jpg"
ANNOTATION = PROJECT_DIR / "Data" / "Players" / "annotations" / "tb2026" / "frame_012.txt"

# Same class order as Player_Detection/prepare_dataset.py's CLASS_NAMES and
# the colours used elsewhere in the pipeline (Pipeline/drawing.py's
# POSITION_COLOR_MAP) -- kept as a local copy since Helpers/ scripts are
# self-contained and don't cross-import from Pipeline/.
CLASS_NAMES = {0: "defense", 1: "oline", 2: "qb", 3: "ref", 4: "skill"}
LEGEND_LABELS = {
    "qb":      "Quarterback",
    "oline":   "Offensive Line",
    "skill":   "Skill",
    "defense": "Defense",
    "ref":     "Referee",
}
COLOR_MAP = {
    "qb":      "#FFFF00",   # yellow
    "skill":   "#FF8C00",   # orange
    "oline":   "#FF69B4",   # pink
    "defense": "#00BFFF",   # blue
    "ref":     "#32CD32",   # green
}


def load_boxes(path: Path, w: int, h: int) -> list[tuple[str, float, float, float, float]]:
    boxes = []
    for line in path.read_text().strip().splitlines():
        cls, cx, cy, bw, bh = map(float, line.split())
        name = CLASS_NAMES[int(cls)]
        x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
        boxes.append((name, x1, y1, x2, y2))
    return boxes


img_bgr = cv2.imread(str(FRAME))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w = img_rgb.shape[:2]

boxes = load_boxes(ANNOTATION, w, h)

fig, ax = plt.subplots(figsize=(12.8, 7.2))
ax.imshow(img_rgb)
ax.axis("off")

for name, x1, y1, x2, y2 in boxes:
    color = COLOR_MAP[name]
    rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                               linewidth=2.0, edgecolor=color, facecolor="none", zorder=3)
    ax.add_patch(rect)

legend_handles = [
    mpatches.Patch(edgecolor=COLOR_MAP[name], facecolor="none", linewidth=2.0, label=LEGEND_LABELS[name])
    for name in ("qb", "oline", "skill", "defense", "ref")
]
ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
          ncol=5, frameon=False, fontsize=11, handlelength=1.5, handleheight=1.5,
          labelcolor="black")

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print("saved ->", OUT)
