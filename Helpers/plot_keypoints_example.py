"""
Draws one annotated tb2026 broadcast frame with GT field keypoints overlaid,
colour-coded by landmark group (sideline/numbers/hashes, top/bottom), for
the thesis Materials section, matching the same play used in
plot_annotated_classes_example.py (clip_012 / play 12, presnap).

Only keypoints that fall inside the frame (0 <= x_norm, y_norm <= 1) are
drawn; visible (vis=2) points are filled circles, occluded (vis=1) points
are hollow circles, matching the 2-state visibility flag used in the
annotations (the third "not in the image" state is implicit -- those
points' normalized coordinates simply fall outside [0, 1]).

Usage:
  python plot_keypoints_example.py
"""

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import cv2
from pathlib import Path

MASTER_THESIS_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT = MASTER_THESIS_DIR / "thesis-template" / "figures" / "keypoints_example.png"

# clip_012 (play 12) starts at global frame 172 (tb2026_dataset_metadata.csv);
# this is the same presnap moment shown in plot_annotated_classes_example.py.
FRAME = PROJECT_DIR / "Data" / "Keypoints" / "frames" / "tb2026" / "frame_000172.png"
ANNOTATION = PROJECT_DIR / "Data" / "Keypoints" / "annotations" / "tb2026" / "labels" / "frame_000172.txt"

N_KPT = 62

# Same 6 landmark groups and colours as
# Field_Keypoint_Detection/utils/visualize_field_keypoints.py's GROUP_COLOR
# (kept as a local copy since Helpers/ scripts are self-contained).
GROUPS = {
    "sideline_top": (range(1, 14),  "#FF4444", "Top sideline (1-13)"),
    "sideline_bot": (range(14, 27), "#FF9900", "Bottom sideline (14-26)"),
    "num_top":      (range(27, 36), "#00CFFF", "Top numbers (27-35)"),
    "num_bot":      (range(36, 45), "#FFE600", "Bottom numbers (36-44)"),
    "hash_top":     (range(45, 54), "#44FF44", "Top hash (45-53)"),
    "hash_bot":     (range(54, 63), "#FF44FF", "Bottom hash (54-62)"),
}


def point_group(point_id: int) -> str:
    for name, (rng, _, _) in GROUPS.items():
        if point_id in rng:
            return name
    raise ValueError(point_id)


def load_keypoints(path: Path) -> list[tuple[int, float, float, int]]:
    """Returns (point_id, x_norm, y_norm, vis) for points that fall inside the frame."""
    vals = list(map(float, path.read_text().strip().split()))
    rest = vals[5:]  # skip the single-object bbox header
    pts = []
    for i in range(N_KPT):
        x, y, vis = rest[i * 3], rest[i * 3 + 1], int(rest[i * 3 + 2])
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and vis > 0:
            pts.append((i + 1, x, y, vis))
    return pts


img_bgr = cv2.imread(str(FRAME))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w = img_rgb.shape[:2]

keypoints = load_keypoints(ANNOTATION)

fig, ax = plt.subplots(figsize=(12.8, 7.2))
ax.imshow(img_rgb)
ax.axis("off")

for point_id, x_n, y_n, vis in keypoints:
    group = point_group(point_id)
    color = GROUPS[group][1]
    x_px, y_px = x_n * w, y_n * h
    if vis == 2:
        ax.scatter(x_px, y_px, c=color, s=110, zorder=4,
                   edgecolors="black", linewidths=0.8)
    else:  # occluded
        ax.scatter(x_px, y_px, facecolors="none", edgecolors=color, s=110,
                   linewidths=2.0, zorder=4)
    ax.text(x_px, y_px - 14, str(point_id), ha="center", va="bottom",
            fontsize=9, color=color, fontweight="bold", zorder=5)

group_handles = [
    mlines.Line2D([], [], color=color, marker="o", linestyle="None",
                  markersize=8, markeredgecolor="black", markeredgewidth=0.6, label=label)
    for _, color, label in GROUPS.values()
]
vis_handles = [
    mlines.Line2D([], [], color="black", marker="o", linestyle="None",
                  markersize=8, markerfacecolor="black", label="Visible"),
    mlines.Line2D([], [], color="black", marker="o", linestyle="None",
                  markersize=8, markerfacecolor="none", markeredgewidth=2.0, label="Occluded"),
]

legend1 = ax.legend(handles=group_handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
                     ncol=3, frameon=False, fontsize=10, labelcolor="black")
ax.add_artist(legend1)
ax.legend(handles=vis_handles, loc="upper center", bbox_to_anchor=(0.5, -0.14),
          ncol=2, frameon=False, fontsize=10, labelcolor="black")

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print("saved ->", OUT)
