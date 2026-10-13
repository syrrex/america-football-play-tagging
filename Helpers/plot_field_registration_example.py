"""
Builds the two "Field Registration end-to-end" figure images
(sec:methods_field_registration): (a) the deployed keypoint model's
detections drawn on the broadcast frame, (b) the resulting bird's-eye view
with GT player boxes projected through the homography fitted from those
detections.

Uses the Q2 fold weights (finetune_quarter_results_1280/Q2), since clip_069
falls in Q2 -- Q2 was the held-out validation quarter for this fold, so the
model has not seen this frame during training.

Usage:
  python plot_field_registration_example.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from ultralytics import YOLO

MASTER_THESIS_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = MASTER_THESIS_DIR / "thesis-template" / "figures"

sys.path.insert(0, str(PROJECT_DIR / "Pipeline"))
from homography import compute_homography_from_keypoints, project_to_field_yards  # noqa: E402
from drawing import create_football_field  # noqa: E402

PLAY_NUM = 69
FRAME = PROJECT_DIR / "Data" / "Players" / "frames" / "tb2026" / f"frame_{PLAY_NUM:03d}.jpg"
GT_ANN = PROJECT_DIR / "Data" / "Players" / "annotations" / "tb2026" / f"frame_{PLAY_NUM:03d}.txt"

KPT_WEIGHTS = PROJECT_DIR / "Field_Keypoint_Detection" / "finetune_quarter_results_1280" / "Q2" / "train" / "weights" / "best.pt"
DET_CONF, KPT_CONF, IMGSZ = 0.25, 0.25, 1280

YOLO_NAMES = {0: "defense", 1: "oline", 2: "qb", 3: "ref", 4: "skill"}
CLASS_REF = 3  # excluded from the bird's-eye figure -- not load-bearing for anything downstream

# Same role colours/shapes as Pipeline/drawing.py's POSITION_COLOR_MAP / MARKER_SHAPE.
POSITION_COLOR_MAP = {
    "qb":      "#FFFF00",
    "skill":   "#FF8C00",
    "oline":   "#FF69B4",
    "defense": "#00BFFF",
}
MARKER_SHAPE = {
    "qb":      "*",
    "skill":   "o",
    "oline":   "s",
    "defense": "^",
}

# Same 6 landmark groups/colours as plot_keypoints_example.py.
GROUPS = {
    "sideline_top": (range(1, 14),  (68, 68, 255)),
    "sideline_bot": (range(14, 27), (0, 153, 255)),
    "num_top":      (range(27, 36), (255, 207, 0)),
    "num_bot":      (range(36, 45), (0, 230, 255)),
    "hash_top":     (range(45, 54), (68, 255, 68)),
    "hash_bot":     (range(54, 63), (255, 68, 255)),
}


def group_color(point_id: int) -> tuple:
    for _, (rng, color) in GROUPS.items():
        if point_id in rng:
            return color
    return (128, 128, 128)


def load_gt_boxes(path: Path, w: int, h: int):
    class_ids, boxes = [], []
    for line in path.read_text().strip().splitlines():
        cls, cx, cy, bw, bh = map(float, line.split())
        x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
        class_ids.append(int(cls))
        boxes.append([x1, y1, x2, y2])
    return np.array(boxes, dtype=np.float64), class_ids


def main():
    frame = cv2.imread(str(FRAME))
    h, w = frame.shape[:2]

    model = YOLO(str(KPT_WEIGHTS))
    results = model(frame, conf=DET_CONF, iou=0.5, max_det=1, imgsz=IMGSZ, verbose=False)
    kpts_data = results[0].keypoints.data[0].cpu().numpy()  # (62, 3)

    # (a) detected landmarks on the broadcast frame
    landmarks_img = frame.copy()
    for idx, (x, y, conf) in enumerate(kpts_data):
        if conf < KPT_CONF:
            continue
        point_id = idx + 1
        color = group_color(point_id)
        cv2.circle(landmarks_img, (int(x), int(y)), 8, color, -1, cv2.LINE_AA)
        cv2.circle(landmarks_img, (int(x), int(y)), 8, (0, 0, 0), 2, cv2.LINE_AA)

    out_a = FIGURES_DIR / "registration_landmarks.jpg"
    cv2.imwrite(str(out_a), landmarks_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"saved -> {out_a}  ({int((kpts_data[:, 2] >= KPT_CONF).sum())} keypoints above conf {KPT_CONF})")

    # (b) bird's-eye projection of GT player boxes through the fitted H
    H, n_inliers = compute_homography_from_keypoints(kpts_data, conf_thresh=KPT_CONF, img_shape=(h, w))
    if H is None:
        raise RuntimeError("Homography estimation failed on this frame")

    boxes, class_ids = load_gt_boxes(GT_ANN, w, h)
    positions = project_to_field_yards(boxes, H)

    fig, ax = create_football_field(figsize=(12, 6.33))
    for (x_yd, y_yd), cls_id in zip(positions, class_ids):
        name = YOLO_NAMES.get(cls_id, "unknown")
        if cls_id == CLASS_REF:
            continue
        ax.scatter(x_yd, y_yd, c=POSITION_COLOR_MAP[name], marker=MARKER_SHAPE[name],
                   s=120, zorder=5, edgecolors="white", linewidths=0.5)

    legend_handles = [
        mlines.Line2D([], [], color=color, marker=MARKER_SHAPE[name], linestyle="None",
                      markersize=10, markeredgecolor="white", markeredgewidth=0.5, label=name)
        for name, color in POSITION_COLOR_MAP.items()
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
              ncol=4, frameon=False, fontsize=11, labelcolor="black")

    out_b = FIGURES_DIR / "registration_bev.png"
    fig.savefig(out_b, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved -> {out_b}  (H from {n_inliers} RANSAC inliers)")


if __name__ == "__main__":
    main()
