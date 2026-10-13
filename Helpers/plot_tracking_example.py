"""
Runs SAM-3 tracking (GT-box init, same convention as
Player_Tracking/run_tracking.py) on clip_042 and renders 4 evenly-spaced
frames (snap, two mid-play checkpoints, last frame) with each player's
segmentation mask colored by role (same POSITION_COLOR_MAP used in the
player-detection figure), for the Segmentation and Tracking figure in the
thesis Methods chapter (sec:methods_tracking). The per-frame tracker ID is
still drawn as a small number on each mask, so identity can be checked
player-by-player even though teammates sharing a role share a color.

Unlike Player_Tracking/run_tracking.py, masks are kept (not stripped) since
they are the whole point of this figure.

Usage:
  python plot_tracking_example.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

MASTER_THESIS_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = MASTER_THESIS_DIR / "thesis-template" / "figures"

DATA_ROOT   = PROJECT_DIR / "Data"
FRAMES_ROOT = DATA_ROOT / "Keypoints" / "frames" / "tb2026"
GT_ANN_DIR  = DATA_ROOT / "Players" / "annotations" / "tb2026"
GT_IMG_DIR  = DATA_ROOT / "Players" / "frames" / "tb2026"

PLAY_NUM = 42
START_FRAME, END_FRAME = 800, 815  # clip_042, tb2026_dataset_metadata.csv

sys.path.insert(0, str(PROJECT_DIR / "Pipeline"))
from sam3_tracking import run_sam3_tracking  # noqa: E402

import supervision as sv  # noqa: E402
from ultralytics import YOLO  # noqa: E402

PLAYER_PT = PROJECT_DIR / "Player_Detection" / "cross_stadium_results" / "3b_1280" / "train" / "weights" / "best.pt"

MASK_OPACITY = 0.5
CONTOUR_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2

# Same role colours as the player-detection figure
# (Pipeline/drawing.py's POSITION_COLOR_MAP / plot_annotated_classes_example.py),
# kept as a local copy since Helpers/ scripts are self-contained.
COLOR_MAP = {
    "qb":      "#FFFF00",   # yellow
    "skill":   "#FF8C00",   # orange
    "oline":   "#FF69B4",   # pink
    "defense": "#00BFFF",   # blue
    "ref":     "#32CD32",   # green
}


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def load_gt_detections(play_num: int) -> sv.Detections:
    ann_path = GT_ANN_DIR / f"frame_{play_num:03d}.txt"
    img_path = GT_IMG_DIR / f"frame_{play_num:03d}.jpg"

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    class_ids, boxes = [], []
    for line in ann_path.read_text().strip().splitlines():
        cls, cx, cy, bw, bh = map(float, line.split())
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        class_ids.append(int(cls))
        boxes.append([x1, y1, x2, y2])

    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
    )


def draw_tracked_frame(frame_bgr: np.ndarray, detections: sv.Detections, yolo_names: dict) -> np.ndarray:
    colors = [
        _hex_to_bgr(COLOR_MAP.get(yolo_names.get(int(cid), "").lower(), "#808080"))
        for cid in detections.class_id
    ]

    overlay = frame_bgr.copy()
    for mask, color in zip(detections.mask, colors):
        overlay[mask] = color
    out = cv2.addWeighted(overlay, MASK_OPACITY, frame_bgr, 1 - MASK_OPACITY, 0)

    for mask, tid, color in zip(detections.mask, detections.tracker_id, colors):
        mask_u8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, CONTOUR_THICKNESS)

        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue
        tx, ty = int(xs.min()), max(int(ys.min()) - 6, 15)
        text = str(int(tid))
        cv2.putText(out, text, (tx, ty), FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS + 3, cv2.LINE_AA)
        cv2.putText(out, text, (tx, ty), FONT, FONT_SCALE, color, FONT_THICKNESS, cv2.LINE_AA)
    return out


def main():
    n_frames = END_FRAME - START_FRAME + 1  # 20
    gt_detections = load_gt_detections(PLAY_NUM)
    yolo_names = YOLO(str(PLAYER_PT)).names

    frame_paths = [FRAMES_ROOT / f"frame_{i:06d}.png" for i in range(START_FRAME, END_FRAME + 1)]

    tmp_frames = Path(tempfile.mkdtemp())
    tmp_out = Path(tempfile.mkdtemp())
    try:
        for i, fp in enumerate(frame_paths):
            shutil.copy(fp, tmp_frames / f"frame_{i:06d}.png")

        all_frame_detections, _ = run_sam3_tracking(
            source_dir=tmp_frames,
            output_dir=tmp_out,
            initial_detections=gt_detections,
            yolo_names=yolo_names,
        )
    finally:
        shutil.rmtree(tmp_frames, ignore_errors=True)
        shutil.rmtree(tmp_out, ignore_errors=True)

    # Snap, ~1/3, ~2/3, last -- matches t=0 / t1 / t2 / T in the figure.
    checkpoints = {
        "t0": 0,
        "t1": round((n_frames - 1) / 3),
        "t2": round(2 * (n_frames - 1) / 3),
        "T":  n_frames - 1,
    }

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for label, local_idx in checkpoints.items():
        global_idx = START_FRAME + local_idx
        frame = cv2.imread(str(FRAMES_ROOT / f"frame_{global_idx:06d}.png"))
        dets = all_frame_detections.get(local_idx)
        if dets is not None and len(dets) > 0:
            frame = draw_tracked_frame(frame, dets, yolo_names)
        out_path = FIGURES_DIR / f"tracking_{label}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"saved -> {out_path}  (local frame {local_idx}, global frame {global_idx})")


if __name__ == "__main__":
    main()
