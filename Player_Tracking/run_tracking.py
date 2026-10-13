"""
Runs SAM-3 tracking on all 107 tb2026 clips, initialized from ground-truth
player boxes instead of YOLO detections.

Using GT boxes for the frame-0 init isolates tracking-specific failures
(ID switches, mask drift) from player-detection failures, the same
isolation logic used for the H_gt/H_pred split in the keypoint-homography
evaluation.

Caches each clip's per-frame box positions (xyxy/tracker_id/class_id only,
no segmentation masks) to a pickle. Masks are dropped here: at full frame
resolution they run ~1-2GB per clip, which doesn't fit at this dataset's
scale, and downstream uses (yard-space projection for the Play-Type
Classification movement signal) only need box positions anyway.

Usage:
  python run_tracking.py
  python run_tracking.py --clips /path/to/other_clip_list.csv
"""

import argparse
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO

BASE      = Path(__file__).parent
DATA_ROOT = BASE.parent / "Data"
FRAMES_ROOT = DATA_ROOT / "Keypoints" / "frames" / "tb2026"
GT_ANN_DIR  = DATA_ROOT / "Players" / "annotations" / "tb2026"
GT_IMG_DIR  = DATA_ROOT / "Players" / "frames" / "tb2026"
META_CSV    = DATA_ROOT / "metadata" / "tb2026_dataset_metadata.csv"

# sam3_tracking.py lives in Pipeline/, reused as-is
sys.path.insert(0, str(BASE.parent / "Pipeline"))
from sam3_tracking import run_sam3_tracking  # noqa: E402

PLAYER_PT = BASE.parent / "Player_Detection" / "cross_stadium_results" / "3b_1280" / "train" / "weights" / "best.pt"

OUT_DIR = BASE / "output" / "tracking_cache"


def load_gt_detections(play_num: int) -> sv.Detections:
    """Ground-truth boxes for a clip's pre-snap frame, YOLO-format -> xyxy pixels."""
    ann_path = GT_ANN_DIR / f"frame_{play_num:03d}.txt"
    img_path = GT_IMG_DIR / f"frame_{play_num:03d}.jpg"

    import cv2
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


def track_clip(row: pd.Series, yolo_names: dict) -> dict:
    start, end = int(row["start_frame"]), int(row["end_frame"])
    frame_paths = [FRAMES_ROOT / f"frame_{i:06d}.png" for i in range(start, end + 1)]

    gt_detections = load_gt_detections(int(row["play_num"]))

    tmp_frames = Path(tempfile.mkdtemp())
    tmp_out    = Path(tempfile.mkdtemp())
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

    return strip_masks(all_frame_detections)


def strip_masks(all_frame_detections: dict) -> dict:
    """Keeps only xyxy/tracker_id/class_id per frame — see module docstring."""
    return {
        frame_idx: {
            "xyxy": det.xyxy,
            "tracker_id": det.tracker_id,
            "class_id": det.class_id,
        }
        for frame_idx, det in all_frame_detections.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", default=str(META_CSV))
    args = parser.parse_args()

    selected = pd.read_csv(args.clips)
    yolo_names = YOLO(str(PLAYER_PT)).names

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for _, row in selected.iterrows():
        clip_name = row["clip"].replace(".mp4", "")
        cache_path = OUT_DIR / f"{clip_name}.pkl"
        if cache_path.exists():
            print(f"skip {clip_name} (cached)")
            continue

        print(f"tracking {clip_name} ({row['n_frames']} frames)...")
        try:
            detections = track_clip(row, yolo_names)
            with open(cache_path, "wb") as f:
                pickle.dump({"detections": detections, "yolo_names": yolo_names}, f)
            print(f"  -> saved {cache_path}")
        except Exception as e:
            print(f"  !! failed on {clip_name}: {e} — skipping, will retry on next run")
            cache_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
