"""
Runs SAM-3 tracking on ALL 107 tb2026 clips, initialized from the deployed
player detector's own frame-0 detections instead of ground-truth boxes.

Counterpart to run_tracking.py (GT-box-initialized, all 107 clips). This
is the phase-2 "real pipeline capability" input for
Play-Type Classification's self-detected ablation: initializing from the
same Stage-3b/1280 detector used by the other Play_Analysis self-detected
ablations, so tracking inherits real detection error from frame 0, not
just SAM-3's own tracking error.

Usage:
  python run_tracking_selfdetected.py
"""

import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO

BASE      = Path(__file__).parent
DATA_ROOT = BASE.parent / "Data"
FRAMES_ROOT = DATA_ROOT / "Keypoints" / "frames" / "tb2026"
GT_IMG_DIR  = DATA_ROOT / "Players" / "frames" / "tb2026"
META_CSV    = DATA_ROOT / "metadata" / "tb2026_dataset_metadata.csv"

sys.path.insert(0, str(BASE.parent / "Pipeline"))
from sam3_tracking import run_sam3_tracking  # noqa: E402

sys.path.insert(0, str(BASE.parent / "Play_Analysis"))
from classify_formation import PLAYER_DET_WEIGHTS, DET_CONF, IMGSZ  # noqa: E402

OUT_DIR = BASE / "output" / "tracking_cache_selfdetected"


def detect_init_boxes(model: YOLO, play_num: int) -> sv.Detections:
    """Runs the deployed player detector on the presnap frame — same model,
    same conf threshold as classify_formation.py/yard_hash_analysis.py's
    self-detected ablations, for a consistent single deployed-pipeline story."""
    img_path = GT_IMG_DIR / f"frame_{play_num:03d}.jpg"
    res = model(str(img_path), conf=DET_CONF, iou=0.5, imgsz=IMGSZ, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return sv.Detections(xyxy=np.zeros((0, 4), dtype=np.float32), class_id=np.zeros((0,), dtype=int))
    return sv.Detections(
        xyxy=res.boxes.xyxy.cpu().numpy().astype(np.float32),
        class_id=res.boxes.cls.cpu().numpy().astype(int),
    )


def track_clip(row: pd.Series, model: YOLO, yolo_names: dict) -> dict:
    start, end = int(row["start_frame"]), int(row["end_frame"])
    frame_paths = [FRAMES_ROOT / f"frame_{i:06d}.png" for i in range(start, end + 1)]

    init_detections = detect_init_boxes(model, int(row["play_num"]))

    tmp_frames = Path(tempfile.mkdtemp())
    tmp_out    = Path(tempfile.mkdtemp())
    try:
        for i, fp in enumerate(frame_paths):
            shutil.copy(fp, tmp_frames / f"frame_{i:06d}.png")

        all_frame_detections, _ = run_sam3_tracking(
            source_dir=tmp_frames,
            output_dir=tmp_out,
            initial_detections=init_detections,
            yolo_names=yolo_names,
        )
    finally:
        shutil.rmtree(tmp_frames, ignore_errors=True)
        shutil.rmtree(tmp_out, ignore_errors=True)

    return strip_masks(all_frame_detections)


def strip_masks(all_frame_detections: dict) -> dict:
    return {
        frame_idx: {
            "xyxy": det.xyxy,
            "tracker_id": det.tracker_id,
            "class_id": det.class_id,
        }
        for frame_idx, det in all_frame_detections.items()
    }


def main():
    meta = pd.read_csv(META_CSV)
    model = YOLO(str(PLAYER_DET_WEIGHTS))
    yolo_names = model.names

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for _, row in meta.iterrows():
        clip_name = row["clip"].replace(".mp4", "")
        cache_path = OUT_DIR / f"{clip_name}.pkl"
        if cache_path.exists():
            print(f"skip {clip_name} (cached)")
            continue

        print(f"tracking {clip_name} ({row['n_frames']} frames)...")
        try:
            detections = track_clip(row, model, yolo_names)
            with open(cache_path, "wb") as f:
                pickle.dump({"detections": detections, "yolo_names": yolo_names}, f)
            print(f"  -> saved {cache_path}")
        except Exception as e:
            print(f"  !! failed on {clip_name}: {e} — skipping, will retry on next run")
            cache_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
