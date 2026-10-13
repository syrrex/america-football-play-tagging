"""
Isolates how much box "looseness" (the mAP50 vs. mAP50-95 gap in Player
Detection's evaluation) actually costs Formation Classification and
Yard-Line/Hash Analysis, which both project a detected box's own foot point
straight into the bird's-eye view (Section~\ref{sec:methods_formation_classification},
Section~\ref{sec:methods_yard_hash_analysis} -- unlike Play-Type
Classification, which projects a SAM3-mask-derived box instead).

For each of the 107 tb2026 presnap frames: run the deployed player detector,
match each detection to its GT box by IoU (>=0.5, greedy one-to-one, so a
missed/false-positive detection -- a different, already-quantified error
source -- never counts here), then project BOTH the GT box's foot point and
the matched detected box's foot point through the SAME ground-truth
homography. The distance between the two projections isolates the box-
tightness error alone, with detection/count error and homography error both
held fixed.

Usage:
  python box_tightness_error.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PLAY_ANALYSIS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLAY_ANALYSIS))
sys.path.insert(0, str(PLAY_ANALYSIS.parent / "Pipeline"))

from classify_formation import (  # noqa: E402
    META_CSV, GT_IMG_DIR, GT_ANN_DIR, GT_KPT_DIR,
    PLAYER_DET_WEIGHTS, DET_CONF, IMGSZ,
    load_gt_boxes, load_gt_keypoints,
)
from homography import compute_homography_from_keypoints, project_to_field_yards  # noqa: E402


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4), b: (M,4) -> (N,M) IoU matrix."""
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def greedy_match(det_xyxy: np.ndarray, gt_xyxy: np.ndarray, iou_thresh: float = 0.5):
    """Returns list of (det_idx, gt_idx) pairs, one-to-one, highest IoU first."""
    if len(det_xyxy) == 0 or len(gt_xyxy) == 0:
        return []
    ious = iou_xyxy(det_xyxy, gt_xyxy)
    pairs = []
    used_det, used_gt = set(), set()
    flat = [(ious[i, j], i, j) for i in range(ious.shape[0]) for j in range(ious.shape[1])]
    flat.sort(reverse=True)
    for iou, i, j in flat:
        if iou < iou_thresh:
            break
        if i in used_det or j in used_gt:
            continue
        used_det.add(i)
        used_gt.add(j)
        pairs.append((i, j))
    return pairs


def foot_points(xyxy: np.ndarray) -> np.ndarray:
    fx = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    fy = xyxy[:, 3]
    return np.stack([fx, fy], axis=1)


def main():
    from ultralytics import YOLO
    model = YOLO(str(PLAYER_DET_WEIGHTS))

    meta = pd.read_csv(META_CSV)
    errors_yd = []
    n_frames_used = 0
    n_gt_total = 0
    n_matched_total = 0

    for _, row in meta.iterrows():
        pnum, start = int(row["play_num"]), int(row["start_frame"])
        img_path = GT_IMG_DIR / f"frame_{pnum:03d}.jpg"
        ann_path = GT_ANN_DIR / f"frame_{pnum:03d}.txt"
        kpt_path = GT_KPT_DIR / f"frame_{start:06d}.txt"
        if not (img_path.exists() and ann_path.exists() and kpt_path.exists()):
            continue

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        gt_rows = load_gt_boxes(ann_path, w, h)
        gt_boxes = [[x1, y1, x2, y2] for _, x1, y1, x2, y2 in gt_rows]
        n_gt_total += len(gt_boxes)
        if len(gt_boxes) == 0:
            continue

        gt_kpts = load_gt_keypoints(kpt_path, w, h)
        H, _ = compute_homography_from_keypoints(gt_kpts, conf_thresh=1.0, min_points=4, img_shape=(h, w))
        if H is None:
            continue

        det_res = model(str(img_path), conf=DET_CONF, iou=0.5, imgsz=IMGSZ, verbose=False)[0]
        if det_res.boxes is None or len(det_res.boxes) == 0:
            n_frames_used += 1
            continue
        det_xyxy = det_res.boxes.xyxy.cpu().numpy()

        pairs = greedy_match(det_xyxy, np.array(gt_boxes), iou_thresh=0.5)
        n_matched_total += len(pairs)
        if not pairs:
            n_frames_used += 1
            continue

        det_idx = [p[0] for p in pairs]
        gt_idx = [p[1] for p in pairs]

        det_feet = foot_points(det_xyxy[det_idx])
        gt_feet = foot_points(np.array(gt_boxes)[gt_idx])

        det_field = project_to_field_yards(
            np.hstack([det_feet, det_feet]), H  # xyxy with x1=x2, y1=y2 -> foot point itself
        )
        gt_field = project_to_field_yards(
            np.hstack([gt_feet, gt_feet]), H
        )

        for (dx, dy), (gx, gy) in zip(det_field, gt_field):
            errors_yd.append(float(np.hypot(dx - gx, dy - gy)))

        n_frames_used += 1

    errors_yd = np.array(errors_yd)
    print(f"Frames used: {n_frames_used}")
    print(f"GT players total: {n_gt_total}, matched (IoU>=0.5): {n_matched_total} "
          f"({100 * n_matched_total / n_gt_total:.1f}%)")
    print(f"Box-tightness position error (yd), matched players only:")
    print(f"  mean   = {errors_yd.mean():.3f}")
    print(f"  median = {np.median(errors_yd):.3f}")
    print(f"  p90    = {np.percentile(errors_yd, 90):.3f}")
    print(f"  max    = {errors_yd.max():.3f}")


if __name__ == "__main__":
    main()
