"""
Quarter-based 4-fold cross-validation for Yard-Line and Hash Analysis.

Same rationale and fold structure as cv_formation.py: refit the
accuracy-tuned constants (HASH_MID_BAND, LOS_STANCE_OFFSET) using only 3
quarters' clips, evaluate on the held-out quarter, pool all four folds'
held-out predictions into one final figure.

HASH_MID_BAND and LOS_STANCE_OFFSET affect *disjoint* outputs -- the first
only ever feeds classify_hash() (via los_y), the second only ever feeds
classify_yardline() (via los_x) -- so they can be grid-searched
independently rather than jointly, with no loss versus a joint search.

OUTLIER_CAP is NOT refit per fold, same reasoning as Formation's
cv_formation.py: it's a physical-plausibility bound (largest legitimate
per-point deviation from the median seen in GT), not a value chosen by
maximizing accuracy, so it's reused as-is via the production _robust_mean().

Usage:
  python cv_yardhash.py
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
    FIELD_KPT_BASE, PLAYER_DET_WEIGHTS, DET_CONF, KPT_CONF, IMGSZ,
    load_gt_boxes, load_gt_keypoints, CLASS_OLINE,
)
from yard_hash_analysis import (  # noqa: E402
    CLASS_DEFENSE, FIELD_MID_X, FIELD_MID_Y, _robust_mean, yardline_error,
)
from homography import compute_homography_from_keypoints, project_to_field_yards  # noqa: E402

QUARTERS = ["Q1", "Q2", "Q3", "Q4"]


# ── Parameterized geometry (OUTLIER_CAP fixed via the imported _robust_mean) ─

def attack_direction_p(oline_pos, def_pos):
    if len(oline_pos) < 3 or len(def_pos) < 1:
        return None
    los_x = _robust_mean([p[0] for p in oline_pos])
    def_x = _robust_mean([p[0] for p in def_pos])
    return "inc" if def_x > los_x else "dec"


def classify_yardline_p(oline_pos, def_pos, los_stance_offset):
    direction = attack_direction_p(oline_pos, def_pos)
    if direction is None:
        return None
    los_x = _robust_mean([p[0] for p in oline_pos])
    los_x += los_stance_offset if direction == "inc" else -los_stance_offset
    los_x = max(10.0 + 1e-3, min(110.0 - 1e-3, los_x))
    if direction == "inc":
        return (10 - los_x) if los_x <= FIELD_MID_X else (110 - los_x)
    else:
        return (los_x - 110) if los_x >= FIELD_MID_X else (los_x - 10)


def classify_hash_p(oline_pos, def_pos, hash_mid_band):
    direction = attack_direction_p(oline_pos, def_pos)
    if direction is None:
        return None
    los_y = _robust_mean([p[1] for p in oline_pos])
    if abs(los_y - FIELD_MID_Y) < hash_mid_band:
        return "M"
    is_bot = los_y > FIELD_MID_Y
    if direction == "dec":
        return "L" if is_bot else "R"
    else:
        return "L" if not is_bot else "R"


# ── Precompute per-clip positions once per (box_source, kpt_source) config ──

def collect_clips(box_source: str, kpt_source: str) -> list[dict]:
    meta = pd.read_csv(META_CSV)
    player_model = None
    kpt_models = None
    if box_source == "detected" or kpt_source == "detected":
        from ultralytics import YOLO
        if box_source == "detected":
            player_model = YOLO(str(PLAYER_DET_WEIGHTS))
        if kpt_source == "detected":
            kpt_models = {q: YOLO(str(FIELD_KPT_BASE / q / "train" / "weights" / "best.pt")) for q in QUARTERS}

    rows = []
    for _, row in meta.iterrows():
        pnum, start = int(row["play_num"]), int(row["start_frame"])
        img_path = GT_IMG_DIR / f"frame_{pnum:03d}.jpg"
        if not img_path.exists() or pd.isna(row["HASH"]):
            continue
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        if kpt_source == "gt":
            kpt_path = GT_KPT_DIR / f"frame_{start:06d}.txt"
            if not kpt_path.exists():
                continue
            kpts = load_gt_keypoints(kpt_path, w, h)
            H, _ = compute_homography_from_keypoints(kpts, conf_thresh=1.0, min_points=4, img_shape=(h, w))
        else:
            kpt_model = kpt_models[row["QTR"]]
            kpt_res = kpt_model(str(img_path), imgsz=IMGSZ, conf=0.10, max_det=1, verbose=False)[0]
            if kpt_res.keypoints is None or len(kpt_res.keypoints.data) == 0:
                continue
            kpts_data = kpt_res.keypoints.data[0].cpu().numpy()
            H, _ = compute_homography_from_keypoints(kpts_data, conf_thresh=KPT_CONF, min_points=4, img_shape=(h, w))
        if H is None:
            continue

        if box_source == "gt":
            ann_path = GT_ANN_DIR / f"frame_{pnum:03d}.txt"
            if not ann_path.exists():
                continue
            boxes = load_gt_boxes(ann_path, w, h)
            xyxy = np.array([[x1, y1, x2, y2] for _, x1, y1, x2, y2 in boxes])
            cids = np.array([c for c, *_ in boxes])
            positions = project_to_field_yards(xyxy, H)
        else:
            det_res = player_model(str(img_path), conf=DET_CONF, iou=0.5, imgsz=IMGSZ, verbose=False)[0]
            if det_res.boxes is not None and len(det_res.boxes) > 0:
                xyxy = det_res.boxes.xyxy.cpu().numpy()
                cids = det_res.boxes.cls.cpu().numpy().astype(int)
                positions = project_to_field_yards(xyxy, H)
            else:
                positions, cids = [], np.array([])

        oline_pos = [p for p, c in zip(positions, cids) if c == CLASS_OLINE]
        def_pos = [p for p, c in zip(positions, cids) if c == CLASS_DEFENSE]
        if len(oline_pos) < 3 or len(def_pos) < 1:
            continue

        rows.append({
            "clip": row["clip"], "qtr": row["QTR"],
            "true_yardline": float(row["YARD_LN"]), "true_hash": row["HASH"],
            "oline_pos": oline_pos, "def_pos": def_pos,
        })
    return rows


# ── Grid search (independent 1D searches) + quarter-fold CV ─────────────────

LOS_STANCE_OFFSET_GRID = np.arange(0.0, 2.55, 0.1)
HASH_MID_BAND_GRID = np.arange(0.5, 5.05, 0.25)


def best_los_stance_offset(clips):
    best = (np.inf, None)
    for offset in LOS_STANCE_OFFSET_GRID:
        errs = [yardline_error(classify_yardline_p(c["oline_pos"], c["def_pos"], offset), c["true_yardline"])
                for c in clips]
        mean_err = np.mean(errs)
        if mean_err < best[0]:
            best = (mean_err, offset)
    return best


def best_hash_mid_band(clips):
    best = (0.0, None)
    for band in HASH_MID_BAND_GRID:
        acc = np.mean([classify_hash_p(c["oline_pos"], c["def_pos"], band) == c["true_hash"] for c in clips])
        if acc > best[0]:
            best = (acc, band)
    return best


def run_cv(clips: list[dict], label: str):
    pooled_yd_true, pooled_yd_pred = [], []
    pooled_hash_true, pooled_hash_pred = [], []
    fold_summaries = []

    for held_out_q in QUARTERS:
        calib = [c for c in clips if c["qtr"] != held_out_q]
        test = [c for c in clips if c["qtr"] == held_out_q]
        if not test:
            continue
        calib_err, best_offset = best_los_stance_offset(calib)
        calib_hash_acc, best_band = best_hash_mid_band(calib)

        for c in test:
            pooled_yd_true.append(c["true_yardline"])
            pooled_yd_pred.append(classify_yardline_p(c["oline_pos"], c["def_pos"], best_offset))
            pooled_hash_true.append(c["true_hash"])
            pooled_hash_pred.append(classify_hash_p(c["oline_pos"], c["def_pos"], best_band))

        fold_summaries.append((held_out_q, len(calib), len(test), best_offset, calib_err, best_band, calib_hash_acc))

    print(f"=== {label}: quarter-fold CV ===")
    for q, n_c, n_t, offset, calib_err, band, calib_hash_acc in fold_summaries:
        print(f"  held out {q}: n_calib={n_c} n_test={n_t}  "
              f"best_los_stance_offset={offset:.2f} (calib mean err={calib_err:.3f})  "
              f"best_hash_mid_band={band:.2f} (calib hash acc={calib_hash_acc:.4f})")

    yd_errs = [yardline_error(p, t) for p, t in zip(pooled_yd_pred, pooled_yd_true)]
    hash_acc = np.mean([t == p for t, p in zip(pooled_hash_true, pooled_hash_pred)])
    print(f"  pooled out-of-fold yardline error: mean={np.mean(yd_errs):.3f}  "
          f"median={np.median(yd_errs):.3f}  max={np.max(yd_errs):.3f}")
    print(f"  pooled out-of-fold hash accuracy: {hash_acc:.4f} "
          f"({sum(t==p for t,p in zip(pooled_hash_true,pooled_hash_pred))}/{len(pooled_hash_true)})")
    confusion = pd.crosstab(pd.Series(pooled_hash_true, name="true"), pd.Series(pooled_hash_pred, name="pred"))
    print(confusion)
    print()


def main():
    gt_clips = collect_clips("gt", "gt")
    run_cv(gt_clips, "GT/GT")

    sd_clips = collect_clips("detected", "detected")
    run_cv(sd_clips, "detected/detected")


if __name__ == "__main__":
    main()
