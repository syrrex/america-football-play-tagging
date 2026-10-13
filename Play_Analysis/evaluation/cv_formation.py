"""
Quarter-based 4-fold cross-validation for Formation Classification.

Answers: how much do the reported headline numbers (94.4% GT/GT, 77.6%
det/det) reflect genuine classification-rule quality vs. fitting the role
thresholds to the same 107 clips they're evaluated on? Each of the 107
clips already has a QTR (Q1-Q4) label, the same fold structure Field
Registration uses for its own held-out-quarter evaluation, so this reuses
an existing, already-justified split rather than introducing a new one.

For each quarter held out in turn: refit WR_GAP/WR_DEPTH/ATT_GAP/ATT_DEPTH
and MATCH_TOLERANCE by grid search using only the other 3 quarters' clips,
then predict on the held-out quarter with those thresholds. Pooling all
four folds' held-out predictions covers all 107 clips exactly once, each
scored with thresholds that never saw that clip during calibration.

OUTLIER_CAP is NOT refit per fold -- it's a physical-plausibility bound
(largest legitimate per-point deviation from the median seen in GT, not a
value chosen by maximizing accuracy), so refitting it doesn't test
anything a genuine accuracy-tuned threshold does.

Usage:
  python cv_formation.py
"""

import itertools
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
    load_gt_boxes, load_gt_keypoints, CLASS_OLINE, CLASS_SKILL,
    OUTLIER_CAP, FORMATION_SIGNATURES,
)
from homography import compute_homography_from_keypoints, project_to_field_yards  # noqa: E402

QUARTERS = ["Q1", "Q2", "Q3", "Q4"]


# ── Parameterized classification (same logic as classify_formation.py,
#    thresholds passed in instead of read from module globals) ────────────

def _tackle_box(oline_positions):
    los_x = float(np.median([px for px, py in oline_positions]))
    ys = np.array([py for px, py in oline_positions])
    center = float(np.median(ys))
    kept = ys[np.abs(ys - center) <= OUTLIER_CAP]
    if len(kept) == 0:
        kept = ys
    return los_x, float(kept.min()), float(kept.max())


def _role(px, py, los_x, box_lo, box_hi, wr_gap, wr_depth, att_gap, att_depth):
    lat_gap = min(abs(py - box_lo), abs(py - box_hi)) if (py < box_lo or py > box_hi) else 0.0
    depth = abs(px - los_x)
    side = "lo" if py < (box_lo + box_hi) / 2 else "hi"
    if depth < wr_depth and lat_gap > wr_gap:
        role = "WR"
    elif depth < att_depth and lat_gap < att_gap:
        role = "attached"
    else:
        role = "other"
    return role, side


def classify_with_params(skill_positions, oline_positions, wr_gap, wr_depth, att_gap, att_depth, match_tolerance):
    if len(oline_positions) < 3:
        return "unknown"
    los_x, box_lo, box_hi = _tackle_box(oline_positions)

    n_wr = {"lo": 0, "hi": 0}
    n_att = {"lo": 0, "hi": 0}
    for px, py in skill_positions:
        role, side = _role(px, py, los_x, box_lo, box_hi, wr_gap, wr_depth, att_gap, att_depth)
        if role == "WR":
            n_wr[side] += 1
        elif role == "attached":
            n_att[side] += 1

    lo, hi = min(n_wr["lo"], n_wr["hi"]), max(n_wr["lo"], n_wr["hi"])
    one_side = "lo" if n_wr["lo"] <= n_wr["hi"] else "hi"
    two_side = "hi" if one_side == "lo" else "lo"

    def resolve_tank_trey():
        if n_att[one_side] > 0:
            return "TANK"
        if n_att[two_side] > 0:
            return "TREY"
        return "TANK"

    if lo == 2 and hi == 2:
        return "Doubles"
    if lo == 1 and hi == 3:
        return "TRIO"
    if {n_wr["lo"], n_wr["hi"]} == {1, 1} and (n_att["lo"] > 0 or n_att["hi"] > 0):
        return "WING T"
    if lo == 1 and hi == 2:
        return resolve_tank_trey()

    dists = {form: abs(lo - s_lo) + abs(hi - s_hi) for form, (s_lo, s_hi) in FORMATION_SIGNATURES.items()}
    best = min(dists, key=dists.get)
    if dists[best] > match_tolerance:
        return "unknown"
    return resolve_tank_trey() if best in ("TANK", "TREY") else best


# ── Precompute per-clip bird's-eye positions once (box_source/kpt_source
#    config), so the grid search re-scores cached positions instead of
#    re-running detection/homography for every candidate threshold ────────

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
        if not img_path.exists():
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
        skill_pos = [p for p, c in zip(positions, cids) if c == CLASS_SKILL]

        rows.append({
            "clip": row["clip"], "qtr": row["QTR"], "true_form": row["OFF_FORM"],
            "oline_pos": oline_pos, "skill_pos": skill_pos,
        })
    return rows


# ── Grid search + quarter-fold cross-validation ────────────────────────────

WR_GAP_GRID = np.arange(1.5, 3.55, 0.25)
WR_DEPTH_GRID = np.arange(2.0, 5.05, 0.5)
ATT_GAP_GRID = np.arange(1.0, 3.05, 0.25)
ATT_DEPTH_GRID = np.arange(1.0, 3.05, 0.5)
MATCH_TOLERANCE_GRID = [0, 1, 2]


def accuracy_for_params(clips, params):
    correct = 0
    for c in clips:
        pred = classify_with_params(c["skill_pos"], c["oline_pos"], *params)
        correct += int(pred == c["true_form"])
    return correct / len(clips)


def grid_search(clips):
    best = (0.0, None)
    for wr_gap, wr_depth, att_gap, att_depth, tol in itertools.product(
        WR_GAP_GRID, WR_DEPTH_GRID, ATT_GAP_GRID, ATT_DEPTH_GRID, MATCH_TOLERANCE_GRID
    ):
        params = (wr_gap, wr_depth, att_gap, att_depth, tol)
        acc = accuracy_for_params(clips, params)
        if acc > best[0]:
            best = (acc, params)
    return best


def run_cv(clips: list[dict], label: str):
    pooled_true, pooled_pred = [], []
    fold_summaries = []
    for held_out_q in QUARTERS:
        calib = [c for c in clips if c["qtr"] != held_out_q]
        test = [c for c in clips if c["qtr"] == held_out_q]
        if not test:
            continue
        calib_acc, best_params = grid_search(calib)
        for c in test:
            pred = classify_with_params(c["skill_pos"], c["oline_pos"], *best_params)
            pooled_true.append(c["true_form"])
            pooled_pred.append(pred)
        test_acc = np.mean([classify_with_params(c["skill_pos"], c["oline_pos"], *best_params) == c["true_form"]
                             for c in test])
        fold_summaries.append((held_out_q, len(calib), len(test), calib_acc, test_acc, best_params))

    print(f"=== {label}: quarter-fold CV ===")
    for q, n_calib, n_test, calib_acc, test_acc, params in fold_summaries:
        print(f"  held out {q}: n_calib={n_calib} n_test={n_test}  "
              f"calib_acc={calib_acc:.4f}  held_out_acc={test_acc:.4f}  params={params}")

    pooled_acc = np.mean([t == p for t, p in zip(pooled_true, pooled_pred)])
    print(f"  pooled out-of-fold accuracy (all 107 clips, each scored on thresholds "
          f"that never saw it): {pooled_acc:.4f} ({sum(t==p for t,p in zip(pooled_true,pooled_pred))}/{len(pooled_true)})")
    confusion = pd.crosstab(pd.Series(pooled_true, name="true"), pd.Series(pooled_pred, name="pred"))
    print(confusion)
    print()
    return pooled_acc, confusion


def main():
    gt_clips = collect_clips("gt", "gt")
    run_cv(gt_clips, "GT/GT")

    sd_clips = collect_clips("detected", "detected")
    run_cv(sd_clips, "detected/detected")


if __name__ == "__main__":
    main()
