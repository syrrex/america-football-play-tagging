"""
Quarter-based 4-fold cross-validation for Play-Type Classification.

Same rationale and fold structure as cv_formation.py/cv_yardhash.py: refit
(qb_thresh, oline_thresh, oline_neg_thresh) using only 3 quarters' clips,
evaluate on the held-out quarter, pool all four folds' held-out predictions
into one final figure. All three thresholds here were chosen by directly
maximizing accuracy on the full 107-clip corpus (evaluate_gt()'s existing
grid search, and the oline-pass-block rule swept afterward), so unlike
OUTLIER_CAP-style bounds elsewhere, there's nothing to hold fixed here --
the whole rule gets refit per fold.

Usage:
  python cv_playtype.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PLAY_ANALYSIS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLAY_ANALYSIS))

from classify_formation import META_CSV  # noqa: E402
from play_type_classification import (  # noqa: E402
    get_attack_direction, compute_trajectory, fit_slope,
    get_attack_direction_selfdetected, compute_trajectory_selfdetected,
    _clean_slope, FPS, FIELD_KPT_BASE,
)

QUARTERS = ["Q1", "Q2", "Q3", "Q4"]

QB_THRESH_GRID = np.arange(0.0, 3.05, 0.25)
OLINE_THRESH_GRID = np.arange(0.0, 3.05, 0.25)
OLINE_NEG_THRESH_GRID = np.arange(0.0, 1.05, 0.1)


def classify_play_type_p(qb, oline, qb_thresh, oline_thresh, oline_neg_thresh):
    """Same as play_type_classification.classify_play_type(), but with the
    oline-pass-block threshold (hardcoded to 0.0 in production) exposed as a
    parameter so it can be refit per fold too."""
    if qb is not None and qb < -qb_thresh:
        return "Pass"
    if oline is not None and oline > oline_thresh:
        return "Run"
    if oline is not None and oline < -oline_neg_thresh:
        return "Pass"
    if qb is not None:
        return "Pass" if qb < 0 else "Run"
    if oline is not None:
        return "Run" if oline > 0 else "Pass"
    return "Run"


def collect_clips_gt() -> list[dict]:
    meta = pd.read_csv(META_CSV)
    rows = []
    for _, row in meta.iterrows():
        direction = get_attack_direction(row)
        traj = compute_trajectory(row)
        if direction is None or not traj:
            continue
        sign = 1.0 if direction == "inc" else -1.0
        qb_slope = fit_slope(traj["qb"])
        oline_slope = fit_slope(traj["oline"])
        qb_slope_yd_s = sign * qb_slope * FPS if qb_slope is not None else None
        oline_slope_yd_s = sign * oline_slope * FPS if oline_slope is not None else None
        rows.append({
            "clip": row["clip"], "qtr": row["QTR"], "true_play_type": row["PLAY_TYPE"],
            "qb": _clean_slope(qb_slope_yd_s), "oline": _clean_slope(oline_slope_yd_s),
        })
    return rows


def collect_clips_selfdetected() -> list[dict]:
    from ultralytics import YOLO
    meta = pd.read_csv(META_CSV)
    kpt_models = {q: YOLO(str(FIELD_KPT_BASE / q / "train" / "weights" / "best.pt")) for q in QUARTERS}

    rows = []
    for _, row in meta.iterrows():
        kpt_model = kpt_models[row["QTR"]]
        direction = get_attack_direction_selfdetected(row, kpt_model)
        traj = compute_trajectory_selfdetected(row, kpt_model)
        if direction is None or not traj:
            continue
        sign = 1.0 if direction == "inc" else -1.0
        qb_slope = fit_slope(traj["qb"])
        oline_slope = fit_slope(traj["oline"])
        qb_slope_yd_s = sign * qb_slope * FPS if qb_slope is not None else None
        oline_slope_yd_s = sign * oline_slope * FPS if oline_slope is not None else None
        rows.append({
            "clip": row["clip"], "qtr": row["QTR"], "true_play_type": row["PLAY_TYPE"],
            "qb": _clean_slope(qb_slope_yd_s), "oline": _clean_slope(oline_slope_yd_s),
        })
    return rows


def accuracy_for_params(clips, qb_thresh, oline_thresh, oline_neg_thresh):
    correct = 0
    for c in clips:
        pred = classify_play_type_p(c["qb"], c["oline"], qb_thresh, oline_thresh, oline_neg_thresh)
        correct += int(pred == c["true_play_type"])
    return correct / len(clips)


def grid_search(clips):
    best = (0.0, None)
    for qb_t in QB_THRESH_GRID:
        for ol_t in OLINE_THRESH_GRID:
            for neg_t in OLINE_NEG_THRESH_GRID:
                acc = accuracy_for_params(clips, qb_t, ol_t, neg_t)
                if acc > best[0]:
                    best = (acc, (qb_t, ol_t, neg_t))
    return best


def run_cv(clips: list[dict], label: str):
    pooled_true, pooled_pred = [], []
    fold_summaries = []

    for held_out_q in QUARTERS:
        calib = [c for c in clips if c["qtr"] != held_out_q]
        test = [c for c in clips if c["qtr"] == held_out_q]
        if not test:
            continue
        calib_acc, (qb_t, ol_t, neg_t) = grid_search(calib)
        for c in test:
            pred = classify_play_type_p(c["qb"], c["oline"], qb_t, ol_t, neg_t)
            pooled_true.append(c["true_play_type"])
            pooled_pred.append(pred)
        fold_summaries.append((held_out_q, len(calib), len(test), calib_acc, qb_t, ol_t, neg_t))

    print(f"=== {label}: quarter-fold CV ===")
    for q, n_c, n_t, calib_acc, qb_t, ol_t, neg_t in fold_summaries:
        print(f"  held out {q}: n_calib={n_c} n_test={n_t}  calib_acc={calib_acc:.4f}  "
              f"qb_thresh={qb_t:.2f}  oline_thresh={ol_t:.2f}  oline_neg_thresh={neg_t:.2f}")

    pooled_acc = np.mean([t == p for t, p in zip(pooled_true, pooled_pred)])
    print(f"  pooled out-of-fold accuracy: {pooled_acc:.4f} "
          f"({sum(t==p for t,p in zip(pooled_true,pooled_pred))}/{len(pooled_true)})")
    confusion = pd.crosstab(pd.Series(pooled_true, name="true"), pd.Series(pooled_pred, name="pred"))
    print(confusion)
    print()


def main():
    gt_clips = collect_clips_gt()
    run_cv(gt_clips, "GT-init")

    sd_clips = collect_clips_selfdetected()
    run_cv(sd_clips, "Self-detected")


if __name__ == "__main__":
    main()
