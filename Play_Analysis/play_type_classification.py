
"""
Rule-based play-type (Run/Pass) classification from tracked player movement.

Compares QB and offensive-line downfield displacement over the course of a
play: a QB dropping straight back is a strong pass tell, offensive linemen
driving forward as a group is a strong run tell. Movement is measured as a
linear-regression slope (yards/frame, converted to yd/s at this dataset's
4fps) fit across every tracked frame in the clip, not a two-point
comparison — a slope is far more robust to a few noisy frames than an
early-vs-late endpoint difference (motivated by the clip_020 case, a real
stretch run whose fitted magnitude was initially implausible).

Trajectories are also filtered for degenerate zero-area boxes before
fitting (`_valid_indices`) — a lost SAM-3 track emits `[0,0,0,0]` rather
than dropping the detection, which otherwise projects to a phantom field
position and corrupts the slope (17/107 clips affected for QB, 9/107 for
oline).

The one component in Play_Analysis that needs tracking rather than just the
presnap frame — formation and yard/hash are single-frame tasks, this one is
inherently about movement. Uses Player_Tracking/run_tracking.py's cached
SAM-3 tracking (GT-box-initialized) and per-frame GT-keypoint homography.

Usage:
  python play_type_classification.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from classify_formation import (  # noqa: E402
    DATA_ROOT, META_CSV, GT_ANN_DIR, GT_KPT_DIR,
    CLASS_OLINE, load_gt_keypoints, load_gt_boxes,
    FIELD_KPT_BASE, KPT_CONF, IMGSZ,
)
from yard_hash_analysis import (  # noqa: E402
    CLASS_DEFENSE, attack_direction,
    compute_homography_from_keypoints, project_to_field_yards,
)

BASE = Path(__file__).parent
TRACKING_CACHE = BASE.parent / "Player_Tracking" / "output" / "tracking_cache"
TRACKING_CACHE_SELFDETECTED = BASE.parent / "Player_Tracking" / "output" / "tracking_cache_selfdetected"
FRAMES_ROOT = DATA_ROOT / "Keypoints" / "frames" / "tb2026"

CLASS_QB = 2
IMG_W, IMG_H = 1920, 1080  # constant across tb2026 (checked directly), skips per-frame image loads
FPS = 4.0


# ── Core geometry (reusable for both GT-init and, later, detected input) ────

def fit_H_for_global_frame(global_frame_idx: int):
    kpt_path = GT_KPT_DIR / f"frame_{global_frame_idx:06d}.txt"
    if not kpt_path.exists():
        return None
    kpts = load_gt_keypoints(kpt_path, IMG_W, IMG_H)
    H, _ = compute_homography_from_keypoints(kpts, conf_thresh=1.0, min_points=4, img_shape=(IMG_H, IMG_W))
    return H


def _clean_slope(slope_yd_s: float | None) -> float | None:
    """A degenerate regression fit (e.g. a homography-projected outlier
    feeding np.polyfit) can produce NaN; treated as missing, same as no
    slope at all, so it can't silently leak into the decision cascade's
    comparisons (NaN < x is always False in Python, so an unguarded NaN
    would fall through every threshold check instead of being excluded)."""
    if slope_yd_s is not None and np.isnan(slope_yd_s):
        return None
    return slope_yd_s


def fit_slope(traj: list[tuple[int, float]]) -> float | None:
    """Linear-regression slope in yd/frame across the whole trajectory; None if <2 points."""
    if len(traj) < 2:
        return None
    frames = np.array([f for f, _ in traj], dtype=float)
    values = np.array([v for _, v in traj], dtype=float)
    slope, _ = np.polyfit(frames, values, 1)
    return float(slope)


def classify_play_type(qb_slope_yd_s: float | None, oline_slope_yd_s: float | None,
                        qb_thresh: float, oline_thresh: float) -> str:
    """
    Both slopes are already signed so positive = moving in the offense's
    attack direction. QB dropping back (clearly negative) is checked first
    since it's the cleanest signal; oline driving forward beyond
    oline_thresh is the next tell. Oline retreating at all (even slightly)
    is checked next -- a real pass-protection drop, distinct from and not
    caught by the forward-drive check above. Any oline retreat, not just a
    large one, is diagnostic here: on true run plays the line is at worst
    driven back off a scheme it's actively trying to win, whereas on a
    real pass its job is to retreat and protect, so even a small negative
    slope reliably lands on the pass side (calibrated empirically against
    all 107 GT clips).
    """
    if qb_slope_yd_s is not None and qb_slope_yd_s < -qb_thresh:
        return "Pass"
    if oline_slope_yd_s is not None and oline_slope_yd_s > oline_thresh:
        return "Run"
    if oline_slope_yd_s is not None and oline_slope_yd_s < 0:
        return "Pass"
    if qb_slope_yd_s is not None:
        return "Pass" if qb_slope_yd_s < 0 else "Run"
    if oline_slope_yd_s is not None:
        return "Run" if oline_slope_yd_s > 0 else "Pass"
    return "Run"  # majority-class fallback if both signals are missing


# ── Trajectory extraction from cached tracking ──────────────────────────────

def get_attack_direction(clip_row: pd.Series) -> str | None:
    pnum, start = int(clip_row["play_num"]), int(clip_row["start_frame"])
    ann_path = GT_ANN_DIR / f"frame_{pnum:03d}.txt"
    if not ann_path.exists():
        return None
    H = fit_H_for_global_frame(start)
    if H is None:
        return None
    boxes = load_gt_boxes(ann_path, IMG_W, IMG_H)
    xyxy = np.array([[x1, y1, x2, y2] for _, x1, y1, x2, y2 in boxes])
    positions = project_to_field_yards(xyxy, H)
    oline_pos = [pos for (cls, *_), pos in zip(boxes, positions) if cls == CLASS_OLINE]
    def_pos = [pos for (cls, *_), pos in zip(boxes, positions) if cls == CLASS_DEFENSE]
    return attack_direction(oline_pos, def_pos)


MIN_BOX_AREA_PX = 1.0  # SAM-3 emits a degenerate [0,0,0,0] box (not a dropped detection)
                        # when a track is lost, e.g. player leaves frame or is fully occluded


def _valid_indices(xyxy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Indices where `mask` is True AND the box has non-degenerate area.

    Without this, a lost track's zero-area box still gets projected through
    the homography and lands at a phantom field position (e.g. clip_020's
    QB "teleporting" to X=120yd for its last 8/40 frames), corrupting the
    trajectory slope. Confirmed by dumping clip_020 frame-by-frame: 17/107
    clips have >=1 zero-area QB box, 9/107 for oline.
    """
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return idx
    boxes = xyxy[idx]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return idx[areas > MIN_BOX_AREA_PX]


def compute_trajectory(clip_row: pd.Series) -> dict:
    """Returns {'qb': [(frame_idx, x_yd), ...], 'oline': [(frame_idx, median_x_yd), ...]}."""
    clip_name = clip_row["clip"].replace(".mp4", "")
    cache_path = TRACKING_CACHE / f"{clip_name}.pkl"
    if not cache_path.exists():
        return {}

    with open(cache_path, "rb") as f:
        cached = pickle.load(f)
    detections_by_frame = cached["detections"]
    start = int(clip_row["start_frame"])

    qb_traj, oline_traj = [], []
    for local_idx, frame_data in detections_by_frame.items():
        H = fit_H_for_global_frame(start + local_idx)
        if H is None:
            continue
        xyxy = frame_data["xyxy"]
        cids = frame_data["class_id"]

        qb_idx = _valid_indices(xyxy, cids == CLASS_QB)
        if len(qb_idx) > 0:
            pos = project_to_field_yards(xyxy[qb_idx], H)
            qb_traj.append((local_idx, float(np.mean([p[0] for p in pos]))))

        oline_idx = _valid_indices(xyxy, cids == CLASS_OLINE)
        if len(oline_idx) > 0:
            pos = project_to_field_yards(xyxy[oline_idx], H)
            oline_traj.append((local_idx, float(np.median([p[0] for p in pos]))))

    return {"qb": qb_traj, "oline": oline_traj}


# ── GT-init evaluation ───────────────────────────────────────────────────────

def evaluate_gt() -> dict:
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

        qb_slope_yd_s = _clean_slope(sign * qb_slope * FPS if qb_slope is not None else None)
        oline_slope_yd_s = _clean_slope(sign * oline_slope * FPS if oline_slope is not None else None)

        rows.append({
            "clip": row["clip"],
            "true_play_type": row["PLAY_TYPE"],
            "qb_slope_yd_s": round(qb_slope_yd_s, 3) if qb_slope_yd_s is not None else None,
            "oline_slope_yd_s": round(oline_slope_yd_s, 3) if oline_slope_yd_s is not None else None,
        })

    df = pd.DataFrame(rows)

    # grid search over (qb_thresh, oline_thresh)
    best = None
    for qb_thresh in np.arange(0.0, 3.05, 0.25):
        for oline_thresh in np.arange(0.0, 3.05, 0.25):
            preds = []
            for _, r in df.iterrows():
                preds.append(classify_play_type(r["qb_slope_yd_s"], r["oline_slope_yd_s"], qb_thresh, oline_thresh))
            acc = float((df["true_play_type"] == pd.Series(preds)).mean())
            if best is None or acc > best[0]:
                best = (acc, qb_thresh, oline_thresh, preds)

    acc, qb_thresh, oline_thresh, preds = best
    df["pred_play_type"] = preds
    confusion = pd.crosstab(df["true_play_type"], df["pred_play_type"])
    recall = {
        pt: float((sub["true_play_type"] == sub["pred_play_type"]).mean())
        for pt, sub in df.groupby("true_play_type")
    }

    return {
        "n_clips": len(df),
        "best_thresholds": {"qb_thresh_yd_s": round(qb_thresh, 2), "oline_thresh_yd_s": round(oline_thresh, 2)},
        "accuracy": round(acc, 4),
        "recall_per_play_type": {k: round(v, 4) for k, v in recall.items()},
        "confusion_matrix": confusion.to_dict(orient="index"),  # {true_label: {pred_label: count}}
        "predictions": df.to_dict(orient="records"),
    }


# ── Self-detected evaluation (phase 2: real pipeline capability) ───────────
# Uses Player_Tracking/run_tracking_selfdetected.py's cache: SAM-3 tracking
# initialized from the deployed Stage-3b/1280 player detector's own frame-0
# boxes (not GT), and per-frame homography from the finetune-init keypoint
# model for that clip's held-out quarter (not GT keypoint labels). Reuses
# the fixed GT-calibrated thresholds (qb_thresh=1.25, oline_thresh=0.5)
# rather than re-grid-searching, the same "deployed rule, not re-tuned per
# phase" approach used for Formation Classification's self-detected ablation.

GT_QB_THRESH = 1.25
GT_OLINE_THRESH = 0.5


def fit_H_for_global_frame_selfdetected(global_frame_idx: int, kpt_model):
    img_path = FRAMES_ROOT / f"frame_{global_frame_idx:06d}.png"
    if not img_path.exists():
        return None
    kpt_res = kpt_model(str(img_path), imgsz=IMGSZ, conf=0.10, max_det=1, verbose=False)[0]
    if kpt_res.keypoints is None or len(kpt_res.keypoints.data) == 0:
        return None
    kpts_data = kpt_res.keypoints.data[0].cpu().numpy()
    H, _ = compute_homography_from_keypoints(kpts_data, conf_thresh=KPT_CONF, min_points=4, img_shape=(IMG_H, IMG_W))
    return H


def get_attack_direction_selfdetected(clip_row: pd.Series, kpt_model) -> str | None:
    """Same idea as get_attack_direction(), but the frame-0 boxes come from the
    self-detected tracking cache (the deployed detector's own init boxes) and H
    comes from the deployed keypoint model, instead of GT boxes/keypoints."""
    clip_name = clip_row["clip"].replace(".mp4", "")
    cache_path = TRACKING_CACHE_SELFDETECTED / f"{clip_name}.pkl"
    if not cache_path.exists():
        return None
    with open(cache_path, "rb") as f:
        cached = pickle.load(f)
    detections_by_frame = cached["detections"]
    if 0 not in detections_by_frame:
        return None

    start = int(clip_row["start_frame"])
    H = fit_H_for_global_frame_selfdetected(start, kpt_model)
    if H is None:
        return None

    xyxy = detections_by_frame[0]["xyxy"]
    cids = detections_by_frame[0]["class_id"]
    oline_idx = _valid_indices(xyxy, cids == CLASS_OLINE)
    def_idx = _valid_indices(xyxy, cids == CLASS_DEFENSE)
    oline_pos = project_to_field_yards(xyxy[oline_idx], H) if len(oline_idx) > 0 else []
    def_pos = project_to_field_yards(xyxy[def_idx], H) if len(def_idx) > 0 else []
    return attack_direction(oline_pos, def_pos)


def compute_trajectory_selfdetected(clip_row: pd.Series, kpt_model) -> dict:
    """Same as compute_trajectory(), but reads the self-detected tracking cache
    and fits H per frame from the deployed keypoint model instead of GT labels."""
    clip_name = clip_row["clip"].replace(".mp4", "")
    cache_path = TRACKING_CACHE_SELFDETECTED / f"{clip_name}.pkl"
    if not cache_path.exists():
        return {}

    with open(cache_path, "rb") as f:
        cached = pickle.load(f)
    detections_by_frame = cached["detections"]
    start = int(clip_row["start_frame"])

    qb_traj, oline_traj = [], []
    for local_idx, frame_data in detections_by_frame.items():
        H = fit_H_for_global_frame_selfdetected(start + local_idx, kpt_model)
        if H is None:
            continue
        xyxy = frame_data["xyxy"]
        cids = frame_data["class_id"]

        qb_idx = _valid_indices(xyxy, cids == CLASS_QB)
        if len(qb_idx) > 0:
            pos = project_to_field_yards(xyxy[qb_idx], H)
            qb_traj.append((local_idx, float(np.mean([p[0] for p in pos]))))

        oline_idx = _valid_indices(xyxy, cids == CLASS_OLINE)
        if len(oline_idx) > 0:
            pos = project_to_field_yards(xyxy[oline_idx], H)
            oline_traj.append((local_idx, float(np.median([p[0] for p in pos]))))

    return {"qb": qb_traj, "oline": oline_traj}


def evaluate_self_detected() -> dict:
    from ultralytics import YOLO

    meta = pd.read_csv(META_CSV)
    kpt_models = {
        q: YOLO(str(FIELD_KPT_BASE / q / "train" / "weights" / "best.pt"))
        for q in ["Q1", "Q2", "Q3", "Q4"]
    }

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

        qb_slope_yd_s = _clean_slope(sign * qb_slope * FPS if qb_slope is not None else None)
        oline_slope_yd_s = _clean_slope(sign * oline_slope * FPS if oline_slope is not None else None)

        pred = classify_play_type(qb_slope_yd_s, oline_slope_yd_s, GT_QB_THRESH, GT_OLINE_THRESH)

        rows.append({
            "clip": row["clip"],
            "true_play_type": row["PLAY_TYPE"],
            "qb_slope_yd_s": round(qb_slope_yd_s, 3) if qb_slope_yd_s is not None else None,
            "oline_slope_yd_s": round(oline_slope_yd_s, 3) if oline_slope_yd_s is not None else None,
            "pred_play_type": pred,
        })

    df = pd.DataFrame(rows)
    acc = float((df["true_play_type"] == df["pred_play_type"]).mean())
    confusion = pd.crosstab(df["true_play_type"], df["pred_play_type"])
    recall = {
        pt: float((sub["true_play_type"] == sub["pred_play_type"]).mean())
        for pt, sub in df.groupby("true_play_type")
    }

    return {
        "n_clips": len(df),
        "thresholds_used": {"qb_thresh_yd_s": GT_QB_THRESH, "oline_thresh_yd_s": GT_OLINE_THRESH},
        "accuracy": round(acc, 4),
        "recall_per_play_type": {k: round(v, 4) for k, v in recall.items()},
        "confusion_matrix": confusion.to_dict(orient="index"),
        "predictions": df.to_dict(orient="records"),
    }


def main():
    results = evaluate_gt()
    print(f"Evaluated {results['n_clips']} clips")
    print(f"\nBest thresholds: {results['best_thresholds']}")
    print(f"Overall accuracy: {results['accuracy']:.1%}")
    print("\nPer-class recall:")
    for pt, r in results["recall_per_play_type"].items():
        print(f"  {pt:6s} {r:.1%}")

    out = BASE / "play_type_eval.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-detected", action="store_true",
                         help="Run phase 2 (real pipeline capability) instead of the GT-init eval")
    args = parser.parse_args()

    if args.self_detected:
        results = evaluate_self_detected()
        print(f"Evaluated {results['n_clips']} clips")
        print(f"\nThresholds used (fixed, GT-calibrated): {results['thresholds_used']}")
        print(f"Overall accuracy: {results['accuracy']:.1%}")
        print("\nPer-class recall:")
        for pt, r in results["recall_per_play_type"].items():
            print(f"  {pt:6s} {r:.1%}")

        out = BASE / "play_type_eval_selfdetected.json"
        out.write_text(json.dumps(results, indent=2))
        print(f"\nSaved -> {out}")
    else:
        main()
