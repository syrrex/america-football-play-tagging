"""
Rule-based field-position (yard line + hash) recognition from bird's-eye
player positions.

Yard-line magnitude (distance from the nearer goal line) is directly
recoverable from the line of scrimmage's downfield homography position. The
*sign* (whose side of the field) and the hash *label* (L/M/R) both depend on
which way the offense is facing, which the field position alone can't give —
but the defense's position relative to the offensive line resolves it: the
defense always lines up on the side of the LOS the offense is attacking
toward. That single geometric check (attack_direction) fixes both, since
turning 180 degrees flips which fixed lateral direction counts as "left" for
the offense the same way it flips which goal is "theirs" for the yard-line
sign.

Thresholds calibrated by grid search against the 107 tb2026 presnap frames.

Two use modes, same as classify_formation.py:
  - GT-input (this script's __main__): GT player boxes + GT keypoints/homography.
  - Self-detected ablation (later): same core functions, fed detected positions.

Usage:
  python yard_hash_analysis.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from classify_formation import (  # noqa: E402
    DATA_ROOT, META_CSV, GT_IMG_DIR, GT_ANN_DIR, GT_KPT_DIR,
    CLASS_OLINE, load_gt_keypoints, load_gt_boxes,
)

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE.parent / "Pipeline"))
from homography import compute_homography_from_keypoints, project_to_field_yards, FIELD_YD_W, FIELD_YD_H  # noqa: E402

CLASS_DEFENSE = 0

FIELD_MID_X = FIELD_YD_W / 2.0    # 60 yd — midfield, downfield axis
FIELD_MID_Y = FIELD_YD_H / 2.0    # ~26.67 yd — field-width center, lateral axis
HASH_MID_BAND = 2.75              # yd either side of FIELD_MID_Y classified as "M"

# The oline sets up a bit behind the true line of scrimmage rather than
# straddling it, so the detected/annotated oline mean is systematically
# short of the real LOS by about this much toward the offense's own side.
# Confirmed by sweeping this constant against the 107 GT clips: yard-line
# mean/median error is minimized right around 1.0yd on both GT and
# self-detected input (mean 1.14->0.54yd / 1.22->0.78yd), matching the
# expected stance depth almost exactly.
LOS_STANCE_OFFSET = 1.0

# Same OUTLIER_CAP pattern as compute_tackle_box() in classify_formation.py:
# trim any point more than OUTLIER_CAP from the group's median before taking
# the mean. Without this, a single wild false-positive detection (e.g. a
# background object misread as "defense" and projected through the
# homography to an implausible field position far outside 0-120 x 0-53.33)
# drags the mean far enough to flip attack_direction(), which flips both
# the yard-line sign and the hash label at once. Calibrated above the
# largest legitimate per-point deviation from the median seen across the
# 107 GT clips for every group this is applied to: oline x 1.13yd, oline y
# 6.23yd, defense x 7.03yd.
OUTLIER_CAP = 10.0


def _robust_mean(values: list[float]) -> float:
    """Mean after trimming points more than OUTLIER_CAP from the median."""
    arr = np.array(values)
    if len(arr) <= 1:
        return float(np.mean(arr))
    med = np.median(arr)
    kept = arr[np.abs(arr - med) <= OUTLIER_CAP]
    if len(kept) == 0:
        kept = arr
    return float(np.mean(kept))


# ── Core geometry (reusable for both GT and detected input) ────────────────

def attack_direction(oline_positions: list[tuple[float, float]],
                      defense_positions: list[tuple[float, float]]) -> str | None:
    """
    Which way the offense is driving, from defense position relative to the
    line of scrimmage. Defense always lines up on the side of the LOS the
    offense is attacking toward.

    Returns "inc" (offense drives toward increasing downfield X) or "dec",
    or None if there aren't enough players to tell.
    """
    if len(oline_positions) < 3 or len(defense_positions) < 1:
        return None
    los_x = _robust_mean([px for px, py in oline_positions])
    def_mean_x = _robust_mean([px for px, py in defense_positions])
    return "inc" if def_mean_x > los_x else "dec"


def classify_yardline(oline_positions: list[tuple[float, float]],
                       defense_positions: list[tuple[float, float]]) -> float | None:
    """
    Signed yard-line value matching the "-49..+50" scouting convention
    (negative = offense's own side, positive = opponent's side, distance
    from midfield). Returns None if attack direction can't be determined.
    """
    direction = attack_direction(oline_positions, defense_positions)
    if direction is None:
        return None
    los_x = _robust_mean([px for px, py in oline_positions])
    # Shift from the oline's own stance to the true LOS, toward the defense
    # (see LOS_STANCE_OFFSET).
    los_x += LOS_STANCE_OFFSET if direction == "inc" else -LOS_STANCE_OFFSET
    # A line of scrimmage can never legally be in the end zone; clamping to
    # the true goal-line bounds keeps the branch below correctly signed even
    # when the LOS estimate is noisy right at a goal line (otherwise the
    # value's sign can flip purely from crossing x=10/110, not from
    # crossing midfield, which would corrupt yardline_error()'s side check).
    # A small inward epsilon keeps the clamped value strictly on the correct
    # side of 0 -- landing exactly on a goal line is otherwise ambiguous
    # between "own goal line" and "opponent's goal line" (both collapse to
    # the same raw value 0), and _field_pos() cannot tell them apart from
    # the value alone.
    los_x = max(10.0 + 1e-3, min(110.0 - 1e-3, los_x))

    if direction == "inc":
        # own goal at the low-X end
        return (10 - los_x) if los_x <= FIELD_MID_X else (110 - los_x)
    else:
        # own goal at the high-X end
        return (los_x - 110) if los_x >= FIELD_MID_X else (los_x - 10)


def classify_hash(oline_positions: list[tuple[float, float]],
                   defense_positions: list[tuple[float, float]]) -> str | None:
    """Returns "L", "M", or "R", or None if attack direction can't be determined."""
    direction = attack_direction(oline_positions, defense_positions)
    if direction is None:
        return None
    los_y = _robust_mean([py for px, py in oline_positions])

    if abs(los_y - FIELD_MID_Y) < HASH_MID_BAND:
        return "M"
    is_bot = los_y > FIELD_MID_Y  # "bottom" of the field-width canvas (higher Y)
    if direction == "dec":
        return "L" if is_bot else "R"
    else:
        return "L" if not is_bot else "R"


# ── GT-input evaluation ──────────────────────────────────────────────────

def _field_pos(v: float) -> float:
    """
    Converts a signed "-49..-1 own / 1..50 opponent" value to a position on
    a single 0-100 axis (0 = that play's own goal line, 100 = opponent's,
    50 = midfield). Needed because the raw signed values are not linear
    across the sign change: -1 and +1 are near-opposite goal lines (~98yd
    apart), while -49 and +49 are both near midfield (~2yd apart) -- a
    plain |pred - true| gets both of these backwards.
    """
    return -v if v < 0 else (100 - v)


def yardline_error(pred: float, true: float) -> float:
    """Absolute error in true field-position yards (see _field_pos)."""
    return abs(_field_pos(pred) - _field_pos(true))


# ── Unified evaluation (all 4 box_source x kpt_source configurations) ──────
# Same deployed models as classify_formation.py's evaluate():
# finetune-init field-keypoint model per held-out quarter for H, Stage-3b/1280
# player detector (not quarter-split) for oline+defense boxes.

from classify_formation import FIELD_KPT_BASE, PLAYER_DET_WEIGHTS, DET_CONF, KPT_CONF, IMGSZ, IMG_W, IMG_H  # noqa: E402


def evaluate(box_source: str, kpt_source: str) -> dict:
    """
    box_source, kpt_source: each "gt" or "detected", independently.
    Mirrors classify_formation.py's evaluate() so the same error-source
    decomposition (player detection vs. field registration) can be run here.
    """
    assert box_source in ("gt", "detected") and kpt_source in ("gt", "detected")

    from ultralytics import YOLO
    player_model = YOLO(str(PLAYER_DET_WEIGHTS)) if box_source == "detected" else None
    kpt_models = (
        {q: YOLO(str(FIELD_KPT_BASE / q / "train" / "weights" / "best.pt")) for q in ["Q1", "Q2", "Q3", "Q4"]}
        if kpt_source == "detected" else None
    )

    meta = pd.read_csv(META_CSV)
    rows = []
    n_no_homography = 0
    n_no_call = 0

    for _, row in meta.iterrows():
        pnum, start = int(row["play_num"]), int(row["start_frame"])
        img_path = GT_IMG_DIR / f"frame_{pnum:03d}.jpg"
        if not img_path.exists() or pd.isna(row["HASH"]):
            continue
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        # --- homography ---
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
                n_no_homography += 1
                continue
            kpts_data = kpt_res.keypoints.data[0].cpu().numpy()
            H, _ = compute_homography_from_keypoints(kpts_data, conf_thresh=KPT_CONF, min_points=4, img_shape=(IMG_H, IMG_W))
        if H is None:
            n_no_homography += 1
            continue

        # --- boxes ---
        if box_source == "gt":
            ann_path = GT_ANN_DIR / f"frame_{pnum:03d}.txt"
            if not ann_path.exists():
                continue
            boxes = load_gt_boxes(ann_path, w, h)
            xyxy = np.array([[x1, y1, x2, y2] for _, x1, y1, x2, y2 in boxes])
            cids = np.array([cls for cls, *_ in boxes])
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

        pred_yardline = classify_yardline(oline_pos, def_pos)
        pred_hash = classify_hash(oline_pos, def_pos)
        if pred_yardline is None or pred_hash is None:
            n_no_call += 1
            continue

        rows.append({
            "clip": row["clip"],
            "true_yardline": float(row["YARD_LN"]),
            "pred_yardline": round(pred_yardline, 2),
            "yardline_err": round(yardline_error(pred_yardline, float(row["YARD_LN"])), 2),
            "true_hash": row["HASH"],
            "pred_hash": pred_hash,
            "n_oline_det": len(oline_pos),
            "n_defense_det": len(def_pos),
        })

    df = pd.DataFrame(rows)
    hash_acc = float((df["true_hash"] == df["pred_hash"]).mean())
    hash_confusion = pd.crosstab(df["true_hash"], df["pred_hash"])

    return {
        "box_source": box_source,
        "kpt_source": kpt_source,
        "n_clips": len(df),
        "n_no_homography": n_no_homography,
        "n_no_call": n_no_call,
        "yardline": {
            "mean_err_yd": round(float(df["yardline_err"].mean()), 3),
            "median_err_yd": round(float(df["yardline_err"].median()), 3),
            "max_err_yd": round(float(df["yardline_err"].max()), 3),
        },
        "hash": {
            "accuracy": round(hash_acc, 4),
            "confusion_matrix": hash_confusion.to_dict(orient="index"),  # {true_label: {pred_label: count}}
        },
        "predictions": df.to_dict(orient="records"),
    }


def evaluate_gt() -> dict:
    """Thin wrapper kept for backward compatibility; equivalent to evaluate("gt", "gt")."""
    return evaluate("gt", "gt")


def evaluate_self_detected() -> dict:
    """Thin wrapper kept for backward compatibility; equivalent to evaluate("detected", "detected")."""
    return evaluate("detected", "detected")


# filenames for each of the four (box_source, kpt_source) configurations
OUT_FILENAMES = {
    ("gt", "gt"):             "yard_hash_eval.json",
    ("detected", "detected"): "yard_hash_eval_selfdetected.json",
    ("gt", "detected"):       "yard_hash_eval_mixed_gtbox_detkpt.json",
    ("detected", "gt"):       "yard_hash_eval_mixed_detbox_gtkpt.json",
}


def main(box_source: str = "gt", kpt_source: str = "gt"):
    results = evaluate(box_source, kpt_source)
    print(f"box_source={box_source}  kpt_source={kpt_source}")
    print(f"Evaluated {results['n_clips']} clips "
          f"({results['n_no_homography']} no homography, {results['n_no_call']} no call)")
    print("\nYard line:")
    for k, v in results["yardline"].items():
        print(f"  {k}: {v}")
    print(f"\nHash accuracy: {results['hash']['accuracy']:.1%}")

    out = BASE / OUT_FILENAMES[(box_source, kpt_source)]
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--box-source", choices=["gt", "detected"], default="gt")
    parser.add_argument("--kpt-source", choices=["gt", "detected"], default="gt")
    parser.add_argument("--self-detected", action="store_true",
                         help="Shortcut for --box-source detected --kpt-source detected")
    args = parser.parse_args()

    if args.self_detected:
        main("detected", "detected")
    else:
        main(args.box_source, args.kpt_source)
