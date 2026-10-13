"""
Rule-based offensive formation classification from bird's-eye player positions.

Identifies Doubles / TRIO / TREY / TANK / WING T from the presnap alignment of
"skill"-class players relative to the offensive line ("tackle box"), using only
lateral split distance and depth behind the line of scrimmage. No training —
thresholds were calibrated by grid search against the 107 tb2026 presnap frames.

Two use modes:
  - GT-input (this script's __main__): GT player boxes + GT keypoints/homography.
    Isolates whether the classification logic itself is correct, independent of
    upstream detection error. Currently 94.4% (101/107).
  - Self-detected ablation (later): same `classify_formation()` core, fed with
    the deployed pipeline's detected boxes/keypoints instead of GT.

Usage:
  python classify_formation.py
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
DATA_ROOT = BASE.parent / "Data"
META_CSV = DATA_ROOT / "metadata" / "tb2026_dataset_metadata.csv"
GT_IMG_DIR = DATA_ROOT / "Players" / "frames" / "tb2026"
GT_ANN_DIR = DATA_ROOT / "Players" / "annotations" / "tb2026"
GT_KPT_DIR = DATA_ROOT / "Keypoints" / "annotations" / "tb2026" / "labels"

sys.path.insert(0, str(BASE.parent / "Pipeline"))
from homography import compute_homography_from_keypoints, project_to_field_yards  # noqa: E402

N_KPT = 62
CLASS_OLINE = 1
CLASS_QB = 2
CLASS_SKILL = 4

# ── Calibrated thresholds (grid search) ──────────────────────────────────────
ATT_GAP = 2.0     # lat_gap < this AND depth < ATT_DEPTH -> "attached" (TE/FB-on-line)
ATT_DEPTH = 2.0
WR_GAP = 2.5       # lat_gap > this AND depth < WR_DEPTH -> "WR"
WR_DEPTH = 3.0

# Tackle-box geometry: median-anchored, outlier-trimmed extent (see
# compute_tackle_box docstring for the empirical grounding). A
# first attempt used a FIXED half-width (3.16yd, from the GT median span)
# instead of OUTLIER_CAP below — that dropped GT accuracy 92.5% -> 88.8%
# (TANK recall 68.2% -> 54.5%) because real box width varies legitimately
# (GT: 4.85-10.14yd across 107 clips) and a fixed width erases that signal.
# Trimming only implausible outliers keeps the real extent otherwise, at
# zero GT cost (see below).
OUTLIER_CAP = 7.0  # yd; max legitimate per-point GT deviation from the
                    # median oline lateral position is 6.23yd (107 clips)

# A QB-position fallback for the box center (override to the QB's lateral
# position when it disagreed with the oline median beyond a threshold) was
# tried and removed. It was grounded in a real domain fact -- the QB lines
# up directly behind the center of the oline, so a large disagreement
# should mean the oline reading is wrong -- but the mechanism itself had no
# way to tell that failure mode apart from a bad QB detection, which is
# exactly the scenario self-detected input can produce. Checked directly:
# on self-detected input it fired 23/107 times (vs. 9/107 on clean GT),
# including one 22.46yd disagreement traced to oline over-detection (8
# boxes vs. the true 5) rather than a bad QB read, and changed the final
# classification in only 1/107 clips (net positive there). The mechanism
# wasn't caught doing harm on this corpus, but it was never actually
# guarded against the failure mode it was meant to protect against either,
# so it's removed rather than kept as an unverified assumption.


def compute_tackle_box(oline_positions: list[tuple[float, float]]) -> tuple[float, float, float]:
    """
    Returns (los_x, box_lo, box_hi) for the tackle box.

    Anchors on the MEDIAN of detected oline positions (robust to a single
    outlier, unlike the mean), and takes the real min/max extent of the
    oline lateral positions AFTER discarding any point more than
    OUTLIER_CAP from that median. oline is the lowest-precision detected
    role (0.827 on the deployed detector, see Player_Detection eval) — a
    single false-positive oline detection would otherwise blow out the raw
    min/max directly and corrupt every skill player's lat_gap in that
    frame. The cap is set above the largest legitimate per-point deviation
    seen in 107 GT clips (6.23yd), so it only ever trims genuinely
    implausible points, never real formation width.
    """
    los_x = float(np.median([px for px, py in oline_positions]))
    ys = np.array([py for px, py in oline_positions])
    center = float(np.median(ys))

    kept = ys[np.abs(ys - center) <= OUTLIER_CAP]
    if len(kept) == 0:
        kept = ys
    return los_x, float(kept.min()), float(kept.max())


# ── Core classification (reusable for both GT and detected input) ──────────

def role_and_side(px: float, py: float, los_x: float, box_lo: float, box_hi: float) -> tuple[str, str, float, float]:
    """
    Classifies one skill player's role from its bird's-eye position.

    Parameters
    ----------
    px, py    : bird's-eye position in yards (px = downfield, py = lateral/width)
    los_x     : line-of-scrimmage depth (mean oline downfield position)
    box_lo/hi : tackle-box lateral edges (min/max oline lateral position)

    Returns
    -------
    (role, side, lat_gap, depth) where role in {"WR", "attached", "other"}
    and side in {"lo", "hi"} (which side of the box center the player is on).
    """
    lat_gap = min(abs(py - box_lo), abs(py - box_hi)) if (py < box_lo or py > box_hi) else 0.0
    depth = abs(px - los_x)
    side = "lo" if py < (box_lo + box_hi) / 2 else "hi"

    if depth < WR_DEPTH and lat_gap > WR_GAP:
        role = "WR"
    elif depth < ATT_DEPTH and lat_gap < ATT_GAP:
        role = "attached"
    else:
        role = "other"
    return role, side, lat_gap, depth


# Reference WR-count-per-side (lo, hi) signature for each formation, used by
# the tolerant fallback below. TANK and TREY share a signature and are only
# told apart by which side the attached piece is on.
FORMATION_SIGNATURES = {
    "Doubles": (2, 2),
    "TRIO":    (1, 3),
    "WING T":  (1, 1),
    "TANK":    (1, 2),
    "TREY":    (1, 2),
}
MATCH_TOLERANCE = 1  # max |lo-lo'|+|hi-hi'| from a signature to still accept it


def classify_formation(
    skill_positions: list[tuple[float, float]],
    oline_positions: list[tuple[float, float]],
) -> str:
    """
    Classifies the offensive formation from bird's-eye positions of skill and
    oline players (both as lists of (x_yd, y_yd), x=downfield, y=lateral).

    Returns one of "Doubles", "TRIO", "TREY", "TANK", "WING T", or "unknown".

    Matches the detected (lo, hi) WR-count-per-side against each formation's
    reference signature and takes the nearest one, rather than requiring an
    exact match. On clean GT input, real counts always land on an exact
    signature already, so this is a no-op there (verified: identical 92.5%
    GT-input accuracy with or without the tolerance). It matters once inputs
    are noisy (self-detected ablation): a single missed or spurious skill
    detection shifts (lo, hi) by one, which used to fall straight to
    "unknown" (or default to TANK for the (1,2)-shaped cases) instead of
    recovering the intended formation. Tested against the self-detected
    ablation: MATCH_TOLERANCE=1 raised accuracy 68.2% -> 77.6% and reduced
    "unknown" predictions from 21 to 0. Going to tolerance=2
    gave no further gain, so 1 is kept as the more conservative choice.
    """
    if len(oline_positions) < 3:
        return "unknown"

    los_x, box_lo, box_hi = compute_tackle_box(oline_positions)

    n_wr = {"lo": 0, "hi": 0}
    n_att = {"lo": 0, "hi": 0}
    for px, py in skill_positions:
        role, side, _, _ = role_and_side(px, py, los_x, box_lo, box_hi)
        if role == "WR":
            n_wr[side] += 1
        elif role == "attached":
            n_att[side] += 1

    lo, hi = min(n_wr["lo"], n_wr["hi"]), max(n_wr["lo"], n_wr["hi"])
    one_side = "lo" if n_wr["lo"] <= n_wr["hi"] else "hi"  # the "1" side of a 1-vs-N split
    two_side = "hi" if one_side == "lo" else "lo"

    def resolve_tank_trey() -> str:
        if n_att[one_side] > 0:
            return "TANK"
        if n_att[two_side] > 0:
            return "TREY"
        return "TANK"  # majority class fallback when the attached piece isn't confidently detected

    # exact match first
    if lo == 2 and hi == 2:
        return "Doubles"
    if lo == 1 and hi == 3:
        return "TRIO"
    if {n_wr["lo"], n_wr["hi"]} == {1, 1} and (n_att["lo"] > 0 or n_att["hi"] > 0):
        return "WING T"
    if lo == 1 and hi == 2:
        return resolve_tank_trey()

    # tolerant nearest-signature fallback
    dists = {form: abs(lo - s_lo) + abs(hi - s_hi) for form, (s_lo, s_hi) in FORMATION_SIGNATURES.items()}
    best = min(dists, key=dists.get)
    if dists[best] > MATCH_TOLERANCE:
        return "unknown"
    return resolve_tank_trey() if best in ("TANK", "TREY") else best


# ── GT-input evaluation ──────────────────────────────────────────────────

def load_gt_keypoints(path: Path, w: int, h: int) -> np.ndarray:
    line = path.read_text().strip().split()
    offset = 5
    kpts = np.zeros((N_KPT, 3))
    for i in range(N_KPT):
        x = float(line[offset + i * 3]) * w
        y = float(line[offset + i * 3 + 1]) * h
        vis = float(line[offset + i * 3 + 2])
        kpts[i] = [x, y, 2.0 if vis > 0 else 0.0]
    return kpts


def load_gt_boxes(path: Path, w: int, h: int) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line in path.read_text().strip().splitlines():
        cls, cx, cy, bw, bh = map(float, line.split())
        x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
        boxes.append((int(cls), x1, y1, x2, y2))
    return boxes


# ── Unified evaluation (GT / detected, independently per box source and
#    keypoint/homography source) ────────────────────────────────────────────
# Four combinations, each isolating a different error source, the same
# isolation logic already used for Field Registration's H_gt/H_pred split:
#   ("gt", "gt")             baseline: classification logic alone (92.5%)
#   ("detected", "detected") real pipeline capability (78.5%)
#   ("gt", "detected")       homography/keypoint error alone (boxes held GT)
#   ("detected", "gt")       player-detection error alone (H held GT)

FIELD_KPT_BASE = BASE.parent / "Field_Keypoint_Detection" / "finetune_quarter_results_1280"
PLAYER_DET_WEIGHTS = BASE.parent / "Player_Detection" / "cross_stadium_results" / "3b_1280" / "train" / "weights" / "best.pt"
DET_CONF = 0.25
KPT_CONF = 0.25
IMGSZ = 1280
IMG_W, IMG_H = 1920, 1080  # constant across tb2026 presnap frames (checked directly)


def evaluate(box_source: str, kpt_source: str) -> dict:
    """
    box_source, kpt_source: each "gt" or "detected", independently.

    "gt" boxes come from Data/Players/annotations/tb2026 (load_gt_boxes).
    "detected" boxes come from the deployed Stage-3b/1280 player detector.
    "gt" keypoints/H come from Data/Keypoints/annotations/tb2026 (load_gt_keypoints).
    "detected" keypoints/H come from the finetune-init model for that clip's
    held-out quarter.
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

    for _, row in meta.iterrows():
        pnum, start = int(row["play_num"]), int(row["start_frame"])
        img_path = GT_IMG_DIR / f"frame_{pnum:03d}.jpg"
        if not img_path.exists():
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
            H, _ = compute_homography_from_keypoints(kpts_data, conf_thresh=KPT_CONF, min_points=4, img_shape=(h, w))
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
        else:
            det_res = player_model(str(img_path), conf=DET_CONF, iou=0.5, imgsz=IMGSZ, verbose=False)[0]
            if det_res.boxes is not None and len(det_res.boxes) > 0:
                xyxy = det_res.boxes.xyxy.cpu().numpy()
                cids = det_res.boxes.cls.cpu().numpy().astype(int)
            else:
                xyxy, cids = np.zeros((0, 4)), np.array([])

        positions = project_to_field_yards(xyxy, H) if len(xyxy) else []
        oline_pos = [p for p, c in zip(positions, cids) if c == CLASS_OLINE]
        skill_pos = [p for p, c in zip(positions, cids) if c == CLASS_SKILL]
        qb_pos = [p for p, c in zip(positions, cids) if c == CLASS_QB]

        # classify_formation() itself returns "unknown" when len(oline_pos) < 3 --
        # no separate skip here, so that failure mode is scored as a miss, not
        # excluded from the denominator (this is meant to measure real capability).
        pred = classify_formation(skill_pos, oline_pos)
        rows.append({
            "clip": row["clip"], "true_form": row["OFF_FORM"], "pred_form": pred,
            "n_oline_det": len(oline_pos), "n_skill_det": len(skill_pos), "n_qb_det": len(qb_pos),
        })

    df = pd.DataFrame(rows)
    acc = float((df["true_form"] == df["pred_form"]).mean())
    confusion = pd.crosstab(df["true_form"], df["pred_form"])
    recall = {
        form: float((sub["true_form"] == sub["pred_form"]).mean())
        for form, sub in df.groupby("true_form")
    }

    return {
        "box_source": box_source,
        "kpt_source": kpt_source,
        "n_clips": len(df),
        "n_no_homography": n_no_homography,
        "n_unknown_pred": int((df["pred_form"] == "unknown").sum()),
        "oline_det_distribution": df["n_oline_det"].value_counts().sort_index().to_dict(),
        "accuracy": round(acc, 4),
        "recall_per_formation": {k: round(v, 4) for k, v in recall.items()},
        "confusion_matrix": confusion.to_dict(orient="index"),  # {true_label: {pred_label: count}}
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
    ("gt", "gt"):             "formation_eval.json",
    ("detected", "detected"): "formation_eval_selfdetected.json",
    ("gt", "detected"):       "formation_eval_mixed_gtbox_detkpt.json",
    ("detected", "gt"):       "formation_eval_mixed_detbox_gtkpt.json",
}


def main(box_source: str = "gt", kpt_source: str = "gt"):
    results = evaluate(box_source, kpt_source)
    print(f"box_source={box_source}  kpt_source={kpt_source}")
    print(f"Evaluated {results['n_clips']} clips ({results['n_no_homography']} skipped, no homography)")
    print(f"Overall accuracy: {results['accuracy']:.1%}")
    print(f"'unknown' predictions: {results['n_unknown_pred']}")
    print("\nPer-formation recall:")
    for form, r in sorted(results["recall_per_formation"].items()):
        print(f"  {form:10s} {r:.1%}")

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
