"""
Field registration (homography) evaluation. Answers RQ2 directly, building on top
of evaluate_keypoints.py's component-level (pixel-space) keypoint accuracy.

Three experiments, run per --weights:

  1. Player-projection error (yards) — the 107 presnap frames that have BOTH GT
     player boxes (Data/Players/annotations/tb2026/) and GT field keypoints.
     Computes H_gt (from GT keypoints) and H_pred (from predicted keypoints),
     projects each GT player's foot point through both, and measures the distance
     between the two projections in yards. This isolates the registration error
     attributable to imperfect keypoint detection (player detection held at GT).

  2. Keypoint-only field-space error — every test frame, no player data needed.
     Every keypoint's true field position is known a priori (fixed field
     template), so this doesn't need GT player boxes at all. Per frame: fit H_gt
     from the GT keypoints, project each *predicted* keypoint through that same
     H_gt, and compare to its known true field position. Re-expresses
     evaluate_keypoints.py's pixel-space error in yards, at full test-set size.

  3. Confidence-threshold sweep (--sweep) — reruns experiment 2 at multiple
     --conf thresholds to show how fewer/higher-confidence correspondence points
     trade off against homography quality.

Usage (run from Field_Keypoint_Detection/):
  python evaluation/evaluate_homography.py --weights quarter_kfold_results_1280/Q1/train/weights/best.pt --quarter Q1
  python evaluation/evaluate_homography.py --weights finetune_quarter_results_1280/Q1/train/weights/best.pt --quarter Q1 --sweep
  python evaluation/evaluate_homography.py --weights combined_results/train/weights/best.pt   # no --quarter: fixed TB2026 test set
"""

import argparse
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import sys
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "utils"))
from field_landmarks import FIELD_POSITIONS  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = HERE.parent  # Field_Keypoint_Detection/
DATA     = BASE.parent / "Data"
KP_DS    = Path(os.environ.get("KP_DS", DATA / "Keypoints" / "datasets"))
TEST_IMGS   = KP_DS / "test" / "images"
TEST_LABELS = KP_DS / "test" / "labels"
PLAYER_ANN  = DATA / "Players" / "annotations" / "tb2026"
META_CSV    = DATA / "metadata" / "tb2026_dataset_metadata.csv"

N_KPT  = 62
IMGSZ  = 1280
SWEEP_THRESHOLDS = [0.10, 0.25, 0.40, 0.55, 0.70, 0.80, 0.85, 0.90, 0.95]
RANSAC_THRESH_YD = 2.0   # inlier tolerance for cv2.findHomography, in yards
MIN_CORRESPONDENCES = 4  # minimum non-collinear points needed to fit a homography

FRAME_RE = re.compile(r"frame_0*(\d+)\.")


def frame_number(path: Path) -> int:
    m = FRAME_RE.search(path.name)
    return int(m.group(1)) if m else -1


def load_metadata() -> pd.DataFrame:
    meta = pd.read_csv(META_CSV)
    meta["clip_num"] = meta["clip"].str.extract(r"clip_0*(\d+)").astype(int)
    return meta


def load_frame_to_quarter() -> dict[int, str]:
    """Map every dense keypoint-dataset frame index -> quarter label."""
    meta = load_metadata()
    mapping = {}
    for _, row in meta.iterrows():
        q = row["QTR"]
        for f in range(int(row["start_frame"]), int(row["end_frame"]) + 1):
            mapping[f] = q
    return mapping


def load_clip_to_kpt_frame() -> dict[int, int]:
    """Map player-annotation clip number -> the clip's first frame in the dense
    keypoint-dataset numbering (Data/Players/.../frame_NNN.* is named by *clip
    number*, e.g. frame_007.txt = presnap frame of clip_007.mp4 = that clip's
    start_frame — NOT global keypoint frame index 7)."""
    meta = load_metadata()
    return dict(zip(meta["clip_num"], meta["start_frame"]))


def load_clip_to_quarter() -> dict[int, str]:
    meta = load_metadata()
    return dict(zip(meta["clip_num"], meta["QTR"]))


# ── GT / prediction loading ────────────────────────────────────────────────────

def load_gt_keypoints(label_path: Path, img_w: int, img_h: int) -> dict[int, tuple[float, float]]:
    """Returns {kpt_id (1-based): (x_px, y_px)} for all annotated keypoints (vis > 0)."""
    line = label_path.read_text().strip().split()
    offset = 5
    gt = {}
    for i in range(N_KPT):
        x_n = float(line[offset + i * 3])
        y_n = float(line[offset + i * 3 + 1])
        vis = int(float(line[offset + i * 3 + 2]))
        if vis > 0:
            gt[i + 1] = (x_n * img_w, y_n * img_h)
    return gt


def load_gt_player_footpoints(ann_path: Path, img_w: int, img_h: int) -> list[tuple[float, float]]:
    """Returns [(x_px, y_px), ...] — bottom-center of each GT player box."""
    footpoints = []
    for line in ann_path.read_text().strip().splitlines():
        _, cx, cy, bw, bh = (float(v) for v in line.split())
        footpoints.append((cx * img_w, (cy + bh / 2) * img_h))
    return footpoints


def run_model_keypoints_raw(model, img_path: Path) -> np.ndarray | None:
    """Runs the model once; returns the raw (62, 3) array of (x_px, y_px, conf) for
    every keypoint slot, unfiltered. Per-keypoint conf is already computed in a
    single forward pass — callers filter by whatever threshold they need via
    filter_kpts() instead of re-running the model per threshold."""
    frame = cv2.imread(str(img_path))
    results = model(frame, imgsz=IMGSZ, conf=0.10, max_det=1, verbose=False)[0]
    if results.keypoints is None or len(results.keypoints.data) == 0:
        return None
    return results.keypoints.data[0].cpu().numpy()  # (62, 3)


def filter_kpts(raw: np.ndarray | None, conf_thresh: float) -> dict[int, tuple[float, float]]:
    """Returns {kpt_id (1-based): (x_px, y_px)} for raw detections above conf_thresh."""
    if raw is None:
        return {}
    return {i + 1: (float(x), float(y)) for i, (x, y, conf) in enumerate(raw) if conf >= conf_thresh}


# ── Homography ──────────────────────────────────────────────────────────────────

def fit_homography(kpt_px: dict[int, tuple[float, float]]) -> np.ndarray | None:
    """Fits image-pixel -> field-yard homography from a set of keypoint correspondences.
    Returns None if fewer than MIN_CORRESPONDENCES points are available."""
    if len(kpt_px) < MIN_CORRESPONDENCES:
        return None
    ids = sorted(kpt_px)
    src = np.array([kpt_px[i] for i in ids], dtype=np.float64)
    dst = np.array([FIELD_POSITIONS[i] for i in ids], dtype=np.float64)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH_YD)
    return H


def project(H: np.ndarray, pts: list[tuple[float, float]]) -> np.ndarray:
    """Projects image-pixel points through H into field-yard coordinates."""
    arr = np.array(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(arr, H).reshape(-1, 2)


# ── Experiment 1: player-projection error (yards) ────────────────────────────────

def eval_player_projection(model, clip_ids: list[int], clip_to_kpt_frame: dict[int, int],
                            conf_thresh: float) -> dict:
    errors = []
    n_used = n_skipped = 0

    for clip_num in clip_ids:
        fidx = clip_to_kpt_frame.get(clip_num)
        ann_path = PLAYER_ANN / f"frame_{clip_num:03d}.txt"
        if fidx is None:
            n_skipped += 1
            continue
        img_path = TEST_IMGS / f"frame_{fidx:06d}.png"
        lbl_path = TEST_LABELS / f"frame_{fidx:06d}.txt"
        if not (img_path.exists() and lbl_path.exists() and ann_path.exists()):
            n_skipped += 1
            continue

        frame = cv2.imread(str(img_path))
        h, w = frame.shape[:2]

        gt_kpts   = load_gt_keypoints(lbl_path, w, h)
        pred_kpts = filter_kpts(run_model_keypoints_raw(model, img_path), conf_thresh)
        H_gt   = fit_homography(gt_kpts)
        H_pred = fit_homography(pred_kpts)
        if H_gt is None or H_pred is None:
            n_skipped += 1
            continue

        footpoints = load_gt_player_footpoints(ann_path, w, h)
        p_gt   = project(H_gt, footpoints)
        p_pred = project(H_pred, footpoints)
        dists  = np.linalg.norm(p_gt - p_pred, axis=1)
        errors.extend(dists.tolist())
        n_used += 1

    return {
        "n_frames_used":    n_used,
        "n_frames_skipped": n_skipped,
        "n_players":        len(errors),
        "mean_err_yd": round(float(np.mean(errors)), 3) if errors else None,
        "med_err_yd":  round(float(np.median(errors)), 3) if errors else None,
        "p90_err_yd":  round(float(np.percentile(errors, 90)), 3) if errors else None,
    }


# ── Experiment 2/3: keypoint-only field-space error ──────────────────────────────

def eval_keypoint_field_error(model, frame_ids: list[int], conf_thresholds: list[float]) -> dict[float, dict]:
    """Evaluates every threshold in conf_thresholds in a single pass over frame_ids —
    the model only runs once per frame regardless of how many thresholds are requested,
    since per-keypoint conf is already available from that one forward pass."""
    errors             = {c: [] for c in conf_thresholds}
    n_correspondences  = {c: [] for c in conf_thresholds}
    n_used             = {c: 0 for c in conf_thresholds}
    n_skipped          = {c: 0 for c in conf_thresholds}
    n_det_total        = {c: 0 for c in conf_thresholds}
    n_below_min        = {c: 0 for c in conf_thresholds}
    n_gt_total = 0

    for fidx in frame_ids:
        img_path = TEST_IMGS / f"frame_{fidx:06d}.png"
        lbl_path = TEST_LABELS / f"frame_{fidx:06d}.txt"
        if not (img_path.exists() and lbl_path.exists()):
            for c in conf_thresholds:
                n_skipped[c] += 1
            continue

        frame = cv2.imread(str(img_path))
        h, w = frame.shape[:2]

        gt_kpts = load_gt_keypoints(lbl_path, w, h)
        H_gt = fit_homography(gt_kpts)
        if H_gt is None:
            for c in conf_thresholds:
                n_skipped[c] += 1
            continue
        n_gt_total += len(gt_kpts)

        raw_pred = run_model_keypoints_raw(model, img_path)  # one inference call per frame

        for c in conf_thresholds:
            pred_kpts = filter_kpts(raw_pred, c)
            n_correspondences[c].append(len(pred_kpts))
            n_used[c] += 1
            if len(pred_kpts) < MIN_CORRESPONDENCES:
                n_below_min[c] += 1

            # Only score keypoints that are both GT-visible (true field position is
            # meaningful for this frame) and detected by the model at this threshold.
            common = [kid for kid in gt_kpts if kid in pred_kpts]
            n_det_total[c] += len(common)
            if not common:
                continue

            pred_px = [pred_kpts[kid] for kid in common]
            p_pred_field = project(H_gt, pred_px)
            true_field    = np.array([FIELD_POSITIONS[kid] for kid in common])
            dists = np.linalg.norm(p_pred_field - true_field, axis=1)
            errors[c].extend(dists.tolist())

    out = {}
    for c in conf_thresholds:
        e = errors[c]
        out[c] = {
            "n_frames_used":    n_used[c],
            "n_frames_skipped": n_skipped[c],
            "n_points":         len(e),
            "det_rate":         round(n_det_total[c] / n_gt_total, 4) if n_gt_total else None,
            "avg_n_correspondences": round(float(np.mean(n_correspondences[c])), 1) if n_correspondences[c] else None,
            "n_below_min_correspondences": n_below_min[c],
            "pct_below_min_correspondences": round(n_below_min[c] / n_used[c], 4) if n_used[c] else None,
            "mean_err_yd": round(float(np.mean(e)), 3) if e else None,
            "med_err_yd":  round(float(np.median(e)), 3) if e else None,
            "p90_err_yd":  round(float(np.percentile(e, 90)), 3) if e else None,
        }
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--quarter", default=None, choices=["Q1", "Q2", "Q3", "Q4"],
                         help="Restrict to this quarter's held-out frames (quarter-based models). "
                              "Omit for models evaluated on the full fixed TB2026 test set.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--sweep", action="store_true",
                         help="Also run experiment 2 across SWEEP_THRESHOLDS.")
    parser.add_argument("--out", default=None, help="Output JSON path (default: next to weights)")
    args = parser.parse_args()

    weights = Path(args.weights)
    model = YOLO(str(weights))

    all_frame_ids = sorted(frame_number(p) for p in TEST_IMGS.glob("*.png"))
    if args.quarter:
        frame_to_q = load_frame_to_quarter()
        all_frame_ids = [f for f in all_frame_ids if frame_to_q.get(f) == args.quarter]

    clip_to_kpt_frame = load_clip_to_kpt_frame()
    clip_to_quarter    = load_clip_to_quarter()
    player_clip_ids = sorted(frame_number(p) for p in PLAYER_ANN.glob("*.txt"))
    if args.quarter:
        player_clip_ids = [c for c in player_clip_ids if clip_to_quarter.get(c) == args.quarter]

    print(f"Weights: {weights}")
    print(f"Quarter filter: {args.quarter or '(none — full test set)'}")
    print(f"Frames: {len(all_frame_ids)} total, {len(player_clip_ids)} with player GT")

    thresholds = sorted(set([args.conf] + (SWEEP_THRESHOLDS if args.sweep else [])))
    print(f"Running keypoint field-error pass over thresholds: {thresholds}")
    kf_by_thresh = eval_keypoint_field_error(model, all_frame_ids, thresholds)

    results = {
        "config": {"weights": str(weights), "quarter": args.quarter, "conf": args.conf},
        "player_projection": eval_player_projection(model, player_clip_ids, clip_to_kpt_frame, args.conf),
        "keypoint_field_error": kf_by_thresh[args.conf],
    }

    if args.sweep:
        results["threshold_sweep"] = [{"conf": c, **kf_by_thresh[c]} for c in SWEEP_THRESHOLDS]

    print("\n" + "=" * 70)
    print(f"HOMOGRAPHY EVALUATION — {weights.parent.parent.name}"
          f"{' / ' + args.quarter if args.quarter else ''}")
    print("=" * 70)
    pp = results["player_projection"]
    print(f"  Player projection ({pp['n_frames_used']} frames, {pp['n_players']} players):")
    print(f"    mean={pp['mean_err_yd']}yd  med={pp['med_err_yd']}yd  p90={pp['p90_err_yd']}yd")
    kf = results["keypoint_field_error"]
    print(f"  Keypoint field error ({kf['n_frames_used']} frames, {kf['n_points']} points):")
    print(f"    mean={kf['mean_err_yd']}yd  med={kf['med_err_yd']}yd  p90={kf['p90_err_yd']}yd")
    print("=" * 70)

    out = Path(args.out) if args.out else weights.parent.parent / "homography_eval.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
