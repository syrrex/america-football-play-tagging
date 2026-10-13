"""
Per-keypoint evaluation for field keypoint detection.

Bypasses YOLO's OKS/box-based mAP. For each of the 62 keypoints:
  - Detection rate : % of frames where GT was annotated and model detected it
  - Localization error : pixel distance between predicted and GT position
                         (for frames where both are present)

Reports overall stats + per-keypoint breakdown + saves a summary JSON.

Usage (run from Field_Keypoint_Detection/, the repo of *_results/ dirs):
  python evaluation/evaluate_keypoints.py --weights path/to/best.pt
  python evaluation/evaluate_keypoints.py --weights cross_stadium_results/2/train/weights/best.pt
  python evaluation/evaluate_keypoints.py --weights cross_stadium_results/2/train/weights/best.pt --conf 0.20
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).parent.parent  # Field_Keypoint_Detection/
KP_DS   = Path(os.environ.get("KP_DS", BASE.parent / "Data" / "Keypoints" / "datasets"))
TEST_IMGS   = KP_DS / "test" / "images"
TEST_LABELS = KP_DS / "test" / "labels"

N_KPT   = 62
IMGSZ   = 1280
DEVICE  = 0

# Keypoint group names for the breakdown table
KPT_GROUPS = {
    range(1,  14): "Sideline top  (1–13)",
    range(14, 27): "Sideline bot  (14–26)",
    range(27, 36): "Numbers top   (27–35)",
    range(36, 45): "Numbers bot   (36–44)",
    range(45, 54): "Hash top      (45–53)",
    range(54, 63): "Hash bot      (54–62)",
}


def group_name(kpt_id: int) -> str:
    for r, name in KPT_GROUPS.items():
        if kpt_id in r:
            return name
    return "unknown"


# ── Load GT from label file ────────────────────────────────────────────────────

def load_gt(label_path: Path, img_w: int, img_h: int) -> dict[int, tuple[float, float]]:
    """
    Returns {kpt_id (1-based): (x_px, y_px)} for all annotated keypoints (vis > 0).
    """
    line = label_path.read_text().strip().split()
    # YOLO pose format: class cx cy bw bh  [x y vis] * 62
    offset = 5
    gt = {}
    for i in range(N_KPT):
        x_n = float(line[offset + i * 3])
        y_n = float(line[offset + i * 3 + 1])
        vis = int(float(line[offset + i * 3 + 2]))
        if vis > 0:
            gt[i + 1] = (x_n * img_w, y_n * img_h)
    return gt


# ── Run model on one image ─────────────────────────────────────────────────────

def run_model(model, img_path: Path, conf_thresh: float) -> dict[int, tuple[float, float]]:
    """
    Returns {kpt_id (1-based): (x_px, y_px)} for all detected keypoints above conf_thresh.
    Uses the highest-confidence detection if multiple fields are found.
    """
    frame = cv2.imread(str(img_path))
    results = model(frame, imgsz=IMGSZ, conf=0.10, max_det=1, verbose=False)[0]

    if results.keypoints is None or len(results.keypoints.data) == 0:
        return {}

    kpts = results.keypoints.data[0].cpu().numpy()  # (62, 3)
    pred = {}
    for i, (x, y, conf) in enumerate(kpts):
        if conf >= conf_thresh:
            pred[i + 1] = (float(x), float(y))
    return pred


# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate(weights: Path, conf_thresh: float) -> dict:
    model = YOLO(str(weights))

    img_paths   = sorted(TEST_IMGS.glob("*.png"))
    label_paths = {p.stem: TEST_LABELS / f"{p.stem}.txt" for p in img_paths}

    # Per-keypoint accumulators
    n_gt        = np.zeros(N_KPT + 1)   # how many frames GT annotated (1-indexed)
    n_detected  = np.zeros(N_KPT + 1)   # detected when GT annotated
    errors      = [[] for _ in range(N_KPT + 1)]  # pixel distances when both present

    print(f"Evaluating {len(img_paths)} test images  (conf_thresh={conf_thresh})")

    for img_path in img_paths:
        lbl_path = label_paths.get(img_path.stem)
        if lbl_path is None or not lbl_path.exists():
            continue

        frame    = cv2.imread(str(img_path))
        h, w     = frame.shape[:2]
        gt       = load_gt(lbl_path, w, h)
        pred     = run_model(model, img_path, conf_thresh)

        for kpt_id, gt_pos in gt.items():
            n_gt[kpt_id] += 1
            if kpt_id in pred:
                n_detected[kpt_id] += 1
                dx = pred[kpt_id][0] - gt_pos[0]
                dy = pred[kpt_id][1] - gt_pos[1]
                err_px = float(np.sqrt(dx**2 + dy**2))
                # Normalise by image diagonal for scale-independence
                err_norm = err_px / np.sqrt(w**2 + h**2)
                errors[kpt_id].append((err_px, err_norm))

    # Aggregate
    kpt_stats = {}
    for kid in range(1, N_KPT + 1):
        if n_gt[kid] == 0:
            continue
        det_rate = float(n_detected[kid]) / float(n_gt[kid])
        errs_px  = [e[0] for e in errors[kid]]
        errs_n   = [e[1] for e in errors[kid]]
        kpt_stats[kid] = {
            "n_gt":        int(n_gt[kid]),
            "n_detected":  int(n_detected[kid]),
            "det_rate":    round(det_rate, 4),
            "mean_err_px": round(float(np.mean(errs_px)),  2) if errs_px else None,
            "med_err_px":  round(float(np.median(errs_px)), 2) if errs_px else None,
            "mean_err_norm": round(float(np.mean(errs_n)),  5) if errs_n else None,
            "group":       group_name(kid),
        }

    # Overall
    all_gt    = int(n_gt[1:].sum())
    all_det   = int(n_detected[1:].sum())
    all_errs  = [e[0] for kid in range(1, N_KPT+1) for e in errors[kid]]
    all_errs_n = [e[1] for kid in range(1, N_KPT+1) for e in errors[kid]]

    overall = {
        "n_frames":       len(img_paths),
        "n_gt_total":     all_gt,
        "n_det_total":    all_det,
        "det_rate":       round(all_det / all_gt, 4) if all_gt else 0,
        "mean_err_px":    round(float(np.mean(all_errs)),   2) if all_errs else None,
        "med_err_px":     round(float(np.median(all_errs)), 2) if all_errs else None,
        "p90_err_px":     round(float(np.percentile(all_errs, 90)), 2) if all_errs else None,
        "mean_err_norm":  round(float(np.mean(all_errs_n)), 5) if all_errs_n else None,
    }

    return {"overall": overall, "per_keypoint": kpt_stats}


# ── Print & save ──────────────────────────────────────────────────────────────

def print_results(results: dict, weights: Path):
    ov = results["overall"]
    kp = results["per_keypoint"]

    print("\n" + "=" * 70)
    print(f"KEYPOINT EVALUATION — {weights.parent.parent.name}")
    print("=" * 70)
    print(f"  Frames evaluated : {ov['n_frames']}")
    print(f"  GT annotations   : {ov['n_gt_total']}")
    print(f"  Detected         : {ov['n_det_total']}  ({ov['det_rate']*100:.1f}%)")
    print(f"  Mean error       : {ov['mean_err_px']} px  ({ov['mean_err_norm']:.4f} norm)")
    print(f"  Median error     : {ov['med_err_px']} px")
    print(f"  90th pct error   : {ov['p90_err_px']} px")

    # Per-group summary
    print("\n  Per group:")
    print(f"  {'Group':<28} {'Det rate':>9} {'Mean err px':>12} {'Med err px':>11}")
    print("  " + "-" * 62)
    groups: dict[str, list] = {}
    for kid, s in kp.items():
        g = s["group"]
        groups.setdefault(g, []).append(s)
    for gname, stats in groups.items():
        det   = np.mean([s["det_rate"] for s in stats])
        mean_e = np.mean([s["mean_err_px"] for s in stats if s["mean_err_px"] is not None])
        med_e  = np.mean([s["med_err_px"]  for s in stats if s["med_err_px"]  is not None])
        print(f"  {gname:<28} {det*100:>8.1f}% {mean_e:>12.1f} {med_e:>11.1f}")

    # Worst 5 keypoints by mean error
    worst = sorted(
        [(kid, s) for kid, s in kp.items() if s["mean_err_px"] is not None],
        key=lambda x: x[1]["mean_err_px"], reverse=True
    )[:5]
    print("\n  Hardest keypoints (highest mean error):")
    for kid, s in worst:
        print(f"    kpt {kid:>2}  det={s['det_rate']*100:.0f}%  "
              f"mean={s['mean_err_px']:.0f}px  group={s['group']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--conf",    type=float, default=0.25)
    parser.add_argument("--out",     default=None,
                        help="Output JSON path (default: next to weights)")
    parser.add_argument("--img-dir", default=None,
                        help="Custom image dir (default: KP_DS/test/images)")
    parser.add_argument("--lbl-dir", default=None,
                        help="Custom label dir (default: KP_DS/test/labels)")
    args = parser.parse_args()

    global TEST_IMGS, TEST_LABELS
    if args.img_dir:
        TEST_IMGS = Path(args.img_dir)
    if args.lbl_dir:
        TEST_LABELS = Path(args.lbl_dir)

    weights = Path(args.weights)
    results = evaluate(weights, args.conf)
    print_results(results, weights)

    out = Path(args.out) if args.out else weights.parent.parent / "kpt_eval.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
