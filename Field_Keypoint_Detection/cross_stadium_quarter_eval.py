"""
Cross-stadium evaluation on per-quarter TB2026 val sets.

No training — re-evaluates existing cross-stadium model weights on each
game quarter separately, using the same quarter boundaries as the k-fold.

Models evaluated:
  stage2 : trained on condition1 (game1+2+5)
  stage3 : stage2 fine-tuned on ravelin2025

For each model × quarter the script:
  1. Builds a temporary per-quarter val dir (symlinks, no copies)
  2. Runs evaluation (same metric as evaluate_keypoints.py)
  3. Saves kpt_eval.json
  4. Removes the temp val dir

Usage:
  python cross_stadium_quarter_eval.py
  python cross_stadium_quarter_eval.py --stage stage2
  TB2026_SRC=~/keypoint_dataset/tb2026_all python cross_stadium_quarter_eval.py
"""

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent

META_CSV = Path(os.environ.get(
    "KP_META",
    BASE.parent / "Data" / "metadata" / "tb2026_dataset_metadata.csv"
))

TB2026_SRC = Path(os.environ.get(
    "TB2026_SRC",
    BASE.parent / "Data" / "Keypoints" / "datasets" / "test"
)).expanduser()

OUT = BASE / "cross_stadium_quarter_results"

QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
N_KPT    = 62
IMGSZ    = 1280
CONF     = 0.25

FRAME_RE = re.compile(r"frame_(\d+)\.")

# Models to evaluate: {name: weights_path}
MODELS = {
    "stage2": BASE / "cross_stadium_results" / "2" / "train" / "weights" / "best.pt",
    "stage3": BASE / "cross_stadium_results" / "3" / "train" / "weights" / "best.pt",
}


# ── Quarter frame sets ─────────────────────────────────────────────────────────

def load_quarter_frames() -> dict[str, set[int]]:
    """Returns {quarter: set_of_frame_indices}."""
    meta = pd.read_csv(META_CSV)
    qframes: dict[str, set[int]] = {q: set() for q in QUARTERS}
    for _, row in meta.iterrows():
        q = row["QTR"]
        if q not in qframes:
            continue
        for f in range(int(row["start_frame"]), int(row["end_frame"]) + 1):
            qframes[q].add(f)
    return qframes


def frame_number(path: Path) -> int:
    m = FRAME_RE.search(path.name)
    return int(m.group(1)) if m else -1


def build_quarter_val_dir(quarter: str, frames: set[int],
                          src_imgs: Path, src_lbls: Path) -> Path:
    """Creates a temporary directory with symlinks for this quarter's frames."""
    tmp = Path(tempfile.mkdtemp(prefix=f"cs_eval_{quarter}_"))
    (tmp / "images").mkdir()
    (tmp / "labels").mkdir()
    for img in sorted(src_imgs.glob("*.png")):
        if frame_number(img) not in frames:
            continue
        lbl = src_lbls / img.with_suffix(".txt").name
        if not lbl.exists():
            continue
        (tmp / "images" / img.name).symlink_to(img.resolve())
        (tmp / "labels" / lbl.name).symlink_to(lbl.resolve())
    return tmp


# ── Evaluation (inline, same logic as evaluate_keypoints.py) ──────────────────

KPT_GROUPS = {
    range(1,  14): "Sideline top  (1–13)",
    range(14, 27): "Sideline bot  (14–26)",
    range(27, 36): "Numbers top   (27–35)",
    range(36, 45): "Numbers bot   (36–44)",
    range(45, 54): "Hash top      (45–53)",
    range(54, 63): "Hash bot      (54–62)",
}

def group_name(kid: int) -> str:
    for r, name in KPT_GROUPS.items():
        if kid in r:
            return name
    return "unknown"

def load_gt(label_path: Path, img_w: int, img_h: int) -> dict[int, tuple]:
    line = label_path.read_text().strip().split()
    gt = {}
    for i in range(N_KPT):
        x_n = float(line[5 + i * 3])
        y_n = float(line[5 + i * 3 + 1])
        vis = int(float(line[5 + i * 3 + 2]))
        if vis > 0:
            gt[i + 1] = (x_n * img_w, y_n * img_h)
    return gt

def run_model(model, img_path: Path) -> dict[int, tuple]:
    frame = cv2.imread(str(img_path))
    results = model(frame, imgsz=IMGSZ, conf=0.10, max_det=1, verbose=False)[0]
    if results.keypoints is None or len(results.keypoints.data) == 0:
        return {}
    kpts = results.keypoints.data[0].cpu().numpy()
    return {i + 1: (float(x), float(y)) for i, (x, y, c) in enumerate(kpts) if c >= CONF}

def evaluate_on_dir(model, img_dir: Path, lbl_dir: Path) -> dict:
    img_paths = sorted(img_dir.glob("*.png"))
    n_gt       = np.zeros(N_KPT + 1)
    n_detected = np.zeros(N_KPT + 1)
    errors     = [[] for _ in range(N_KPT + 1)]

    for img_path in img_paths:
        lbl_path = lbl_dir / img_path.with_suffix(".txt").name
        if not lbl_path.exists():
            continue
        frame = cv2.imread(str(img_path))
        h, w  = frame.shape[:2]
        gt    = load_gt(lbl_path, w, h)
        pred  = run_model(model, img_path)
        for kid, gt_pos in gt.items():
            n_gt[kid] += 1
            if kid in pred:
                n_detected[kid] += 1
                dx = pred[kid][0] - gt_pos[0]
                dy = pred[kid][1] - gt_pos[1]
                err_px   = float(np.sqrt(dx**2 + dy**2))
                err_norm = err_px / np.sqrt(w**2 + h**2)
                errors[kid].append((err_px, err_norm))

    kpt_stats = {}
    for kid in range(1, N_KPT + 1):
        if n_gt[kid] == 0:
            continue
        errs_px = [e[0] for e in errors[kid]]
        errs_n  = [e[1] for e in errors[kid]]
        kpt_stats[kid] = {
            "n_gt":          int(n_gt[kid]),
            "n_detected":    int(n_detected[kid]),
            "det_rate":      round(float(n_detected[kid]) / float(n_gt[kid]), 4),
            "mean_err_px":   round(float(np.mean(errs_px)),   2) if errs_px else None,
            "med_err_px":    round(float(np.median(errs_px)), 2) if errs_px else None,
            "mean_err_norm": round(float(np.mean(errs_n)),    5) if errs_n  else None,
            "group":         group_name(kid),
        }

    all_gt   = int(n_gt[1:].sum())
    all_det  = int(n_detected[1:].sum())
    all_errs = [e[0] for kid in range(1, N_KPT+1) for e in errors[kid]]
    all_n    = [e[1] for kid in range(1, N_KPT+1) for e in errors[kid]]

    overall = {
        "n_frames":      len(img_paths),
        "n_gt_total":    all_gt,
        "n_det_total":   all_det,
        "det_rate":      round(all_det / all_gt, 4) if all_gt else 0,
        "mean_err_px":   round(float(np.mean(all_errs)),      2) if all_errs else None,
        "med_err_px":    round(float(np.median(all_errs)),    2) if all_errs else None,
        "p90_err_px":    round(float(np.percentile(all_errs, 90)), 2) if all_errs else None,
        "mean_err_norm": round(float(np.mean(all_n)),         5) if all_n    else None,
    }
    return {"overall": overall, "per_keypoint": kpt_stats}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=list(MODELS.keys()), default=None,
                        help="Run a single stage (default: all)")
    args = parser.parse_args()

    stages = {args.stage: MODELS[args.stage]} if args.stage else MODELS

    src_imgs = TB2026_SRC / "images"
    src_lbls = TB2026_SRC / "labels"

    print("=" * 60)
    print("CROSS-STADIUM QUARTER EVALUATION")
    print(f"Source   : {TB2026_SRC}")
    print(f"Metadata : {META_CSV}")
    print(f"Output   : {OUT}")
    print(f"Stages   : {list(stages.keys())}")
    print("=" * 60)

    quarter_frames = load_quarter_frames()

    for stage_name, weights in stages.items():
        if not Path(weights).exists():
            print(f"  [skip] {stage_name} — weights not found: {weights}")
            continue

        print(f"\nLoading {stage_name} ({weights})")
        model = YOLO(str(weights))

        for quarter in QUARTERS:
            out_path = OUT / stage_name / quarter / "kpt_eval.json"
            if out_path.exists():
                print(f"  [skip] {stage_name}/{quarter} — already evaluated")
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)
            frames = quarter_frames[quarter]
            print(f"\n  Evaluating {stage_name} on {quarter} ({len(frames)} frames)...")

            tmp = build_quarter_val_dir(quarter, frames, src_imgs, src_lbls)
            try:
                results = evaluate_on_dir(model, tmp / "images", tmp / "labels")
                o = results["overall"]
                print(f"    det={o['det_rate']*100:.1f}%  "
                      f"mean={o['mean_err_px']}px  med={o['med_err_px']}px")
                out_path.write_text(json.dumps(results, indent=2))
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

    print("\nAll done.")


if __name__ == "__main__":
    main()
