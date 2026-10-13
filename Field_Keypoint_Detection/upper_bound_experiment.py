"""
Upper bound experiment — 5-fold cross-validation on TB2026 keypoints.

Each fold trains on 4/5 of TB2026 clips and evaluates on the held-out 1/5.
The mean val mAP across folds is the upper bound reference ceiling.

Starting weights: yolo11l-pose.pt (COCO pretrained)
Resolution: 1280 px

Usage:
  python upper_bound_experiment.py
  python upper_bound_experiment.py --fold 2
  python upper_bound_experiment.py --device 1
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE  = Path(__file__).parent
KP_DS = Path(os.environ.get("KP_DS", BASE.parent / "Data" / "Keypoints" / "datasets"))

TB_DS  = KP_DS / "tb2026"
OUT    = BASE / "upper_bound_results"

MODEL  = "yolo11l-pose.pt"
IMGSZ  = 1280
BATCH  = -1
DEVICE = 0
EPOCHS = 100

N_FOLDS = 5
FOLDS   = [f"fold_{k}" for k in range(N_FOLDS)]


# ── Yaml path fix ─────────────────────────────────────────────────────────────

def fix_yaml_path(yaml_path: Path, dataset_dir: Path) -> Path:
    """Rewrite the 'path:' field to match the actual location on this machine."""
    fixed = OUT / f"_fixed_{yaml_path.parent.name}.yaml"
    fixed.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for line in yaml_path.read_text().splitlines():
        lines.append(f"path: {dataset_dir}" if line.startswith("path:") else line)
    fixed.write_text("\n".join(lines) + "\n")
    return fixed


# ── Metrics ───────────────────────────────────────────────────────────────────

def extract_metrics(results, elapsed: float, n: int) -> dict:
    pose = results.pose
    box  = results.box
    return {
        "pose": {
            "mAP50":    round(float(pose.map50), 4),
            "mAP50-95": round(float(pose.map),   4),
        },
        "box": {
            "mAP50":    round(float(box.map50), 4),
            "mAP50-95": round(float(box.map),   4),
            "precision": round(float(np.mean(box.p)), 4),
            "recall":    round(float(np.mean(box.r)), 4),
        },
        "fps": round(n / elapsed, 1),
    }


# ── Train & eval ──────────────────────────────────────────────────────────────

def n_val_images(fold_dir: Path) -> int:
    return len(list((fold_dir / "images" / "val").glob("*.png")))


def run_fold(fold: str) -> dict:
    fold_dir  = TB_DS / fold
    run_dir   = OUT / fold
    run_dir.mkdir(parents=True, exist_ok=True)

    yaml      = fix_yaml_path(fold_dir / "data.yaml", fold_dir)
    best_pt   = run_dir / "train" / "weights" / "best.pt"
    metrics_cache = run_dir / "metrics.json"

    if metrics_cache.exists() and best_pt.exists():
        print(f"    [cache] {fold} loaded")
        return json.loads(metrics_cache.read_text())

    if not best_pt.exists():
        mdl = YOLO(MODEL)
        mdl.train(
            data=str(yaml), epochs=EPOCHS, batch=BATCH,
            imgsz=IMGSZ, device=DEVICE,
            project=str(run_dir), name="train", exist_ok=True,
            lr0=0.001, lrf=0.01, cos_lr=True,
            warmup_epochs=3, patience=20,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            translate=0.1, scale=0.5,
            fliplr=0.5, flipud=0.0, mosaic=1.0, close_mosaic=10,
            save=True, save_period=-1, verbose=True, workers=8,
        )

    n   = n_val_images(fold_dir)
    mdl = YOLO(str(best_pt))
    t0  = time.perf_counter()
    res = mdl.val(
        data=str(yaml), imgsz=IMGSZ, batch=8,
        device=DEVICE, split="val",
        project=str(run_dir), name="eval", exist_ok=True,
        half=True, workers=8,
    )
    elapsed = time.perf_counter() - t0
    m = extract_metrics(res, elapsed, n)
    metrics_cache.write_text(json.dumps(m, indent=2))
    print(f"    pose mAP50={m['pose']['mAP50']:.4f}  box mAP50={m['box']['mAP50']:.4f}")
    return m


# ── Summary ───────────────────────────────────────────────────────────────────

def load_summary() -> dict:
    p = OUT / "summary.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_summary(data: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(data, indent=2))


def print_summary(data: dict):
    print("\n" + "=" * 55)
    print("UPPER BOUND — 5-fold CV on TB2026 (1280px)")
    print("=" * 55)
    print(f"{'Fold':<10} {'mAP50-pose':>12} {'mAP50-box':>12}")
    print("-" * 55)
    pose_vals, box_vals = [], []
    for fold in FOLDS:
        m    = data.get(fold, {})
        pose = m.get("pose", {}).get("mAP50", None)
        box  = m.get("box",  {}).get("mAP50", None)
        fp   = f"{pose:.4f}" if pose is not None else "—"
        fb   = f"{box:.4f}"  if box  is not None else "—"
        print(f"{fold:<10} {fp:>12} {fb:>12}")
        if pose is not None: pose_vals.append(pose)
        if box  is not None: box_vals.append(box)
    print("-" * 55)
    if pose_vals:
        print(f"{'mean':<10} {np.mean(pose_vals):>12.4f} {np.mean(box_vals):>12.4f}")
        print(f"{'std':<10} {np.std(pose_vals):>12.4f} {np.std(box_vals):>12.4f}")
    print("=" * 55)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=FOLDS, default=None,
                        help="Run a single fold (default: all)")
    parser.add_argument("--device", type=int, default=DEVICE)
    args   = parser.parse_args()
    DEVICE = args.device

    folds   = [args.fold] if args.fold else FOLDS
    summary = load_summary()

    for fold in folds:
        if fold in summary:
            print(f"  [skip] {fold} already done")
            continue
        print(f"\n{'─'*55}")
        print(f"  {fold}  |  upper bound CV")
        print(f"{'─'*55}")
        if not (TB_DS / fold / "data.yaml").exists():
            print(f"  [SKIP] dataset not found: {TB_DS / fold}")
            continue
        m = run_fold(fold)
        summary[fold] = m
        save_summary(summary)

    print_summary(summary)
    print(f"\nResults: {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
