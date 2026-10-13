"""
Condition1 learning curve — train on subsets of game1+2+5, eval on TB2026.

Shows how cross-stadium keypoint detection performance scales with the amount
of non-Ravelin training data. Val is the fixed condition1 held-out clips.
Test (reported metric) is the TB2026 held-out test set.

Starting weights: yolo11l-pose.pt (COCO pretrained)
Resolution: 1280 px

Usage:
  python condition1_learning_curve.py
  python condition1_learning_curve.py --frac frac_025
  python condition1_learning_curve.py --device 1
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

LC_DS     = KP_DS / "condition1_lc"
EVAL_YAML = KP_DS / "eval_only.yaml"
OUT       = BASE / "condition1_lc_results"

MODEL  = "yolo11l-pose.pt"
IMGSZ  = 1280
BATCH  = -1
DEVICE = 0
EPOCHS = 150

FRACS = ["frac_005", "frac_010", "frac_025", "frac_050", "frac_075", "frac_100"]

FRAC_LABELS = {
    "frac_005":  "5%",
    "frac_010": "10%",
    "frac_025": "25%",
    "frac_050": "50%",
    "frac_075": "75%",
    "frac_100": "100%",
}


# ── Yaml path fix ─────────────────────────────────────────────────────────────

def fix_yaml_path(yaml_path: Path, dataset_dir: Path) -> Path:
    fixed = OUT / f"_fixed_{yaml_path.parent.name}.yaml"
    fixed.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for line in yaml_path.read_text().splitlines():
        lines.append(f"path: {dataset_dir}" if line.startswith("path:") else line)
    fixed.write_text("\n".join(lines) + "\n")
    return fixed


def fix_eval_yaml() -> Path:
    fixed = OUT / "_fixed_eval_only.yaml"
    fixed.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for line in EVAL_YAML.read_text().splitlines():
        lines.append(f"path: {KP_DS}" if line.startswith("path:") else line)
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

def n_test_images() -> int:
    return len(list((KP_DS / "test" / "images").glob("*.png")))


def n_train_frames(frac_dir: Path) -> int:
    return len(list((frac_dir / "images" / "train").glob("*.png")))


def run_frac(frac: str, eval_yaml: Path) -> dict:
    frac_dir  = LC_DS / frac
    run_dir   = OUT / frac
    run_dir.mkdir(parents=True, exist_ok=True)

    train_yaml    = fix_yaml_path(frac_dir / "data.yaml", frac_dir)
    best_pt       = run_dir / "train" / "weights" / "best.pt"
    metrics_cache = run_dir / "metrics.json"

    if metrics_cache.exists() and best_pt.exists():
        print(f"    [cache] {frac} loaded")
        return json.loads(metrics_cache.read_text())

    n_train = n_train_frames(frac_dir)
    print(f"    training on {n_train} frames")

    if not best_pt.exists():
        mdl = YOLO(MODEL)
        mdl.train(
            data=str(train_yaml), epochs=EPOCHS, batch=BATCH,
            imgsz=IMGSZ, device=DEVICE,
            project=str(run_dir), name="train", exist_ok=True,
            lr0=0.001, lrf=0.01, cos_lr=True,
            warmup_epochs=3, patience=20,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            translate=0.1, scale=0.5,
            fliplr=0.5, flipud=0.0, mosaic=1.0, close_mosaic=10,
            save=True, save_period=-1, verbose=True, workers=8,
        )

    # Evaluate on TB2026 test set
    n   = n_test_images()
    mdl = YOLO(str(best_pt))
    t0  = time.perf_counter()
    res = mdl.val(
        data=str(eval_yaml), imgsz=IMGSZ, batch=8,
        device=DEVICE, split="test",
        project=str(run_dir), name="eval", exist_ok=True,
        half=True, workers=8,
    )
    elapsed = time.perf_counter() - t0
    m = extract_metrics(res, elapsed, n)
    m["n_train_frames"] = n_train
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
    print("\n" + "=" * 65)
    print("CONDITION1 LC — cross-stadium → eval TB2026 (1280px)")
    print("=" * 65)
    print(f"{'Frac':<10} {'Train frames':>14} {'mAP50-pose':>12} {'mAP50-box':>12}")
    print("-" * 65)
    for frac in FRACS:
        m    = data.get(frac, {})
        n    = m.get("n_train_frames", "—")
        pose = m.get("pose", {}).get("mAP50", None)
        box  = m.get("box",  {}).get("mAP50", None)
        fn   = f"{n}" if isinstance(n, int) else n
        fp   = f"{pose:.4f}" if pose is not None else "—"
        fb   = f"{box:.4f}"  if box  is not None else "—"
        print(f"{FRAC_LABELS.get(frac, frac):<10} {fn:>14} {fp:>12} {fb:>12}")
    print("=" * 65)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--frac", choices=FRACS, default=None,
                        help="Run a single fraction (default: all)")
    parser.add_argument("--device", type=int, default=DEVICE)
    args   = parser.parse_args()
    DEVICE = args.device

    fracs     = [args.frac] if args.frac else FRACS
    summary   = load_summary()
    eval_yaml = fix_eval_yaml()

    for frac in fracs:
        if frac in summary:
            print(f"  [skip] {frac} already done")
            continue
        print(f"\n{'─'*55}")
        print(f"  {frac}  ({FRAC_LABELS.get(frac, frac)} of condition1 train clips)")
        print(f"{'─'*55}")
        if not (LC_DS / frac / "data.yaml").exists():
            print(f"  [SKIP] dataset not found: {LC_DS / frac}")
            continue
        m = run_frac(frac, eval_yaml)
        summary[frac] = m
        save_summary(summary)

    print_summary(summary)
    print(f"\nResults: {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
