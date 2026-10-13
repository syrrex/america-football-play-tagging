"""
Flip augmentation ablation — train on full condition1, eval on TB2026 test.

Compares two runs:
  no_flip   : fliplr=0.0  (no horizontal flip augmentation) — trained here
  with_flip : fliplr=0.5  — reuses Stage 2 weights from cross_stadium_results

Usage:
  python flip_ablation.py
  python flip_ablation.py --device 1
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

COND1_YAML    = KP_DS / "condition1" / "data.yaml"
EVAL_YAML     = KP_DS / "eval_only.yaml"
OUT           = BASE / "flip_ablation_results"
STAGE2_PT     = BASE / "cross_stadium_results" / "2" / "train" / "weights" / "best.pt"

MODEL  = "yolo11l-pose.pt"
IMGSZ  = 1280
BATCH  = -1
DEVICE = 0
EPOCHS = 150


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

def extract_metrics(results, elapsed: float, n: int, fliplr: float) -> dict:
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
        "fps":    round(n / elapsed, 1),
        "fliplr": fliplr,
    }


# ── Train & eval ──────────────────────────────────────────────────────────────

def n_test_images() -> int:
    return len(list((KP_DS / "test" / "images").glob("*.png")))


def eval_weights(weights: Path, run_dir: Path, fliplr: float, eval_yaml: Path) -> dict:
    metrics_cache = run_dir / "metrics.json"
    if metrics_cache.exists():
        print(f"    [cache] loaded")
        return json.loads(metrics_cache.read_text())

    n   = n_test_images()
    mdl = YOLO(str(weights))
    t0  = time.perf_counter()
    res = mdl.val(
        data=str(eval_yaml), imgsz=IMGSZ, batch=8,
        device=DEVICE, split="test",
        project=str(run_dir), name="eval", exist_ok=True,
        half=True, workers=8,
    )
    elapsed = time.perf_counter() - t0
    m = extract_metrics(res, elapsed, n, fliplr)
    metrics_cache.write_text(json.dumps(m, indent=2))
    print(f"    pose mAP50={m['pose']['mAP50']:.4f}  box mAP50={m['box']['mAP50']:.4f}")
    return m


def run_no_flip(train_yaml: Path, eval_yaml: Path) -> dict:
    run_dir = OUT / "no_flip"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_pt = run_dir / "train" / "weights" / "best.pt"

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
            fliplr=0.0, flipud=0.0, mosaic=1.0, close_mosaic=10,
            save=True, save_period=-1, verbose=True, workers=8,
        )

    return eval_weights(best_pt, run_dir, 0.0, eval_yaml)


def run_with_flip(eval_yaml: Path) -> dict:
    run_dir = OUT / "with_flip"
    run_dir.mkdir(parents=True, exist_ok=True)

    if not STAGE2_PT.exists():
        raise RuntimeError(
            f"Stage 2 weights not found at {STAGE2_PT}. "
            "Run cross_stadium_experiment.py stage 2 first."
        )
    print(f"    reusing Stage 2 weights: {STAGE2_PT}")
    return eval_weights(STAGE2_PT, run_dir, 0.5, eval_yaml)


# ── Summary ───────────────────────────────────────────────────────────────────

RUNS = ["no_flip", "with_flip"]


def load_summary() -> dict:
    p = OUT / "summary.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_summary(data: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(data, indent=2))


def print_summary(data: dict):
    print("\n" + "=" * 65)
    print("FLIP ABLATION — condition1 train → eval TB2026 test (1280px)")
    print("=" * 65)
    print(f"{'Run':<12} {'fliplr':>8} {'mAP50-pose':>12} {'mAP50-box':>12}")
    print("-" * 65)
    for name in RUNS:
        m      = data.get(name, {})
        fliplr = m.get("fliplr", "—")
        pose   = m.get("pose", {}).get("mAP50", None)
        box    = m.get("box",  {}).get("mAP50", None)
        ff     = f"{fliplr:.1f}" if isinstance(fliplr, float) else fliplr
        fp     = f"{pose:.4f}"   if pose is not None else "—"
        fb     = f"{box:.4f}"    if box  is not None else "—"
        print(f"{name:<12} {ff:>8} {fp:>12} {fb:>12}")
    print("=" * 65)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=DEVICE)
    args   = parser.parse_args()
    DEVICE = args.device

    OUT.mkdir(parents=True, exist_ok=True)
    summary    = load_summary()
    train_yaml = fix_yaml_path(COND1_YAML, KP_DS / "condition1")
    eval_yaml  = fix_eval_yaml()

    if "no_flip" not in summary:
        print(f"\n{'─'*55}\n  no_flip  (fliplr=0.0 — training from scratch)\n{'─'*55}")
        summary["no_flip"] = run_no_flip(train_yaml, eval_yaml)
        save_summary(summary)
    else:
        print("  [skip] no_flip already done")

    if "with_flip" not in summary:
        print(f"\n{'─'*55}\n  with_flip  (fliplr=0.5 — reusing Stage 2 weights)\n{'─'*55}")
        try:
            summary["with_flip"] = run_with_flip(eval_yaml)
            save_summary(summary)
        except RuntimeError as e:
            print(f"  [SKIP] {e}")
    else:
        print("  [skip] with_flip already done")

    print_summary(summary)
    print(f"\nResults: {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()