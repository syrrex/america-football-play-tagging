"""
Cross-stadium progressive fine-tuning experiment — yolo11l-pose, 1280 px.

All stages are evaluated on the same held-out TB2026 test set (2920 frames).

Stage 1 : COCO zero-shot                   → eval TB2026
Stage 2 : train condition1 (game1+2+5)     → eval TB2026
Stage 3 : Stage 2 → fine-tune ravelin2025  → eval TB2026

Usage:
  python cross_stadium_experiment.py
  python cross_stadium_experiment.py --stage 2
  python cross_stadium_experiment.py --device 1
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

# Override with KP_DATASETS env var on the cluster
KP_DS = Path(os.environ.get("KP_DATASETS", BASE.parent / "Data" / "Keypoints" / "datasets"))

COND1_YAML = KP_DS / "condition1"   / "data.yaml"
R25_YAML   = KP_DS / "ravelin2025"  / "data.yaml"
EVAL_YAML  = KP_DS / "eval_only.yaml"

OUT = BASE / "cross_stadium_results"


def fix_yaml_path(yaml_path: Path, dataset_dir: Path) -> Path:
    """Rewrite the 'path:' field in a data.yaml to match the actual location.

    The yaml files were built on a different machine and contain a hardcoded
    absolute path. This rewrites that line so YOLO finds images correctly
    wherever the datasets are stored.
    """
    fixed = OUT / f"_fixed_{yaml_path.parent.name}.yaml"
    text  = yaml_path.read_text()
    lines = []
    for line in text.splitlines():
        if line.startswith("path:"):
            lines.append(f"path: {dataset_dir}")
        else:
            lines.append(line)
    fixed.write_text("\n".join(lines) + "\n")
    return fixed

MODEL    = "yolo11l-pose.pt"   # COCO pretrained starting point
IMGSZ    = 1280
BATCH    = -1    # auto-detect optimal batch size for available GPU VRAM
DEVICE   = 0

STAGE_ORDER = ["1", "2", "3"]

STAGE_LABELS = {
    "1": "COCO zero-shot",
    "2": "train condition1 (game1+2+5)",
    "3": "cond1 → fine-tune ravelin2025",
}

EPOCHS_STAGE2 = 150
EPOCHS_STAGE3 = 50


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


# ── Eval & train helpers ──────────────────────────────────────────────────────

def n_test_images() -> int:
    return len(list((KP_DS / "test" / "images").glob("*.png")))


def evaluate(weights: str | Path, run_dir: Path,
             cache_name: str = "metrics.json", eval_yaml: Path = None) -> dict:
    cache = run_dir / cache_name
    if cache.exists():
        print(f"    [cache] {cache_name} loaded")
        return json.loads(cache.read_text())

    yaml  = eval_yaml or EVAL_YAML
    n     = n_test_images()
    mdl   = YOLO(str(weights))
    t0    = time.perf_counter()
    res   = mdl.val(
        data=str(yaml), imgsz=IMGSZ, batch=8,
        device=DEVICE, split="test",
        project=str(run_dir), name="eval", exist_ok=True,
        half=True, workers=8,
    )
    elapsed = time.perf_counter() - t0
    m = extract_metrics(res, elapsed, n)
    cache.write_text(json.dumps(m, indent=2))
    print(f"    pose mAP50={m['pose']['mAP50']:.4f}  "
          f"mAP50-95={m['pose']['mAP50-95']:.4f}  "
          f"box mAP50={m['box']['mAP50']:.4f}")
    return m


def train_and_eval(weights: str | Path, data_yaml: Path,
                   run_dir: Path, epochs: int,
                   eval_yaml: Path = None) -> tuple[Path, dict]:
    """Train from `weights` on `data_yaml`, return (best_pt, eval_metrics)."""
    best_pt = run_dir / "train" / "weights" / "best.pt"

    if not best_pt.exists():
        mdl = YOLO(str(weights))
        mdl.train(
            data=str(data_yaml), epochs=epochs, batch=BATCH,
            imgsz=IMGSZ, device=DEVICE,
            project=str(run_dir), name="train", exist_ok=True,
            lr0=0.001, lrf=0.01, cos_lr=True,
            warmup_epochs=3, patience=20,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            translate=0.1, scale=0.5,
            fliplr=0.5, flipud=0.0, mosaic=1.0, close_mosaic=10,
            save=True, save_period=-1, verbose=True, workers=8,
        )

    m = evaluate(best_pt, run_dir, cache_name="metrics.json", eval_yaml=eval_yaml)
    return best_pt, m


# ── Summary I/O ───────────────────────────────────────────────────────────────

def load_summary() -> dict:
    p = OUT / "summary.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_summary(data: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(data, indent=2))


def print_summary(data: dict):
    print("\n" + "=" * 65)
    print("KEYPOINT CROSS-STADIUM LADDER — eval on TB2026 test (1280px)")
    print("=" * 65)
    print(f"{'Stage':<6} {'Description':<38} {'mAP50-pose':>10} {'mAP50-box':>10}")
    print("-" * 65)
    for s in STAGE_ORDER:
        m     = data.get(s, {})
        pose  = m.get("pose", {}).get("mAP50", "—")
        box   = m.get("box",  {}).get("mAP50", "—")
        fp    = f"{pose:.4f}" if isinstance(pose, float) else pose
        fb    = f"{box:.4f}"  if isinstance(box,  float) else box
        print(f"{s:<6} {STAGE_LABELS.get(s,''):<38} {fp:>10} {fb:>10}")
    print("=" * 65)


# ── Stage runner ──────────────────────────────────────────────────────────────

def run_stage(stage: str, summary: dict) -> dict:
    run_dir = OUT / stage
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*55}")
    print(f"  Stage {stage}  |  {STAGE_LABELS[stage]}")
    print(f"{'─'*55}")

    cond1_yaml = fix_yaml_path(COND1_YAML, KP_DS / "condition1")
    r25_yaml   = fix_yaml_path(R25_YAML,   KP_DS / "ravelin2025")
    eval_yaml  = fix_yaml_path(EVAL_YAML,  KP_DS)

    if stage == "1":
        m = evaluate(MODEL, run_dir, eval_yaml=eval_yaml)

    elif stage == "2":
        _, m = train_and_eval(MODEL, cond1_yaml, run_dir, EPOCHS_STAGE2, eval_yaml=eval_yaml)

    elif stage == "3":
        prev_pt = OUT / "2" / "train" / "weights" / "best.pt"
        if not prev_pt.exists():
            raise RuntimeError(f"Stage 2 weights not found at {prev_pt}. Run stage 2 first.")
        _, m = train_and_eval(prev_pt, r25_yaml, run_dir, EPOCHS_STAGE3, eval_yaml=eval_yaml)

    else:
        raise ValueError(f"Unknown stage: {stage}")

    summary[stage] = m
    save_summary(summary)
    return m


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGE_ORDER, default=None,
                        help="Run a single stage (default: all)")
    parser.add_argument("--device", type=int, default=DEVICE)
    args = parser.parse_args()
    DEVICE = args.device

    stages  = [args.stage] if args.stage else STAGE_ORDER
    summary = load_summary()

    for stage in stages:
        if stage in summary:
            print(f"  [skip] stage {stage} already done")
            continue
        try:
            run_stage(stage, summary)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  [SKIP] {e}")

    print_summary(summary)
    print(f"\nResults: {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
