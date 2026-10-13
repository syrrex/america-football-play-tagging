"""
Cross-stadium progressive fine-tuning experiment — yolo11m, 640 and 1280 px.

All stages are evaluated on the same held-out TB2026 test set (107 images).

Stage 1  : COCO zero-shot          → eval TB2026
Stage 2  : Roboflow pretrained      → eval TB2026  (weights already exist)
Stage 3a : COCO → train non-Ravelin → eval TB2026
Stage 3b : Stage 2 → train non-Ravelin → eval TB2026
Stage 4a : Stage 3a → fine-tune g8  → eval TB2026
Stage 4b : Stage 3b → fine-tune g8  → eval TB2026

Usage:
  python cross_stadium_experiment.py            # run all stages / resolutions
  python cross_stadium_experiment.py --imgsz 640
  python cross_stadium_experiment.py --stage 3a --imgsz 1280
  python cross_stadium_experiment.py --device 1
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
PROJECT   = BASE.parent
CS        = PROJECT / "Data" / "Players" / "datasets" / "cross_stadium"

NR_YAML   = CS / "non_ravelin"  / "data.yaml"   # non-Ravelin train + TB2026 test
R25_YAML  = CS / "ravelin2025"  / "data.yaml"   # Ravelin2025 train + TB2026 test

# Eval-only yaml — just points to TB2026 test (written below if missing)
EVAL_YAML = CS / "eval_only.yaml"

OUT       = BASE / "cross_stadium_results"

ROBOFLOW_M = {
    640:  Path("/mnt/c/Users/Lukas/Documents/Semester_5/Seminar_Project"
               "/Ravelin_Object_Detection/roboflow_models/yolo11m/weights/best.pt"),
    1280: Path("/mnt/c/Users/Lukas/Documents/Semester_5/Seminar_Project"
               "/Ravelin_Object_Detection/roboflow_models/yolo11m/weights/best.pt"),
}

CLASS_NAMES = ["defense", "oline", "qb", "ref", "skill"]
MODEL_SIZE  = "m"
IMAGE_SIZES = [640, 1280]
EPOCHS      = 50
BATCH       = {640: 12, 1280: 3}
DEVICE      = 0

STAGE_ORDER = ["1", "2", "3a", "3b", "4a", "4b"]

# ─────────────────────────────────────────────────────────────────────────────


def ensure_eval_yaml():
    if not EVAL_YAML.exists():
        EVAL_YAML.write_text(
            f"path: {CS}\n"
            f"train: test/images\n"
            f"val:   test/images\n"
            f"test:  test/images\n\n"
            f"nc: {len(CLASS_NAMES)}\n"
            f"names: {CLASS_NAMES}\n"
        )


def n_test_images() -> int:
    return len(list((CS / "test" / "images").glob("*")))


def extract_metrics(results, elapsed: float, n: int) -> dict:
    return {
        "overall": {
            "mAP50":      round(float(results.box.map50),      4),
            "mAP50-95":   round(float(results.box.map),        4),
            "precision":  round(float(np.mean(results.box.p)), 4),
            "recall":     round(float(np.mean(results.box.r)), 4),
            "fps":        round(n / elapsed, 1),
        },
        "per_class": {
            name: {
                "mAP50":     round(float(results.box.ap50[i]), 4),
                "mAP50-95":  round(float(results.box.ap[i]),   4),
                "precision": round(float(results.box.p[i]),    4),
                "recall":    round(float(results.box.r[i]),    4),
            }
            for i, name in enumerate(CLASS_NAMES)
        },
    }


def evaluate(weights: str | Path, imgsz: int, run_dir: Path,
             cache_name: str = "metrics.json") -> dict:
    cache = run_dir / cache_name
    if cache.exists():
        print(f"    [cache] {cache_name} loaded")
        return json.loads(cache.read_text())

    n   = n_test_images()
    mdl = YOLO(str(weights))
    t0  = time.perf_counter()
    res = mdl.val(
        data=str(EVAL_YAML), imgsz=imgsz, batch=BATCH[imgsz],
        device=DEVICE, split="test",
        project=str(run_dir), name="eval", exist_ok=True,
        half=(imgsz == 1280),
    )
    elapsed = time.perf_counter() - t0
    m = extract_metrics(res, elapsed, n)
    cache.write_text(json.dumps(m, indent=2))
    print(f"    mAP50={m['overall']['mAP50']:.4f}  "
          f"mAP50-95={m['overall']['mAP50-95']:.4f}")
    return m


def train_and_eval(weights: str | Path, data_yaml: Path,
                   imgsz: int, run_dir: Path) -> tuple[Path, dict]:
    """Train from `weights` on `data_yaml`, return (best_weights_path, eval_metrics)."""
    metrics_cache = run_dir / "metrics.json"
    best_pt       = run_dir / "train" / "weights" / "best.pt"

    # Full cache hit — both training and evaluation done
    if metrics_cache.exists() and best_pt.exists():
        print(f"    [cache] training results loaded")
        return best_pt, json.loads(metrics_cache.read_text())

    # Training done but evaluation was killed — skip straight to eval
    if not best_pt.exists():
        mdl = YOLO(str(weights))
        mdl.train(
            data=str(data_yaml), epochs=EPOCHS, batch=BATCH[imgsz],
            imgsz=imgsz, device=DEVICE,
            project=str(run_dir), name="train", exist_ok=True,
            lr0=0.001, lrf=0.01, cos_lr=True,
            warmup_epochs=3, patience=15,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            translate=0.1, scale=0.5,
            fliplr=0.5, flipud=0.0, mosaic=1.0, close_mosaic=10,
            save=True, save_period=-1, verbose=True,
        )

    m = evaluate(best_pt, imgsz, run_dir, cache_name="metrics.json")
    return best_pt, m


def load_summary() -> dict:
    p = OUT / "summary.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_summary(data: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(data, indent=2))


def print_summary(data: dict):
    print("\n" + "=" * 65)
    print("CROSS-STADIUM LADDER — mAP50 on TB2026 test")
    print("=" * 65)
    print(f"{'Stage':<8} {'Description':<35} {'640':>7} {'1280':>7}")
    print("-" * 65)
    labels = {
        "1":  "COCO zero-shot",
        "2":  "Roboflow pretrained",
        "3a": "COCO → non-Ravelin",
        "3b": "Roboflow → non-Ravelin",
        "4a": "3a → Ravelin 2025 (g8)",
        "4b": "3b → Ravelin 2025 (g8)",
    }
    for s in STAGE_ORDER:
        v640  = data.get(f"{s}_640",  {}).get("overall", {}).get("mAP50", "—")
        v1280 = data.get(f"{s}_1280", {}).get("overall", {}).get("mAP50", "—")
        f640  = f"{v640:.4f}" if isinstance(v640,  float) else v640
        f1280 = f"{v1280:.4f}" if isinstance(v1280, float) else v1280
        print(f"{s:<8} {labels.get(s,''):<35} {f640:>7} {f1280:>7}")
    print("=" * 65)


def run_stage(stage: str, imgsz: int, summary: dict) -> dict:
    key     = f"{stage}_{imgsz}"
    run_dir = OUT / key
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*55}")
    print(f"  Stage {stage}  |  imgsz={imgsz}")
    print(f"{'─'*55}")

    if stage == "1":
        m = evaluate(f"yolo11{MODEL_SIZE}.pt", imgsz, run_dir)

    elif stage == "2":
        rf_w = ROBOFLOW_M[imgsz]
        if not rf_w.exists():
            raise FileNotFoundError(f"Roboflow weights not found: {rf_w}")
        m = evaluate(rf_w, imgsz, run_dir)

    elif stage == "3a":
        start = f"yolo11{MODEL_SIZE}.pt"
        _, m  = train_and_eval(start, NR_YAML, imgsz, run_dir)

    elif stage == "3b":
        start = ROBOFLOW_M[imgsz]
        if not start.exists():
            raise FileNotFoundError(f"Roboflow weights not found: {start}")
        _, m = train_and_eval(start, NR_YAML, imgsz, run_dir)

    elif stage == "4a":
        prev_key = f"3a_{imgsz}"
        prev_pt  = OUT / prev_key / "train" / "weights" / "best.pt"
        if not prev_pt.exists():
            raise RuntimeError(f"Stage 3a weights not found at {prev_pt}. Run stage 3a first.")
        _, m = train_and_eval(prev_pt, R25_YAML, imgsz, run_dir)

    elif stage == "4b":
        prev_key = f"3b_{imgsz}"
        prev_pt  = OUT / prev_key / "train" / "weights" / "best.pt"
        if not prev_pt.exists():
            raise RuntimeError(f"Stage 3b weights not found at {prev_pt}. Run stage 3b first.")
        _, m = train_and_eval(prev_pt, R25_YAML, imgsz, run_dir)

    else:
        raise ValueError(f"Unknown stage: {stage}")

    summary[key] = m
    save_summary(summary)
    return m


def main():
    global DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGE_ORDER, default=None,
                        help="Run a single stage (default: all)")
    parser.add_argument("--imgsz", type=int, choices=IMAGE_SIZES, default=None,
                        help="Run a single resolution (default: both)")
    parser.add_argument("--device", type=int, default=DEVICE)
    args = parser.parse_args()

    DEVICE = args.device
    ensure_eval_yaml()

    stages = [args.stage] if args.stage else STAGE_ORDER
    imgszs = [args.imgsz] if args.imgsz else IMAGE_SIZES

    summary = load_summary()

    for stage in stages:
        for imgsz in imgszs:
            key = f"{stage}_{imgsz}"
            if key in summary:
                print(f"  [skip] stage {stage} @ {imgsz} already done")
                continue
            try:
                run_stage(stage, imgsz, summary)
            except (FileNotFoundError, RuntimeError) as e:
                print(f"  [SKIP] {e}")

    print_summary(summary)
    print(f"\nResults: {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
