"""
Unified experiment grid: starting point × model size × image size × (baseline / finetuned)

Starting points:
  coco      — COCO pretrained weights (yolo11{size}.pt)
  roboflow  — Roboflow football-finetuned weights (train_roboflow.py must run first)

Models  : yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
Img sizes: 640, 1280
Per cell : baseline eval + K-fold finetune on Ravelin data

Run a specific starting point:
  python experiment_grid.py --start coco
  python experiment_grid.py --start roboflow
  python experiment_grid.py              # runs both sequentially

Results are saved incrementally — already-finished experiments are skipped on re-run.

Output layout:
  grid_results/
    coco_yolo11n_640/
    coco_yolo11n_1280/
    …
    roboflow_yolo11n_640/
    …
    grid_summary.json
    grid_summary.csv
"""

import argparse
import shutil
import json
import time
import csv
import numpy as np
from pathlib import Path

from sklearn.model_selection import KFold
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent
DATASET     = BASE / "dataset"
OUTPUT_DIR  = BASE / "grid_results"

CLASS_NAMES = ["defense", "oline", "qb", "ref", "skill"]
K           = 5
EPOCHS      = 50
DEVICE      = 0

MODEL_SIZES    = ["n", "s", "m", "l", "x"]
STARTING_POINTS = ["coco", "roboflow"]

# Batch sizes — reduced for larger models / higher resolution to fit 24 GB VRAM
BATCH_TABLE = {
    ("n", 640): 16, ("n", 1280): 4,
    ("s", 640): 16, ("s", 1280): 4,
    ("m", 640): 12, ("m", 1280): 3,
    ("l", 640): 8,  ("l", 1280): 1,
    ("x", 640): 6,  ("x", 1280): 1,
}

IMAGE_SIZES = [640, 1280]
# ─────────────────────────────────────────────────────────────────────────────


def weights_path(size: str, starting_point: str = "coco") -> str:
    """Return the correct starting weights for a given size and starting point."""
    if starting_point == "roboflow":
        # yolo11n already trained separately; others trained by train_roboflow.py
        roboflow_paths = {
            "n": Path("/mnt/c/Users/Lukas/Documents/Semester_5/Seminar_Project"
                      "/object_detection/football-models-yolo11/yolo11n-optimized/weights/best.pt"),
        }
        for s in ["s", "m", "l", "x"]:
            p = BASE / "roboflow_models" / f"yolo11{s}" / "weights" / "best.pt"
            if p.exists():
                roboflow_paths[s] = p

        if size not in roboflow_paths or not Path(roboflow_paths[size]).exists():
            raise FileNotFoundError(
                f"Roboflow weights for yolo11{size} not found. "
                f"Run train_roboflow.py --size {size} first."
            )
        return str(roboflow_paths[size])

    # COCO pretrained
    coco_local = {
        "n": "/mnt/c/Users/Lukas/Documents/Semester_5/Seminar_Project/object_detection/yolo11n.pt",
        "m": "/mnt/c/Users/Lukas/Documents/Semester_5/Seminar_Project/object_detection/yolo11m.pt",
    }
    return coco_local.get(size, f"yolo11{size}.pt")


def extract_metrics(results, elapsed_sec: float, n_images: int) -> dict:
    """Pull overall + per-class metrics and speed from a YOLO val results object."""
    overall = {
        "mAP50":     round(float(results.box.map50),      4),
        "mAP50-95":  round(float(results.box.map),        4),
        "precision": round(float(np.mean(results.box.p)), 4),
        "recall":    round(float(np.mean(results.box.r)), 4),
        "ms_per_img": round(elapsed_sec / n_images * 1000, 2),
        "fps":        round(n_images / elapsed_sec, 1),
    }
    per_class = {
        name: {
            "mAP50":     round(float(results.box.ap50[i]), 4),
            "mAP50-95":  round(float(results.box.ap[i]),   4),
            "precision": round(float(results.box.p[i]),    4),
            "recall":    round(float(results.box.r[i]),    4),
        }
        for i, name in enumerate(CLASS_NAMES)
    }
    return {"overall": overall, "per_class": per_class}


def make_fold_dirs(fold_dir: Path, train_imgs: list, val_imgs: list) -> Path:
    for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
        img_out = fold_dir / "images" / split
        lbl_out = fold_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy(img, img_out / img.name)
            lbl = DATASET / "labels" / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy(lbl, lbl_out / lbl.name)
    data_yaml = fold_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {fold_dir}\ntrain: images/train\nval: images/val\n"
        f"\nnc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n"
    )
    return data_yaml


def run_baseline(size: str, imgsz: int, batch: int,
                 all_images: list, run_dir: Path,
                 starting_point: str = "coco") -> dict:
    cache = run_dir / "baseline_metrics.json"
    if cache.exists():
        print(f"  [cache] baseline loaded")
        return json.loads(cache.read_text())

    base_dir = run_dir / "baseline"
    img_dir  = base_dir / "images" / "val"
    lbl_dir  = base_dir / "labels" / "val"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for img in all_images:
        shutil.copy(img, img_dir / img.name)
        lbl = DATASET / "labels" / (img.stem + ".txt")
        if lbl.exists():
            shutil.copy(lbl, lbl_dir / lbl.name)

    data_yaml = base_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {base_dir}\ntrain: images/val\nval: images/val\n"
        f"\nnc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n"
    )

    model = YOLO(weights_path(size, starting_point))
    t0 = time.perf_counter()
    results = model.val(
        data=str(data_yaml), imgsz=imgsz, batch=batch,
        device=DEVICE, split="val",
        project=str(base_dir), name="eval", exist_ok=True,
        half=(imgsz == 1280),
    )
    elapsed = time.perf_counter() - t0

    metrics = extract_metrics(results, elapsed, len(all_images))
    cache.write_text(json.dumps(metrics, indent=2))
    return metrics


def run_kfold(size: str, imgsz: int, batch: int,
              all_images: list, run_dir: Path,
              starting_point: str = "coco") -> list[dict]:
    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_images)):
        fold_num = fold_idx + 1
        cache    = run_dir / f"fold_{fold_num}_metrics.json"

        if cache.exists():
            print(f"  [cache] fold {fold_num} loaded")
            fold_metrics.append(json.loads(cache.read_text()))
            continue

        train_imgs = [all_images[i] for i in train_idx]
        val_imgs   = [all_images[i] for i in val_idx]
        print(f"  Fold {fold_num}/{K}: train={len(train_imgs)}  val={len(val_imgs)}")

        fold_dir  = run_dir / f"fold_{fold_num}"
        data_yaml = make_fold_dirs(fold_dir, train_imgs, val_imgs)

        model = YOLO(weights_path(size, starting_point))
        model.train(
            data=str(data_yaml), epochs=EPOCHS, batch=batch,
            imgsz=imgsz, device=DEVICE,
            project=str(run_dir), name=f"fold_{fold_num}", exist_ok=True,
            lr0=0.001, lrf=0.01, cos_lr=True,
            warmup_epochs=3, patience=15,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            degrees=5.0, translate=0.1, scale=0.5,
            fliplr=0.5, flipud=0.0, mosaic=1.0, close_mosaic=10,
            save=True, save_period=-1, verbose=False,
        )

        best = run_dir / f"fold_{fold_num}" / "weights" / "best.pt"
        finetuned = YOLO(str(best))
        t0 = time.perf_counter()
        val_results = finetuned.val(
            data=str(data_yaml), imgsz=imgsz, batch=batch,
            device=DEVICE, split="val", verbose=False,
            half=(imgsz == 1280),
        )
        elapsed = time.perf_counter() - t0

        metrics = {"fold": fold_num,
                   **extract_metrics(val_results, elapsed, len(val_imgs))}
        cache.write_text(json.dumps(metrics, indent=2))
        fold_metrics.append(metrics)
        print(f"    mAP50={metrics['overall']['mAP50']:.4f}  "
              f"mAP50-95={metrics['overall']['mAP50-95']:.4f}")

    return fold_metrics


def avg_metrics(fold_metrics: list[dict]) -> dict:
    """Average overall + per-class metrics across folds."""
    keys = ["mAP50", "mAP50-95", "precision", "recall", "ms_per_img", "fps"]
    overall_avg = {
        k: round(sum(m["overall"][k] for m in fold_metrics) / K, 4)
        for k in keys
    }
    per_class_avg = {
        cls: {
            k: round(sum(m["per_class"][cls][k] for m in fold_metrics) / K, 4)
            for k in ["mAP50", "mAP50-95", "precision", "recall"]
        }
        for cls in CLASS_NAMES
    }
    return {"overall": overall_avg, "per_class": per_class_avg}


def save_summary(all_results: dict):
    # JSON
    json_path = OUTPUT_DIR / "grid_summary.json"
    json_path.write_text(json.dumps(all_results, indent=2))

    # CSV — one row per (model, imgsz, mode)
    csv_path = OUTPUT_DIR / "grid_summary.csv"
    fields = (["start", "model", "imgsz", "mode"] +
              ["mAP50", "mAP50-95", "precision", "recall", "ms_per_img", "fps"] +
              [f"{cls}_{m}" for cls in CLASS_NAMES
               for m in ["mAP50", "mAP50-95", "precision", "recall"]])

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key, entry in all_results.items():
            start, model, imgsz = key.split("_")
            for mode in ["baseline", "finetuned_avg"]:
                m = entry[mode]
                row = {"start": start, "model": model, "imgsz": imgsz, "mode": mode,
                       **m["overall"]}
                for cls in CLASS_NAMES:
                    for metric in ["mAP50", "mAP50-95", "precision", "recall"]:
                        row[f"{cls}_{metric}"] = m["per_class"][cls][metric]
                writer.writerow(row)

    print(f"\nSummary saved to:\n  {json_path}\n  {csv_path}")


def print_grid(all_results: dict):
    print("\n" + "=" * 80)
    print("EXPERIMENT GRID SUMMARY — mAP50 (baseline → finetuned)")
    print("=" * 80)
    header = f"{'Start':>10} {'Model':<10} {'imgsz':>6}  {'Baseline':>10}  {'Finetuned':>10}  {'Δ mAP50':>8}  {'FPS':>6}"
    print(header)
    print("-" * 80)
    for key, entry in sorted(all_results.items()):
        start, model, imgsz = key.split("_")
        b   = entry["baseline"]["overall"]["mAP50"]
        f   = entry["finetuned_avg"]["overall"]["mAP50"]
        fps = entry["finetuned_avg"]["overall"]["fps"]
        print(f"{start:>10} {model:<10} {imgsz:>6}  {b:>10.4f}  {f:>10.4f}  {f-b:>+8.4f}  {fps:>6.1f}")
    print("=" * 80)


def main():
    global DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", choices=["coco", "roboflow", "both"],
                        default="both",
                        help="Which starting weights to use (default: both)")
    parser.add_argument("--size", choices=MODEL_SIZES, default=None,
                        help="Run only one model size (useful for cluster jobs)")
    parser.add_argument("--imgsz", type=int, choices=IMAGE_SIZES, default=None,
                        help="Run only one image size")
    parser.add_argument("--device", type=int, default=DEVICE)
    args = parser.parse_args()

    DEVICE = args.device

    if not DATASET.exists() or not any((DATASET / "images").glob("*.jpg")):
        print("Dataset not found — running prepare_dataset.py first...")
        from prepare_dataset import prepare
        prepare()

    all_images = sorted((DATASET / "images").glob("*.jpg"))
    print(f"Dataset: {len(all_images)} images")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    starts  = ["coco", "roboflow"] if args.start == "both" else [args.start]
    sizes   = [args.size]  if args.size  else MODEL_SIZES
    imgszs  = [args.imgsz] if args.imgsz else IMAGE_SIZES

    # Load existing results
    summary_path = OUTPUT_DIR / "grid_summary.json"
    all_results  = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    for start in starts:
        for size in sizes:
            for imgsz in imgszs:
                key     = f"{start}_yolo11{size}_{imgsz}"
                run_dir = OUTPUT_DIR / key
                run_dir.mkdir(parents=True, exist_ok=True)
                batch   = BATCH_TABLE[(size, imgsz)]

                print(f"\n{'='*70}")
                print(f"  [{start}]  yolo11{size}  |  imgsz={imgsz}  |  batch={batch}")
                print(f"{'='*70}")

                try:
                    # Baseline
                    print("\n  [BASELINE]")
                    baseline = run_baseline(size, imgsz, batch, all_images,
                                            run_dir, starting_point=start)

                    # K-Fold finetune
                    print("\n  [K-FOLD FINETUNE]")
                    fold_metrics  = run_kfold(size, imgsz, batch, all_images,
                                              run_dir, starting_point=start)
                    finetuned_avg = avg_metrics(fold_metrics)

                except FileNotFoundError as e:
                    print(f"  [SKIP] {e}")
                    continue

                all_results[key] = {
                    "starting_point": start,
                    "baseline":       baseline,
                    "folds":          fold_metrics,
                    "finetuned_avg":  finetuned_avg,
            }

                # Save incrementally after each combo
                save_summary(all_results)

                b = baseline["overall"]["mAP50"]
                f = finetuned_avg["overall"]["mAP50"]
                print(f"\n  [{start}] yolo11{size} @ {imgsz}: {b:.4f} → {f:.4f}  (Δ={f-b:+.4f})")

    print_grid(all_results)
    print("\nDone.")


if __name__ == "__main__":
    main()
