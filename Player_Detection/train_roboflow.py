"""
Train all YOLO11 model sizes on the Roboflow Football-Presnap-Tracker-1 dataset.

On local GPU — train all sizes sequentially:
  python train_roboflow.py

On a cluster — submit one job per size:
  python train_roboflow.py --size n
  python train_roboflow.py --size s
  python train_roboflow.py --size m
  python train_roboflow.py --size l
  python train_roboflow.py --size x

Results saved to:
  roboflow_models/yolo11{size}/weights/best.pt
"""

import argparse
import json
from pathlib import Path
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
ROBOFLOW_YAML = Path(
    "/mnt/c/Users/Lukas/Documents/Semester_5/Seminar_Project"
    "/object_detection/datasets/Football-Presnap-Tracker-1"
    "/football_dataset/data.yaml"
)
OUT_DIR      = Path(__file__).parent / "roboflow_models"
MODEL_SIZES  = ["n", "s", "m", "l", "x"]

# Already trained — skip to avoid overwriting
ALREADY_DONE = {
    "n": Path(
        "/mnt/c/Users/Lukas/Documents/Semester_5/Seminar_Project"
        "/object_detection/football-models-yolo11/yolo11n-optimized/weights/best.pt"
    )
}

# Batch sizes for 24 GB VRAM at imgsz=640
BATCH = {"n": 16, "s": 16, "m": 12, "l": 8, "x": 6}

EPOCHS   = 150
IMGSZ    = 640
DEVICE   = 0
# ─────────────────────────────────────────────────────────────────────────────


def coco_weights(size: str) -> str:
    local = {
        "n": "/mnt/c/Users/Lukas/Documents/Semester_5/Seminar_Project/object_detection/yolo11n.pt",
        "m": "/mnt/c/Users/Lukas/Documents/Semester_5/Seminar_Project/object_detection/yolo11m.pt",
    }
    return local.get(size, f"yolo11{size}.pt")


def train_size(size: str):
    out_dir   = OUT_DIR / f"yolo11{size}"
    best_path = out_dir / "weights" / "best.pt"

    # Point to existing weights if already trained
    if size in ALREADY_DONE and ALREADY_DONE[size].exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        # Symlink or note the existing path
        info = {"size": size, "weights": str(ALREADY_DONE[size]), "skipped": True}
        (out_dir / "info.json").write_text(json.dumps(info, indent=2))
        print(f"  yolo11{size}: already trained → {ALREADY_DONE[size]}")
        return

    if best_path.exists():
        print(f"  yolo11{size}: already trained → {best_path}")
        return

    print(f"\n{'='*60}")
    print(f"  Training yolo11{size} on Roboflow dataset")
    print(f"  batch={BATCH[size]}  imgsz={IMGSZ}  epochs={EPOCHS}")
    print(f"{'='*60}")

    model = YOLO(coco_weights(size))
    model.train(
        data=str(ROBOFLOW_YAML),
        epochs=EPOCHS,
        patience=20,
        batch=BATCH[size],
        imgsz=IMGSZ,
        device=DEVICE,
        project=str(OUT_DIR),
        name=f"yolo11{size}",
        exist_ok=True,

        # Same hyperparameters as the original yolo11n training
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=True,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0001,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,
        erasing=0.4,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        save=True,
        save_period=10,
        verbose=True,
    )
    print(f"\n  ✓ yolo11{size} saved to {best_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=MODEL_SIZES, default=None,
                        help="Model size to train. Omit to train all sequentially.")
    parser.add_argument("--device", type=int, default=DEVICE,
                        help="GPU device id (default: 0)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sizes = [args.size] if args.size else MODEL_SIZES

    if len(sizes) > 1:
        print(f"Training {len(sizes)} model sizes sequentially on device {args.device}")
        print("Tip: on a cluster, submit one job per size with --size n/s/m/l/x\n")

    for size in sizes:
        train_size(size)

    print("\nAll done. Weights at:")
    for size in sizes:
        p = OUT_DIR / f"yolo11{size}" / "weights" / "best.pt"
        exists = "✓" if p.exists() else "— (check ALREADY_DONE)"
        print(f"  {exists}  yolo11{size}: {p}")


if __name__ == "__main__":
    main()
