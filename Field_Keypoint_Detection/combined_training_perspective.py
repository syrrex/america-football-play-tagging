"""
Combined training with perspective augmentation — trains on condition1 + ravelin2025.

Identical to combined_training.py but adds perspective=0.001 and degrees=5 to
simulate different camera angles across stadiums.

Usage:
  python combined_training_perspective.py
  python combined_training_perspective.py --device 1
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE  = Path(__file__).parent
KP_DS = Path(os.environ.get("KP_DATASETS", BASE.parent / "Data" / "Keypoints" / "datasets"))

OUT = BASE / "combined_perspective_results"
OUT.mkdir(parents=True, exist_ok=True)

FLIP_IDX = [
    12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
    25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13,
    34, 33, 32, 31, 30, 29, 28, 27, 26,
    43, 42, 41, 40, 39, 38, 37, 36, 35,
    52, 51, 50, 49, 48, 47, 46, 45, 44,
    61, 60, 59, 58, 57, 56, 55, 54, 53,
]

MODEL  = "yolo11l-pose.pt"
IMGSZ  = 1280
BATCH  = -1
EPOCHS = 150
DEVICE = 0


def build_combined_yaml() -> Path:
    flip_str = ", ".join(str(i) for i in FLIP_IDX)
    yaml_text = f"""path: {KP_DS}
train:
  - condition1/images/train
  - ravelin2025/images/train
val:
  - condition1/images/val
  - ravelin2025/images/val
test: test/images

kpt_shape: [62, 3]
flip_idx: [{flip_str}]

nc: 1
names:
  0: Field
"""
    out_yaml = OUT / "combined_data.yaml"
    out_yaml.write_text(yaml_text)
    return out_yaml


def train(device: int):
    yaml = build_combined_yaml()
    best_pt = OUT / "train" / "weights" / "best.pt"

    if best_pt.exists():
        print(f"[skip] Weights already exist at {best_pt}")
    else:
        model = YOLO(MODEL)
        model.train(
            data=str(yaml),
            epochs=EPOCHS,
            batch=BATCH,
            imgsz=IMGSZ,
            device=device,
            project=str(OUT),
            name="train",
            exist_ok=True,
            lr0=0.001, lrf=0.01, cos_lr=True,
            warmup_epochs=3, patience=20,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            translate=0.1, scale=0.5,
            fliplr=0.5, flipud=0.0, mosaic=1.0, close_mosaic=10,
            perspective=0.001, degrees=5.0,
            save=True, save_period=-1, verbose=True, workers=8,
        )

    return best_pt


def main():
    global DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=DEVICE)
    args = parser.parse_args()
    DEVICE = args.device

    print("=" * 60)
    print("COMBINED TRAINING + PERSPECTIVE AUG — condition1 + ravelin2025")
    print(f"Dataset root : {KP_DS}")
    print(f"Output       : {OUT}")
    print("=" * 60)

    best_pt = train(DEVICE)
    print(f"\nTraining complete. Best weights: {best_pt}")
    print(f"\nRun evaluation with:")
    print(f"  KP_DS={KP_DS} python evaluate_keypoints.py --weights {best_pt}")


if __name__ == "__main__":
    main()
