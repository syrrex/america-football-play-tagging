"""
Hash-only training — trains on condition1 + ravelin2025 but supervises
only the 18 hash mark keypoints (kpt_id 45–62, 0-based indices 44–61).

All other keypoints have visibility=0 in the modified labels so YOLO
ignores them during loss computation. Architecture stays identical
(kpt_shape=[62, 3]) — no re-annotation needed.

Usage:
  python hash_only_training.py
  python hash_only_training.py --device 1
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE  = Path(__file__).parent
KP_DS = Path(os.environ.get("KP_DATASETS", BASE.parent / "Data" / "Keypoints" / "datasets"))

OUT = BASE / "hash_only_results"
OUT.mkdir(parents=True, exist_ok=True)

# Hash keypoints: kpt_id 45–62 → 0-based indices 44–61
HASH_INDICES = set(range(44, 62))

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


def mask_labels(src_dir: Path, dst_dir: Path):
    """Copy label files with non-hash keypoints zeroed out."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for lbl in src_dir.glob("*.txt"):
        parts = lbl.read_text().strip().split()
        new_parts = parts[:5]  # class cx cy bw bh
        for i in range(62):
            x   = parts[5 + i * 3]
            y   = parts[5 + i * 3 + 1]
            vis = parts[5 + i * 3 + 2]
            if i in HASH_INDICES:
                new_parts.extend([x, y, vis])
            else:
                new_parts.extend(["0.0", "0.0", "0"])
        (dst_dir / lbl.name).write_text(" ".join(new_parts) + "\n")


def build_masked_dataset(tmp: Path) -> Path:
    """Build a temporary dataset with hash-only labels, symlink images."""
    print("Building hash-only label set...")
    for ds in ("condition1", "ravelin2025"):
        for split in ("train", "val"):
            src_img = KP_DS / ds / "images" / split
            src_lbl = KP_DS / ds / "labels" / split
            dst_img = tmp / ds / "images" / split
            dst_lbl = tmp / ds / "labels" / split

            # Symlink images (no copying)
            dst_img.mkdir(parents=True, exist_ok=True)
            for img in src_img.glob("*.png"):
                link = dst_img / img.name
                if not link.exists():
                    link.symlink_to(img)

            mask_labels(src_lbl, dst_lbl)

    # Symlink test images
    dst_test = tmp / "test" / "images"
    dst_test.mkdir(parents=True, exist_ok=True)
    for img in (KP_DS / "test" / "images").glob("*.png"):
        link = dst_test / img.name
        if not link.exists():
            link.symlink_to(img)

    # Write data.yaml
    flip_str = ", ".join(str(i) for i in FLIP_IDX)
    yaml_text = f"""path: {tmp}
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
    yaml_path = tmp / "data.yaml"
    yaml_path.write_text(yaml_text)
    print(f"Dataset ready at {tmp}")
    return yaml_path


def train(device: int):
    best_pt = OUT / "train" / "weights" / "best.pt"

    if best_pt.exists():
        print(f"[skip] Weights already exist at {best_pt}")
        return best_pt

    tmp = Path(tempfile.mkdtemp(prefix="hash_only_ds_"))
    try:
        yaml = build_masked_dataset(tmp)
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
            save=True, save_period=-1, verbose=True, workers=8,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return best_pt


def main():
    global DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=DEVICE)
    args = parser.parse_args()
    DEVICE = args.device

    print("=" * 60)
    print("HASH-ONLY TRAINING — kpt 45–62 supervised, rest masked")
    print(f"Dataset root : {KP_DS}")
    print(f"Output       : {OUT}")
    print("=" * 60)

    best_pt = train(DEVICE)
    print(f"\nTraining complete. Best weights: {best_pt}")


if __name__ == "__main__":
    main()
