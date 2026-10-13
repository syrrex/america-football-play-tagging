"""
Quarter-based 4-fold cross-validation on TB2026 keypoints.

Each game quarter is one held-out val fold; the model trains on the other 3.
Quarter assignments come from the QTR column in tb2026_dataset_metadata.csv.

  Q1 : clips  7– 39  →  744 frames
  Q2 : clips 40– 76  →  652 frames
  Q3 : clips 77–105  →  748 frames
  Q4 : clips107–144  →  776 frames

Uses 640 px to keep training time reasonable.

After training, run evaluate_keypoints.py with --img-dir / --lbl-dir pointing
to each fold's val images/labels to get per-fold kpt_eval.json.

Usage:
  python quarter_kfold_training.py
  python quarter_kfold_training.py --fold Q1
  python quarter_kfold_training.py --device 1
  TB2026_SRC=~/keypoint_dataset/tb2026_all python quarter_kfold_training.py
"""

import argparse
import os
import re
import shutil
import tempfile
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent

# Override with KP_META env var on the cluster
META_CSV = Path(os.environ.get(
    "KP_META",
    BASE.parent / "Data" / "metadata" / "tb2026_dataset_metadata.csv"
))

# Directory with all TB2026 frames:
#   <TB2026_SRC>/images/frame_XXXXXX.png
#   <TB2026_SRC>/labels/frame_XXXXXX.txt
# On cluster: ~/keypoint_dataset/tb2026_all  (built by merge_tb2026_folds.py)
# Locally   : .../Data/Keypoints/datasets/test
TB2026_SRC = Path(os.environ.get(
    "TB2026_SRC",
    BASE.parent / "Data" / "Keypoints" / "datasets" / "test"
)).expanduser()

OUT = BASE / "quarter_kfold_results"
OUT.mkdir(parents=True, exist_ok=True)

QUARTERS = ["Q1", "Q2", "Q3", "Q4"]

FLIP_IDX = [
    12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
    25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13,
    34, 33, 32, 31, 30, 29, 28, 27, 26,
    43, 42, 41, 40, 39, 38, 37, 36, 35,
    52, 51, 50, 49, 48, 47, 46, 45, 44,
    61, 60, 59, 58, 57, 56, 55, 54, 53,
]

MODEL  = "yolo11l-pose.pt"
IMGSZ  = 640   # overridden by --imgsz
BATCH  = -1
EPOCHS = 150
DEVICE = 0

FRAME_RE = re.compile(r"frame_(\d+)\.")


def load_frame_to_quarter() -> dict[int, str]:
    """Map every frame index → quarter label using the metadata CSV."""
    meta = pd.read_csv(META_CSV)
    mapping = {}
    for _, row in meta.iterrows():
        q = row["QTR"]
        for f in range(int(row["start_frame"]), int(row["end_frame"]) + 1):
            mapping[f] = q
    return mapping


def frame_number(path: Path) -> int:
    m = FRAME_RE.search(path.name)
    return int(m.group(1)) if m else -1


def build_fold(val_quarter: str, frame_to_q: dict, tmp: Path) -> Path:
    src_imgs = TB2026_SRC / "images"
    src_lbls = TB2026_SRC / "labels"

    all_imgs = sorted(src_imgs.glob("*.png"))
    if not all_imgs:
        raise FileNotFoundError(f"No images found in {src_imgs}")

    # Permanent val dir used later by evaluate_keypoints.py
    perm_val_imgs = OUT / val_quarter / "val_images"
    perm_val_lbls = OUT / val_quarter / "val_labels"
    perm_val_imgs.mkdir(parents=True, exist_ok=True)
    perm_val_lbls.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        (tmp / "images" / split).mkdir(parents=True, exist_ok=True)
        (tmp / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_train = n_val = 0
    for img in all_imgs:
        fidx = frame_number(img)
        lbl  = src_lbls / img.with_suffix(".txt").name
        if not lbl.exists():
            continue
        q = frame_to_q.get(fidx)
        if q is None:
            continue
        split = "val" if q == val_quarter else "train"
        (tmp / "images" / split / img.name).symlink_to(img)
        (tmp / "labels" / split / lbl.name).symlink_to(lbl)
        if split == "val":
            # Also write to permanent val dir for evaluation
            p = perm_val_imgs / img.name
            if not p.exists():
                p.symlink_to(img.resolve())
            p = perm_val_lbls / lbl.name
            if not p.exists():
                p.symlink_to(lbl.resolve())
            n_val += 1
        else:
            n_train += 1

    flip_str = ", ".join(str(i) for i in FLIP_IDX)
    (tmp / "data.yaml").write_text(f"""path: {tmp}
train: images/train
val:   images/val

kpt_shape: [62, 3]
flip_idx: [{flip_str}]

nc: 1
names:
  0: Field
""")
    print(f"  Dataset: {n_train} train / {n_val} val frames")
    return tmp / "data.yaml"


def train_fold(val_quarter: str, frame_to_q: dict) -> Path:
    out_dir = OUT / val_quarter
    best_pt = out_dir / "train" / "weights" / "best.pt"

    if best_pt.exists():
        print(f"  [skip] weights exist: {best_pt}")
        return best_pt

    tmp = Path(tempfile.mkdtemp(prefix=f"qkfold_{val_quarter}_"))
    try:
        yaml = build_fold(val_quarter, frame_to_q, tmp)
        model = YOLO(MODEL)
        model.train(
            data=str(yaml),
            epochs=EPOCHS,
            batch=BATCH,
            imgsz=IMGSZ,
            device=DEVICE,
            project=str(out_dir),
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
    global DEVICE, IMGSZ, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",  choices=QUARTERS, default=None,
                        help="Run a single fold (default: all)")
    parser.add_argument("--device", type=int, default=DEVICE)
    parser.add_argument("--imgsz",  type=int, default=IMGSZ,
                        help="Training resolution (default: 640)")
    args = parser.parse_args()
    DEVICE = args.device
    IMGSZ  = args.imgsz
    # Keep results for different resolutions separate
    OUT = BASE / f"quarter_kfold_results_{IMGSZ}"
    OUT.mkdir(parents=True, exist_ok=True)

    folds = [args.fold] if args.fold else QUARTERS

    print("=" * 60)
    print("QUARTER K-FOLD — TB2026 keypoints, 640 px")
    print(f"Source   : {TB2026_SRC}")
    print(f"Metadata : {META_CSV}")
    print(f"Output   : {OUT}")
    print(f"Folds    : {folds}")
    print("=" * 60)

    frame_to_q = load_frame_to_quarter()

    for q in folds:
        print(f"\n{'─'*60}")
        print(f"  Fold {q}  (val = {q} quarter)")
        print(f"{'─'*60}")
        best_pt = train_fold(q, frame_to_q)
        print(f"  Best weights: {best_pt}")

    print("\nAll folds done.")
    print("\nEvaluate each fold on its val quarter:")
    for q in folds:
        pt  = OUT / q / "train" / "weights" / "best.pt"
        print(f"  python evaluate_keypoints.py --weights {pt} --out {OUT}/{q}/kpt_eval.json")


if __name__ == "__main__":
    main()