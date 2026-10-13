"""
Fine-tuning from cross-stadium weights on TB2026 quarter k-fold.

Starting from stage-3 weights (condition1 → ravelin2025 fine-tune),
trains on the 3 TB2026 training quarters and evaluates on the held-out quarter.

Comparison:
  quarter_kfold_training.py  — COCO pretrained  → train on 3 TB2026 quarters
  finetune_quarter_training  — stage-3 weights  → train on 3 TB2026 quarters  (this)

Same training data, different initialisation. Answers: does pre-training on
external stadium data give a better starting point for TB2026 fine-tuning?

Usage:
  python finetune_quarter_training.py
  python finetune_quarter_training.py --fold Q1
  TB2026_SRC=~/keypoint_dataset/tb2026_all python finetune_quarter_training.py
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

META_CSV = Path(os.environ.get(
    "KP_META",
    BASE.parent / "Data" / "metadata" / "tb2026_dataset_metadata.csv"
))

TB2026_SRC = Path(os.environ.get(
    "TB2026_SRC",
    BASE.parent / "Data" / "Keypoints" / "datasets" / "test"
)).expanduser()

# Stage-3 weights: condition1 → ravelin2025 fine-tune
# Override with PRETRAIN_WEIGHTS env var on the cluster
PRETRAIN_WEIGHTS = Path(os.environ.get(
    "PRETRAIN_WEIGHTS",
    BASE / "cross_stadium_results" / "3" / "train" / "weights" / "best.pt"
)).expanduser()

OUT = BASE / "finetune_quarter_results_1280"

QUARTERS = ["Q1", "Q2", "Q3", "Q4"]

FLIP_IDX = [
    12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
    25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13,
    34, 33, 32, 31, 30, 29, 28, 27, 26,
    43, 42, 41, 40, 39, 38, 37, 36, 35,
    52, 51, 50, 49, 48, 47, 46, 45, 44,
    61, 60, 59, 58, 57, 56, 55, 54, 53,
]

IMGSZ  = 1280
BATCH  = -1
EPOCHS = 150
DEVICE = 0

FRAME_RE = re.compile(r"frame_(\d+)\.")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_frame_to_quarter() -> dict[int, str]:
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


# ── Dataset build ──────────────────────────────────────────────────────────────

def build_fold(val_quarter: str, frame_to_q: dict, tmp: Path) -> Path:
    src_imgs = TB2026_SRC / "images"
    src_lbls = TB2026_SRC / "labels"

    all_imgs = sorted(src_imgs.glob("*.png"))
    if not all_imgs:
        raise FileNotFoundError(f"No images in {src_imgs}")

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
            p = perm_val_imgs / img.name
            if not p.exists():
                p.symlink_to(img.resolve())
            p = perm_val_lbls / lbl.name
            if not p.exists():
                p.symlink_to(lbl.resolve())
            n_val += 1
        else:
            n_train += 1

    print(f"  Dataset: {n_train} train / {n_val} val frames")

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
    return tmp / "data.yaml"


# ── Training ───────────────────────────────────────────────────────────────────

def train_fold(val_quarter: str, frame_to_q: dict) -> Path:
    out_dir = OUT / val_quarter
    best_pt = out_dir / "train" / "weights" / "best.pt"

    if best_pt.exists():
        print(f"  [skip] {val_quarter} — weights exist")
        return best_pt

    tmp = Path(tempfile.mkdtemp(prefix=f"ft_{val_quarter}_"))
    try:
        yaml = build_fold(val_quarter, frame_to_q, tmp)
        model = YOLO(str(PRETRAIN_WEIGHTS))
        model.train(
            data=str(yaml),
            epochs=EPOCHS,
            batch=BATCH,
            imgsz=IMGSZ,
            device=DEVICE,
            project=str(out_dir),
            name="train",
            exist_ok=True,
            lr0=0.0001, lrf=0.01, cos_lr=True,   # lower lr for fine-tuning
            warmup_epochs=3, patience=20,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            translate=0.1, scale=0.5,
            fliplr=0.5, flipud=0.0, mosaic=1.0, close_mosaic=10,
            save=True, save_period=-1, verbose=True, workers=8,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return best_pt


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global DEVICE, IMGSZ, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",   choices=QUARTERS, default=None)
    parser.add_argument("--device", type=int, default=DEVICE)
    parser.add_argument("--imgsz",  type=int, default=IMGSZ)
    args = parser.parse_args()
    DEVICE = args.device
    IMGSZ  = args.imgsz
    OUT = BASE / f"finetune_quarter_results_{IMGSZ}"
    OUT.mkdir(parents=True, exist_ok=True)

    folds = [args.fold] if args.fold else QUARTERS

    if not PRETRAIN_WEIGHTS.exists():
        raise FileNotFoundError(f"Stage-3 weights not found: {PRETRAIN_WEIGHTS}")

    print("=" * 60)
    print(f"FINE-TUNE QUARTER K-FOLD — {IMGSZ}px")
    print(f"Starting weights : {PRETRAIN_WEIGHTS}")
    print(f"TB2026           : {TB2026_SRC}")
    print(f"Metadata         : {META_CSV}")
    print(f"Output           : {OUT}")
    print(f"Folds            : {folds}")
    print("=" * 60)

    frame_to_q = load_frame_to_quarter()

    for q in folds:
        print(f"\n{'─'*60}")
        print(f"  Fold {q}  (val = {q})")
        print(f"{'─'*60}")
        train_fold(q, frame_to_q)

    print("\nAll folds done.")


if __name__ == "__main__":
    main()
