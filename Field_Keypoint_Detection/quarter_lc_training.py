"""
Quarter-based learning curve for TB2026 keypoints.

Training levels are defined in half-quarter increments, where "half a quarter"
means the first half of CLIPS (by play_num) in that quarter — never splitting
a clip mid-way to avoid frame leakage.

  Level  |  Training set
  -------+---------------------------------------------------------
  0.5Q   |  first half of Q_earliest
  1.0Q   |  all of Q_earliest
  1.5Q   |  all of Q_earliest  + first half of Q_middle
  2.0Q   |  all of Q_earliest  + all of Q_middle
  2.5Q   |  all of Q_earliest  + all of Q_middle  + first half of Q_latest
  3.0Q   |  all of Q_earliest  + all of Q_middle  + all of Q_latest  (full fold)

Quarter order within each fold is chronological (Q1 < Q2 < Q3 < Q4).
Val set is always the single held-out quarter, same across all levels.

Usage:
  python quarter_lc_training.py
  python quarter_lc_training.py --fold Q1
  python quarter_lc_training.py --fold Q2 --level 1.5Q
  TB2026_SRC=~/keypoint_dataset/tb2026_all python quarter_lc_training.py
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

QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
LEVELS   = ["0.5Q", "1.0Q", "1.5Q", "2.0Q", "2.5Q", "3.0Q"]

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

FRAME_RE = re.compile(r"frame_(\d+)\.")

OUT = BASE / "quarter_lc_results_1280"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_clip_info() -> dict[str, list[tuple[int, int, int]]]:
    """
    Returns {quarter: [(play_num, start_frame, end_frame), ...]} sorted by play_num.
    Half-quarter splits use the first half of this list (clip-level boundary).
    """
    meta = pd.read_csv(META_CSV)
    info: dict[str, list] = {}
    for q in QUARTERS:
        qdf = meta[meta["QTR"] == q].sort_values("play_num")
        info[q] = [
            (int(row["play_num"]), int(row["start_frame"]), int(row["end_frame"]))
            for _, row in qdf.iterrows()
        ]
    return info


def get_train_frames(val_quarter: str, level: str,
                     clip_info: dict) -> set[int]:
    """
    Returns the set of frame indices to use for training at a given level.
    Builds up from the earliest available training quarter chronologically.
    Half-quarter = first half of clips by play_num (no frame splitting).
    """
    train_qs = sorted([q for q in QUARTERS if q != val_quarter])  # Q1 < Q2 < Q3 < Q4
    level_map = {
        "0.5Q": [(train_qs[0], "half")],
        "1.0Q": [(train_qs[0], "full")],
        "1.5Q": [(train_qs[0], "full"), (train_qs[1], "half")],
        "2.0Q": [(train_qs[0], "full"), (train_qs[1], "full")],
        "2.5Q": [(train_qs[0], "full"), (train_qs[1], "full"), (train_qs[2], "half")],
        "3.0Q": [(train_qs[0], "full"), (train_qs[1], "full"), (train_qs[2], "full")],
    }
    frames: set[int] = set()
    for q, mode in level_map[level]:
        clips = clip_info[q]
        selected = clips[:len(clips) // 2] if mode == "half" else clips
        for _, start, end in selected:
            frames.update(range(start, end + 1))
    return frames


def frame_number(path: Path) -> int:
    m = FRAME_RE.search(path.name)
    return int(m.group(1)) if m else -1


# ── Dataset & training ─────────────────────────────────────────────────────────

def build_val_dir(val_quarter: str, src_imgs: Path, src_lbls: Path,
                  val_frames: set[int]):
    """Write permanent val_images / val_labels for this fold (once per fold)."""
    perm_imgs = OUT / val_quarter / "val_images"
    perm_lbls = OUT / val_quarter / "val_labels"
    if perm_imgs.exists() and any(perm_imgs.iterdir()):
        return  # already built
    perm_imgs.mkdir(parents=True, exist_ok=True)
    perm_lbls.mkdir(parents=True, exist_ok=True)
    for img in src_imgs.glob("*.png"):
        if frame_number(img) in val_frames:
            lbl = src_lbls / img.with_suffix(".txt").name
            if not lbl.exists():
                continue
            p = perm_imgs / img.name
            if not p.exists():
                p.symlink_to(img.resolve())
            p = perm_lbls / lbl.name
            if not p.exists():
                p.symlink_to(lbl.resolve())


def build_and_train(val_quarter: str, level: str, train_frames: set[int],
                    val_frames: set[int]):
    out_dir  = OUT / val_quarter / level
    best_pt  = out_dir / "train" / "weights" / "best.pt"
    if best_pt.exists():
        print(f"  [skip] {val_quarter}/{level} — weights exist")
        return best_pt

    src_imgs = TB2026_SRC / "images"
    src_lbls = TB2026_SRC / "labels"
    all_imgs = sorted(src_imgs.glob("*.png"))
    if not all_imgs:
        raise FileNotFoundError(f"No images in {src_imgs}")

    tmp = Path(tempfile.mkdtemp(prefix=f"qlc_{val_quarter}_{level}_"))
    try:
        for split in ("train", "val"):
            (tmp / "images" / split).mkdir(parents=True, exist_ok=True)
            (tmp / "labels" / split).mkdir(parents=True, exist_ok=True)

        n_train = n_val = 0
        for img in all_imgs:
            fidx = frame_number(img)
            lbl  = src_lbls / img.with_suffix(".txt").name
            if not lbl.exists():
                continue
            if fidx in train_frames:
                split = "train"
            elif fidx in val_frames:
                split = "val"
            else:
                continue
            (tmp / "images" / split / img.name).symlink_to(img)
            (tmp / "labels" / split / lbl.name).symlink_to(lbl)
            if split == "train":
                n_train += 1
            else:
                n_val += 1

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
        model = YOLO(MODEL)
        model.train(
            data=str(tmp / "data.yaml"),
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global DEVICE, IMGSZ, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",   choices=QUARTERS, default=None)
    parser.add_argument("--level",  choices=LEVELS,   default=None)
    parser.add_argument("--device", type=int, default=DEVICE)
    parser.add_argument("--imgsz",  type=int, default=IMGSZ)
    args = parser.parse_args()
    DEVICE = args.device
    IMGSZ  = args.imgsz
    OUT = BASE / f"quarter_lc_results_{IMGSZ}"
    OUT.mkdir(parents=True, exist_ok=True)

    folds  = [args.fold]  if args.fold  else QUARTERS
    levels = [args.level] if args.level else LEVELS

    print("=" * 60)
    print(f"QUARTER LEARNING CURVE — {IMGSZ}px")
    print(f"Source   : {TB2026_SRC}")
    print(f"Metadata : {META_CSV}")
    print(f"Output   : {OUT}")
    print(f"Folds    : {folds}")
    print(f"Levels   : {levels}")
    print("=" * 60)

    clip_info = load_clip_info()
    src_imgs  = TB2026_SRC / "images"
    src_lbls  = TB2026_SRC / "labels"

    for val_q in folds:
        val_clips  = clip_info[val_q]
        val_frames = {f for _, s, e in val_clips for f in range(s, e + 1)}

        # Build permanent val dir once per fold
        build_val_dir(val_q, src_imgs, src_lbls, val_frames)

        for level in levels:
            train_frames = get_train_frames(val_q, level, clip_info)
            print(f"\n{'─'*60}")
            print(f"  Fold {val_q} | Level {level}  "
                  f"({len(train_frames)} train / {len(val_frames)} val frames)")
            print(f"{'─'*60}")
            build_and_train(val_q, level, train_frames, val_frames)

    print("\nAll runs done.")


if __name__ == "__main__":
    main()