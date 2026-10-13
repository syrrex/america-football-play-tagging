"""
Builds TB2026 keypoint dataset splits for:
  1. Upper bound  — 5-fold cross-validation (fold_0 … fold_4)
  2. Learning curve — fixed 80/20 clip split at 6 training fractions

Splitting is done at clip level to avoid temporal leakage — consecutive
frames within a clip are nearly identical, so frame-level random splits
would inflate val performance.

Clip boundaries are read from Data/metadata/tb2026_dataset_metadata.csv.

Source: Data/Keypoints/datasets/test/  (2920 frames, labels already fixed)

Output structure:
  Data/Keypoints/datasets/tb2026/
    fold_0/           train: 86 clips (~2336 frames), val: 21 clips (~584 frames)
    fold_1/ … fold_4/
    learning_curve/
      frac_005/       ~5%  of train clips
      frac_010/
      frac_025/
      frac_050/
      frac_075/
      frac_100/       all train clips (same as LC baseline)
"""

import csv
import os
import random
import shutil
from pathlib import Path

PROJECT  = Path(__file__).parent.parent
DATA     = PROJECT / "Data"

# Override KP_DS with env var when running on the cluster
KP_DS    = Path(os.environ.get("KP_DS", DATA / "Keypoints" / "datasets"))
META_CSV = Path(os.environ.get("KP_META", DATA / "metadata" / "tb2026_dataset_metadata.csv"))

SRC_IMG  = KP_DS / "test" / "images"
SRC_LBL  = KP_DS / "test" / "labels"
OUT      = KP_DS / "tb2026"

KPT_SHAPE = "[62, 3]"
FLIP_IDX  = (
    "[12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,\n"
    " 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13,\n"
    " 34, 33, 32, 31, 30, 29, 28, 27, 26,\n"
    " 43, 42, 41, 40, 39, 38, 37, 36, 35,\n"
    " 52, 51, 50, 49, 48, 47, 46, 45, 44,\n"
    " 61, 60, 59, 58, 57, 56, 55, 54, 53]"
)

SEED     = 42
N_FOLDS  = 5
LC_FRACS = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]


# ── Load clip metadata ────────────────────────────────────────────────────────

def load_clips() -> list[dict]:
    """Return list of {clip, start_frame, end_frame, n_frames}."""
    clips = []
    with open(META_CSV, newline="") as f:
        for row in csv.DictReader(f):
            clips.append({
                "clip":        row["clip"],
                "start_frame": int(row["start_frame"]),
                "end_frame":   int(row["end_frame"]),
                "n_frames":    int(row["n_frames"]),
            })
    return clips


def stems_for_clips(clips: list[dict]) -> list[str]:
    stems = []
    for c in clips:
        for i in range(c["start_frame"], c["end_frame"] + 1):
            stems.append(f"frame_{i:06d}")
    return stems


# ── Copy helpers ──────────────────────────────────────────────────────────────

def copy_stems(stems: list[str], dst_img: Path, dst_lbl: Path):
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        src_i = SRC_IMG / f"{stem}.png"
        src_l = SRC_LBL / f"{stem}.txt"
        if src_i.exists():
            shutil.copy2(src_i, dst_img / f"{stem}.png")
        else:
            print(f"  [WARN] missing image: {stem}.png")
        if src_l.exists():
            shutil.copy2(src_l, dst_lbl / f"{stem}.txt")
        else:
            print(f"  [WARN] missing label: {stem}.txt")


def write_yaml(dataset_dir: Path):
    (dataset_dir / "data.yaml").write_text(
        f"path: {dataset_dir}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  ../../test/images\n\n"
        f"kpt_shape: {KPT_SHAPE}\n"
        f"flip_idx: {FLIP_IDX}\n\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: Field\n"
    )


# ── 5-fold CV ─────────────────────────────────────────────────────────────────

def build_folds(clips: list[dict]):
    rng = random.Random(SEED)
    shuffled = clips[:]
    rng.shuffle(shuffled)

    fold_size = len(shuffled) // N_FOLDS
    folds = [shuffled[i * fold_size: (i + 1) * fold_size] for i in range(N_FOLDS)]
    folds[-1] += shuffled[N_FOLDS * fold_size:]   # remainder into last fold

    for k in range(N_FOLDS):
        dst = OUT / f"fold_{k}"
        if dst.exists():
            shutil.rmtree(dst)

        val_clips   = folds[k]
        train_clips = [c for i, f in enumerate(folds) for c in f if i != k]

        train_stems = stems_for_clips(train_clips)
        val_stems   = stems_for_clips(val_clips)

        copy_stems(train_stems, dst / "images" / "train", dst / "labels" / "train")
        copy_stems(val_stems,   dst / "images" / "val",   dst / "labels" / "val")
        write_yaml(dst)

        print(f"  fold_{k}: {len(train_clips)} train clips ({len(train_stems)} frames) | "
              f"{len(val_clips)} val clips ({len(val_stems)} frames)")


# ── Learning curve ────────────────────────────────────────────────────────────

def build_learning_curve(clips: list[dict]):
    rng = random.Random(SEED)
    shuffled = clips[:]
    rng.shuffle(shuffled)

    n_val       = max(1, round(len(shuffled) * 0.20))
    val_clips   = shuffled[:n_val]
    train_pool  = shuffled[n_val:]
    val_stems   = stems_for_clips(val_clips)

    lc_dir = OUT / "learning_curve"
    if lc_dir.exists():
        shutil.rmtree(lc_dir)

    for frac in LC_FRACS:
        n_train     = max(1, round(len(train_pool) * frac))
        train_clips = random.Random(SEED).sample(train_pool, n_train)
        train_stems = stems_for_clips(train_clips)

        tag = f"frac_{int(frac * 100):03d}"
        dst = lc_dir / tag

        copy_stems(train_stems, dst / "images" / "train", dst / "labels" / "train")
        copy_stems(val_stems,   dst / "images" / "val",   dst / "labels" / "val")
        write_yaml(dst)

        print(f"  {tag}: {len(train_clips)} train clips ({len(train_stems)} frames) | "
              f"{len(val_clips)} val clips ({len(val_stems)} frames)")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    clips = load_clips()
    total_frames = sum(c["n_frames"] for c in clips)
    print(f"Total TB2026 clips: {len(clips)}  frames: {total_frames}")

    print("\nBuilding 5-fold CV splits...")
    build_folds(clips)

    print("\nBuilding learning curve splits...")
    build_learning_curve(clips)

    print(f"\nDone. Datasets at: {OUT}")
