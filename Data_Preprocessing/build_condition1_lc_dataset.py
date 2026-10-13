"""
Builds condition1 learning curve splits from the already-assembled condition1 dataset.

Sources from condition1/images/train/ and condition1/labels/train/ — no raw frames needed.
Clips are identified by the game-prefixed filenames (e.g. game2_000097.png).
Val set is always the full condition1 val (107 frames).
Eval target is the TB2026 test set.

Output: Data/Keypoints/datasets/condition1_lc/
  frac_005/  frac_010/  frac_025/  frac_050/  frac_075/  frac_100/
"""

import os
import random
import shutil
from pathlib import Path

KP_DS = Path(os.environ.get("KP_DS",
    Path(__file__).parent.parent / "Data" / "Keypoints" / "datasets"))

SRC       = KP_DS / "condition1"
SRC_TRAIN_IMG = SRC / "images" / "train"
SRC_TRAIN_LBL = SRC / "labels" / "train"
SRC_VAL_IMG   = SRC / "images" / "val"
SRC_VAL_LBL   = SRC / "labels" / "val"
OUT           = KP_DS / "condition1_lc"

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
LC_FRACS = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]

# Clip ranges derived from keypoint_range_data.csv (local indices per game)
# Only train clips (val clips are excluded from condition1/images/train/)
TRAIN_CLIP_RANGES = [
    ("game1", "clip22",  48,  77),
    ("game1", "clip27",  78, 102),
    ("game1", "clip46", 103, 139),
    ("game1", "clip53", 140, 164),
    ("game2", "clip7",    0,  41),
    ("game2", "clip89",  42,  96),
    ("game2", "clip58",  97, 145),
    ("game2", "clip162",146, 213),
    ("game2", "clip149",214, 252),
    ("game2", "clip159",253, 290),
    ("game2", "clip123",320, 369),
    ("game2", "clip43", 370, 411),
    ("game5", "clip2",    0,  92),
    ("game5", "clip23",  93, 183),
    ("game5", "clip60", 184, 251),
    ("game5", "clip9",  282, 311),
    ("game5", "clip102",312, 364),
    ("game5", "clip119",365, 399),
]


def stems_for_clip(game: str, start: int, end: int) -> list[str]:
    return [f"{game}_{i:06d}" for i in range(start, end + 1)]


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


def copy_split(stems: list[str], src_img: Path, src_lbl: Path,
               dst_img: Path, dst_lbl: Path):
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        img = src_img / f"{stem}.png"
        lbl = src_lbl / f"{stem}.txt"
        if img.exists():
            shutil.copy2(img, dst_img / f"{stem}.png")
        else:
            print(f"  [WARN] missing: {stem}.png")
        if lbl.exists():
            shutil.copy2(lbl, dst_lbl / f"{stem}.txt")


def build():
    if OUT.exists():
        shutil.rmtree(OUT)

    # Build clip → stems mapping
    clips = [(game, clip_id, stems_for_clip(game, s, e))
             for game, clip_id, s, e in TRAIN_CLIP_RANGES]

    # Fixed val stems (all files in condition1/images/val/)
    val_stems = [p.stem for p in sorted(SRC_VAL_IMG.glob("*.png"))]

    print(f"Total train clips: {len(clips)}  val frames: {len(val_stems)}")

    rng = random.Random(SEED)
    shuffled = clips[:]
    rng.shuffle(shuffled)

    for frac in LC_FRACS:
        n_clips     = max(1, round(len(shuffled) * frac))
        sel_clips   = random.Random(SEED).sample(shuffled, n_clips)
        train_stems = [s for _, _, stems in sel_clips for s in stems]

        tag = f"frac_{int(frac * 100):03d}"
        dst = OUT / tag

        copy_split(train_stems, SRC_TRAIN_IMG, SRC_TRAIN_LBL,
                   dst / "images" / "train", dst / "labels" / "train")
        copy_split(val_stems, SRC_VAL_IMG, SRC_VAL_LBL,
                   dst / "images" / "val", dst / "labels" / "val")
        write_yaml(dst)

        print(f"  {tag}: {n_clips} clips ({len(train_stems)} frames) | "
              f"{len(val_stems)} val frames")


if __name__ == "__main__":
    build()
    print(f"\nDone. Datasets at: {OUT}")
