"""
Merge all TB2026 fold val sets into a single directory.

In a k-fold split each frame appears in exactly one fold's val set,
so fold_0/val + fold_1/val + ... + fold_4/val = all unique frames.

Run once on the cluster:
  python merge_tb2026_folds.py
  python merge_tb2026_folds.py --src ~/keypoint_dataset/tb2026 --dst ~/keypoint_dataset/tb2026_all
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="~/keypoint_dataset/tb2026",
                        help="TB2026 fold root directory")
    parser.add_argument("--dst", default="~/keypoint_dataset/tb2026_all",
                        help="Output directory for merged frames")
    args = parser.parse_args()

    src = Path(args.src).expanduser()
    dst = Path(args.dst).expanduser()

    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)

    n_imgs = n_lbls = 0
    for fold_dir in sorted(src.glob("fold_*")):
        val_imgs = fold_dir / "images" / "val"
        val_lbls = fold_dir / "labels" / "val"
        if not val_imgs.exists():
            print(f"  [skip] {fold_dir.name} — no images/val")
            continue
        for img in sorted(val_imgs.glob("*.png")):
            link = dst / "images" / img.name
            if not link.exists():
                link.symlink_to(img.resolve())
                n_imgs += 1
        for lbl in sorted(val_lbls.glob("*.txt")):
            link = dst / "labels" / lbl.name
            if not link.exists():
                link.symlink_to(lbl.resolve())
                n_lbls += 1

    print(f"Done: {n_imgs} images, {n_lbls} labels → {dst}")


if __name__ == "__main__":
    main()