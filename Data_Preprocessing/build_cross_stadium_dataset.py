"""
Builds Data/Players/datasets/cross_stadium/ for the cross-stadium experiment.

Dataset splits:
  non_ravelin/  : g1, g2, g4, g5, g6, g7  (120 images — stages 3a/3b training)
  ravelin2025/  : g8                        ( 20 images — stages 4a/4b training)
  test/         : all 107 TB2026 frames     (held-out evaluation for all stages)

Val sets (for early-stopping monitoring during training):
  non_ravelin_val/ : random 20% of non-ravelin (24 images, seed=42)
  ravelin2025_val/ : random 20% of g8          (  4 images, seed=42)

Class mapping (identical across all sources):
  0=defense  1=oline  2=qb  3=ref  4=skill
"""

import shutil
import random
from pathlib import Path

random.seed(42)

PROJECT       = Path(__file__).parent.parent
DATA          = PROJECT / "Data"
PLAYERS       = DATA / "Players"
FRAMES        = PLAYERS / "frames"
ANNOTS        = PLAYERS / "annotations"
OUT           = PLAYERS / "datasets" / "cross_stadium"

CLASS_NAMES   = ["defense", "oline", "qb", "ref", "skill"]
NON_RAVELIN   = ["g1", "g2", "g4", "g5", "g6", "g7"]
RAVELIN_2025  = "g8"
TB2026        = "tb2026"
VAL_FRAC      = 0.2


def collect(games: list[str]) -> list[Path]:
    imgs = []
    for game in games:
        imgs += sorted((FRAMES / game).glob("*.jpg"))
        imgs += sorted((FRAMES / game).glob("*.png"))
    return imgs


def copy_split(imgs: list[Path], split_name: str, val_frac: float = 0.0):
    """Copy images+labels to OUT/<split_name>/images/train (and /val if val_frac>0)."""
    shuffled = imgs[:]
    random.shuffle(shuffled)
    n_val  = max(1, round(len(shuffled) * val_frac)) if val_frac > 0 else 0
    val_set = set(p.name for p in shuffled[:n_val])

    for img in imgs:
        game    = img.parent.name
        lbl_src = ANNOTS / game / (img.stem + ".txt")
        subset  = "val" if img.name in val_set else "train"

        img_dst = OUT / split_name / "images" / subset / img.name
        lbl_dst = OUT / split_name / "labels" / subset / (img.stem + ".txt")
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        lbl_dst.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(img, img_dst)
        if lbl_src.exists():
            shutil.copy2(lbl_src, lbl_dst)
        else:
            print(f"  [WARN] missing label: {lbl_src.name}")

    n_train = len(imgs) - n_val
    return n_train, n_val


def write_yaml(split_name: str):
    split_dir = OUT / split_name
    # test always points to the shared test split
    yaml = (
        f"path: {OUT}\n"
        f"train: {split_name}/images/train\n"
        f"val:   {split_name}/images/val\n"
        f"test:  test/images\n\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    (split_dir / "data.yaml").write_text(yaml)


def build():
    # ── Non-Ravelin games (stages 3a / 3b training) ──────────────────────────
    nr_imgs = collect(NON_RAVELIN)
    n_train, n_val = copy_split(nr_imgs, "non_ravelin", VAL_FRAC)
    write_yaml("non_ravelin")
    print(f"non_ravelin : {n_train} train  {n_val} val  ({len(nr_imgs)} total)")

    # ── Ravelin 2025 (stages 4a / 4b training) ───────────────────────────────
    r25_imgs = collect([RAVELIN_2025])
    n_train, n_val = copy_split(r25_imgs, "ravelin2025", VAL_FRAC)
    write_yaml("ravelin2025")
    print(f"ravelin2025 : {n_train} train  {n_val} val  ({len(r25_imgs)} total)")

    # ── TB2026 test split (shared evaluation for all stages) ─────────────────
    test_dir   = OUT / "test" / "images"
    test_lbls  = OUT / "test" / "labels"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_lbls.mkdir(parents=True, exist_ok=True)

    tb_imgs = collect([TB2026])
    for img in tb_imgs:
        lbl_src = ANNOTS / TB2026 / (img.stem + ".txt")
        shutil.copy2(img, test_dir / img.name)
        if lbl_src.exists():
            shutil.copy2(lbl_src, test_lbls / (img.stem + ".txt"))
        else:
            print(f"  [WARN] missing label: {lbl_src.name}")
    print(f"test (TB2026): {len(tb_imgs)} images")

    print(f"\nDataset root: {OUT}")


if __name__ == "__main__":
    build()
