"""
Prepares the Player_Detection_Ravelin CVAT export for YOLO training.

Steps:
  1. Remaps class IDs from CVAT order to the original model's class order.
  2. Copies images + remapped labels into a clean dataset folder.
  3. Writes a data.yaml for Ultralytics YOLO.

CVAT export classes (obj.names):  qb=0, skill=1, oline=2, defense=3, ref=4
Original model classes:          defense=0, oline=1, qb=2, ref=3, skill=4
"""

import shutil
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent
SRC         = BASE / "Player_Detection_Ravelin" / "obj_train_data"
DATASET_OUT = BASE / "dataset"
# ─────────────────────────────────────────────────────────────────────────────

# CVAT id → original model id
REMAP = {0: 2, 1: 4, 2: 1, 3: 0, 4: 3}

CLASS_NAMES = ["defense", "oline", "qb", "ref", "skill"]


def remap_label_file(src_txt: Path, dst_txt: Path) -> int:
    """Copy label file with remapped class IDs. Returns number of boxes."""
    lines = src_txt.read_text().splitlines()
    remapped = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        cls = int(parts[0])
        parts[0] = str(REMAP[cls])
        remapped.append(" ".join(parts))
    dst_txt.write_text("\n".join(remapped))
    return len(remapped)


def prepare():
    images_out = DATASET_OUT / "images"
    labels_out = DATASET_OUT / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    img_files = sorted(SRC.glob("*.jpg"))
    total_boxes = 0
    empty = 0

    for img in img_files:
        txt = SRC / (img.stem + ".txt")
        if not txt.exists():
            print(f"  [SKIP] no label for {img.name}")
            continue

        shutil.copy(img, images_out / img.name)
        n = remap_label_file(txt, labels_out / txt.name)
        total_boxes += n
        if n == 0:
            empty += 1

    n_images = len(list(images_out.glob("*.jpg")))
    print(f"Dataset ready: {n_images} images, {total_boxes} boxes, {empty} empty label files")
    print(f"Output: {DATASET_OUT}")

    # Write data.yaml
    yaml_path = DATASET_OUT / "data.yaml"
    yaml_path.write_text(
        f"path: {DATASET_OUT}\n"
        f"train: images\n"
        f"val: images\n"
        f"\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    print(f"data.yaml written to: {yaml_path}")

    return DATASET_OUT, n_images


if __name__ == "__main__":
    prepare()
