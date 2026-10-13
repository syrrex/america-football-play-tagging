"""
Builds keypoint detection datasets for stages 1–3.

Output structure:
  Data/Keypoints/datasets/
    condition1/          stage 2 training  — game1+2+5
      images/train/
      images/val/        held-out clips (one per angle per game)
      labels/train/
      labels/val/
      data.yaml
    ravelin2025/         stage 3 fine-tune — game8 only
      images/train/
      images/val/
      labels/train/
      labels/val/
      data.yaml
    test/                shared held-out test set — all TB2026 frames
      images/
      labels/

Val holdout clips (one per game, covering l/m/r angles):
  game1 : clip 13  angle=m  frames local  0– 47   (48 frames)
  game2 : clip 70  angle=l  frames local 291–319   (29 frames)
  game5 : clip  3  angle=r  frames local 252–281   (30 frames)
  game8 : clip  8  angle=m  frames local  63–151   (89 frames)

All label files are fixed (out-of-bounds keypoints clamped to 0,0,0).
"""

import csv
import shutil
from pathlib import Path

PROJECT   = Path(__file__).parent.parent
DATA      = PROJECT / "Data"
KP        = DATA / "Keypoints"
FRAMES    = KP / "frames"
ANNOTS    = KP / "annotations"
OUT       = KP / "datasets"
CSV_PATH  = DATA / "metadata" / "keypoint_range_data.csv"

GAME_FOLDER = {"Game1": "game1", "Game2": "game2", "Game5": "game5", "Game8": "game8"}

# Val clip IDs (clip column in CSV) per game
VAL_CLIPS = {
    "game1": {"13"},   # m angle, 48 frames
    "game2": {"70"},   # l angle, 29 frames
    "game5": {"3"},    # r angle, 30 frames
    "game8": {"8"},    # m angle, 89 frames
}

# YOLO pose data.yaml fields
KPT_SHAPE = "[62, 3]"
FLIP_IDX  = (
    "[12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,\n"
    " 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13,\n"
    " 34, 33, 32, 31, 30, 29, 28, 27, 26,\n"
    " 43, 42, 41, 40, 39, 38, 37, 36, 35,\n"
    " 52, 51, 50, 49, 48, 47, 46, 45, 44,\n"
    " 61, 60, 59, 58, 57, 56, 55, 54, 53]"
)


# ── Label fix ────────────────────────────────────────────────────────────────

def fix_label(text: str) -> str:
    """Clamp bbox coords; zero-out any keypoint with out-of-bounds x or y."""
    lines = []
    for line in text.strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        cls = parts[0]
        cx, cy = float(parts[1]), float(parts[2])
        w,  h  = float(parts[3]), float(parts[4])
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w  = max(1e-4, min(1.0, w))
        h  = max(1e-4, min(1.0, h))
        bbox = [cls, f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
        kpts = parts[5:]
        fixed = []
        for j in range(0, len(kpts), 3):
            x, y, v = float(kpts[j]), float(kpts[j+1]), float(kpts[j+2])
            if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
                x, y, v = 0.0, 0.0, 0.0
            fixed.extend([f"{x:.6f}", f"{y:.6f}", f"{int(v)}"])
        lines.append(" ".join(bbox + fixed))
    return "\n".join(lines) + "\n"


# ── CSV parsing — compute local frame index per clip ─────────────────────────

def load_clip_ranges() -> dict[str, list[dict]]:
    """
    Returns dict: game_folder → list of clip info dicts with local frame indices.
    Each dict: {clip, angle, fps, local_start, local_end}
    """
    rows = []
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            folder = GAME_FOLDER[r["Game"]]
            frm, to, fps = int(r["from"]), int(r["to"]), int(r["FPS"])
            indices = list(range(frm, to + 1, 6)) if fps == 30 else list(range(frm, to + 1))
            rows.append({"folder": folder, "clip": r["clip"], "angle": r["angle"],
                         "fps": fps, "indices": indices})

    # Group by game and assign local indices
    from collections import defaultdict
    game_clips: dict[str, list[dict]] = defaultdict(list)
    cursor = defaultdict(int)

    for r in rows:
        folder = r["folder"]
        n = len(r["indices"])
        game_clips[folder].append({
            "clip":        r["clip"],
            "angle":       r["angle"],
            "fps":         r["fps"],
            "local_start": cursor[folder],
            "local_end":   cursor[folder] + n - 1,
            "n":           n,
        })
        cursor[folder] += n

    return dict(game_clips)


# ── Copy helpers ──────────────────────────────────────────────────────────────

def copy_frame(src_img: Path, src_lbl: Path, dst_img: Path, dst_lbl: Path):
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_img, dst_img)
    if src_lbl.exists():
        dst_lbl.write_text(fix_label(src_lbl.read_text()))
    else:
        print(f"  [WARN] missing label: {src_lbl.name}")


def write_yaml(dataset_dir: Path):
    yaml = (
        f"path: {dataset_dir}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  ../test/images\n\n"
        f"kpt_shape: {KPT_SHAPE}\n"
        f"flip_idx: {FLIP_IDX}\n\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: Field\n"
    )
    (dataset_dir / "data.yaml").write_text(yaml)


# ── Build condition1 (game1+2+5) ─────────────────────────────────────────────

def build_condition1(clip_ranges: dict):
    dst = OUT / "condition1"
    if dst.exists():
        shutil.rmtree(dst)

    stats = {"train": 0, "val": 0}

    for game in ["game1", "game2", "game5"]:
        val_clip_ids = VAL_CLIPS[game]
        src_frames = FRAMES / game
        src_labels = ANNOTS / game

        for clip in clip_ranges[game]:
            is_val = clip["clip"] in val_clip_ids
            split  = "val" if is_val else "train"

            for local_i in range(clip["local_start"], clip["local_end"] + 1):
                src_name = f"frame_{local_i:06d}"
                dst_name = f"{game}_{local_i:06d}"   # prefix prevents cross-game collisions
                copy_frame(
                    src_frames / f"{src_name}.png",
                    src_labels / f"{src_name}.txt",
                    dst / "images" / split / f"{dst_name}.png",
                    dst / "labels" / split / f"{dst_name}.txt",
                )
                stats[split] += 1

        print(f"  {game}: done")

    write_yaml(dst)
    print(f"condition1 → train={stats['train']}  val={stats['val']}")


# ── Build ravelin2025 (game8 only, for stage 3 fine-tune) ────────────────────

def build_ravelin2025(clip_ranges: dict):
    dst = OUT / "ravelin2025"
    if dst.exists():
        shutil.rmtree(dst)

    stats   = {"train": 0, "val": 0}
    game    = "game8"
    val_ids = VAL_CLIPS[game]
    src_frames = FRAMES / game
    src_labels = ANNOTS / game

    for clip in clip_ranges[game]:
        is_val = clip["clip"] in val_ids
        split  = "val" if is_val else "train"

        for local_i in range(clip["local_start"], clip["local_end"] + 1):
            src_name = f"frame_{local_i:06d}"
            dst_name = f"{game}_{local_i:06d}"
            copy_frame(
                src_frames / f"{src_name}.png",
                src_labels / f"{src_name}.txt",
                dst / "images" / split / f"{dst_name}.png",
                dst / "labels" / split / f"{dst_name}.txt",
            )
            stats[split] += 1

    write_yaml(dst)
    print(f"ravelin2025 → train={stats['train']}  val={stats['val']}")


# ── Build test set (all TB2026 frames) ────────────────────────────────────────

def build_test():
    dst = OUT / "test"
    if dst.exists():
        shutil.rmtree(dst)

    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)

    src_frames = FRAMES / "tb2026"
    src_labels = ANNOTS / "tb2026" / "labels"

    count = 0
    for img in sorted(src_frames.glob("*.png")):
        lbl = src_labels / (img.stem + ".txt")
        dst_img = dst / "images" / img.name
        dst_lbl = dst / "labels" / (img.stem + ".txt")
        shutil.copy2(img, dst_img)
        if lbl.exists():
            dst_lbl.write_text(fix_label(lbl.read_text()))
        else:
            print(f"  [WARN] missing label: {lbl.name}")
        count += 1

    # Shared eval yaml
    eval_yaml = OUT / "eval_only.yaml"
    eval_yaml.write_text(
        f"path: {OUT}\n"
        f"train: test/images\n"
        f"val:   test/images\n"
        f"test:  test/images\n\n"
        f"kpt_shape: {KPT_SHAPE}\n"
        f"flip_idx: {FLIP_IDX}\n\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: Field\n"
    )
    print(f"test → {count} frames")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    clip_ranges = load_clip_ranges()

    # Print clip summary for verification
    print("Clip assignment summary:")
    for game in ["game1", "game2", "game5", "game8"]:
        val_ids = VAL_CLIPS[game]
        for c in clip_ranges[game]:
            tag = " ← VAL" if c["clip"] in val_ids else ""
            print(f"  {game}  clip {c['clip']:>4}  {c['angle']}  "
                  f"local {c['local_start']:>4}–{c['local_end']:>4}  "
                  f"({c['n']} frames){tag}")
    print()

    print("Building condition1 (game1+2+5)...")
    build_condition1(clip_ranges)

    print("\nBuilding ravelin2025 (game8)...")
    build_ravelin2025(clip_ranges)

    print("\nBuilding test set (TB2026)...")
    build_test()

    print(f"\nDone. Datasets at: {OUT}")
