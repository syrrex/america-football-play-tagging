"""
Populates Data/Keypoints/frames/{game}/ and Data/Keypoints/annotations/{game}/
by reading Data/metadata/keypoint_range_data.csv as the authoritative source.

Source for all images + labels: Data/Original_CVAT_Field_Data/ (frames 0–4461).

Rules:
  - FPS=30 rows: subsample every 6th frame (≈ 5 fps equivalent)
  - FPS=5  rows: copy all frames in [from, to] inclusive

Game folder mapping (from CSV 'Game' column):
  Game1 → game1   (Tirol2,       Raiders vs Prag)
  Game2 → game2   (Tirol,        Raiders vs Ducks)
  Game5 → game5   (Fehervar,     Enthroners vs Invaders)
  Game8 → game8   (Ravelin 2025, Tigers vs Bears)
"""

import csv
import shutil
from pathlib import Path

PROJECT = Path(__file__).parent.parent          # Master_Thesis/Project/
DATA    = PROJECT / "Data"
SRC     = DATA / "Original_CVAT_Field_Data"     # fresh TestV4 download (raw CVAT export)
TV4_IMG = SRC / "images" / "train"
TV4_LBL = SRC / "labels" / "train"
CSV     = DATA / "metadata" / "keypoint_range_data.csv"

GAME_FOLDER = {
    "Game1": "game1",
    "Game2": "game2",
    "Game5": "game5",
    "Game8": "game8",
}

# ── Load CSV ──────────────────────────────────────────────────────────────────
rows = []
with open(CSV, newline="") as f:
    for r in csv.DictReader(f):
        rows.append({
            "game":   r["Game"],
            "folder": GAME_FOLDER[r["Game"]],
            "frm":    int(r["from"]),
            "to":     int(r["to"]),
            "fps":    int(r["FPS"]),
            "angle":  r["angle"],
            "clip":   r["clip"],
        })

# ── Compute frame selections per game (preserving CSV order) ──────────────────
from collections import defaultdict

game_frames = defaultdict(list)   # folder → list of TestV4 global indices
for r in rows:
    tv4_indices = list(range(r["frm"], r["to"] + 1))
    if r["fps"] == 30:
        tv4_indices = tv4_indices[::6]   # every 6th → ≈ 5 fps
    game_frames[r["folder"]].extend(tv4_indices)

# ── Copy frames + labels ──────────────────────────────────────────────────────
print(f"Source: {SRC}")
print(f"Target: {DATA}\n")

for folder in ["game1", "game2", "game5", "game8"]:
    indices  = game_frames[folder]
    frm_dir  = DATA / "Keypoints" / "frames" / folder
    lbl_dir  = DATA / "Keypoints" / "annotations" / folder
    frm_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    copied = missing_img = missing_lbl = 0
    for local_i, tv4_i in enumerate(indices):
        src_img = TV4_IMG / f"frame_{tv4_i:06d}.png"
        src_lbl = TV4_LBL / f"frame_{tv4_i:06d}.txt"
        dst_img = frm_dir  / f"frame_{local_i:06d}.png"
        dst_lbl = lbl_dir  / f"frame_{local_i:06d}.txt"

        if not src_img.exists():
            print(f"  [WARN] image missing: {src_img.name}")
            missing_img += 1
            continue

        shutil.copy2(src_img, dst_img)

        if src_lbl.exists():
            shutil.copy2(src_lbl, dst_lbl)
        else:
            missing_lbl += 1

        copied += 1

    warn = ""
    if missing_img: warn += f"  {missing_img} images missing!"
    if missing_lbl: warn += f"  {missing_lbl} labels missing"
    print(f"{folder:8s}: {copied:4d} frames copied{warn}")

print("\nDone.")
