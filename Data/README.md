# Data

All raw images, annotations, and assembled YOLO datasets for both pipeline components.

> **Not included in the repository** (~39 GB of footage, frames and annotations). This README documents the expected layout and schema.

---

## Players/

Player bounding box data across all games. Class mapping (YOLO IDs):
`0=defense  1=oline  2=qb  3=ref  4=skill`

### frames/
Raw pre-snap images, one per game folder.

| Folder | Stadium | Teams | Images |
|--------|---------|-------|--------|
| g1 | Tirol2 | Raiders vs. Prag | 20 |
| g2 | Tirol | Raiders vs. Ducks | 20 |
| g4 | Fehervar | Enthroners vs. Vikings | 20 |
| g5 | Fehervar | Enthroners vs. Invaders | 20 |
| g6 | Mödling | Rangers vs. Invaders | 20 |
| g7 | Mödling | Rangers vs. Tigers | 20 |
| g8 | Ravelin | Bears vs. Tigers (2025) | 20 |
| tb2026 | Ravelin | Bears vs. Tigers (2026) | 107 |

### annotations/
YOLO-format `.txt` label files (one per image, matching frames/). All files use the remapped class ordering above. Note: tb2026 labels were originally exported from CVAT with a different class ordering and have been remapped.

### datasets/
Assembled YOLO dataset splits ready for training.

- **roboflow/** — Public Roboflow football dataset (Football-Presnap-Tracker-1). Used as stage 2 training data and as starting weights source for the cross-stadium experiment.
- **cross_stadium/** — Cross-stadium generalization dataset. Contains:
  - `non_ravelin/` — g1,g2,g4,g5,g6,g7 (96 train / 24 val) for stages 3a/3b
  - `ravelin2025/` — g8 (16 train / 4 val) for stages 4a/4b
  - `test/` — all 107 TB2026 frames, shared held-out test set for all stages
  - `eval_only.yaml` — data.yaml pointing only to the test split
- **tb2026/** — Placeholder for the 5-fold CV upper bound dataset (built by `prepare_dataset.py` in Player_Detection/).

---

## Keypoints/

Field keypoint data for homography estimation. 62 keypoints per frame (field line intersections and landmarks).

### frames/
Video frames sampled at ~5 fps from game clips.

| Folder | Stadium | Teams | Frames |
|--------|---------|-------|--------|
| game1 | Tirol2 | Raiders vs. Prag | 165 |
| game2 | Tirol | Raiders vs. Ducks | 412 |
| game5 | Fehervar | Enthroners vs. Invaders | 400 |
| game8 | Ravelin | Bears vs. Tigers (2025) | 411 |
| tb2026 | Ravelin | Bears vs. Tigers (2026) | 2920 |

30fps clips were subsampled to every 6th frame (~5fps equivalent). Source frame ranges are defined in `metadata/keypoint_range_data.csv`.

### annotations/
YOLO pose-format `.txt` label files. tb2026 labels come from the original CVAT export (`Original_CVAT_Field_Data/`).

### datasets/
Assembled YOLO pose datasets for keypoint detection experiments.

- **condition1/** — Training on game1 + game2 + game5 (cross-stadium, no Ravelin)
- **ravelin2025/** — game8 (411 frames, train/val split) for the sequential and joint cross-stadium fine-tuning stages
- **test/** — all 2920 tb2026 frames, the fixed held-out test set shared across every stage
- `eval_only.yaml` — data.yaml pointing only at the test split

---

## Original_CVAT_Field_Data/
Raw CVAT export of the full TestV4 dataset (4462 frames, indices 0–4461, all games combined, before any per-game splitting). Source for every game's `Keypoints/frames/{game}/` and `Keypoints/annotations/{game}/`, including tb2026. Frame ranges are defined in `metadata/keypoint_range_data.csv`; the split into per-game folders is done by `Data_Preprocessing/populate_data.py`.

## raw/
Raw unprocessed clip frames per game, used as the source before preprocessing.

---

## metadata/

| File | Description |
|------|-------------|
| `tb2026_dataset_metadata.csv` | Per-play metadata for the 107 tb2026 plays (QTR, YARD_LN, PLAY_TYPE, OFF_FORM, etc.), extended in place by `Data/datainsights.ipynb`. |
| `PlaylistData_2026-06-03.xlsx` | Original scouting spreadsheet, source of the scouting columns merged into `tb2026_dataset_metadata.csv`. |
| `keypoint_range_data.csv` | Maps TestV4 frame ranges to games, stadiums, teams, FPS, and clip IDs. Authoritative source for keypoint data population (`Data_Preprocessing/populate_data.py`). |
| `player_detection_data.csv` | Per-play metadata for all 107 TB2026 plays (ODK, down, distance, hash, formation, etc.) plus bounding box counts. Not currently read by any script. |
| `player_detection_old_games.csv` | Per-image metadata and box counts for all 140 old-game player detection frames (g1–g8). Not currently read by any script. |