# Data_Preprocessing

Scripts for assembling and transforming raw data into structured datasets.

---

## Scripts

### populate_data.py
Populates `Data/Keypoints/frames/` and `Data/Keypoints/annotations/` from `Data/Original_CVAT_Field_Data/` (TestV4, 4462 frames). Reads `Data/metadata/keypoint_range_data.csv` as the authoritative source for frame ranges and game assignments.
- FPS=30 rows: subsamples every 6th frame (~5fps equivalent)
- FPS=5 rows: copies all frames in the specified range
- Output games: game1, game2, game5, game8

### build_cross_stadium_dataset.py
Assembles the cross-stadium player detection dataset from raw frames and annotations into `Data/Players/datasets/cross_stadium/`.
- `non_ravelin/` split: g1, g2, g4, g5, g6, g7 (80/20 train/val, seed=42)
- `ravelin2025/` split: g8 (80/20 train/val, seed=42)
- `test/` split: all 107 TB2026 frames (shared held-out test set)
- Writes `data.yaml` files for each split

### build_tb2026_dataset.py
Builds `Data/Keypoints/datasets/tb2026/` from the already-assembled `test/` split (2920 tb2026 frames), for the keypoint model's 5-fold CV upper bound and 6-fraction learning curve experiments (superseded by the quarter-based CV used for the deployed model, but still what `learning_curve.py`/`upper_bound_experiment.py` read from). Splits at clip level (via `Data/metadata/tb2026_dataset_metadata.csv`) to avoid temporal leakage between near-identical consecutive frames.
- `fold_0/` … `fold_4/` — 5-fold CV, clip-level split, seed=42
- `learning_curve/frac_005/` … `frac_100/` — fixed 80/20 clip split at 6 training fractions
- Not currently materialized on disk — already consumed to produce `Field_Keypoint_Detection/upper_bound_results/` and `learning_curve_results/`, regenerable by re-running this script.

### build_condition1_lc_dataset.py
Builds `Data/Keypoints/datasets/condition1_lc/` from the already-assembled `condition1/` dataset's train split, for `condition1_learning_curve.py`'s learning-curve experiment. Val is always the full `condition1` val set (107 frames); eval target is the tb2026 test set.
- `frac_005/` … `frac_100/` — 6 training fractions of `condition1`'s train clips
- Not currently materialized on disk — already consumed to produce `Field_Keypoint_Detection/condition1_lc_results/`, regenerable by re-running this script.
