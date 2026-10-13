# Field_Keypoint_Detection

YOLO11-pose based field keypoint detection for homography estimation. Detects 62 field landmarks (line intersections, end zone corners, hash marks) from broadcast footage.

Starting weights `yolo11l-pose.pt` (COCO pretrained) are loaded by bare filename and auto-downloaded by ultralytics into this folder on first run — not tracked in git.

---

## Layout

| Folder | Contents |
|--------|----------|
| `utils/` | `field_landmarks.py` (62-landmark table + real-world field coordinates), `visualize.py`, `visualize_field_keypoints.py` |
| `evaluation/` | `evaluate_keypoints.py`, `evaluate_homography.py`, `zone_eval.py`, `main_evaluation.ipynb` (Ch. 7 main results), `additional_experiments_evaluation.ipynb` (cross-stadium generalization chapter), `evaluation_output/` (saved chart PNGs) |
| `logs/` | Old SLURM `.out` job logs |
| *(repo root)* | Training scripts (`*_training.py`, `*_experiment.py`), SLURM submission scripts (`run_*.sh`), and their result folders (`*_results/`) — all still flat, one folder per experiment |

---

## Scripts

| Script | Result folder | Description |
|--------|---------------|--------------|
| `build_datasets.py` | — | Builds `Data/Keypoints/datasets/condition1\|ravelin2025\|test` from the per-game frames/annotations. |
| `cross_stadium_experiment.py` | `cross_stadium_results/` | Cross-stadium ladder: Stage 1 (COCO zero-shot), Stage 2 (train condition1), Stage 3 (Stage 2 → fine-tune on ravelin2025). All evaluated on the fixed tb2026 test set. |
| `combined_training.py` | `combined_results/` | Joint training on condition1 + ravelin2025 simultaneously, to test whether it avoids the catastrophic forgetting seen in Stage 3's sequential fine-tune. |
| `combined_training_perspective.py` | `combined_perspective_results/` | Same as above, plus `perspective`/`degrees` augmentation to simulate cross-stadium camera-angle variation. |
| `condition1_learning_curve.py` | `condition1_lc_results/` | Learning curve over fractions of condition1's training data, evaluated on tb2026. Dataset built by `Data_Preprocessing/build_condition1_lc_dataset.py`. |
| `flip_ablation.py` | `flip_ablation_results/` | Ablation: `fliplr=0.0` vs. `fliplr=0.5`, trained on condition1, evaluated on tb2026. |
| `hash_only_training.py` | `hash_only_results/` | Ablation: supervising only the 18 hash-mark keypoints vs. all 62, trained on condition1+ravelin2025. |
| `learning_curve.py` | `learning_curve_results/` | Pre-quarter-fold learning curve: tb2026 data fractions with a fixed random val split. Dataset built by `Data_Preprocessing/build_tb2026_dataset.py`. Superseded by `quarter_lc_training.py`. |
| `upper_bound_experiment.py` | `upper_bound_results/` | Pre-quarter-fold 5-fold CV upper bound on tb2026. Same dataset source as `learning_curve.py`. Superseded by `quarter_kfold_training.py`. |
| `merge_tb2026_folds.py` | — | Cluster utility: merges the 5-fold `tb2026/fold_*/val` splits back into one deduplicated frame set (`tb2026_all`), the source `quarter_kfold_training.py` reads via `TB2026_SRC` on the cluster. |
| `quarter_kfold_training.py` | `quarter_kfold_results_640/`, `quarter_kfold_results_1280/` | Quarter-based 4-fold CV on tb2026, COCO-pretrained init (Stage 4a). Deployed-model comparison baseline — isolates the value of cross-stadium init. |
| `quarter_lc_training.py` | `quarter_lc_results_1280/` | Quarter-based learning curve: half-quarter training increments per held-out fold, never splitting a clip. |
| `finetune_quarter_training.py` | `finetune_quarter_results_1280/` | **Deployed model (Stage 4b).** Starts from Stage 3 (cross-stadium) weights, trains on 3 tb2026 quarters, evaluates on the held-out quarter. |
| `cross_stadium_quarter_eval.py` | `cross_stadium_quarter_results/` | No training — re-evaluates the Stage 2/Stage 3 cross-stadium weights per tb2026 quarter, for the per-quarter breakdown in Additional Experiments. |

---

## utils/ and evaluation/ scripts of note

- `evaluation/evaluate_homography.py` — end-to-end field-registration evaluation: compute homographies from GT vs. predicted keypoints, project GT player foot-points through both, measure distance error in yards. Answers RQ2 directly (component-level keypoint accuracy alone only partially answers it).
- `evaluation/zone_eval.py` — re-evaluates the deployed quarter-CV models per field-position zone (midfield vs. field-ends), checking whether registration accuracy degrades toward either end of the field as the camera's viewing angle gets shallower.
