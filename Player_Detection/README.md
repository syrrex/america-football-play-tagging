# Player_Detection

YOLO11-based player detection experiments. Two experiments: cross-stadium generalization (main) and 5-fold CV upper bound (reference ceiling). Model fixed to yolo11m, both 640px and 1280px input resolution.

Class mapping: `0=defense  1=oline  2=qb  3=ref  4=skill`

---

## Experiments

### Cross-Stadium Generalization (main experiment)
Progressive fine-tuning ladder evaluated on the held-out TB2026 test set (107 frames).

| Stage | Starting weights | Training data | mAP50 @ 640 | mAP50 @ 1280 |
|-------|-----------------|---------------|-------------|--------------|
| 1 | COCO zero-shot | — | 0.039 | 0.028 |
| 2 | Roboflow pretrained | — (eval only) | 0.613 | 0.741 |
| 3a | COCO | non-Ravelin games (g1-g7) | 0.832 | 0.835 |
| 3b | Stage 2 weights | non-Ravelin games (g1-g7) | 0.885 | 0.917 |
| 4a | Stage 3a weights | Ravelin 2025 (g8) | 0.834 | 0.714 |
| 4b | Stage 3b weights | Ravelin 2025 (g8) | 0.863 | 0.906 |

Results saved in `cross_stadium_results/summary.json`.

### 5-Fold CV Upper Bound (reference ceiling)
Full experiment grid (2 starting points × 5 model sizes × 2 resolutions) evaluated via 5-fold cross-validation on the 107 TB2026 frames. Best result: roboflow+yolo11m+1280px → mAP50=0.985.

Results in `upper_bound_results/grid_summary.json` and `grid_summary.csv`.

---

## Scripts

| File | Description |
|------|-------------|
| `cross_stadium_experiment.py` | Main cross-stadium ladder experiment. Runs all 6 stages at 640 and 1280px. Saves results incrementally — safe to interrupt and resume. |
| `upper_bound_experiment.py` | 5-fold CV grid experiment (adapted from seminar project). Runs all 20 configurations. |
| `train_roboflow.py` | Trains yolo11 models on the public Roboflow football dataset to produce stage 2 starting weights. |
| `learning_curve.py` | Generates learning curve by training at multiple dataset fractions. |
| `annotation_analysis.py` | Checks annotation completeness against expected player counts per class. |
| `scientific_analysis.py` | Statistical significance testing (Wilcoxon, t-test) and visualization for upper bound results. |
| `prepare_dataset.py` | Prepares the TB2026 CVAT export for YOLO training, remapping class IDs from CVAT order to model order. |

---

## models/
Pre-trained starting weights used as initialization.

```
models/
  roboflow/
    yolo11n/weights/best.pt   — trained on Roboflow football dataset
    yolo11s/weights/best.pt
    yolo11m/weights/best.pt   — used for cross-stadium stages 2/3b/4b
    yolo11l/weights/best.pt
    yolo11x/weights/best.pt
```

## cross_stadium_results/
Output of `cross_stadium_experiment.py`. One subfolder per stage+resolution (e.g. `3b_1280/`), each containing trained weights and evaluation metrics.

## upper_bound_results/
Output of the 5-fold CV grid experiment.
- `grid_summary.json` / `grid_summary.csv` — full results for all 20 configurations
- `roboflow_yolo11m_1280/` — fold weights and per-fold metrics for the best configuration
