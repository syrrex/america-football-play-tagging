# Pipeline

The integrated end-to-end pipeline: pre-snap YOLO player detection → SAM3 tracking → keypoint-based homography → bird's-eye player positions → Formation/Yard-Hash/Play-Type classification. Also the shared library (`drawing.py`, `homography.py`, `sam3_tracking.py`) imported by `Player_Tracking/`, `Play_Analysis/`, and `Helpers/`.

---

## Scripts

| File | Description |
|------|-------------|
| `homography.py` | Maps the 62 YOLO-pose field landmarks to their known real-world field positions and fits the perspective homography (`run_field_pose_model`, `kpt_world`, `PX_PER_YARD`). Field dimension constants (`FIELD_YD_H = 53.33`, etc.) live here. |
| `sam3_tracking.py` | SAM-3 video object segmentation: takes first-frame YOLO detections and propagates tracking masks through all frames. |
| `drawing.py` | Visualization utilities: bird's-eye field canvas, projected player-position plotting, SAM3 mask/label annotation on frames. |
| `evaluate_pipeline.py` | Reprojection error per clip across all quarter folds — uses the matching fold's keypoint model (no data leakage), runs detection + homography on every clip in that fold's val set, saves a CSV with clip metadata (yard line, play type, etc.) for the camera-angle-vs-accuracy analysis. Output: `pipeline_eval.csv`. |
| `pipeline.ipynb` | Full pipeline demo notebook, one clip at a time: pre-snap detection (Stage 4b) → SAM3 tracking → keypoint homography (in-domain upper-bound fold model) → bird's-eye player cards → Formation/Yard-Hash/Play-Type classification (imports directly from `Play_Analysis/`, same self-detected logic as the thesis's RQ3 ablation). |

`pipeline_eval.csv` is the current, up-to-date output of `evaluate_pipeline.py` (post yard-number field-template fix).
