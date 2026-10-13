# Helpers

Standalone scripts that generate specific thesis figures/sheets directly into `thesis-template/figures/` (or, for the extrinsic-evaluation sheets, `Play_Analysis/extrinsic_evaluation/`). Each one is a one-off illustration or example figure, not part of any evaluation pipeline.

---

## Scripts

| Script | Output | Thesis location |
|--------|--------|------------------|
| `plot_annotated_classes_example.py` | `annotated_classes_example.png` | Materials — one annotated tb2026 presnap frame, GT boxes colour-coded by class, with legend. |
| `plot_keypoints_example.py` | `keypoints_example.png` | Materials — one broadcast frame with GT field keypoints overlaid, colour-coded by landmark group. Same play as the classes example (clip_012). |
| `plot_tacklebox_concept.py` | `tacklebox_concept.png` | Background — Doubles-formation diagram with the tackle box overlaid, showing attached vs. detached players. |
| `plot_field_registration_example.py` | `registration_landmarks.jpg`, `registration_bev.png` | Methods (field registration) — deployed keypoint detections on the broadcast frame, and the resulting bird's-eye projection. |
| `plot_tracking_example.py` | `tracking_t0.jpg`, `tracking_t1.jpg`, `tracking_t2.jpg`, `tracking_T.jpg` | Methods (segmentation and tracking) — 4 evenly-spaced frames of SAM-3 tracking on clip_042 (GT-box init), masks colored by role. |
| `plot_wr_routes_example.py` | `wr_routes_example_*.png` | Additional Experiments (Trajectory Visualization) — tracked bird's-eye WR/skill routes for one play. Purely qualitative, no route-type ground truth exists. |
| `plot_attack_direction_concept.py` | `attack_direction_concept.png` | Methods (yard-line/hash) — top-down diagram of how the Oline vs. defense mean position resolves attack direction. |
| `plot_hash_zones_concept.py` | `hash_zones_concept.png` | Background (field position) — field template with the hash-mark lines highlighted. |
| `generate_extrinsic_eval_sheets.py` | `Play_Analysis/extrinsic_evaluation/*.xlsx` | Additional Experiments (Extrinsic Evaluation) — the four Hudl-style tagging sheets used for the manual-vs.-pipeline-assisted timing comparison. |

## README assets

Written to `assets/` instead of `thesis-template/figures/`.

| Script | Output | Description |
|--------|--------|-------------|
| `render_readme_demo.py` | `demo.gif`, `demo_presnap.png` | One clip rendered from the pipeline's self-detected outputs: role boxes from the tracking cache, bird's-eye minimap via the held-out fold's keypoint model, and the three tags vs. the coach's tags. Needs the raw clip, tracking cache and keypoint weights locally. |
| `plot_readme_results.py` | `results_tags.png` | Tag accuracy, ground-truth input vs. end-to-end, with majority-class baselines, read from `Play_Analysis/*_eval*.json`. Runs without any data. |
