# Player_Tracking

SAM-3 video object tracking for all 107 tb2026 clips, providing the per-frame player trajectories that Play-Type Classification's movement signal (QB dropback / Oline drive slope) is built on. Reuses `Pipeline/sam3_tracking.py` as-is.

Both scripts cache each clip's per-frame box positions (xyxy/tracker_id/class_id, no segmentation masks — full-resolution masks run ~1-2GB/clip, and the downstream yard-space projection only needs box positions) to a pickle in their respective `output/` cache folder.

---

## Scripts

| Script | Output | Description |
|--------|--------|-------------|
| `run_tracking.py` | `output/tracking_cache/` | GT-input phase: SAM-3 tracking initialized from ground-truth frame-0 player boxes. Isolates tracking-specific failures (ID switches, mask drift) from player-detection error. |
| `run_tracking_selfdetected.py` | `output/tracking_cache_selfdetected/` | Self-detected phase: SAM-3 tracking initialized from the deployed Stage-3b/1280 detector's own frame-0 detections, so tracking inherits real detection error too. |

Both are safe to interrupt and resume — already-cached clips are skipped.

No ground-truth tracking data exists for tb2026 (no annotated per-frame track IDs), so tracking quality itself is never independently evaluated, only indirectly through Play-Type Classification's own accuracy numbers (see `7_0_evaluation.tex`'s Play-Type Limitations paragraph).
