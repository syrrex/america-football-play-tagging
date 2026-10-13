"""
Per-field-zone breakdown of field-registration accuracy (deployed model).

Checks the expectation stated in 6_0_Materials.tex: landmarks toward either
end of the field are viewed at a shallower angle from the fixed camera near
midfield, so homography accuracy is expected to be worse in the own-20 and
red-zone regions than at midfield. Never previously checked -- this script
closes that gap.

Reuses evaluate_homography.py's fit_homography/project/eval logic as-is,
just tags each clip/frame by zone before aggregating, instead of pooling
everything into one global stat. Zone convention matches the existing
field-position distribution table exactly (Data/datainsights.ipynb cell 12):
  abs(YARD_LN) <= 20  ->  Own 20 if YARD_LN < 0, else Red zone
  abs(YARD_LN) > 20   ->  Midfield

Runs all four quarter-CV finetune-init models (the deployed configuration),
pooling zone-tagged errors across quarters, same as every other CV result
in the thesis.

Usage (run from Field_Keypoint_Detection/, SAM3_ENV):
  python evaluation/zone_eval.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate_homography import (  # noqa: E402
    BASE, TEST_IMGS, TEST_LABELS, PLAYER_ANN, FIELD_POSITIONS,
    load_metadata, load_clip_to_kpt_frame,
    load_gt_keypoints, load_gt_player_footpoints,
    run_model_keypoints_raw, filter_kpts, fit_homography, project,
)

QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
CONF = 0.25
WEIGHTS_TMPL = BASE / "finetune_quarter_results_1280" / "{q}" / "train" / "weights" / "best.pt"
MIN_CORRESPONDENCES = 4


def zone(yard_ln: float) -> str:
    """Own-20 and red-zone are team/drive-relative labels (whose end zone is
    near), not fixed camera-relative positions -- PLAY_DIR flips per drive, so
    keeping them separate pools plays from both physical ends of the field
    into each bucket. Collapsed here into a single "Field ends" class vs.
    midfield, which is what the shallow-angle-toward-either-end hypothesis
    actually claims: distance from the camera's centered midfield position,
    symmetric in both directions, regardless of team framing."""
    return "Field ends" if abs(yard_ln) <= 20 else "Midfield"


def build_maps():
    meta = load_metadata()
    clip_to_zone = dict(zip(meta["clip_num"], meta["YARD_LN"].apply(zone)))
    clip_to_quarter = dict(zip(meta["clip_num"], meta["QTR"]))
    frame_to_zone = {}
    frame_to_quarter = {}
    for _, row in meta.iterrows():
        z = zone(row["YARD_LN"])
        q = row["QTR"]
        for f in range(int(row["start_frame"]), int(row["end_frame"]) + 1):
            frame_to_zone[f] = z
            frame_to_quarter[f] = q
    return clip_to_zone, clip_to_quarter, frame_to_zone, frame_to_quarter


def frame_number(path: Path) -> int:
    import re
    m = re.search(r"frame_0*(\d+)\.", path.name)
    return int(m.group(1)) if m else -1


def main():
    clip_to_zone, clip_to_quarter, frame_to_zone, frame_to_quarter = build_maps()
    clip_to_kpt_frame = load_clip_to_kpt_frame()

    zones = ["Midfield", "Field ends"]
    player_err_by_zone = {z: [] for z in zones}
    kpt_err_by_zone = {z: [] for z in zones}
    n_frames_by_zone = {z: 0 for z in zones}
    n_skipped_by_zone = {z: 0 for z in zones}

    all_frame_ids = sorted(frame_number(p) for p in TEST_IMGS.glob("*.png"))
    player_clip_ids = sorted(frame_number(p) for p in PLAYER_ANN.glob("*.txt"))

    for q in QUARTERS:
        weights = Path(str(WEIGHTS_TMPL).format(q=q))
        print(f"=== {q}: loading {weights} ===")
        model = YOLO(str(weights))

        # --- player-projection error, tagged by clip zone ---
        clips_q = [c for c in player_clip_ids if clip_to_quarter.get(c) == q]
        for clip_num in clips_q:
            z = clip_to_zone.get(clip_num)
            if z is None:
                continue
            fidx = clip_to_kpt_frame.get(clip_num)
            ann_path = PLAYER_ANN / f"frame_{clip_num:03d}.txt"
            if fidx is None:
                continue
            img_path = TEST_IMGS / f"frame_{fidx:06d}.png"
            lbl_path = TEST_LABELS / f"frame_{fidx:06d}.txt"
            if not (img_path.exists() and lbl_path.exists() and ann_path.exists()):
                continue

            frame = cv2.imread(str(img_path))
            h, w = frame.shape[:2]
            gt_kpts = load_gt_keypoints(lbl_path, w, h)
            pred_kpts = filter_kpts(run_model_keypoints_raw(model, img_path), CONF)
            H_gt = fit_homography(gt_kpts)
            H_pred = fit_homography(pred_kpts)
            if H_gt is None or H_pred is None:
                continue

            footpoints = load_gt_player_footpoints(ann_path, w, h)
            p_gt = project(H_gt, footpoints)
            p_pred = project(H_pred, footpoints)
            dists = np.linalg.norm(p_gt - p_pred, axis=1)
            player_err_by_zone[z].extend(dists.tolist())

        # --- keypoint-only field error, tagged by frame zone ---
        frames_q = [f for f in all_frame_ids if frame_to_quarter.get(f) == q]
        for fidx in frames_q:
            z = frame_to_zone.get(fidx)
            if z is None:
                continue
            img_path = TEST_IMGS / f"frame_{fidx:06d}.png"
            lbl_path = TEST_LABELS / f"frame_{fidx:06d}.txt"
            if not (img_path.exists() and lbl_path.exists()):
                n_skipped_by_zone[z] += 1
                continue

            frame = cv2.imread(str(img_path))
            h, w = frame.shape[:2]
            gt_kpts = load_gt_keypoints(lbl_path, w, h)
            H_gt = fit_homography(gt_kpts)
            if H_gt is None:
                n_skipped_by_zone[z] += 1
                continue

            raw_pred = run_model_keypoints_raw(model, img_path)
            pred_kpts = filter_kpts(raw_pred, CONF)
            n_frames_by_zone[z] += 1

            common = [kid for kid in gt_kpts if kid in pred_kpts]
            if not common:
                continue
            pred_px = [pred_kpts[kid] for kid in common]
            p_pred_field = project(H_gt, pred_px)
            true_field = np.array([FIELD_POSITIONS[kid] for kid in common])
            dists = np.linalg.norm(p_pred_field - true_field, axis=1)
            kpt_err_by_zone[z].extend(dists.tolist())

    results = {}
    print("\n" + "=" * 70)
    print("PER-ZONE FIELD-REGISTRATION ACCURACY (pooled across Q1-Q4, deployed finetune-init)")
    print("=" * 70)
    for z in zones:
        pe = player_err_by_zone[z]
        ke = kpt_err_by_zone[z]
        results[z] = {
            "player_projection": {
                "n_players": len(pe),
                "mean_err_yd": round(float(np.mean(pe)), 3) if pe else None,
                "med_err_yd": round(float(np.median(pe)), 3) if pe else None,
                "p90_err_yd": round(float(np.percentile(pe, 90)), 3) if pe else None,
            },
            "keypoint_field_error": {
                "n_points": len(ke),
                "n_frames_used": n_frames_by_zone[z],
                "mean_err_yd": round(float(np.mean(ke)), 3) if ke else None,
                "med_err_yd": round(float(np.median(ke)), 3) if ke else None,
                "p90_err_yd": round(float(np.percentile(ke, 90)), 3) if ke else None,
            },
        }
        pp, kf = results[z]["player_projection"], results[z]["keypoint_field_error"]
        print(f"\n{z}:")
        print(f"  Player projection  (n={pp['n_players']}): mean={pp['mean_err_yd']}yd "
              f"med={pp['med_err_yd']}yd p90={pp['p90_err_yd']}yd")
        print(f"  Keypoint field err (n={kf['n_points']}): mean={kf['mean_err_yd']}yd "
              f"med={kf['med_err_yd']}yd p90={kf['p90_err_yd']}yd")

    out = HERE / "zone_eval_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
