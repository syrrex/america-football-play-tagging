"""
Pipeline evaluation — reprojection error per clip across all folds.

For each fold, uses the matching fold model (no data leakage) and runs
keypoint detection + homography on every clip in that fold's val set.
Results are saved to a CSV with clip metadata (yard line, play type, etc.)
so you can analyse the correlation between camera angle and model accuracy.

Usage:
  python evaluate_pipeline.py
  python evaluate_pipeline.py --out pipeline_eval.csv
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from homography import run_field_pose_model, kpt_world, PX_PER_YARD

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent
DATA_ROOT   = BASE.parent / "Data"
FRAMES_ROOT = DATA_ROOT / "Keypoints" / "frames" / "tb2026"
META_CSV    = DATA_ROOT / "metadata" / "tb2026_dataset_metadata.csv"
KP_BASE     = BASE.parent / "Field_Keypoint_Detection" / "upper_bound_results"

KP_DET_CONF = 0.25
KP_KPT_CONF = 0.25
IMGSZ       = 1280

FOLD_MODEL = {
    0: KP_BASE / "fold_0" / "train" / "weights" / "best.pt",
    1: KP_BASE / "fold_1" / "train" / "weights" / "best.pt",
    2: KP_BASE / "fold_2" / "train" / "weights" / "best.pt",
    3: KP_BASE / "fold_3" / "train" / "weights" / "best.pt",
}

FOLD_CLIPS = {
    0: ["clip_008.mp4","clip_018.mp4","clip_024.mp4","clip_037.mp4","clip_043.mp4",
        "clip_044.mp4","clip_048.mp4","clip_056.mp4","clip_064.mp4","clip_069.mp4",
        "clip_078.mp4","clip_081.mp4","clip_082.mp4","clip_093.mp4","clip_094.mp4",
        "clip_101.mp4","clip_105.mp4","clip_107.mp4","clip_113.mp4","clip_135.mp4",
        "clip_144.mp4"],
    1: ["clip_013.mp4","clip_017.mp4","clip_019.mp4","clip_031.mp4","clip_041.mp4",
        "clip_055.mp4","clip_058.mp4","clip_059.mp4","clip_061.mp4","clip_065.mp4",
        "clip_079.mp4","clip_084.mp4","clip_087.mp4","clip_092.mp4","clip_095.mp4",
        "clip_100.mp4","clip_108.mp4","clip_114.mp4","clip_115.mp4","clip_133.mp4",
        "clip_141.mp4"],
    2: ["clip_009.mp4","clip_012.mp4","clip_016.mp4","clip_025.mp4","clip_028.mp4",
        "clip_032.mp4","clip_033.mp4","clip_034.mp4","clip_045.mp4","clip_047.mp4",
        "clip_050.mp4","clip_053.mp4","clip_054.mp4","clip_060.mp4","clip_068.mp4",
        "clip_088.mp4","clip_089.mp4","clip_097.mp4","clip_099.mp4","clip_121.mp4",
        "clip_128.mp4"],
    3: ["clip_007.mp4","clip_021.mp4","clip_029.mp4","clip_030.mp4","clip_035.mp4",
        "clip_057.mp4","clip_063.mp4","clip_070.mp4","clip_080.mp4","clip_098.mp4",
        "clip_103.mp4","clip_112.mp4","clip_117.mp4","clip_119.mp4","clip_120.mp4",
        "clip_122.mp4","clip_123.mp4","clip_127.mp4","clip_129.mp4","clip_134.mp4",
        "clip_137.mp4"],
}


def eval_clip(model, frame_paths: list[Path]) -> dict:
    errors = []
    valid_h = 0

    for fp in frame_paths:
        frame_bgr = cv2.imread(str(fp))
        if frame_bgr is None:
            continue

        H, _ = run_field_pose_model(frame_bgr, model,
                                    det_conf=KP_DET_CONF, kpt_conf=KP_KPT_CONF)
        if H is None:
            continue
        valid_h += 1

        res = model(frame_bgr, imgsz=IMGSZ, conf=KP_DET_CONF,
                    max_det=1, verbose=False)[0]
        if res.keypoints is None or len(res.keypoints.data) == 0:
            continue

        kpts = res.keypoints.data[0].cpu().numpy()
        for idx, (x, y, conf) in enumerate(kpts):
            kpt_id = idx + 1
            if conf < KP_KPT_CONF or kpt_id not in kpt_world:
                continue
            pt_pred = cv2.perspectiveTransform(
                np.array([[[float(x), float(y)]]], dtype=np.float64), H
            )[0, 0]
            gt_x, gt_y = kpt_world[kpt_id]
            pt_gt = np.array([gt_x * PX_PER_YARD, gt_y * PX_PER_YARD])
            errors.append(float(np.linalg.norm(pt_pred - pt_gt) / PX_PER_YARD))

    n = len(frame_paths)
    return {
        "n_frames":          n,
        "valid_homographies": valid_h,
        "homography_rate":   round(valid_h / n, 4) if n else 0,
        "n_keypoints":       len(errors),
        "mean_err_yards":    round(float(np.mean(errors)),   3) if errors else None,
        "median_err_yards":  round(float(np.median(errors)), 3) if errors else None,
        "p90_err_yards":     round(float(np.percentile(errors, 90)), 3) if errors else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(BASE / "pipeline_eval.csv"))
    parser.add_argument("--fold", type=int, default=None,
                        help="Evaluate a single fold (default: all)")
    args = parser.parse_args()

    meta = pd.read_csv(META_CSV)
    rows = []

    folds = [args.fold] if args.fold is not None else list(FOLD_CLIPS.keys())

    for fold in folds:
        print(f"\n{'='*55}")
        print(f"  Fold {fold}  —  {len(FOLD_CLIPS[fold])} clips")
        print(f"{'='*55}")

        model = YOLO(str(FOLD_MODEL[fold]))

        for clip_name in FOLD_CLIPS[fold]:
            clip_row = meta[meta["clip"] == clip_name]
            if clip_row.empty:
                print(f"  [skip] {clip_name} not in metadata")
                continue
            clip_row = clip_row.iloc[0]

            start = int(clip_row["start_frame"])
            end   = int(clip_row["end_frame"])
            frame_paths = [FRAMES_ROOT / f"frame_{i:06d}.png"
                           for i in range(start, end + 1)]

            print(f"  {clip_name}  yard={clip_row['YARD_LN']}  "
                  f"dn={clip_row['DN']}  type={clip_row['PLAY_TYPE']} ...", end=" ", flush=True)

            stats = eval_clip(model, frame_paths)
            print(f"H={stats['homography_rate']*100:.0f}%  "
                  f"err={stats['mean_err_yards']} yd")

            rows.append({
                "fold":             fold,
                "clip":             clip_name,
                "yard_ln":          clip_row["YARD_LN"],
                "hash":             clip_row["HASH"],
                "dn":               clip_row["DN"],
                "dist":             clip_row["DIST"],
                "play_type":        clip_row["PLAY_TYPE"],
                "odk":              clip_row["ODK"],
                **stats,
            })

        del model

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")

    # Quick summary
    print("\nCorrelation with yard line (distance from midfield = |yard_ln - 50|):")
    df["dist_from_mid"] = (df["yard_ln"] - 50).abs()
    valid = df.dropna(subset=["mean_err_yards"])
    corr = valid[["dist_from_mid", "mean_err_yards",
                  "homography_rate"]].corr()["dist_from_mid"]
    print(corr.to_string())


if __name__ == "__main__":
    main()
