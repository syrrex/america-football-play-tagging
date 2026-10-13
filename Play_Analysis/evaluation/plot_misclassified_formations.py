"""
Renders one diagnostic figure per misclassified formation clip, for the
GT-input and self-detected configurations separately.

Each figure: left panel is the presnap frame with the player boxes used for
that configuration drawn and colored by role; right panel is the bird's-eye
projection of the same players plus the computed tackle box, with the true
and predicted formation in the title.

Only rerurns detection/keypoint models on the clips that were actually
misclassified (read from the existing formation_eval*.json files), not the
full 107-clip corpus.

Usage (needs ultralytics, e.g. `conda activate SAM3_ENV`):
    python plot_misclassified_formations.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PLAY_ANALYSIS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLAY_ANALYSIS))
sys.path.insert(0, str(PLAY_ANALYSIS.parent / "Pipeline"))

from classify_formation import (  # noqa: E402
    META_CSV, GT_IMG_DIR, GT_ANN_DIR, GT_KPT_DIR,
    FIELD_KPT_BASE, PLAYER_DET_WEIGHTS, DET_CONF, KPT_CONF, IMGSZ,
    load_gt_boxes, load_gt_keypoints, compute_tackle_box,
)
from homography import compute_homography_from_keypoints, project_to_field_yards  # noqa: E402
from drawing import create_football_field, draw_birdseye_cards, POSITION_COLOR_MAP  # noqa: E402

OUT_ROOT = Path(__file__).parent / "evaluation_output"
YOLO_NAMES = {0: "defense", 1: "oline", 2: "qb", 3: "ref", 4: "skill"}

CONFIGS = {
    "gt": dict(
        json_file=PLAY_ANALYSIS / "formation_eval.json",
        box_source="gt", kpt_source="gt",
        out_dir=OUT_ROOT / "misclassified_gt",
    ),
    "selfdetected": dict(
        json_file=PLAY_ANALYSIS / "formation_eval_selfdetected.json",
        box_source="detected", kpt_source="detected",
        out_dir=OUT_ROOT / "misclassified_selfdetected",
    ),
}


def get_homography(row, w, h, kpt_source, kpt_models):
    if kpt_source == "gt":
        kpt_path = GT_KPT_DIR / f"frame_{int(row['start_frame']):06d}.txt"
        if not kpt_path.exists():
            return None
        kpts = load_gt_keypoints(kpt_path, w, h)
        H, _ = compute_homography_from_keypoints(kpts, conf_thresh=1.0, min_points=4, img_shape=(h, w))
    else:
        img_path = GT_IMG_DIR / f"frame_{int(row['play_num']):03d}.jpg"
        kpt_model = kpt_models[row["QTR"]]
        kpt_res = kpt_model(str(img_path), imgsz=IMGSZ, conf=0.10, max_det=1, verbose=False)[0]
        if kpt_res.keypoints is None or len(kpt_res.keypoints.data) == 0:
            return None
        kpts_data = kpt_res.keypoints.data[0].cpu().numpy()
        H, _ = compute_homography_from_keypoints(kpts_data, conf_thresh=KPT_CONF, min_points=4, img_shape=(h, w))
    return H


def get_boxes(row, box_source, w, h, player_model):
    if box_source == "gt":
        ann_path = GT_ANN_DIR / f"frame_{int(row['play_num']):03d}.txt"
        if not ann_path.exists():
            return np.zeros((0, 4)), np.array([], dtype=int)
        boxes = load_gt_boxes(ann_path, w, h)
        xyxy = np.array([[x1, y1, x2, y2] for _, x1, y1, x2, y2 in boxes]) if boxes else np.zeros((0, 4))
        cids = np.array([cls for cls, *_ in boxes], dtype=int)
    else:
        img_path = GT_IMG_DIR / f"frame_{int(row['play_num']):03d}.jpg"
        det_res = player_model(str(img_path), conf=DET_CONF, iou=0.5, imgsz=IMGSZ, verbose=False)[0]
        if det_res.boxes is not None and len(det_res.boxes) > 0:
            xyxy = det_res.boxes.xyxy.cpu().numpy()
            cids = det_res.boxes.cls.cpu().numpy().astype(int)
        else:
            xyxy, cids = np.zeros((0, 4)), np.array([], dtype=int)
    return xyxy, cids


def draw_frame_panel(ax, img_rgb, xyxy, cids):
    ax.imshow(img_rgb)
    for (x1, y1, x2, y2), cid in zip(xyxy, cids):
        name = YOLO_NAMES.get(int(cid), "unknown")
        color = POSITION_COLOR_MAP.get(name, "#808080")
        ax.add_patch(plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor=color, linewidth=1.6,
        ))
        ax.text(x1, max(y1 - 4, 0), name, fontsize=6, color=color,
                 bbox=dict(facecolor="black", alpha=0.55, pad=0.5, linewidth=0))
    ax.axis("off")
    ax.set_title("Presnap frame + detections", fontsize=10)


def draw_field_panel(ax, oline_pos, skill_pos, qb_pos, box_lo, box_hi, los_x):
    create_football_field(ax=ax)
    positions = oline_pos + skill_pos + qb_pos
    cids = [1] * len(oline_pos) + [4] * len(skill_pos) + [2] * len(qb_pos)
    if positions:
        draw_birdseye_cards(ax, positions, cids, YOLO_NAMES)
    ax.plot([los_x - 3, los_x + 3], [box_lo, box_lo], color="cyan", linewidth=1.2, linestyle="--", zorder=4)
    ax.plot([los_x - 3, los_x + 3], [box_hi, box_hi], color="cyan", linewidth=1.2, linestyle="--", zorder=4)
    ax.set_title("Bird's-eye projection + tackle box", fontsize=10)


def run_config(name, cfg):
    with open(cfg["json_file"]) as f:
        results = json.load(f)
    mis = [p for p in results["predictions"] if p["true_form"] != p["pred_form"]]
    print(f"[{name}] {len(mis)} misclassified clips")
    if not mis:
        return

    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    box_source, kpt_source = cfg["box_source"], cfg["kpt_source"]

    player_model = None
    kpt_models = None
    if box_source == "detected" or kpt_source == "detected":
        from ultralytics import YOLO
        if box_source == "detected":
            player_model = YOLO(str(PLAYER_DET_WEIGHTS))
        if kpt_source == "detected":
            kpt_models = {q: YOLO(str(FIELD_KPT_BASE / q / "train" / "weights" / "best.pt"))
                          for q in ["Q1", "Q2", "Q3", "Q4"]}

    meta = pd.read_csv(META_CSV).set_index("clip")
    mis_clips = {m["clip"] for m in mis}
    mis_by_clip = {m["clip"]: m for m in mis}

    for clip in mis_clips:
        row = meta.loc[clip]
        pnum = int(row["play_num"])
        img_path = GT_IMG_DIR / f"frame_{pnum:03d}.jpg"
        if not img_path.exists():
            print(f"  skip {clip}: no frame")
            continue
        img_bgr = cv2.imread(str(img_path))
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        H = get_homography(row, w, h, kpt_source, kpt_models)
        xyxy, cids = get_boxes(row, box_source, w, h, player_model)
        if H is None:
            print(f"  skip {clip}: no homography")
            continue

        positions = project_to_field_yards(xyxy, H) if len(xyxy) else []
        oline_pos = [p for p, c in zip(positions, cids) if c == 1]
        skill_pos = [p for p, c in zip(positions, cids) if c == 4]
        qb_pos = [p for p, c in zip(positions, cids) if c == 2]

        if len(oline_pos) < 3:
            los_x, box_lo, box_hi = (np.mean([p[0] for p in oline_pos]) if oline_pos else 0.0), 0.0, 0.0
        else:
            los_x, box_lo, box_hi = compute_tackle_box(oline_pos)

        pred = mis_by_clip[clip]
        fig, (ax_img, ax_field) = plt.subplots(1, 2, figsize=(16, 6.5))
        draw_frame_panel(ax_img, img_rgb, xyxy, cids)
        draw_field_panel(ax_field, oline_pos, skill_pos, qb_pos, box_lo, box_hi, los_x)
        fig.suptitle(f"{clip}   true={pred['true_form']}   pred={pred['pred_form']}", fontsize=13)
        fig.tight_layout()

        out_path = cfg["out_dir"] / f"{Path(clip).stem}_true-{pred['true_form']}_pred-{pred['pred_form']}.png"
        fig.savefig(out_path, dpi=150, facecolor="white")
        plt.close(fig)
        print(f"  saved {out_path.name}")


def main():
    for name, cfg in CONFIGS.items():
        run_config(name, cfg)


if __name__ == "__main__":
    main()
