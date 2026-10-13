"""
Keypoint-based homography for football field mapping.

Maps the 62 field landmarks (1-based IDs 1–62) detected by the YOLO pose model
to their known real-world positions on the canonical top-down field, then
computes a perspective homography for bird's-eye view warping and coordinate
projection.
"""

import cv2
import numpy as np
from pathlib import Path

# ── Field dimensions (NFL) ─────────────────────────────────────────────────────
FIELD_YD_W  = 120.0
FIELD_YD_H  = 53 + 1 / 3      # 53.333... yards
PX_PER_YARD = 10
CANVAS_W    = int(FIELD_YD_W * PX_PER_YARD)           # 1200 px
CANVAS_H    = int(round(FIELD_YD_H * PX_PER_YARD))    # 533 px

# ── y-positions (yards from top sideline, NFL standard) ───────────────────────
Y_SIDELINE_TOP = 0.0
Y_SIDELINE_BOT = FIELD_YD_H           # 53.333
Y_NUMBER_TOP   = 8.0                  # yard numbers: centre of drawn text (top of numeral 9 yd
                                       # from sideline, minus half of the 6 ft/2 yd max numeral height)
Y_NUMBER_BOT   = FIELD_YD_H - 8.0    # 45.333
Y_HASH_TOP     = 20                   # hashes 20 yards away from sideline
Y_HASH_BOT     = FIELD_YD_H - 20      #

# ── x-positions for the two keypoint groups ────────────────────────────────────
# 13-point groups (sidelines): left end line → midfield → right end line
X_SIDELINE = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
# 9-point groups (numbers, hashes): 10→50→10 yd lines inside the end zones
X_NINE = [20, 30, 40, 50, 60, 70, 80, 90, 100]

# ── Build world-coordinate lookup (0-based YOLO index → (x_yd, y_yd)) ─────────
kpt_world: dict[int, tuple[float, float]] = {}

for _i, _x in enumerate(X_SIDELINE):
    kpt_world[1  + _i]  = (float(_x), Y_SIDELINE_TOP)  # 1–13  : top sideline
    kpt_world[14 + _i]  = (float(_x), Y_SIDELINE_BOT)  # 14–26 : bottom sideline

for _i, _x in enumerate(X_NINE):
    kpt_world[27 + _i]  = (float(_x), Y_NUMBER_TOP)    # 27–35 : top numbers
    kpt_world[36 + _i]  = (float(_x), Y_NUMBER_BOT)    # 36–44 : bottom numbers
    kpt_world[45 + _i]  = (float(_x), Y_HASH_TOP)      # 45–53 : top hashes
    kpt_world[54 + _i]  = (float(_x), Y_HASH_BOT)      # 54–62 : bottom hashes

# Canvas coordinates in pixels (x_yd * PX_PER_YARD, y_yd * PX_PER_YARD)
kpt_canvas: dict[int, tuple[float, float]] = {
    idx: (x * PX_PER_YARD, y * PX_PER_YARD)
    for idx, (x, y) in kpt_world.items()
}


# ── Core functions ─────────────────────────────────────────────────────────────

def compute_homography_from_keypoints(
    kpts_data: np.ndarray,
    conf_thresh: float = 0.25,
    min_points: int = 4,
    img_shape: tuple | None = None,
) -> tuple[np.ndarray | None, int]:
    """
    Compute H mapping full-resolution image pixels → bird's-eye canvas pixels.

    Parameters
    ----------
    kpts_data   : (62, 3) float array [x, y, confidence] in full-res image px
    conf_thresh : minimum keypoint confidence to use
    min_points  : minimum correspondences required
    img_shape   : optional (h, w) — keypoints outside the image are ignored

    Returns
    -------
    H          : 3×3 homography (image px → canvas px), or None if failed
    n_inliers  : number of RANSAC inliers (0 if failed)
    """
    src_pts, dst_pts = [], []

    for idx, (x, y, conf) in enumerate(kpts_data):
        kpt_id = idx + 1  # YOLO array is 0-based; keypoints are 1-based (1–62)
        if conf < conf_thresh:
            continue
        if kpt_id not in kpt_canvas:
            continue
        if img_shape is not None:
            h, w = img_shape
            if not (0 <= x < w and 0 <= y < h):
                continue
        src_pts.append([float(x), float(y)])
        dst_pts.append(list(kpt_canvas[kpt_id]))

    if len(src_pts) < min_points:
        return None, 0

    src_arr = np.array(src_pts, dtype=np.float64)
    dst_arr = np.array(dst_pts, dtype=np.float64)

    H, mask = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, ransacReprojThreshold=8.0)
    n_inliers = int(mask.sum()) if mask is not None else len(src_pts)
    return H, n_inliers


def derive_H_for_resized(H_kpt_warp: np.ndarray, resize_ratio: float) -> np.ndarray:
    """
    Derive a homography compatible with the existing pipeline.

    The existing pipeline calls `project_players_to_field` which scales
    bounding-box coordinates by `resize_ratio` before applying H.  This
    function converts H_kpt_warp (full-res px → canvas px) into H_compat
    (resized-image px → field yards), so all downstream code is unchanged.

    Math:
        H_compat = S_yd  @ H_kpt_warp @ S_inv_r
    where
        S_inv_r  = diag(1/r, 1/r, 1)  (undo the resize scaling)
        S_yd     = diag(1/P, 1/P, 1)  (convert canvas px to yards)
    """
    P = float(PX_PER_YARD)
    r = float(resize_ratio)
    S_inv_r = np.diag([1.0 / r, 1.0 / r, 1.0])
    S_yd    = np.diag([1.0 / P, 1.0 / P, 1.0])
    return S_yd @ H_kpt_warp @ S_inv_r


def project_to_field_yards(
    bboxes_xyxy: np.ndarray,
    H_kpt_warp: np.ndarray,
) -> list[tuple[float, float]]:
    """
    Project player bounding-box foot positions to field yard coordinates.

    Parameters
    ----------
    bboxes_xyxy  : (N, 4) [x1, y1, x2, y2] in full-resolution pixels
    H_kpt_warp   : 3×3 homography (full-res image px → canvas px)

    Returns
    -------
    List of (x_yd, y_yd) in absolute field yards (0–120, 0–53.33)
    """
    if len(bboxes_xyxy) == 0:
        return []

    foot_x = (bboxes_xyxy[:, 0] + bboxes_xyxy[:, 2]) / 2.0
    foot_y = bboxes_xyxy[:, 3]
    pts = np.stack([foot_x, foot_y], axis=1).astype(np.float64)

    canvas_pts = cv2.perspectiveTransform(pts.reshape(1, -1, 2), H_kpt_warp).reshape(-1, 2)
    return [(float(px / PX_PER_YARD), float(py / PX_PER_YARD)) for px, py in canvas_pts]


def run_field_pose_model(
    frame_bgr: np.ndarray,
    model,
    det_conf: float = 0.25,
    kpt_conf: float = 0.25,
    min_points: int = 4,
) -> tuple[np.ndarray | None, int]:
    """
    Run the YOLO field-pose model on a BGR frame and return the homography.

    Parameters
    ----------
    frame_bgr  : full-resolution BGR image (numpy array)
    model      : loaded ultralytics.YOLO instance
    det_conf   : detection confidence threshold
    kpt_conf   : keypoint confidence threshold
    min_points : minimum keypoints required to compute H

    Returns
    -------
    H_kpt_warp : 3×3 homography (full-res → canvas px), or None
    n_inliers  : RANSAC inlier count
    """
    results = model(frame_bgr, conf=det_conf, iou=0.5, max_det=1, verbose=False)
    result = results[0]

    if result.keypoints is None or len(result.keypoints.data) == 0:
        return None, 0

    kpts_data = result.keypoints.data[0].cpu().numpy()  # (62, 3)
    return compute_homography_from_keypoints(
        kpts_data,
        conf_thresh=kpt_conf,
        min_points=min_points,
        img_shape=frame_bgr.shape[:2],
    )
