"""
Drawing and visualization utilities for the play analysis pipeline.

Provides:
  - create_football_field()   matplotlib bird's-eye field canvas
  - draw_birdseye_cards()     plot projected player positions on the field
  - annotate()                draw SAM3 masks + labels on a frame
  - create_class_color_palette()
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import supervision as sv
from PIL import Image
from typing import Dict, List, Optional


# ── Position colours ──────────────────────────────────────────────────────────
POSITION_COLOR_MAP = {
    "qb":      "#FFFF00",   # yellow
    "skill":   "#FF8C00",   # orange
    "oline":   "#FF69B4",   # pink
    "defense": "#00BFFF",   # blue
    "ref":     "#32CD32",   # green
}

MARKER_SHAPE = {
    "qb":      "*",
    "skill":   "o",
    "oline":   "s",
    "defense": "^",
    "ref":     "D",
}


def create_class_color_palette(yolo_names: Dict[int, str]) -> sv.ColorPalette:
    hex_colors = [
        POSITION_COLOR_MAP.get(yolo_names.get(i, "").lower(), "#808080")
        for i in range(len(yolo_names))
    ]
    return sv.ColorPalette.from_hex(hex_colors)


def annotate(
    image: Image.Image,
    detections: sv.Detections,
    labels: Optional[List[str]] = None,
    class_color_palette: Optional[sv.ColorPalette] = None,
) -> Image.Image:
    """Draw SAM3 masks and optional labels onto a PIL frame."""
    palette = class_color_palette or sv.ColorPalette.DEFAULT
    mask_annotator = sv.MaskAnnotator(
        color=palette, color_lookup=sv.ColorLookup.CLASS, opacity=0.6
    )
    out = mask_annotator.annotate(image.copy(), detections)
    if labels:
        label_annotator = sv.LabelAnnotator(
            color=palette, color_lookup=sv.ColorLookup.CLASS,
            text_color=sv.Color.BLACK, text_padding=10,
        )
        out = label_annotator.annotate(out, detections, labels)
    return out


# ── Bird's-eye field ──────────────────────────────────────────────────────────

def create_football_field(
    highlight_line: bool = False,
    highlight_line_number: int = 50,
    highlighted_name: str = "Line of Scrimmage",
    figsize: tuple = (12, 6.33),
    ax: Optional[plt.Axes] = None,
) -> tuple:
    """
    Create a matplotlib figure of an NFL football field (top-down view).

    Pass an existing `ax` to draw the field into it (e.g. as one panel of a
    larger figure) instead of creating a new standalone figure.

    Returns (fig, ax). y=0 is the top sideline, matching image convention.
    """
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    else:
        fig = ax.figure

    # Field background
    ax.add_patch(patches.Rectangle(
        (0, 0), 120, 53.3,
        linewidth=0.1, edgecolor="r", facecolor="darkgreen", zorder=0,
    ))

    # End zones
    for x0 in (0, 110):
        ax.add_patch(patches.Rectangle(
            (x0, 0), 10, 53.3,
            linewidth=0.1, edgecolor="r", facecolor="blue", alpha=0.2, zorder=0,
        ))

    # Yard lines
    ax.plot(
        [10,10,20,20,30,30,40,40,50,50,60,60,70,70,80,80,90,90,100,100,110,110,0,0,120,120],
        [0,53.3,53.3,0,0,53.3,53.3,0,0,53.3,53.3,0,0,53.3,53.3,0,0,53.3,53.3,0,0,53.3,53.3,0,0,53.3],
        color="white", linewidth=0.8,
    )

    # Yard numbers (NCAA: 9 yards from each sideline)
    for x in range(20, 110, 10):
        n = x if x <= 50 else 120 - x
        ax.text(x, 9,      str(n - 10), ha="center", fontsize=14, color="white")
        ax.text(x, 44.3,   str(n - 10), ha="center", fontsize=14, color="white", rotation=180)

    # Hash marks (NCAA: 60 feet = 20 yards from each sideline)
    y_hash_top = 20.0
    y_hash_bot = 53.3 - 20.0  # 33.3
    for x in range(11, 110):
        for yh in (y_hash_top, y_hash_bot):
            ax.plot([x, x], [yh - 0.3, yh + 0.3], color="white", linewidth=0.6)
        ax.plot([x, x], [0.2, 0.5],    color="white", linewidth=0.6)
        ax.plot([x, x], [52.8, 53.1],  color="white", linewidth=0.6)

    # Optional highlighted line
    if highlight_line:
        hl = highlight_line_number + 10
        ax.plot([hl, hl], [0, 53.3], color="yellow", linewidth=1.5)
        ax.text(hl + 1, 50, f"← {highlighted_name}", color="yellow", fontsize=9)

    ax.set_xlim(0, 120)
    ax.set_ylim(58.3, -5)
    ax.axis("off")
    return fig, ax


def draw_birdseye_cards(
    ax: plt.Axes,
    player_positions: List[tuple],
    class_ids: List[int],
    yolo_names: Dict[int, str],
    player_ids: Optional[List[int]] = None,
    marker_size: int = 120,
):
    """
    Plot projected player positions onto a bird's-eye field axes.

    Parameters
    ----------
    ax              : axes from create_football_field()
    player_positions: list of (x_yd, y_yd) in field yards
    class_ids       : YOLO class ID per player
    yolo_names      : {class_id: class_name}
    player_ids      : optional tracker IDs for labelling
    marker_size     : scatter marker size
    """
    for i, (pos, cls_id) in enumerate(zip(player_positions, class_ids)):
        x_yd, y_yd = pos
        name  = yolo_names.get(cls_id, "unknown").lower()
        color = POSITION_COLOR_MAP.get(name, "#808080")
        shape = MARKER_SHAPE.get(name, "o")

        ax.scatter(x_yd, y_yd, c=color, marker=shape,
                   s=marker_size, zorder=5, edgecolors="white", linewidths=0.5)

        label = name if player_ids is None else f"{name}\n#{player_ids[i]}"
        ax.text(x_yd + 0.4, y_yd, label, fontsize=6.5, color="white",
                va="center", zorder=6)


def draw_keypoint_overlay(
    ax_img: plt.Axes,
    kpts_data: np.ndarray,
    conf_thresh: float = 0.25,
):
    """
    Overlay detected keypoints on an image axes (for debugging / thesis figures).

    Parameters
    ----------
    ax_img     : matplotlib axes showing the frame image
    kpts_data  : (62, 3) array [x, y, conf]
    conf_thresh: minimum confidence to draw
    """
    from homography import kpt_world
    for idx, (x, y, conf) in enumerate(kpts_data):
        if conf < conf_thresh:
            continue
        kpt_id = idx + 1
        ax_img.scatter(x, y, c="lime", s=20, zorder=5)
        ax_img.text(x + 4, y, str(kpt_id), fontsize=5, color="lime", zorder=6)
