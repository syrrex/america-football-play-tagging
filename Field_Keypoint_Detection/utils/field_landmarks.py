"""
American Football Field Landmark Points

Layout (viewed from above, left-to-right):
  Left EZ | 10 | 20 | 30 | 40 | 50 | 40 | 30 | 20 | 10 | Right EZ

Point numbering:
  1–13   : Top sideline,    left end line → right end line (13 points)
  14–26  : Bottom sideline, left end line → right end line (13 points)
  27–35  : Top yard numbers,    10→50→10  (9 points)
  36–44  : Bottom yard numbers, 10→50→10  (9 points)
  45–53  : Top hash marks,      10→50→10  (9 points)
  54–62  : Bottom hash marks,   10→50→10  (9 points)
"""

from dataclasses import dataclass
from enum import Enum


class LandmarkType(Enum):
    SIDELINE_TOP = "sideline_top"
    SIDELINE_BOTTOM = "sideline_bottom"
    NUMBER_TOP = "number_top"
    NUMBER_BOTTOM = "number_bottom"
    HASH_TOP = "hash_top"
    HASH_BOTTOM = "hash_bottom"


@dataclass
class FieldLandmark:
    point_id: int          # 1-based landmark index used by YOLO
    landmark_type: LandmarkType
    yard_line: int | None  # actual yard line (0–50), None for end lines
    side: str              # "left" or "right" or "center"
    description: str


# ---------------------------------------------------------------------------
# Build the full landmark table
# ---------------------------------------------------------------------------

def _sideline_landmarks(start_id: int, landmark_type: LandmarkType) -> list[FieldLandmark]:
    """13 points along one sideline: end-line, goal-line, 10…50…10, goal-line, end-line."""
    entries = [
        (None,  "left",    "left end line"),
        (0,     "left",    "left goal line (0 yd)"),
        (10,    "left",    "10 yd line"),
        (20,    "left",    "20 yd line"),
        (30,    "left",    "30 yd line"),
        (40,    "left",    "40 yd line"),
        (50,    "center",  "50 yd line (midfield)"),
        (40,    "right",   "40 yd line"),
        (30,    "right",   "30 yd line"),
        (20,    "right",   "20 yd line"),
        (10,    "right",   "10 yd line"),
        (0,     "right",   "right goal line (0 yd)"),
        (None,  "right",   "right end line"),
    ]
    side_label = "top" if "top" in landmark_type.value else "bottom"
    return [
        FieldLandmark(start_id + i, landmark_type, yd, side, f"{side_label} sideline – {desc}")
        for i, (yd, side, desc) in enumerate(entries)
    ]


def _nine_point_landmarks(start_id: int, landmark_type: LandmarkType) -> list[FieldLandmark]:
    """9 points at the major yard lines: 10, 20, 30, 40, 50, 40, 30, 20, 10."""
    entries = [
        (10, "left"),
        (20, "left"),
        (30, "left"),
        (40, "left"),
        (50, "center"),
        (40, "right"),
        (30, "right"),
        (20, "right"),
        (10, "right"),
    ]
    type_label = landmark_type.value.replace("_", " ")
    return [
        FieldLandmark(start_id + i, landmark_type, yd, side, f"{type_label} – {yd} yd ({side})")
        for i, (yd, side) in enumerate(entries)
    ]


ALL_LANDMARKS: list[FieldLandmark] = (
    _sideline_landmarks(1,  LandmarkType.SIDELINE_TOP)      # 1–13
    + _sideline_landmarks(14, LandmarkType.SIDELINE_BOTTOM) # 14–26
    + _nine_point_landmarks(27, LandmarkType.NUMBER_TOP)    # 27–35
    + _nine_point_landmarks(36, LandmarkType.NUMBER_BOTTOM) # 36–44
    + _nine_point_landmarks(45, LandmarkType.HASH_TOP)      # 45–53
    + _nine_point_landmarks(54, LandmarkType.HASH_BOTTOM)   # 54–62
)

# Quick lookup by point_id (1-based)
LANDMARK_BY_ID: dict[int, FieldLandmark] = {lm.point_id: lm for lm in ALL_LANDMARKS}


def get_landmark(point_id: int) -> FieldLandmark:
    """Return the FieldLandmark for a given 1-based YOLO keypoint index."""
    if point_id not in LANDMARK_BY_ID:
        raise KeyError(f"No landmark with point_id={point_id}. Valid range: 1–62.")
    return LANDMARK_BY_ID[point_id]


# ---------------------------------------------------------------------------
# Real-world field coordinates (yards), for homography estimation
# ---------------------------------------------------------------------------
# Field x: 0=left end line, 120=right end line (includes 10-yd end zones each side)
# Field y: 0=top sideline, 53.3=bottom sideline
SIDELINE_X = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
NINE_PT_X  = [20, 30, 40, 50, 60, 70, 80, 90, 100]

Y_SIDELINE_TOP    = 0.0
Y_SIDELINE_BOTTOM = 53.3
Y_NUMBER_TOP      = 8.0    # top of numeral 9 yd from sideline, minus half of the 6 ft/2 yd max height
Y_NUMBER_BOTTOM   = 45.3    # 53.3 - 8
Y_HASH_TOP        = 20.0
Y_HASH_BOTTOM     = 33.3    # 53.3 - 20.0


def _field_positions() -> dict[int, tuple[float, float]]:
    """Return {point_id: (x_yd, y_yd)} for all 62 keypoints on the field template."""
    positions = {}
    for lm in ALL_LANDMARKS:
        pid = lm.point_id
        if lm.landmark_type == LandmarkType.SIDELINE_TOP:
            x, y = SIDELINE_X[pid - 1], Y_SIDELINE_TOP
        elif lm.landmark_type == LandmarkType.SIDELINE_BOTTOM:
            x, y = SIDELINE_X[pid - 14], Y_SIDELINE_BOTTOM
        elif lm.landmark_type == LandmarkType.NUMBER_TOP:
            x, y = NINE_PT_X[pid - 27], Y_NUMBER_TOP
        elif lm.landmark_type == LandmarkType.NUMBER_BOTTOM:
            x, y = NINE_PT_X[pid - 36], Y_NUMBER_BOTTOM
        elif lm.landmark_type == LandmarkType.HASH_TOP:
            x, y = NINE_PT_X[pid - 45], Y_HASH_TOP
        elif lm.landmark_type == LandmarkType.HASH_BOTTOM:
            x, y = NINE_PT_X[pid - 54], Y_HASH_BOTTOM
        positions[pid] = (x, y)
    return positions


# {point_id (1-based): (x_yd, y_yd)} — shared by visualize_field_keypoints.py and
# evaluate_homography.py so both agree on the same field template.
FIELD_POSITIONS: dict[int, tuple[float, float]] = _field_positions()


if __name__ == "__main__":
    for lm in ALL_LANDMARKS:
        print(f"  [{lm.point_id:2d}] {lm.landmark_type.value:<20s}  {lm.description}")
