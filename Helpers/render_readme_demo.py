"""
Renders the README demo (assets/demo.gif + assets/demo_presnap.png) for one
tb2026 clip from the pipeline's own self-detected outputs:

  - player boxes per frame, colored by predicted role, from the SAM-3
    self-detected tracking cache (Player_Tracking/output/tracking_cache_selfdetected/)
  - a bird's-eye minimap, projected through a per-frame homography from the
    deployed keypoint model of the fold that held this clip's quarter OUT of
    training (Field_Keypoint_Detection/finetune_quarter_results_1280/<QTR>/)
  - the three tags exactly as stored in Play_Analysis/*_eval_selfdetected.json,
    with the coach's ground-truth tag next to each

Formation and field position are pre-snap tags and are shown from frame 0.
Play type is classified from the whole tracked clip, so it is revealed only
on the last frame.

Usage:
  python render_readme_demo.py               # clip_108
  python render_readme_demo.py --clip clip_069
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "Pipeline"))
from drawing import POSITION_COLOR_MAP  # noqa: E402
from homography import FIELD_YD_H, FIELD_YD_W, project_to_field_yards, run_field_pose_model  # noqa: E402

RAW_DIR    = PROJECT_DIR / "Data" / "raw" / "tb2026"
META_CSV   = PROJECT_DIR / "Data" / "metadata" / "tb2026_dataset_metadata.csv"
CACHE_DIR  = PROJECT_DIR / "Player_Tracking" / "output" / "tracking_cache_selfdetected"
EVAL_DIR   = PROJECT_DIR / "Play_Analysis"
KP_RESULTS = PROJECT_DIR / "Field_Keypoint_Detection" / "finetune_quarter_results_1280"
ASSETS_DIR = PROJECT_DIR / "assets"

FONT_DIR = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"

VID_W, VID_H = 960, 540           # video panel
STRIP_H      = 240                # bottom strip: tag panel | bird's-eye field
TAG_W        = VID_W - int(FIELD_YD_W * STRIP_H / FIELD_YD_H)
HOLD_LAST_S  = 3.0
FPS          = 4                  # tb2026 clips are stored at 4 fps
MAX_STEP_YD  = 3.0                # max plausible per-frame (0.25 s) move drawn in a trail

BG      = (17, 24, 32)
FG      = (235, 238, 242)
MUTED   = (140, 150, 160)
OK      = (80, 200, 120)
BAD     = (230, 90, 80)
TURF    = (38, 104, 56)
LINE    = (225, 235, 225)


def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


ROLE_RGB = {k: hex_rgb(v) for k, v in POSITION_COLOR_MAP.items()}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size)


def load_tags(clip: str) -> dict:
    def pred(name):
        data = json.load(open(EVAL_DIR / f"{name}_eval_selfdetected.json"))
        return next(p for p in data["predictions"] if p["clip"] == f"{clip}.mp4")

    f, y, t = pred("formation"), pred("yard_hash"), pred("play_type")
    return {
        "formation": (f["pred_form"].upper(), f["true_form"].upper()),
        "yardline":  (y["pred_yardline"], y["true_yardline"]),
        "hash":      (y["pred_hash"], y["true_hash"]),
        "play_type": (t["pred_play_type"].upper(), t["true_play_type"].upper()),
    }


def fmt_yardline(v: float) -> str:
    side = "OPP" if v > 0 else "OWN"
    return f"{side} {abs(round(v))}"


def draw_field(w: int, h: int) -> tuple[Image.Image, float]:
    s = w / FIELD_YD_W
    img = Image.new("RGB", (w, h), TURF)
    d = ImageDraw.Draw(img)
    for ez in (0, 110):
        d.rectangle([ez * s, 0, (ez + 10) * s, h], fill=(30, 86, 46))
    for x in range(10, 111, 5):
        d.line([x * s, 0, x * s, h], fill=LINE if x % 10 == 0 else (120, 160, 125), width=1)
    for x in range(11, 110):
        for y in (20.0, FIELD_YD_H - 20.0):  # hash marks
            d.line([x * s, y * s - 1, x * s, y * s + 1], fill=(160, 190, 165))
    d.rectangle([0, 0, w - 1, h - 1], outline=LINE, width=2)
    return img, s


def render_frame(frame_bgr, det, names, H, tags, show_play_type, field_base, scale, trails):
    vid = cv2.resize(frame_bgr, (VID_W, VID_H))
    rx, ry = VID_W / frame_bgr.shape[1], VID_H / frame_bgr.shape[0]
    canvas = Image.new("RGB", (VID_W, VID_H + STRIP_H), BG)
    canvas.paste(Image.fromarray(cv2.cvtColor(vid, cv2.COLOR_BGR2RGB)), (0, 0))
    d = ImageDraw.Draw(canvas)

    # boxes on the video
    for (x1, y1, x2, y2), cid in zip(det["xyxy"], det["class_id"]):
        role = names[int(cid)]
        d.rectangle([x1 * rx, y1 * ry, x2 * rx, y2 * ry], outline=ROLE_RGB[role], width=2)

    # role legend (top-right of video)
    lf = font(13, bold=True)
    lx = VID_W - 10
    for role in ["ref", "defense", "skill", "qb", "oline"]:
        tw = d.textlength(role.upper(), font=lf)
        lx -= tw + 22
        d.rectangle([lx, 12, lx + 12, 24], fill=ROLE_RGB[role])
        d.text((lx + 16, 10), role.upper(), font=lf, fill=FG, stroke_width=2, stroke_fill=(0, 0, 0))

    # bird's-eye field
    field = field_base.copy()
    fd = ImageDraw.Draw(field)
    if H is not None and len(det["xyxy"]):
        pts = project_to_field_yards(np.asarray(det["xyxy"], dtype=float), H)
        for (xy, cid, tid) in zip(pts, det["class_id"], det["tracker_id"]):
            role = names[int(cid)]
            if role in ("qb", "skill"):
                trails.setdefault(int(tid), []).append(xy)
            px, py = xy[0] * scale, xy[1] * scale
            r = 4 if role == "qb" else 3
            fd.ellipse([px - r, py - r, px + r, py + r], fill=ROLE_RGB[role], outline=(0, 0, 0))
    for tr in trails.values():
        for (x0, y0), (x1, y1) in zip(tr, tr[1:]):
            if np.hypot(x1 - x0, y1 - y0) <= MAX_STEP_YD:   # skip homography glitches
                fd.line([x0 * scale, y0 * scale, x1 * scale, y1 * scale], fill=(255, 220, 120), width=1)
    canvas.paste(field, (TAG_W, VID_H))

    # tag panel
    tx, ty = 18, VID_H + 14
    d.text((tx, ty), "PIPELINE TAGS", font=font(13, bold=True), fill=MUTED)
    d.text((tx + 140, ty), "vs. coach", font=font(13), fill=MUTED)
    rows = [
        ("Formation", tags["formation"][0], tags["formation"][1], True),
        ("Field pos.", f"{fmt_yardline(tags['yardline'][0])} · {tags['hash'][0]} hash",
         f"{fmt_yardline(tags['yardline'][1])} · {tags['hash'][1]}", True),
        ("Play type", tags["play_type"][0] if show_play_type else "tracking …",
         tags["play_type"][1], show_play_type),
    ]
    for i, (label, pred, gt, reveal) in enumerate(rows):
        y = ty + 34 + i * 56
        d.text((tx, y), label.upper(), font=font(11, bold=True), fill=MUTED)
        d.text((tx, y + 15), pred, font=font(24, bold=True), fill=FG if reveal else MUTED)
        if reveal:
            match = pred.split(" · ")[0] == gt.split(" · ")[0] if label != "Field pos." else \
                tags["hash"][0] == tags["hash"][1] and abs(tags["yardline"][0] - tags["yardline"][1]) < 1
            d.text((tx + 250, y + 20), ("✓ " if match else "✗ ") + gt, font=font(13),
                   fill=OK if match else BAD)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="clip_108")
    args = ap.parse_args()

    meta = pd.read_csv(META_CSV).set_index("clip")
    qtr = meta.loc[f"{args.clip}.mp4", "QTR"]
    kp_model = YOLO(str(KP_RESULTS / qtr / "train" / "weights" / "best.pt"))

    cache = pickle.load(open(CACHE_DIR / f"{args.clip}.pkl", "rb"))
    names = cache["yolo_names"]
    tags = load_tags(args.clip)

    cap = cv2.VideoCapture(str(RAW_DIR / f"{args.clip}.mp4"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)

    field_base, scale = draw_field(VID_W - TAG_W, STRIP_H)
    trails, out = {}, []
    for i, f in enumerate(frames):
        # The camera pans during the play, so a stale H from an earlier frame
        # would misplace players; frames without a homography show no minimap dots.
        H, _ = run_field_pose_model(f, kp_model)
        det = cache["detections"].get(i, {"xyxy": [], "class_id": [], "tracker_id": []})
        out.append(render_frame(f, det, names, H, tags, i == len(frames) - 1,
                                field_base, scale, trails))

    ASSETS_DIR.mkdir(exist_ok=True)
    out[0].save(ASSETS_DIR / "demo_presnap.png")
    durations = [int(1000 / FPS)] * (len(out) - 1) + [int(HOLD_LAST_S * 1000)]
    small = [im.resize((im.width * 3 // 4, im.height * 3 // 4), Image.LANCZOS) for im in out]
    # One shared palette keeps role colors stable across frames. It is built from
    # a few frames plus large swatches of the UI/role colors, which would otherwise
    # be averaged away as small areas.
    w, h = small[0].size
    sample = [small[0], small[len(small) // 2], small[-1]]
    swatches = Image.new("RGB", (w, h // 2))
    sd = ImageDraw.Draw(swatches)
    colors = list(ROLE_RGB.values()) + [BG, FG, MUTED, OK, BAD, TURF, LINE, (255, 220, 120)]
    for i, c in enumerate(colors):
        sd.rectangle([i * w // len(colors), 0, (i + 1) * w // len(colors), h // 2], fill=c)
    ref = Image.new("RGB", (w, h * len(sample) + h // 2))
    for i, im in enumerate(sample + [swatches]):
        ref.paste(im, (0, i * h))
    palette = ref.quantize(colors=224, method=Image.MEDIANCUT, dither=Image.NONE)
    gif_frames = [im.quantize(palette=palette, dither=Image.NONE) for im in small]
    gif_frames[0].save(ASSETS_DIR / "demo.gif", save_all=True, append_images=gif_frames[1:],
                       duration=durations, loop=0, optimize=True)
    print(f"{args.clip} ({qtr} held-out keypoint model): {len(out)} frames -> {ASSETS_DIR / 'demo.gif'}")


if __name__ == "__main__":
    main()
