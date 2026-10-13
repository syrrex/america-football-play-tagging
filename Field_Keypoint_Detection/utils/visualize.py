"""
Run trained YOLO pose model on a video and paint detected landmarks onto each frame.
Usage:
    python visualize.py                          # uses defaults below
    python visualize.py --model path/to/best.pt --video path/to/video.mp4
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ── Defaults ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
VIDEO_IN    = BASE_DIR / "Wide - Clip 033.mp4"
VIDEO_OUT   = BASE_DIR / "output_landmarks_033.mp4"
CONF        = 0.25    # detection confidence threshold
KPT_CONF    = 0.25    # minimum keypoint confidence to draw



def draw_keypoints(frame, keypoints, kpt_conf_thresh):
    """Draw keypoints coloured by confidence with landmark number labels."""
    h, w = frame.shape[:2]
    for idx, kp in enumerate(keypoints):
        x, y, conf = float(kp[0]), float(kp[1]), float(kp[2])
        if conf < kpt_conf_thresh:
            continue
        px, py = int(x), int(y)
        if not (0 <= px < w and 0 <= py < h):
            continue
        # colour by confidence
        if conf > 0.7:
            colour = (0, 220, 0)       # green
        elif conf > 0.5:
            colour = (0, 200, 200)     # yellow
        else:
            colour = (0, 80, 220)      # orange-red

        cv2.circle(frame, (px, py), 5, colour, -1)
        cv2.circle(frame, (px, py), 6, (0, 0, 0), 1)   # thin black outline

        # landmark number (1-based)
        label = str(idx + 1)
        cv2.putText(frame, label, (px + 7, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)   # black shadow
        cv2.putText(frame, label, (px + 7, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)


def main(model_path, video_in, video_out, conf, kpt_conf):
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_in}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(str(video_out), fourcc, fps, (width, height))

    print(f"[INFO] Video: {width}x{height} @ {fps:.1f} fps  |  {total} frames")
    print(f"[INFO] Output: {video_out}")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, iou=0.5, max_det=1, verbose=False, device=0)

        for result in results:
            # ── Bounding box ─────────────────────────────────────────────────
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    box_conf = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                    cv2.putText(frame, f"Field {box_conf:.2f}",
                                (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (255, 200, 0), 2)

            # ── Keypoints ────────────────────────────────────────────────────
            if result.keypoints is not None:
                for kps in result.keypoints.data:   # shape [62, 3]
                    draw_keypoints(frame, kps, kpt_conf)

        # Frame counter overlay
        cv2.putText(frame, f"Frame {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  processed {frame_idx}/{total}", flush=True)

    cap.release()
    out.release()
    print(f"[DONE] Saved to {video_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Path to best.pt (auto-detected if omitted)")
    parser.add_argument("--video", default=str(VIDEO_IN),
                        help="Input video path")
    parser.add_argument("--out",   default=str(VIDEO_OUT),
                        help="Output video path")
    parser.add_argument("--conf",     type=float, default=CONF)
    parser.add_argument("--kpt-conf", type=float, default=KPT_CONF)
    args = parser.parse_args()

    model_path = "/mnt/c/Users/Lukas/Documents/Semester_5/Seminar_Project/YOLOField/runs_v4/pose/football_field_v53/weights/best.pt"
    main(model_path, Path(args.video), Path(args.out),
         args.conf, args.kpt_conf)
