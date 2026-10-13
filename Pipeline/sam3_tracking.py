"""
SAM-3 Video Object Segmentation (VOS) Module

This module handles multi-object tracking across video frames using SAM-3.
It takes initial object detections (from YOLO) and propagates tracking
masks through all frames using the SAM-3 VOS model.

Pipeline: YOLO detection (first frame) -> SAM-3 tracking (all frames)
"""

import cv2
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from typing import Tuple, Dict
import supervision as sv
import os
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
from accelerate import Accelerator

from drawing import annotate, create_class_color_palette

# Maps SAM tracker IDs to YOLO class IDs for consistent labeling across frames
_TRACK_ID_TO_CLASS: Dict[int, int] = {}


def _load_frame(frame_path: Path) -> np.ndarray:
    """
    Load a single frame from disk.

    Args:
        frame_path: Path to the image file

    Returns:
        Frame as numpy array in BGR format (OpenCV default)

    Raises:
        FileNotFoundError: If the frame cannot be loaded
    """
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise FileNotFoundError(f"Failed to load frame: {frame_path}")
    return frame


def _convert_sam_output_to_detections(sam_output: dict) -> sv.Detections:
    """
    Convert SAM-3 VOS output to supervision Detections format.

    Args:
        sam_output: Dictionary with 'out_binary_masks' and 'out_obj_ids' keys

    Returns:
        sv.Detections object with bounding boxes, masks, and tracker IDs
    """
    return sv.Detections(
        xyxy=sv.mask_to_xyxy(sam_output["out_binary_masks"]),
        mask=sam_output["out_binary_masks"],
        tracker_id=sam_output["out_obj_ids"],
    )


def run_sam3_tracking(
    source_dir: Path,
    output_dir: Path,
    initial_detections: sv.Detections,
    yolo_names: Dict[int, str]
) -> Tuple[Dict[int, sv.Detections], Dict[int, str]]:
    """
    Run SAM-3 VOS tracking using pre-computed detections from the first frame.

    This function:
    1. Loads the SAM-3 model and initializes a tracking session
    2. Uses YOLO bounding boxes to initialize object masks on frame 0
    3. Propagates masks through all video frames
    4. Maps SAM tracker IDs back to YOLO class labels
    5. Saves annotated frames with tracked objects

    Args:
        source_dir: Directory containing extracted video frames (JPG/PNG)
        output_dir: Directory to save annotated output frames
        initial_detections: YOLO detections from the first frame
        yolo_names: Dictionary mapping class IDs to class names (e.g., {0: 'qb', 1: 'skill'})

    Returns:
        Tuple containing:
        - all_frame_detections: Dict mapping frame_idx to sv.Detections
        - yolo_names: The class name mapping (passed through for convenience)
    """
    global _TRACK_ID_TO_CLASS
    _TRACK_ID_TO_CLASS = {}

    model = None

    try:
        # === 1. Model Setup ===
        DEVICE = Accelerator().device
        print(f"Loading SAM-3 VOS Model on device: {DEVICE}...")

        model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3").to(DEVICE)
        processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")

        # === 2. Load Video Frames ===
        frame_files = sorted([
            f for f in os.listdir(source_dir)
            if f.endswith(('.jpg', '.jpeg', '.png'))
        ])
        if not frame_files:
            print(f"Error: No image frames found in {source_dir}")
            return {}, yolo_names

        video_frames_pil = [
            Image.open(source_dir / f).convert("RGB")
            for f in frame_files
        ]

        # === 3. Validate Initial Detections ===
        num_detections = len(initial_detections)
        if num_detections == 0:
            print("Error: No detections provided. Aborting SAM-3 tracking.")
            return {}, yolo_names

        print(f"Initializing SAM-3 VOS tracking for {num_detections} objects...")

        # === 4. Initialize VOS Session ===
        inference_session = processor.init_video_session(
            video=video_frames_pil,
            inference_device=DEVICE,
        )

        # === 5. Prepare Inputs from YOLO Detections ===
        # Object IDs: 1-indexed for SAM-3
        obj_ids = list(range(1, num_detections + 1))

        # Bounding box inputs (from YOLO)
        input_boxes_xyxy = initial_detections.xyxy.tolist()
        input_boxes = torch.tensor(
            [input_boxes_xyxy], device=DEVICE, dtype=torch.float32
        ).cpu()

        # Label inputs (all positive = object present)
        point_level_labels = [[1] for _ in range(num_detections)]
        input_labels = torch.tensor(
            [point_level_labels], device=DEVICE, dtype=torch.int32
        ).cpu()

        # Point inputs (empty - using boxes only)
        input_points = [[[[]]] * num_detections]

        # === 6. Register Objects for Tracking ===
        processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=0,
            obj_ids=obj_ids,
            input_boxes=input_boxes,
            input_labels=input_labels,
            input_points=input_points,
        )

        # === 7. Propagate Tracking Through Video ===
        all_frame_detections = {}
        frame_outputs = {}
        print("\nStarting VOS propagation across all frames...")

        for output in model.propagate_in_video_iterator(
            inference_session, start_frame_idx=0
        ):
            frame_idx = output.frame_idx

            # Post-process masks to original resolution
            video_res_masks = processor.post_process_masks(
                [output.pred_masks],
                    original_sizes=[[
                    inference_session.video_height,
                    inference_session.video_width
                ]],
                binarize=True
            )[0]

            output_data = {
                "out_binary_masks": video_res_masks.squeeze(1).cpu().numpy(),
                "out_obj_ids": np.array(inference_session.obj_ids, dtype=np.int64),
            }

            detections = _convert_sam_output_to_detections(output_data)

            # On first frame: map SAM tracker IDs to YOLO class IDs
            if frame_idx == 0:
                if initial_detections.class_id is not None and detections.tracker_id is not None:
                    for sam_id, yolo_cls in zip(detections.tracker_id, initial_detections.class_id):
                        _TRACK_ID_TO_CLASS[sam_id] = yolo_cls

            # Apply class mapping to current frame
            class_ids = [
                _TRACK_ID_TO_CLASS.get(tid, -1)
                for tid in detections.tracker_id
            ]
            detections.class_id = np.array(class_ids)

            all_frame_detections[frame_idx] = detections
            frame_outputs[frame_idx] = output_data

        # === 8. Annotate and Save Frames ===
        color_palette = create_class_color_palette(yolo_names)
        total_frames = len(frame_files)
        print(f"\nAnnotating {total_frames} frames from VOS output...")

        for i, frame_name in enumerate(frame_files):
            if i not in frame_outputs:
                continue

            # Load original frame
            frame_np = _load_frame(source_dir / frame_name)
            frame_pil = Image.fromarray(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB))

            # Get detections for this frame
            output = frame_outputs[i]
            detections = _convert_sam_output_to_detections(output)

            class_ids = [
                _TRACK_ID_TO_CLASS.get(tid, -1)
                for tid in detections.tracker_id
            ]
            detections.class_id = np.array(class_ids)

            # Skip if no valid detections
            if detections.tracker_id is None or len(detections) == 0:
                frame_pil.save(output_dir / frame_name)
                continue

            # Create labels with ID and class name
            labels = [
                f"ID {int(tid)}: {yolo_names.get(cid, 'UNKNOWN')}"
                for tid, cid in zip(detections.tracker_id, detections.class_id)
            ]

            # Draw annotations and save
            annotated = annotate(
                frame_pil,
                detections,
                labels=labels,
                class_color_palette=color_palette
            )
            annotated.save(output_dir / frame_name)

        print(f"✅ SAM-3 VOS Tracking Complete. Frames saved to: {output_dir}")
        return all_frame_detections, yolo_names

    finally:
        # === Cleanup GPU Resources ===
        if model is not None:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Shutting down SAM-3 VOS resources...")
