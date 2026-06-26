"""Optical flow motion context (Mt) using OpenCV Farnebäck.

Adapted from Explanation_Distillation/OFM.py for DriveSafe caption generation.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def compute_patch_flow_labels(
    flow: np.ndarray,
    patch_h: int = 50,
    patch_w: int = 50,
) -> np.ndarray:
    """Summarize dense flow into a coarse grid of dominant directions."""
    height, width = flow.shape[:2]
    direction_labels = np.empty((height, width), dtype=object)
    direction_labels[flow[..., 0] > 0] = "right"
    direction_labels[flow[..., 0] < 0] = "left"
    direction_labels[flow[..., 0] == 0] = "none"

    grid_h = height // patch_h
    grid_w = width // patch_w
    dominant_labels = np.empty((grid_h, grid_w), dtype=object)

    for row in range(grid_h):
        for col in range(grid_w):
            patch = direction_labels[
                row * patch_h : (row + 1) * patch_h,
                col * patch_w : (col + 1) * patch_w,
            ]
            dominant_labels[row, col] = Counter(patch.flatten()).most_common(1)[0][0]

    return dominant_labels


def extract_motion_context(
    video_path: str | Path,
    tail_frames: int = 10,
) -> dict[str, list[list[str]]]:
    """Extract per-frame patch-wise motion summaries for the last N frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(total_frames - tail_frames, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return {}

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    motion_by_frame: dict[str, list[list[str]]] = {}
    frame_idx = start_frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        dominant = compute_patch_flow_labels(flow)
        motion_by_frame[f"frame_{frame_idx}"] = dominant.tolist()
        prev_gray = gray
        frame_idx += 1

    cap.release()
    return motion_by_frame


def export_motion_json(
    video_paths: list[str | Path],
    output_path: str | Path,
) -> dict[str, dict[str, list[list[str]]]]:
    """Batch-export motion context JSON keyed by video filename."""
    payload: dict[str, dict[str, list[list[str]]]] = {}
    for video_path in video_paths:
        video_path = Path(video_path)
        payload[video_path.name] = extract_motion_context(video_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract optical flow context JSON.")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    result = {Path(args.video).name: extract_motion_context(args.video)}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote motion context to {args.output}")
