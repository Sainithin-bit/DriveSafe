"""Assemble multimodal input Xv for DriveSafe caption generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from context.loaders import (
    get_video_entry,
    load_json,
    load_videollama_description,
    summarize_depth_context,
    summarize_detections,
    summarize_motion_json,
    summarize_spatial_context,
)


def build_multimodal_bundle(
    clip_id: str,
    spatial_json: str | Path,
    motion_json: str | Path,
    detections_json: str | Path,
    depth_json: str | Path | None = None,
    videollama_txt: str | Path | None = None,
    frame_captions: list[str] | None = None,
) -> dict[str, str]:
    spatial_data = load_json(spatial_json)
    motion_data = load_json(motion_json)
    detections_data = load_json(detections_json)

    spatial = get_video_entry(spatial_data, clip_id)
    motion = get_video_entry(motion_data, clip_id)
    detections = get_video_entry(detections_data, clip_id)

    depth_summary = "No depth context available."
    if depth_json and Path(depth_json).exists():
        depth_data = load_json(depth_json)
        depth_summary = summarize_depth_context(get_video_entry(depth_data, clip_id))

    video_description = ""
    if videollama_txt and Path(videollama_txt).exists():
        video_description = load_videollama_description(videollama_txt, clip_id)
    if frame_captions:
        video_description = f"{video_description} {' '.join(frame_captions)}".strip()

    return {
        "spatial_context": summarize_spatial_context(spatial),
        "motion_context": summarize_motion_json(motion),
        "depth_context": depth_summary,
        "video_description": video_description or "No video-level description available.",
        "detections_context": summarize_detections(detections),
    }
