"""Load multimodal context JSON produced by external tools or sample inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def get_video_entry(data: dict[str, Any], clip_id: str) -> Any:
    """Fetch context for a clip, tolerating keys with or without extension."""
    if clip_id in data:
        return data[clip_id]
    stem = Path(clip_id).stem
    for key, value in data.items():
        if Path(key).stem == stem:
            return value
    raise KeyError(f"No entry found for clip_id={clip_id!r}")


def summarize_spatial_context(spatial: Any, max_frames: int = 8) -> str:
    if not spatial:
        return "No spatial context available."
    if isinstance(spatial, dict):
        keys = sorted(spatial.keys())[-max_frames:]
        parts = []
        for key in keys:
            value = spatial[key]
            if isinstance(value, (list, tuple)):
                flat = " ".join(str(v) for v in value)
            else:
                flat = str(value)
            parts.append(f'"{key}": "{flat}"')
        return ", ".join(parts)
    return str(spatial)


def summarize_detections(detections: Any, max_items: int = 12) -> str:
    if not detections:
        return "No surrounding detections available."
    if isinstance(detections, list):
        items = detections[:max_items]
        return json.dumps(items, ensure_ascii=False)
    if isinstance(detections, dict):
        keys = sorted(detections.keys())[-max_items:]
        return json.dumps({k: detections[k] for k in keys}, ensure_ascii=False)
    return str(detections)


def summarize_depth_context(depth: Any, max_items: int = 12) -> str:
    if not depth:
        return "No depth context available."
    if isinstance(depth, dict):
        keys = sorted(depth.keys())[:max_items]
        return json.dumps({k: depth[k] for k in keys}, ensure_ascii=False)
    return str(depth)


def summarize_motion_json(motion: Any, max_frames: int = 8) -> str:
    """Compact string summary of optical-flow JSON for LLM prompts."""
    if not motion:
        return "No motion context available."

    keys = sorted(motion.keys())[-max_frames:]
    chunks = []
    for key in keys:
        value = motion[key]
        if isinstance(value, list):
            flat = " ".join(str(cell) for row in value for cell in row)
            chunks.append(f'"{key}": "{flat}"')
    return ", ".join(chunks)


def load_videollama_description(path: str | Path, clip_id: str) -> str:
    """Parse Explanation_Distillation-style videollama text dumps."""
    clip_name = Path(clip_id).name
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    sentences: list[str] = []
    current: str | None = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.endswith((".avi", ".mp4")):
            if current == clip_name and sentences:
                return " ".join(sentences)
            current = Path(line).name
            sentences = []
        elif current == clip_name:
            sentences.append(line)

    if current == clip_name and sentences:
        return " ".join(sentences)
    return ""
