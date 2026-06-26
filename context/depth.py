"""Depth context (Dt) helpers.

Full reproduction uses DepthAnything-v2 per detected object (see paper Sec. IV-B).
This module provides a JSON loader plus an optional inference stub.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_depth_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def attach_depth_to_detections(
    detections: list[dict[str, Any]],
    depth_map: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Merge object-level depth estimates into detection records."""
    enriched = []
    for det in detections:
        record = dict(det)
        obj_id = str(record.get("track_id") or record.get("id") or record.get("label", ""))
        if depth_map and obj_id in depth_map:
            record["depth"] = depth_map[obj_id]
        enriched.append(record)
    return enriched


def export_depth_stub(output_path: str | Path, clip_id: str) -> None:
    """Write an empty depth JSON placeholder for pipeline wiring tests."""
    payload = {clip_id: {}}
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
