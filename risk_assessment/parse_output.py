"""Parse structured risk outputs from LLM responses."""

from __future__ import annotations

import re
from typing import Any


def parse_risk_response(text: str) -> dict[str, Any]:
    risk_match = re.search(r"Risk:\s*(Yes|No)", text, re.IGNORECASE)
    caption_match = re.search(
        r"Risk Caption:\s*(.+?)(?:\nKeywords:|\nBounding Box:|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    keywords_match = re.search(
        r"Keywords:\s*(.+?)(?:\nBounding Box:|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    bbox_match = re.search(
        r"Bounding Box:\s*\[?\s*([\d.,\s]+)\s*\]?",
        text,
        re.IGNORECASE,
    )

    keywords: list[str] = []
    if keywords_match:
        keywords = [k.strip() for k in keywords_match.group(1).split(",") if k.strip()]

    bbox = None
    if bbox_match:
        parts = [float(x.strip()) for x in bbox_match.group(1).split(",") if x.strip()]
        if len(parts) == 4:
            bbox = parts

    return {
        "risk": (risk_match.group(1).capitalize() if risk_match else "No"),
        "risk_caption": caption_match.group(1).strip() if caption_match else "",
        "keywords": keywords,
        "bounding_box": bbox,
        "raw_response": text,
    }
