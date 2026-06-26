"""Map risk-related keywords to safety suggestion classes (Table I)."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP_PATH = REPO_ROOT / "configs" / "safety_keyword_map.json"


def load_keyword_map(path: str | Path | None = None) -> dict[str, list[str]]:
    map_path = Path(path or DEFAULT_MAP_PATH)
    with map_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def keyword_to_safety_suggestion(
    keywords: list[str],
    keyword_map: dict[str, list[str]] | None = None,
    default: str = "NA",
) -> str:
    keyword_map = keyword_map or load_keyword_map()
    normalized = [k.strip().lower() for k in keywords if k.strip()]

    best_label = default
    best_score = 0
    for label, candidates in keyword_map.items():
        score = 0
        for candidate in candidates:
            cand = candidate.lower()
            for keyword in normalized:
                if cand in keyword or keyword in cand:
                    score += 1
        if score > best_score:
            best_score = score
            best_label = label

    return best_label


def suggestions_from_text(
    keyword_text: str,
    keyword_map: dict[str, list[str]] | None = None,
) -> str:
    keywords = [part.strip() for part in keyword_text.replace(";", ",").split(",")]
    return keyword_to_safety_suggestion(keywords, keyword_map=keyword_map)
