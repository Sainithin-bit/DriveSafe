#!/usr/bin/env python3
"""Prepare caption-instruction-response triplets for LLaMA-Adapter finetuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DriveSafe finetuning triplets")
    parser.add_argument("--annotations", required=True, help="DRAMA-style JSON annotations")
    parser.add_argument("--output", default="outputs/finetuning/triplets.jsonl")
    return parser.parse_args()


def format_response(record: dict) -> str:
    keywords = ", ".join(record.get("keywords", []))
    bbox = record.get("bounding_box") or []
    return (
        f"Risk: {record.get('risk', 'No')}\n"
        f"Risk Caption: {record.get('risk_caption', '')}\n"
        f"Keywords: {keywords}\n"
        f"Bounding Box: {bbox}"
    )


def main() -> None:
    args = parse_args()
    annotations = json.loads(Path(args.annotations).read_text(encoding="utf-8"))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for clip_id, record in annotations.items():
            triplet = {
                "clip_id": clip_id,
                "input": record.get("caption", ""),
                "instruction": "Analyze the caption and produce structured risk outputs.",
                "response": format_response(record),
            }
            handle.write(json.dumps(triplet) + "\n")

    print(f"Wrote triplets to {output_path}")


if __name__ == "__main__":
    main()
