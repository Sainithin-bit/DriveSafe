#!/usr/bin/env python3
"""Stage 1: generate geometry-aware driving captions (Cv)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from caption_generation.build_multimodal_input import build_multimodal_bundle
from caption_generation.prompts import build_caption_messages
from drivesafe.llm import DEFAULT_MODEL_ID, build_llm_pipeline, run_chat, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DriveSafe caption generation (Stage 1)")
    parser.add_argument("--clip-id", required=True, help="Video filename or id")
    parser.add_argument("--spatial-json", required=True)
    parser.add_argument("--motion-json", required=True)
    parser.add_argument("--detections-json", required=True)
    parser.add_argument("--depth-json", default=None)
    parser.add_argument("--videollama-txt", default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output", default="outputs/captions")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = build_multimodal_bundle(
        clip_id=args.clip_id,
        spatial_json=args.spatial_json,
        motion_json=args.motion_json,
        detections_json=args.detections_json,
        depth_json=args.depth_json,
        videollama_txt=args.videollama_txt,
    )

    messages = build_caption_messages(
        spatial_context=bundle["spatial_context"],
        motion_context=bundle["motion_context"],
        depth_context=bundle["depth_context"],
        video_description=f"{bundle['video_description']} {bundle['detections_context']}".strip(),
    )

    pipeline = build_llm_pipeline(model_id=args.model_id)
    caption = run_chat(pipeline, messages, max_new_tokens=args.max_new_tokens)

    output = {
        "clip_id": args.clip_id,
        "caption": caption,
        "context": bundle,
    }
    out_path = Path(args.output) / f"{Path(args.clip_id).stem}.json"
    save_json(out_path, output)
    print(f"Wrote caption to {out_path}")


if __name__ == "__main__":
    main()
