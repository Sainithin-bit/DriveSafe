#!/usr/bin/env python3
"""Stage 2: risk assessment and safety suggestion from caption Cv."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from drivesafe.llm import DEFAULT_MODEL_ID, build_llm_pipeline, run_chat, save_json
from risk_assessment.parse_output import parse_risk_response
from risk_assessment.prompts import build_risk_messages
from risk_assessment.safety_mapping import suggestions_from_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DriveSafe risk assessment (Stage 2)")
    parser.add_argument("--caption-file", help="JSON file from Stage 1")
    parser.add_argument("--caption", help="Raw caption text (alternative to --caption-file)")
    parser.add_argument("--clip-id", default="clip")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--keyword-map", default=str(REPO_ROOT / "configs" / "safety_keyword_map.json"))
    parser.add_argument("--output", default="outputs/risk")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    return parser.parse_args()


def load_caption(args: argparse.Namespace) -> tuple[str, str]:
    if args.caption_file:
        payload = json.loads(Path(args.caption_file).read_text(encoding="utf-8"))
        clip_id = payload.get("clip_id", args.clip_id)
        return clip_id, payload.get("caption", "")
    if args.caption:
        return args.clip_id, args.caption
    raise SystemExit("Provide --caption-file or --caption")


def main() -> None:
    args = parse_args()
    clip_id, caption = load_caption(args)

    messages = build_risk_messages(caption)
    pipeline = build_llm_pipeline(model_id=args.model_id)
    response = run_chat(pipeline, messages, max_new_tokens=args.max_new_tokens)
    parsed = parse_risk_response(response)
    parsed["safety_suggestion"] = suggestions_from_text(
        ", ".join(parsed["keywords"]),
    )

    output = {
        "clip_id": clip_id,
        "caption": caption,
        "prediction": parsed,
    }
    out_path = Path(args.output) / f"{Path(clip_id).stem}_risk.json"
    save_json(out_path, output)
    print(f"Wrote risk assessment to {out_path}")


if __name__ == "__main__":
    main()
