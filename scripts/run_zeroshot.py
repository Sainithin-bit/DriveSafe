#!/usr/bin/env python3
"""End-to-end DriveSafe zero-shot pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    sample = REPO_ROOT / "testing_set_inputs" / "sample"
    parser = argparse.ArgumentParser(description="Run DriveSafe zero-shot on one clip")
    parser.add_argument(
        "--clip-id",
        default="337b694e-40f3-45f9-b0a7-88a1432ba0b3.mp4",
        help="Clip id present in sample JSON files",
    )
    parser.add_argument("--spatial-json", default=str(sample / "lane_change_sample_patch.json"))
    parser.add_argument("--motion-json", default=str(sample / "optical_flow_output_sample.json"))
    parser.add_argument("--detections-json", default=str(sample / "detections_sample.json"))
    parser.add_argument("--depth-json", default=None)
    parser.add_argument("--videollama-txt", default=str(sample / "videollama_sample.txt"))
    parser.add_argument("--model-id", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--output-dir", default="outputs/zeroshot")
    return parser.parse_args()


def run_step(command: list[str]) -> None:
    print("$", " ".join(command))
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    caption_dir = output_dir / "captions"
    risk_dir = output_dir / "risk"

    caption_cmd = [
        sys.executable,
        str(REPO_ROOT / "caption_generation" / "generate_captions.py"),
        "--clip-id",
        args.clip_id,
        "--spatial-json",
        args.spatial_json,
        "--motion-json",
        args.motion_json,
        "--detections-json",
        args.detections_json,
        "--videollama-txt",
        args.videollama_txt,
        "--model-id",
        args.model_id,
        "--output",
        str(caption_dir),
    ]
    if args.depth_json:
        caption_cmd.extend(["--depth-json", args.depth_json])

    run_step(caption_cmd)

    caption_file = caption_dir / f"{Path(args.clip_id).stem}.json"
    risk_cmd = [
        sys.executable,
        str(REPO_ROOT / "risk_assessment" / "infer_risk.py"),
        "--caption-file",
        str(caption_file),
        "--model-id",
        args.model_id,
        "--output",
        str(risk_dir),
    ]
    run_step(risk_cmd)
    print(f"Done. Results in {output_dir}")


if __name__ == "__main__":
    main()
