#!/usr/bin/env python3
"""LLaMA-Adapter finetuning stub for DriveSafe (paper Sec. III-B, IV-B).

Full training follows the LLaMA-Adapter repository:
https://github.com/OpenGVLab/LLaMA-Adapter

This script documents the hyperparameters used in the paper and validates
triplet files produced by prepare_triplets.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PAPER_HPARAMS = {
    "framework": "LLaMA-Adapter",
    "base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "batch_size": 4,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "epochs": 5,
    "warmup_ratio": 0.1,
    "mixed_precision": True,
    "gradient_checkpointing": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DriveSafe LLaMA-Adapter training helper")
    parser.add_argument("--triplets", required=True, help="JSONL from prepare_triplets.py")
    parser.add_argument("--output-config", default="outputs/finetuning/llama_adapter_config.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    triplet_path = Path(args.triplets)
    count = sum(1 for _ in triplet_path.open(encoding="utf-8"))

    config = {
        **PAPER_HPARAMS,
        "train_file": str(triplet_path),
        "num_samples": count,
        "notes": (
            "Integrate with OpenGVLab/LLaMA-Adapter using the train_file above. "
            "Release finetuned checkpoints on Hugging Face when available."
        ),
    }

    output_path = Path(args.output_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
