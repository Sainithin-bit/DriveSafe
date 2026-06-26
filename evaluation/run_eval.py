#!/usr/bin/env python3
"""Run DriveSafe evaluation metrics on prediction JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.caption_metrics import evaluate_captions
from evaluation.grounding_metrics import evaluate_grounding
from evaluation.safety_metrics import evaluate_safety_suggestions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DriveSafe predictions")
    parser.add_argument("--predictions", required=True, help="JSON list of prediction records")
    parser.add_argument("--ground-truth", required=True, help="JSON list of ground-truth records")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    ground_truth = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))

    gt_by_id = {item["clip_id"]: item for item in ground_truth}

    caption_pairs = []
    grounding_pairs = []
    y_true, y_pred = [], []

    for pred in predictions:
        clip_id = pred["clip_id"]
        gt = gt_by_id.get(clip_id)
        if not gt:
            continue

        caption_pairs.append((gt.get("caption", ""), pred.get("caption", "")))

        if gt.get("bounding_box") and pred.get("prediction", {}).get("bounding_box"):
            grounding_pairs.append((pred["prediction"]["bounding_box"], gt["bounding_box"]))

        if gt.get("safety_suggestion") and pred.get("prediction", {}).get("safety_suggestion"):
            y_true.append(gt["safety_suggestion"])
            y_pred.append(pred["prediction"]["safety_suggestion"])

    report = {
        "caption_metrics": evaluate_captions(caption_pairs),
        "grounding_metrics": evaluate_grounding(grounding_pairs),
        "safety_metrics": evaluate_safety_suggestions(y_true, y_pred),
        "num_samples": len(caption_pairs),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
