"""Risky object grounding metrics (Mean IoU, Acc@0.5)."""

from __future__ import annotations

from typing import Iterable, Sequence


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def evaluate_grounding(
    pairs: Iterable[tuple[Sequence[float], Sequence[float]]],
    threshold: float = 0.5,
) -> dict[str, float]:
    ious = [iou(pred, gt) for pred, gt in pairs]
    if not ious:
        return {"mean_iou": 0.0, f"acc@{threshold}": 0.0}

    acc = sum(1 for value in ious if value >= threshold) / len(ious)
    return {"mean_iou": sum(ious) / len(ious), f"acc@{threshold}": acc}
