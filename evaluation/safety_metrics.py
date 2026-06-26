"""Safety suggestion metrics (Table III)."""

from __future__ import annotations

from typing import Iterable

from sklearn.metrics import accuracy_score, f1_score


EVAL_CLASSES = [
    "(Must) Stop",
    "Be aware / cautious",
    "Slow down",
    "Carefully maneuver",
    "Follow the vehicle ahead",
    "Yield",
    "Start moving",
]


def evaluate_safety_suggestions(
    y_true: Iterable[str],
    y_pred: Iterable[str],
) -> dict[str, float]:
    labels = list(y_true)
    preds = list(y_pred)
    if not labels:
        return {"accuracy": 0.0, "f1_weighted": 0.0}

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_weighted": float(
            f1_score(labels, preds, average="weighted", labels=EVAL_CLASSES, zero_division=0)
        ),
    }
