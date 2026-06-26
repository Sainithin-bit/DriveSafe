"""Caption metrics used in DriveSafe Table II."""

from __future__ import annotations

from typing import Iterable


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def bleu1(reference: str, hypothesis: str) -> float:
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)
    if not hyp_tokens:
        return 0.0
    overlap = sum(1 for token in hyp_tokens if token in ref_tokens)
    return overlap / len(hyp_tokens)


def rouge_l(reference: str, hypothesis: str) -> float:
    ref = _tokenize(reference)
    hyp = _tokenize(hypothesis)
    if not ref or not hyp:
        return 0.0

    table = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    lcs = table[-1][[-1]]
    precision = lcs / len(hyp)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_captions(pairs: Iterable[tuple[str, str]]) -> dict[str, float]:
    scores = {"bleu1": [], "rouge_l": []}
    for reference, hypothesis in pairs:
        scores["bleu1"].append(bleu1(reference, hypothesis))
        scores["rouge_l"].append(rouge_l(reference, hypothesis))

    return {
        metric: (sum(values) / len(values) if values else 0.0)
        for metric, values in scores.items()
    }
