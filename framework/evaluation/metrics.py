from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


def accuracy(labels: Sequence[bool], predictions: Sequence[bool]) -> float:
    if not labels:
        return 0.0
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length")
    correct = sum(int(l == p) for l, p in zip(labels, predictions))
    return correct / len(labels)


@dataclass
class ScalarMetric:
    """Simple wrapper for a single scalar metric."""

    name: str
    value: float

