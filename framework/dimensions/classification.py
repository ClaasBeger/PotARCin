from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from framework.dimensions.base import (
    DimensionConfig,
    DimensionEvaluator,
    DimensionInstance,
    DimensionResult,
)
from framework.grids import Grid, GridPair
from framework.tasks.base import ArcTask, DimensionName


Label = bool


@dataclass
class ClassificationCandidate:
    """Candidate example to classify as same-rule or not."""

    input: Grid | None
    output: Grid


@dataclass
class ClassificationInstance(DimensionInstance[tuple[list[GridPair], ClassificationCandidate], Label]):
    """Training examples plus a candidate pair and a ground-truth label."""

    pass


@dataclass
class ClassificationMetrics:
    accuracy: float


class ClassificationEvaluator(
    DimensionEvaluator[
        tuple[list[GridPair], ClassificationCandidate],
        Label,
        Label,
        ClassificationMetrics,
    ]
):
    """Skeleton evaluator for the classification dimension."""

    def __init__(self, config: DimensionConfig | None = None) -> None:
        self.config = config or DimensionConfig(name=DimensionName.CLASSIFICATION)

    def generate_instances(
        self,
        tasks: Iterable[ArcTask],
    ) -> List[ClassificationInstance]:
        # Placeholder: no real negative sampling yet, just a trivial instance per task.
        instances: List[ClassificationInstance] = []
        for task in tasks:
            if not task.train_pairs:
                continue
            first_pair = task.train_pairs[0]
            candidate = ClassificationCandidate(
                input=first_pair.input,
                output=first_pair.output,
            )
            instances.append(
                ClassificationInstance(
                    task_id=task.task_id,
                    input=([first_pair], candidate),
                    target=True,
                )
            )
        return instances

    def score_predictions(
        self,
        instances: Sequence[ClassificationInstance],
        predictions: Sequence[Label],
    ) -> DimensionResult[ClassificationMetrics]:
        if len(instances) != len(predictions):
            raise ValueError("instances and predictions must have the same length")
        if not instances:
            metrics = ClassificationMetrics(accuracy=0.0)
            return DimensionResult(
                dimension=DimensionName.CLASSIFICATION,
                metrics=metrics,
            )

        correct = 0
        for inst, pred in zip(instances, predictions):
            if inst.target == pred:
                correct += 1
        accuracy = correct / len(instances)
        metrics = ClassificationMetrics(accuracy=accuracy)
        return DimensionResult(
            dimension=DimensionName.CLASSIFICATION,
            metrics=metrics,
        )

