from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from framework.dimensions.base import (
    DimensionConfig,
    DimensionEvaluator,
    DimensionInstance,
    DimensionResult,
)
from framework.grids import Grid, GridPair
from framework.tasks.base import ArcTask, DimensionName


GeneratedPair = Tuple[Grid, Grid]


@dataclass
class ConstrainedGenerationInstance(DimensionInstance[list[GridPair], None]):
    """Training examples; model must generate a new (input, output) pair."""

    pass


@dataclass
class ConstrainedGenerationMetrics:
    success_rate: float


class ConstrainedGenerationEvaluator(
    DimensionEvaluator[list[GridPair], None, GeneratedPair, ConstrainedGenerationMetrics]
):
    """Skeleton evaluator for the constrained-generation dimension."""

    def __init__(self, config: DimensionConfig | None = None) -> None:
        self.config = config or DimensionConfig(
            name=DimensionName.CONSTRAINED_GENERATION
        )

    def generate_instances(
        self,
        tasks: Iterable[ArcTask],
    ) -> List[ConstrainedGenerationInstance]:
        instances: List[ConstrainedGenerationInstance] = []
        for task in tasks:
            if not task.train_pairs:
                continue
            instances.append(
                ConstrainedGenerationInstance(
                    task_id=task.task_id,
                    input=task.train_pairs,
                    target=None,
                )
            )
        return instances

    def score_predictions(
        self,
        instances: Sequence[ConstrainedGenerationInstance],
        predictions: Sequence[GeneratedPair],
    ) -> DimensionResult[ConstrainedGenerationMetrics]:
        if len(instances) != len(predictions):
            raise ValueError("instances and predictions must have the same length")

        # Placeholder: real implementation should call into a verifier.
        metrics = ConstrainedGenerationMetrics(success_rate=0.0)
        return DimensionResult(
            dimension=DimensionName.CONSTRAINED_GENERATION,
            metrics=metrics,
        )

