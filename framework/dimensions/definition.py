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
from framework.tasks.base import ArcTask, DimensionName, Verifier


# In the definition dimension, the prediction is typically a Python program
# (as a string) or a callable. For now, we keep this abstract as `str`.
ProgramText = str


@dataclass
class DefinitionInstance(DimensionInstance[list[GridPair], list[Grid]]):
    """Training pairs plus held-out test inputs."""

    pass


@dataclass
class DefinitionMetrics:
    """Placeholder for definition-dimension metrics."""

    success_rate: float


class DefinitionEvaluator(
    DimensionEvaluator[list[GridPair], list[Grid], ProgramText, DefinitionMetrics]
):
    """Skeleton evaluator for the definition dimension."""

    def __init__(self, config: DimensionConfig | None = None) -> None:
        self.config = config or DimensionConfig(name=DimensionName.DEFINITION)

    def generate_instances(self, tasks: Iterable[ArcTask]) -> List[DefinitionInstance]:
        instances: List[DefinitionInstance] = []
        for task in tasks:
            instances.append(
                DefinitionInstance(
                    task_id=task.task_id,
                    input=task.train_pairs,
                    target=task.test_inputs,
                )
            )
        return instances

    def score_predictions(
        self,
        instances: Sequence[DefinitionInstance],
        predictions: Sequence[ProgramText],
    ) -> DimensionResult[DefinitionMetrics]:
        if len(instances) != len(predictions):
            raise ValueError("instances and predictions must have the same length")

        # Placeholder scoring: no real execution, just a dummy metric.
        metrics = DefinitionMetrics(success_rate=0.0)
        return DimensionResult(dimension=DimensionName.DEFINITION, metrics=metrics)

