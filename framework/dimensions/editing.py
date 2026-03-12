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


@dataclass
class EditingInstance(
    DimensionInstance[tuple[list[GridPair], Grid], Grid],
):
    """Training examples plus a corrupted output that must be repaired."""

    pass


@dataclass
class EditingMetrics:
    success_rate: float


class EditingEvaluator(
    DimensionEvaluator[
        tuple[list[GridPair], Grid],
        Grid,
        Grid,
        EditingMetrics,
    ]
):
    """Skeleton evaluator for the editing dimension."""

    def __init__(self, config: DimensionConfig | None = None) -> None:
        self.config = config or DimensionConfig(name=DimensionName.EDITING)

    def generate_instances(
        self,
        tasks: Iterable[ArcTask],
    ) -> List[EditingInstance]:
        # Placeholder: no real mutation; just echo an existing output as the
        # "corrupted" one so that the API is exercised.
        instances: List[EditingInstance] = []
        for task in tasks:
            if not task.train_pairs:
                continue
            pair = task.train_pairs[0]
            corrupted_output = pair.output
            instances.append(
                EditingInstance(
                    task_id=task.task_id,
                    input=([pair], corrupted_output),
                    target=pair.output,
                )
            )
        return instances

    def score_predictions(
        self,
        instances: Sequence[EditingInstance],
        predictions: Sequence[Grid],
    ) -> DimensionResult[EditingMetrics]:
        if len(instances) != len(predictions):
            raise ValueError("instances and predictions must have the same length")

        # Placeholder: in a real setup, compare against verifier outputs.
        metrics = EditingMetrics(success_rate=0.0)
        return DimensionResult(
            dimension=DimensionName.EDITING,
            metrics=metrics,
        )

