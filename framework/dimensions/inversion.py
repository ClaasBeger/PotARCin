from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from framework.dimensions.base import (
    DimensionConfig,
    DimensionEvaluator,
    DimensionInstance,
    DimensionResult,
)
from framework.grids import Grid, GridPair, is_equal_grid
from framework.tasks.base import ArcTask, DimensionName, Verifier


@dataclass
class InversionPrompt:
    """Demonstrations plus a desired output grid to invert."""

    demonstrations: list[GridPair]
    desired_output: Grid


@dataclass
class InversionInstance(DimensionInstance[InversionPrompt, Grid]):
    """Given demos and desired output, predict an input grid."""

    verifier: Verifier | None = None


@dataclass
class InversionMetrics:
    success_rate: float


class InversionEvaluator(
    DimensionEvaluator[InversionPrompt, Grid, Grid, InversionMetrics]
):
    """Evaluator for inversion: find an input that maps to a given output."""

    def __init__(self, config: DimensionConfig | None = None) -> None:
        self.config = config or DimensionConfig(name=DimensionName.INVERSION)

    def generate_instances(self, tasks: Iterable[ArcTask]) -> List[InversionInstance]:
        instances: List[InversionInstance] = []
        for task in tasks:
            if not task.train_pairs:
                continue

            # Prefer the primary verifier and fall back to alternates.
            verifier = (
                task.verifier
                or task.secondary_verifier
                or task.tertiary_verifier
                or task.quaternary_verifier
                or task.quinary_verifier
            )
            if verifier is None:
                continue

            for test_input, test_output in zip(task.test_inputs, task.test_outputs):
                instances.append(
                    InversionInstance(
                        task_id=task.task_id,
                        input=InversionPrompt(
                            demonstrations=task.train_pairs,
                            desired_output=test_output,
                        ),
                        target=test_input,
                        verifier=verifier,
                    )
                )
        return instances

    def score_predictions(
        self,
        instances: Sequence[InversionInstance],
        predictions: Sequence[Grid],
    ) -> DimensionResult[InversionMetrics]:
        if len(instances) != len(predictions):
            raise ValueError("instances and predictions must have the same length")
        if not instances:
            return DimensionResult(
                dimension=DimensionName.INVERSION,
                metrics=InversionMetrics(success_rate=0.0),
            )

        correct = 0
        for inst, pred in zip(instances, predictions):
            if inst.verifier is None:
                continue
            try:
                produced = inst.verifier(pred)
            except Exception:
                continue
            if is_equal_grid(produced, inst.input.desired_output):
                correct += 1

        return DimensionResult(
            dimension=DimensionName.INVERSION,
            metrics=InversionMetrics(success_rate=correct / len(instances)),
        )
