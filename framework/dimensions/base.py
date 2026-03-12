from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Iterable, List, Protocol, Sequence, TypeVar

from framework.tasks.base import ArcTask, DimensionName


InputT = TypeVar("InputT")
TargetT = TypeVar("TargetT")
PredictionT = TypeVar("PredictionT")
MetricT = TypeVar("MetricT")


@dataclass
class DimensionConfig:
    """Generic configuration shared across dimensions."""

    name: DimensionName
    num_instances_per_task: int = 1
    random_seed: int | None = None


@dataclass
class DimensionInstance(Generic[InputT, TargetT]):
    """Single evaluation instance for a given dimension."""

    task_id: str
    input: InputT
    target: TargetT


@dataclass
class DimensionResult(Generic[MetricT]):
    """Aggregated result for a dimension."""

    dimension: DimensionName
    metrics: MetricT


class DimensionEvaluator(Protocol, Generic[InputT, TargetT, PredictionT, MetricT]):
    """Protocol for all dimension evaluators."""

    config: DimensionConfig

    def generate_instances(self, tasks: Iterable[ArcTask]) -> List[DimensionInstance[InputT, TargetT]]:
        ...

    def score_predictions(
        self,
        instances: Sequence[DimensionInstance[InputT, TargetT]],
        predictions: Sequence[PredictionT],
    ) -> DimensionResult[MetricT]:
        ...

