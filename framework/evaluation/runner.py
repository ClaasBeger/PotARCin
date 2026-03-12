from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

from framework.dimensions.base import (
    DimensionEvaluator,
    DimensionInstance,
    DimensionResult,
)
from framework.tasks.base import ArcTask, DimensionName


@dataclass
class EvaluationSummary:
    """Container for per-dimension results."""

    results: Mapping[DimensionName, DimensionResult[object]]


def run_dimension(
    evaluator: DimensionEvaluator[object, object, object, object],
    tasks: Iterable[ArcTask],
    predict_fn,
) -> DimensionResult[object]:
    """Run a single dimension given an evaluator and a prediction function.

    The `predict_fn` is responsible for turning generated instances into
    predictions appropriate for that dimension. This keeps the runner
    agnostic to model interfaces.
    """
    instances: Sequence[DimensionInstance[object, object]] = evaluator.generate_instances(
        tasks
    )
    predictions = predict_fn(instances)
    return evaluator.score_predictions(instances, predictions)


def run_all_dimensions(
    evaluators: Mapping[DimensionName, DimensionEvaluator[object, object, object, object]],
    tasks: Iterable[ArcTask],
    predict_fns: Mapping[DimensionName, callable],
) -> EvaluationSummary:
    """Run multiple dimensions and aggregate their results.

    Both `evaluators` and `predict_fns` are keyed by `DimensionName`.
    This function performs no concurrency; it is a thin orchestration
    layer intended to make experiments easier to structure.
    """
    results: Dict[DimensionName, DimensionResult[object]] = {}
    for name, evaluator in evaluators.items():
        predict_fn = predict_fns[name]
        result = run_dimension(evaluator, tasks, predict_fn)
        results[name] = result
    return EvaluationSummary(results=results)

