import json
from pathlib import Path

from framework.dimensions.classification import (
    ClassificationEvaluator,
    ClassificationInstance,
    DistributionClassificationEvaluator,
    DistributionClassificationInstance,
)
from framework.dimensions import constrained_generation as cg_mod
from framework.dimensions.constrained_generation import (
    ConstrainedGenerationEvaluator,
    ConstrainedGenerationInstance,
    run_constrained_generation_api_evaluation_once,
)
from framework.dimensions.definition import (
    DefinitionEvaluator,
    DefinitionInstance,
)
from framework.dimensions.editing import EditingEvaluator, EditingInstance
from framework.dimensions.inversion import InversionEvaluator, InversionInstance
from framework.grids import Grid, GridPair
from framework.tasks.base import ArcTask


def _dummy_task() -> ArcTask:
    train_pair = GridPair(input=[[1]], output=[[2]])

    def _verifier(grid: Grid) -> Grid:
        return [[cell + 1 for cell in row] for row in grid]

    def _arc_gen_generator(n: int) -> list[GridPair]:
        out: list[GridPair] = []
        for i in range(n):
            inp = [[10 + i]]
            out.append(GridPair(input=inp, output=_verifier(inp)))
        return out

    return ArcTask(
        task_id="dummy",
        train_pairs=[train_pair],
        test_inputs=[[[3]]],
        test_outputs=[[[4]]],
        verifier=_verifier,
        arc_gen_synthetic_pairs=[GridPair(input=[[5]], output=[[6]])],
        arc_gen_generator=_arc_gen_generator,
    )


def test_definition_dimension_roundtrip():
    evaluator = DefinitionEvaluator()
    instances = evaluator.generate_instances([_dummy_task()])
    assert len(instances) == 1
    assert isinstance(instances[0], DefinitionInstance)
    predictions = ["def f(grid):\n    return grid"]
    result = evaluator.score_predictions(instances, predictions)
    assert result.dimension.value == "definition"


def test_classification_dimension_roundtrip():
    evaluator = ClassificationEvaluator()
    instances = evaluator.generate_instances([_dummy_task()])
    assert instances
    assert isinstance(instances[0], ClassificationInstance)
    predictions = [True for _ in instances]
    result = evaluator.score_predictions(instances, predictions)
    assert result.dimension.value == "classification"


def test_distribution_classification_dimension_roundtrip():
    """Five binary labels per instance; no API (mirrors definition roundtrip style)."""
    demos = [GridPair([[0]], [[1]])]
    pairs = [
        GridPair([[0]], [[1]]),
        GridPair([[1]], [[2]]),
        GridPair([[0]], [[1]]),
        GridPair([[2]], [[3]]),
        GridPair([[0]], [[2]]),
    ]
    target = [True, False, True, False, True]
    inst = DistributionClassificationInstance(
        task_id="dummy",
        input=(demos, pairs),
        target=target,
    )
    evaluator = DistributionClassificationEvaluator()
    predictions = [[1, 0, 1, 0, 1]]
    result = evaluator.score_predictions([inst], predictions)
    assert result.dimension.value == "classification"
    assert result.metrics.accuracy == 1.0
    assert result.metrics.exact_match_rate == 1.0


def test_constrained_generation_dimension_roundtrip():
    evaluator = ConstrainedGenerationEvaluator()
    instances = evaluator.generate_instances([_dummy_task()])
    assert instances
    assert isinstance(instances[0], ConstrainedGenerationInstance)
    predictions = [([[0]], [[0]]) for _ in instances]
    result = evaluator.score_predictions(instances, predictions)
    assert result.dimension.value == "constrained_generation"


def test_constrained_generation_eval_persists_raw_when_parse_fails(tmp_path, monkeypatch):
    """Model transcript must be on disk even when the answer is not a valid dict."""

    def _fake_completion(**kwargs):
        return "sorry, here is prose only — no dict"

    monkeypatch.setattr(cg_mod, "_openai_responses_completion", _fake_completion)
    res = run_constrained_generation_api_evaluation_once(
        task_id="a85d4709",
        api_key="fake-key",
        out_dir=tmp_path,
    )
    assert res.get("evaluation_ok") is False
    assert res.get("raw_response") == _fake_completion()
    path = res.get("result_path")
    assert path
    saved = json.loads(Path(path).read_text(encoding="utf-8"))
    assert saved["raw_response"] == _fake_completion()
    assert saved.get("prediction") is None


def test_editing_dimension_roundtrip():
    evaluator = EditingEvaluator()
    instances = evaluator.generate_instances([_dummy_task()])
    assert instances
    assert isinstance(instances[0], EditingInstance)
    predictions: list[GridPair] = [instances[0].target]
    result = evaluator.score_predictions(instances, predictions)
    assert result.dimension.value == "editing"


def test_inversion_dimension_roundtrip():
    evaluator = InversionEvaluator()
    instances = evaluator.generate_instances([_dummy_task()])
    assert instances
    assert isinstance(instances[0], InversionInstance)
    predictions: list[Grid] = [instances[0].target]
    result = evaluator.score_predictions(instances, predictions)
    assert result.dimension.value == "inversion"
    assert result.metrics.success_rate == 1.0

