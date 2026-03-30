from framework.dimensions.classification import (
    ClassificationEvaluator,
    ClassificationInstance,
)
from framework.dimensions.constrained_generation import (
    ConstrainedGenerationEvaluator,
    ConstrainedGenerationInstance,
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

    return ArcTask(
        task_id="dummy",
        train_pairs=[train_pair],
        test_inputs=[[[3]]],
        test_outputs=[[[4]]],
        verifier=_verifier,
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


def test_constrained_generation_dimension_roundtrip():
    evaluator = ConstrainedGenerationEvaluator()
    instances = evaluator.generate_instances([_dummy_task()])
    assert instances
    assert isinstance(instances[0], ConstrainedGenerationInstance)
    predictions = [([[0]], [[0]]) for _ in instances]
    result = evaluator.score_predictions(instances, predictions)
    assert result.dimension.value == "constrained_generation"


def test_editing_dimension_roundtrip():
    evaluator = EditingEvaluator()
    instances = evaluator.generate_instances([_dummy_task()])
    assert instances
    assert isinstance(instances[0], EditingInstance)
    predictions: list[Grid] = [instances[0].target]
    result = evaluator.score_predictions(instances, predictions)
    assert result.dimension.value == "editing"


def test_inversion_dimension_roundtrip():
    evaluator = InversionEvaluator()
    instances = evaluator.generate_instances([_dummy_task()])
    assert instances
    assert isinstance(instances[0], InversionInstance)
    predictions: list[Grid] = [[[3]]]
    result = evaluator.score_predictions(instances, predictions)
    assert result.dimension.value == "inversion"
    assert result.metrics.success_rate == 1.0

