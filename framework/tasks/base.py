from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Iterable, List, Optional, Protocol, Sequence

from framework.grids import Grid, GridPair


class DimensionName(str, Enum):
    DEFINITION = "definition"
    CLASSIFICATION = "classification"
    CONSTRAINED_GENERATION = "constrained_generation"
    EDITING = "editing"
    INVERSION = "inversion"


class Verifier(Protocol):
    """Callable that applies the underlying ARC transformation.

    Given an input grid, returns the corresponding output grid.
    """

    def __call__(self, grid: Grid) -> Grid:  # pragma: no cover - protocol
        ...


class ProgramHandle(Protocol):
    """Opaque handle to an underlying ground-truth program, if available."""

    def as_callable(self) -> Verifier:  # pragma: no cover - protocol
        ...


@dataclass
class ArcTask:
    """In-memory representation of an ARC task.

    This is intentionally minimal and independent from any particular
    dataset format (ARC JSON, re_arc structures, arc_gen, etc.).
    """

    task_id: str

    # Canonical ARC training examples and test inputs (typically from the
    # original ARC JSONs).
    train_pairs: List[GridPair]
    test_inputs: List[Grid]
    # Ground-truth outputs for the original ARC test inputs, when available.
    test_outputs: List[Grid]

    # Primary verifier, ideally sourced from re_arc's ground-truth programs.
    verifier: Optional[Verifier] = None

    # Optional secondary verifier (e.g. from an independent solution set).
    secondary_verifier: Optional[Verifier] = None

    # Optional additional verifiers (e.g. other independent solution sets).
    tertiary_verifier: Optional[Verifier] = None
    quaternary_verifier: Optional[Verifier] = None
    quinary_verifier: Optional[Verifier] = None

    # Optional handle to an underlying program representation, if you
    # want to keep richer metadata than just a callable.
    program: Optional[ProgramHandle] = None

    # Optional synthetic example pools and generators.
    re_arc_synthetic_pairs: Optional[List[GridPair]] = None
    arc_gen_synthetic_pairs: Optional[List[GridPair]] = None

    # Generators return new synthetic pairs when called with the desired
    # number of examples.
    re_arc_generator: Optional[Callable[[int], List[GridPair]]] = None
    arc_gen_generator: Optional[Callable[[int], List[GridPair]]] = None

    # Which evaluation dimensions this task meaningfully supports.
    dimensions: Sequence[DimensionName] = field(
        default_factory=lambda: tuple(DimensionName)
    )


class TaskSource(Enum):
    ORIGINAL_ARC = auto()
    SYNTHETIC_ARC_GEN = auto()
    RE_ARC_BENCHMARK = auto()
    OTHER = auto()


@dataclass
class TaskMetadata:
    task_id: str
    source: TaskSource = TaskSource.OTHER
    description: Optional[str] = None

