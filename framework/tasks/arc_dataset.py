from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from framework.grids import Grid, GridPair
from framework.tasks.base import ArcTask, TaskSource
from framework.integrations.re_arc_adapter import get_re_arc_verifier


ROOT_DIR = Path(__file__).resolve().parents[2]

ARC_ORIGINAL_ZIP = ROOT_DIR / "external" / "re_arc" / "arc_original.zip"
RE_ARC_ZIP = ROOT_DIR / "external" / "re_arc" / "re_arc.zip"
ARC_GEN_STABLE_ZIP = ROOT_DIR / "external" / "arc_gen_stable.zip"

_ARC_GEN_TASK_LIST_MODULE: ModuleType | None = None


def _load_arc_original(task_id: str) -> Tuple[List[GridPair], List[Grid]]:
    """Load canonical ARC train/test examples for a task from arc_original.zip."""
    import zipfile

    if not ARC_ORIGINAL_ZIP.exists():
        raise FileNotFoundError(f"Missing arc_original.zip at {ARC_ORIGINAL_ZIP}")

    with zipfile.ZipFile(ARC_ORIGINAL_ZIP) as zf:
        path = f"arc_original/training/{task_id}.json"
        try:
            raw = zf.read(path)
        except KeyError as exc:
            raise KeyError(f"No original ARC task {task_id!r} in arc_original.zip") from exc
    data = json.loads(raw)

    train_pairs = [
        GridPair(example["input"], example["output"]) for example in data["train"]
    ]
    test_inputs: List[Grid] = [example["input"] for example in data["test"]]
    return train_pairs, test_inputs


def _load_re_arc_synthetic_pairs(task_id: str) -> Optional[List[GridPair]]:
    """Load stable re_arc synthetic input/output pairs from re_arc.zip, if present."""
    import zipfile

    if not RE_ARC_ZIP.exists():
        return None

    with zipfile.ZipFile(RE_ARC_ZIP) as zf:
        path = f"re_arc/tasks/{task_id}.json"
        try:
            raw = zf.read(path)
        except KeyError:
            return None
    data = json.loads(raw)
    return [GridPair(entry["input"], entry["output"]) for entry in data]


def _load_arc_gen_stable_pairs(task_id: str) -> Optional[List[GridPair]]:
    """Load stable ARC-GEN synthetic pairs from arc_gen_stable.zip, if present."""
    import zipfile

    if not ARC_GEN_STABLE_ZIP.exists():
        return None

    with zipfile.ZipFile(ARC_GEN_STABLE_ZIP) as zf:
        path = f"{task_id}.json"
        try:
            raw = zf.read(path)
        except KeyError:
            return None
    data = json.loads(raw)
    return [GridPair(entry["input"], entry["output"]) for entry in data]


def _load_re_arc_generators_module() -> Optional[ModuleType]:
    """Load external/re_arc/generators.py as a module."""
    path = ROOT_DIR / "external" / "re_arc" / "generators.py"
    if not path.exists():
        return None

    # Ensure relative imports like `from dsl import *` and `from utils import *`
    # inside generators.py can be resolved by temporarily adding the re_arc
    # directory to sys.path.
    re_arc_dir = path.parent
    original_sys_path = list(sys.path)
    try:
        if str(re_arc_dir) not in sys.path:
            sys.path.insert(0, str(re_arc_dir))

        spec = importlib.util.spec_from_file_location("re_arc_generators", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ModuleNotFoundError:
            # If re_arc's own dependencies are not importable, just skip
            # exposing generators rather than failing task loading entirely.
            return None
        return module
    finally:
        sys.path = original_sys_path


def _make_re_arc_generator(task_id: str) -> Optional[Callable[[int], List[GridPair]]]:
    """Return a callable that generates re_arc synthetic pairs for a task."""
    module = _load_re_arc_generators_module()
    if module is None:
        return None
    gen_fn = getattr(module, f"generate_{task_id}", None)
    if not callable(gen_fn):
        return None

    def generator(num_examples: int, *, diff_lb: float = 0.0, diff_ub: float = 1.0) -> List[GridPair]:
        pairs: List[GridPair] = []
        for _ in range(num_examples):
            example = gen_fn(diff_lb, diff_ub)  # type: ignore[misc]
            pairs.append(GridPair(example["input"], example["output"]))
        return pairs

    # Expose as a simple Callable[[int], List[GridPair]] by fixing difficulty.
    return lambda n: generator(n)


def _load_arc_gen_task_list_module() -> Optional[ModuleType]:
    """Load external/ARC-GEN/task_list.py as a module."""
    global _ARC_GEN_TASK_LIST_MODULE
    if _ARC_GEN_TASK_LIST_MODULE is not None:
        return _ARC_GEN_TASK_LIST_MODULE

    path = ROOT_DIR / "external" / "ARC-GEN" / "task_list.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("arc_gen_task_list", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _ARC_GEN_TASK_LIST_MODULE = module
    return module


def _arc_gen_id_to_task_num_and_generator(
    task_id: str,
) -> Optional[Tuple[int, Callable[[], Dict[str, Grid]]]]:
    """Map ARC task_id to (task_num, generator) using ARC-GEN's task_list."""
    module = _load_arc_gen_task_list_module()
    if module is None:
        return None
    task_list_fn = getattr(module, "task_list", None)
    if not callable(task_list_fn):
        return None
    mapping: Dict[int, Tuple[str, Callable[[], Dict[str, Grid]], Callable]] = task_list_fn()  # type: ignore[misc]
    for num, (arc_id, generator, _validator) in mapping.items():
        if arc_id == task_id:
            return num, generator
    return None


def _make_arc_gen_generator(task_id: str) -> Optional[Callable[[int], List[GridPair]]]:
    """Return a callable that generates ARC-GEN synthetic pairs for a task."""
    lookup = _arc_gen_id_to_task_num_and_generator(task_id)
    if lookup is None:
        return None
    task_num, generator = lookup

    def gen(num_examples: int) -> List[GridPair]:
        pairs: List[GridPair] = []
        for example_id in range(num_examples):
            random.seed(task_num + example_id)
            example = generator()
            pairs.append(GridPair(example["input"], example["output"]))
        return pairs

    return gen


def _load_secondary_verifier(task_id: str) -> Optional[Callable[[Grid], Grid]]:
    """Best-effort loader for a secondary verifier from NeurIPS Code Golf solutions.

    This expects the user to have cloned the repository
    `external/NeurIPS-Code-Golf-2025` themselves (this project is not a git
    repository, so we cannot add a submodule automatically).

    The function looks for a solution file whose name contains the ARC task
    id under `solutions/` and then tries a couple of common patterns to
    obtain a callable, such as a top-level `solve` function.
    """
    solutions_root = ROOT_DIR / "external" / "NeurIPS-Code-Golf-2025" / "solutions"
    if not solutions_root.exists():
        return None

    # Simple heuristic: any file in solutions/ whose stem contains the task id.
    candidates = [
        p for p in solutions_root.glob("*.py") if task_id in p.stem
    ]
    if not candidates:
        return None

    path = candidates[0]
    spec = importlib.util.spec_from_file_location(f"cg_solutions_{task_id}", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Prefer a function named `solve`, otherwise fall back to any callable.
    solve_fn = getattr(module, "solve", None)
    if callable(solve_fn):
        return solve_fn  # type: ignore[return-value]

    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        attr = getattr(module, attr_name)
        if callable(attr):
            return attr  # type: ignore[return-value]

    return None


def load_task(task_id: str) -> ArcTask:
    """Load a single ARC task by ID, with attached synthetic data and verifiers."""
    # Canonical ARC examples.
    train_pairs, test_inputs = _load_arc_original(task_id)

    # Optional synthetic data.
    re_arc_pairs = _load_re_arc_synthetic_pairs(task_id)
    arc_gen_pairs = _load_arc_gen_stable_pairs(task_id)

    # Optional generators.
    re_arc_gen = _make_re_arc_generator(task_id)
    arc_gen_gen = _make_arc_gen_generator(task_id)

    # Primary and secondary verifiers.
    verifier = get_re_arc_verifier(task_id)
    secondary_verifier = _load_secondary_verifier(task_id)

    return ArcTask(
        task_id=task_id,
        train_pairs=train_pairs,
        test_inputs=test_inputs,
        verifier=verifier,
        secondary_verifier=secondary_verifier,
        re_arc_synthetic_pairs=re_arc_pairs,
        arc_gen_synthetic_pairs=arc_gen_pairs,
        re_arc_generator=re_arc_gen,
        arc_gen_generator=arc_gen_gen,
    )


def iter_tasks(
    split: Optional[str] = None,
    *,
    source: Optional[TaskSource] = None,
) -> Iterator[ArcTask]:
    """Iterate over tasks from the original ARC JSONs."""
    import zipfile

    if source not in (None, TaskSource.ORIGINAL_ARC):
        raise ValueError(
            f"iter_tasks currently only supports source={TaskSource.ORIGINAL_ARC!r}"
        )

    # For now we only expose the training split from the original ARC set.
    if split not in (None, "train"):
        raise ValueError("Only split='train' (or None) is supported at the moment.")

    if not ARC_ORIGINAL_ZIP.exists():
        raise FileNotFoundError(f"Missing arc_original.zip at {ARC_ORIGINAL_ZIP}")

    with zipfile.ZipFile(ARC_ORIGINAL_ZIP) as zf:
        for name in zf.namelist():
            if not name.startswith("arc_original/training/") or not name.endswith(
                ".json"
            ):
                continue
            task_id = Path(name).stem
            yield load_task(task_id)


def to_grid_pairs(
    inputs: List[Grid],
    outputs: List[Grid],
) -> List[GridPair]:
    """Utility to create `GridPair` objects from separate input/output lists."""
    if len(inputs) != len(outputs):
        raise ValueError("inputs and outputs must have the same length")
    return [GridPair(inp, out) for inp, out in zip(inputs, outputs)]

