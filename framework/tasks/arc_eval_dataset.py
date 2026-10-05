"""Loader for the ARC-AGI-1 public **evaluation** split (400 tasks).

The training split lives as plain JSON under ``external/arc_original_train/``; the
evaluation split ships inside ``external/re_arc/arc_original.zip`` under
``arc_original/evaluation/``. This module reads the zip directly, but prefers an
extracted ``external/arc_original_eval/`` directory when one exists.

Unlike training tasks, evaluation tasks carry **no** verifiers or synthetic
generators (re_arc / ARC-GEN only cover the training split), so the returned
:class:`~framework.tasks.base.ArcTask` holds grids only. That is sufficient for
the output-grid dimension, which needs demonstrations + test input/output.
"""

from __future__ import annotations

import json
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, List

from framework.grids import Grid, GridPair, is_int_grid
from framework.tasks.base import ArcTask


ROOT_DIR = Path(__file__).resolve().parents[2]

ARC_EVAL_DIR = ROOT_DIR / "external" / "arc_original_eval"
ARC_ORIGINAL_ZIP = ROOT_DIR / "external" / "re_arc" / "arc_original.zip"
ZIP_EVAL_PREFIX = "arc_original/evaluation/"


def _source_description() -> str:
    if ARC_EVAL_DIR.is_dir():
        return str(ARC_EVAL_DIR)
    return f"{ARC_ORIGINAL_ZIP}!{ZIP_EVAL_PREFIX}"


@lru_cache(maxsize=1)
def arc_eval_task_ids() -> tuple[str, ...]:
    """Sorted task ids of the ARC-AGI-1 evaluation split."""
    if ARC_EVAL_DIR.is_dir():
        ids = sorted(p.stem for p in ARC_EVAL_DIR.glob("*.json"))
        if ids:
            return tuple(ids)

    if not ARC_ORIGINAL_ZIP.exists():
        raise FileNotFoundError(
            "ARC-AGI-1 evaluation split not found: neither "
            f"{ARC_EVAL_DIR} nor {ARC_ORIGINAL_ZIP} is available."
        )
    with zipfile.ZipFile(ARC_ORIGINAL_ZIP) as zf:
        names = [
            n
            for n in zf.namelist()
            if n.startswith(ZIP_EVAL_PREFIX) and n.endswith(".json")
        ]
    return tuple(sorted(Path(n).stem for n in names))


def _read_eval_json(task_id: str) -> dict[str, Any]:
    path = ARC_EVAL_DIR / f"{task_id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    if not ARC_ORIGINAL_ZIP.exists():
        raise FileNotFoundError(
            f"No evaluation task {task_id!r}: {path} is missing and "
            f"{ARC_ORIGINAL_ZIP} is not available."
        )
    with zipfile.ZipFile(ARC_ORIGINAL_ZIP) as zf:
        try:
            raw = zf.read(f"{ZIP_EVAL_PREFIX}{task_id}.json")
        except KeyError as e:
            raise KeyError(
                f"No ARC-AGI-1 evaluation task {task_id!r} in {ARC_ORIGINAL_ZIP}"
            ) from e
    return json.loads(raw)


def load_arc_eval_task(task_id: str, *, task_id_override: str | None = None) -> ArcTask:
    """Load one ARC-AGI-1 evaluation task as an :class:`ArcTask` (grids only).

    ``task_id_override`` sets :attr:`ArcTask.task_id` without changing which file is
    read; callers use it to keep per-artifact filenames unique when evaluating more
    than one test example of the same task.
    """
    data = _read_eval_json(task_id)
    src = _source_description()
    if "train" not in data or "test" not in data:
        raise ValueError(f"{task_id} in {src} must contain keys 'train' and 'test'")

    train_pairs: List[GridPair] = []
    for i, ex in enumerate(data["train"]):
        inp, out = ex.get("input"), ex.get("output")
        if not is_int_grid(inp) or not is_int_grid(out):
            raise ValueError(f"{task_id} train[{i}] has invalid grid(s)")
        train_pairs.append(GridPair(input=inp, output=out))

    test_inputs: List[Grid] = []
    test_outputs: List[Grid] = []
    for i, ex in enumerate(data["test"]):
        inp, out = ex.get("input"), ex.get("output")
        if not is_int_grid(inp) or not is_int_grid(out):
            raise ValueError(f"{task_id} test[{i}] has invalid grid(s)")
        test_inputs.append(inp)
        test_outputs.append(out)

    if not train_pairs or not test_inputs:
        raise ValueError(f"{task_id} in {src} has no train pairs or no test examples")

    return ArcTask(
        task_id=task_id_override or task_id,
        train_pairs=train_pairs,
        test_inputs=test_inputs,
        test_outputs=test_outputs,
    )


def test_example_count(task_id: str) -> int:
    """Number of test examples for an evaluation task (19 of 400 have 2)."""
    return len(_read_eval_json(task_id)["test"])


def iter_arc_eval_tasks() -> Iterator[ArcTask]:
    """Iterate the full evaluation split in sorted task-id order."""
    for task_id in arc_eval_task_ids():
        yield load_arc_eval_task(task_id)
