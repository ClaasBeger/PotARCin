"""P-ARC (PotARC) plausible wrong outputs for distribution classification / editing.

- **Test2**: hand-authored corrupt JSON files ``t<n>_corrupt*.json`` (single test pair).
- **ARC-AGI-1 training**: H-ARC incorrect submission pools (:mod:`framework.corruption.h_arc`).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from framework.corruption.h_arc import load_h_arc_incorrect_outputs_for_task
from framework.grids import GridPair
from framework.tasks.base import ArcTask
from framework.tasks.test2_dataset import parse_test2_task_num, test2_task_subdir


def sample_p_arc_pair(task: ArcTask, rng: random.Random) -> GridPair | None:
    """Return one plausible-wrong (input, output) pair, or None if unavailable."""
    try:
        n = parse_test2_task_num(task.task_id)
    except ValueError:
        n = None
    if n is not None:
        return _sample_test2_hand_corrupt_pair(n, rng)
    if not task.test_inputs:
        return None
    rows = load_h_arc_incorrect_outputs_for_task(task.task_id, exclude_evaluation=True)
    if not rows:
        return None
    weights = [max(1, r.count) for r in rows]
    (picked,) = rng.choices(rows, weights=weights, k=1)
    inp = copy.deepcopy(rng.choice(task.test_inputs))
    out = copy.deepcopy(picked.output_grid)
    pair = GridPair(inp, out)
    return pair


def _sample_test2_hand_corrupt_pair(task_num: int, rng: random.Random) -> GridPair | None:
    task_dir = test2_task_subdir(task_num)
    paths = sorted(task_dir.glob(f"t{task_num}_corrupted*.json"))
    if not paths:
        paths = sorted(task_dir.glob(f"t{task_num}_corrupt*.json"))
    if not paths:
        return None
    rng.shuffle(paths)
    for p in paths:
        try:
            payload = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            continue
        test = payload.get("test")
        if not isinstance(test, list) or not test:
            continue
        ex = test[0]
        if not isinstance(ex, dict):
            continue
        inp, out = ex.get("input"), ex.get("output")
        if not isinstance(inp, list) or not isinstance(out, list):
            continue
        return GridPair(copy.deepcopy(inp), copy.deepcopy(out))
    return None
