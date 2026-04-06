"""Same-task ARC-GEN dynamic mismatch: one input instance, output from another instance.

All pairs come from **one** task’s ARC-GEN dynamic generator. Given an anchor pair
``(input_A, output_A)``, we search other generated instances ``(input_B, output_B)``
from the same task (different **input** instance), and pick ``output_B`` whose
padded edit distance to ``output_A`` is minimal, excluding outputs grid-equal to
``output_A``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from framework.grids import (
    Grid,
    GridPair,
    cell_edit_distance_padded,
    is_equal_grid,
    normalized_cell_edit_distance_padded,
)
from framework.tasks.arc_dataset import _make_arc_gen_generator


@dataclass(frozen=True)
class CrossDynamicMismatch:
    """Anchor dynamic pair plus the closest non-identical output from another instance."""

    task_id: str
    anchor_pair: GridPair
    other_instance_pair: GridPair
    normalized_output_distance: float
    raw_cell_output_distance: int


def find_cross_dynamic_mismatch(
    task_id: str,
    anchor_pair: GridPair,
    *,
    pool_size: int = 50,
) -> Optional[CrossDynamicMismatch]:
    """Return the best mismatch within the same task, or None if no candidate exists.

    Draws *pool_size* additional ARC-GEN dynamic pairs for *task_id* (independent of
    the anchor draw). Candidates must differ from the anchor on **input** (another
    instance) and on **output** (not grid-equal to the anchor output). The winner
    minimizes (normalized padded distance, raw distance) between candidate and
    anchor **outputs**.
    """
    gen = _make_arc_gen_generator(task_id)
    if gen is None:
        return None
    try:
        pool = gen(pool_size)
    except Exception:
        return None
    if not pool:
        return None

    anchor_in = anchor_pair.input
    anchor_out = anchor_pair.output
    best: Optional[CrossDynamicMismatch] = None
    best_key: Optional[tuple[float, int]] = None

    for cand in pool:
        if is_equal_grid(cand.input, anchor_in):
            continue
        cand_out: Grid = cand.output
        if is_equal_grid(cand_out, anchor_out):
            continue
        nd = normalized_cell_edit_distance_padded(anchor_out, cand_out)
        rd = cell_edit_distance_padded(anchor_out, cand_out)
        key = (nd, rd)
        if best_key is None or key < best_key:
            best_key = key
            best = CrossDynamicMismatch(
                task_id=task_id,
                anchor_pair=anchor_pair,
                other_instance_pair=cand,
                normalized_output_distance=nd,
                raw_cell_output_distance=rd,
            )
    return best


def synthetic_input_borrowed_output_pair(m: CrossDynamicMismatch) -> GridPair:
    """``(anchor input, other instance output)`` for display."""
    return GridPair(m.anchor_pair.input, m.other_instance_pair.output)
