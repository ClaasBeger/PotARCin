"""Same-task ARC-GEN dynamic mismatch: one input instance, output from another instance.

All pairs come from **one** task’s ARC-GEN dynamic generator. Given an anchor pair
``(input_A, output_A)``, we search other generated instances ``(input_B, output_B)``
from the same task (different **input** instance), and among those whose **output**
is not grid-equal to ``output_A`` we rank by padded output edit distance (normalized,
then raw). The caller selects ``rank_index`` (0 = closest mismatch, 1 = next, …).

Pass ``rng=`` to draw the pool with independent seeds per example (stochastic but
reproducible for a seeded batch). Omit ``rng`` for the canonical deterministic
ARC-GEN sequence.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

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
    """Anchor dynamic pair plus the *rank_index*-th best mismatch by output distance."""

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
    rank_index: int = 0,
    rng: Optional[random.Random] = None,
) -> Optional[CrossDynamicMismatch]:
    """Return the *rank_index*-th mismatch by output distance, or None.

    Draws *pool_size* additional ARC-GEN dynamic pairs for *task_id*. If *rng* is
    set, the pool is stochastic (see ``gen(..., rng=rng)``); otherwise the default
    deterministic sequence is used. Candidates must differ from the anchor on
    **input** (another instance) and on **output** (not grid-equal to the anchor
    output). Valid candidates are sorted by ``(normalized padded distance, raw
    distance)`` between candidate and anchor **outputs**; then the *rank_index*-th
    is returned (0 = tightest mismatch).
    """
    gen = _make_arc_gen_generator(task_id)
    if gen is None:
        return None
    try:
        if rng is not None:
            pool = gen(pool_size, rng=rng)
        else:
            pool = gen(pool_size)
    except Exception:
        return None
    if not pool:
        return None

    anchor_in = anchor_pair.input
    anchor_out = anchor_pair.output
    valid: List[CrossDynamicMismatch] = []

    for cand in pool:
        if is_equal_grid(cand.input, anchor_in):
            continue
        cand_out: Grid = cand.output
        if is_equal_grid(cand_out, anchor_out):
            continue
        nd = normalized_cell_edit_distance_padded(anchor_out, cand_out)
        rd = cell_edit_distance_padded(anchor_out, cand_out)
        valid.append(
            CrossDynamicMismatch(
                task_id=task_id,
                anchor_pair=anchor_pair,
                other_instance_pair=cand,
                normalized_output_distance=nd,
                raw_cell_output_distance=rd,
            )
        )
    if not valid:
        return None
    valid.sort(
        key=lambda m: (m.normalized_output_distance, m.raw_cell_output_distance),
    )
    if rank_index < 0 or rank_index >= len(valid):
        return None
    return valid[rank_index]


def synthetic_input_borrowed_output_pair(m: CrossDynamicMismatch) -> GridPair:
    """``(anchor input, other instance output)`` for display."""
    return GridPair(m.anchor_pair.input, m.other_instance_pair.output)
