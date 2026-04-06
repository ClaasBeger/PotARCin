"""Find closest grid *instances* (pairs from other tasks) to a query *pair*.

Distance uses padded bounding-box Hamming per grid (handles different sizes). For a
candidate pair ``(c_in, c_out)`` vs query ``(q_in, q_out)`` we report normalized
distances for input and output and their **average** — used as the ranking key.

Pool sources (excluding the reference task):

- All **train** pairs from ``arc_original_train``
- All **ARC-GEN stable** pairs from ``arc_gen_stable.zip`` (when present)
- Up to *num_dynamic* **ARC-GEN dynamic** samples from other tasks' generators

re_arc synthetic pairs are **not** included.

Scanning the full pool can take several minutes; the demo script supports
``--skip-neighbors`` for a fast corruption-only run.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

import random

from framework.grids import (
    GridPair,
    cell_edit_distance_padded,
    normalized_cell_edit_distance_padded,
    normalized_cell_edit_distance_padded_pair,
)
from framework.tasks.arc_dataset import (
    ARC_ORIGINAL_DIR,
    ROOT_DIR,
    _load_arc_gen_stable_pairs,
    _load_arc_original,
    _make_arc_gen_generator,
)

ARC_GEN_STABLE_ZIP = ROOT_DIR / "external" / "arc_gen_stable.zip"


@dataclass(frozen=True)
class NeighborInstance:
    """A labeled pair from another task, scored against a query *pair* (input + output)."""

    source: str
    ref_task_id: str
    detail: str
    pair: GridPair
    cell_edit_distance_in: int
    cell_edit_distance_out: int
    normalized_distance_in: float
    normalized_distance_out: float
    normalized_distance_avg: float


def _iter_arc_gen_stable_task_ids() -> List[str]:
    if not ARC_GEN_STABLE_ZIP.exists():
        return []
    with zipfile.ZipFile(ARC_GEN_STABLE_ZIP) as zf:
        return sorted(
            Path(n).stem for n in zf.namelist() if n.endswith(".json") and not n.startswith("__")
        )


def _collect_dynamic_pairs(
    exclude_task_id: str,
    rng: random.Random,
    *,
    max_pairs: int,
) -> List[Tuple[str, str, str, GridPair]]:
    """Return (source, task_id, detail, pair) for ARC-GEN dynamic samples."""
    tids = sorted(
        p.stem for p in ARC_ORIGINAL_DIR.glob("*.json") if p.stem != exclude_task_id
    )
    rng.shuffle(tids)
    out: List[Tuple[str, str, str, GridPair]] = []
    for tid in tids:
        if len(out) >= max_pairs:
            break
        gen = _make_arc_gen_generator(tid)
        if gen is None:
            continue
        try:
            pair = gen(1)[0]
        except Exception:
            continue
        out.append(("arc_gen_dynamic", tid, f"dynamic[{len(out)}]", pair))
    return out


def _iter_pool_rows(
    exclude_task_id: str,
    rng: random.Random,
    *,
    num_dynamic: int,
) -> Iterator[Tuple[str, str, str, GridPair]]:
    for path in sorted(ARC_ORIGINAL_DIR.glob("*.json")):
        tid = path.stem
        if tid == exclude_task_id:
            continue
        train, _, _ = _load_arc_original(tid)
        for i, p in enumerate(train):
            yield ("train", tid, f"train[{i}]", p)

    for tid in _iter_arc_gen_stable_task_ids():
        if tid == exclude_task_id:
            continue
        pairs = _load_arc_gen_stable_pairs(tid)
        if not pairs:
            continue
        for i, p in enumerate(pairs):
            yield ("arc_gen_stable", tid, f"stable[{i}]", p)

    for row in _collect_dynamic_pairs(exclude_task_id, rng, max_pairs=num_dynamic):
        yield row


def find_nearest_alternative_instances(
    query: GridPair,
    exclude_task_id: str,
    rng: random.Random,
    *,
    k: int = 3,
    num_dynamic: int = 50,
    fill: int = -1,
) -> List[NeighborInstance]:
    """Return *k* closest instances by average of normalized padded in/out distances."""
    q_in, q_out = query.input, query.output
    scored: List[NeighborInstance] = []
    for source, tid, detail, pair in _iter_pool_rows(
        exclude_task_id, rng, num_dynamic=num_dynamic
    ):
        d_in = cell_edit_distance_padded(q_in, pair.input, fill=fill)
        d_out = cell_edit_distance_padded(q_out, pair.output, fill=fill)
        ni, no, navg = normalized_cell_edit_distance_padded_pair(
            q_in, q_out, pair.input, pair.output, fill=fill
        )
        scored.append(
            NeighborInstance(
                source=source,
                ref_task_id=tid,
                detail=detail,
                pair=pair,
                cell_edit_distance_in=d_in,
                cell_edit_distance_out=d_out,
                normalized_distance_in=ni,
                normalized_distance_out=no,
                normalized_distance_avg=navg,
            )
        )

    scored.sort(
        key=lambda n: (
            n.normalized_distance_avg,
            n.normalized_distance_in + n.normalized_distance_out,
            n.cell_edit_distance_in + n.cell_edit_distance_out,
            n.ref_task_id,
            n.detail,
        )
    )

    picked: List[NeighborInstance] = []
    seen: set[tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]] = set()
    for n in scored:
        key = (
            tuple(tuple(row) for row in n.pair.input),
            tuple(tuple(row) for row in n.pair.output),
        )
        if key in seen:
            continue
        seen.add(key)
        picked.append(n)
        if len(picked) >= k:
            break
    return picked
