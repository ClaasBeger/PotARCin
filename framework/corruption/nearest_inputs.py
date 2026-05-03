"""Find closest alternative instances for retrieval negatives.

Distance uses padded bounding-box Hamming per grid (handles different sizes). For a
candidate pair ``(c_in, c_out)`` vs query ``(q_in, q_out)`` we report normalized
distances for input and output and their **average** — used as the ranking key.

Pool sources (excluding the reference task):

- All **train** pairs from ``arc_original_train``
- All **ARC-GEN stable** pairs from ``arc_gen_stable.zip`` (when present)
- Up to *num_dynamic* **ARC-GEN dynamic** samples from other tasks' generators

re_arc synthetic pairs are **not** included.

Two ranking modes are available:

- ``edit_distance``: original padded in/out grid distance over full instance pool.
- ``task_embedding``: nearest tasks by precomputed task embeddings (fast path),
  then pick representative instances from those tasks.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterator, List, Literal, Optional, Tuple

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
from framework.tasks.test2_dataset import (
    TEST2_DIR,
    canonical_test2_task_id,
    iter_test2_tasks,
    parse_test2_task_num,
)

ARC_GEN_STABLE_ZIP = ROOT_DIR / "external" / "arc_gen_stable.zip"
TASK_EMBEDDINGS_DIR = ROOT_DIR / "external" / "Task_Embeddings"

DistanceCriterion = Literal["edit_distance", "task_embedding"]

_EMBEDDING_INDEX: list[tuple[str, list[float]]] | None = None
_EMBEDDING_INDEX_READY = False


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


def _maybe_extract_vector(obj: object) -> Optional[list[float]]:
    """Best-effort conversion from torch-loaded object to a 1D float list."""
    # torch tensor path (most common)
    if hasattr(obj, "detach") and hasattr(obj, "cpu") and hasattr(obj, "flatten"):
        try:
            vals = obj.detach().cpu().flatten().tolist()
            return [float(v) for v in vals]
        except Exception:
            return None
    # nested container paths
    if isinstance(obj, dict):
        for key in ("embedding", "task_embedding", "task_token_embed", "vector", "emb"):
            if key in obj:
                vec = _maybe_extract_vector(obj[key])
                if vec:
                    return vec
        for v in obj.values():
            vec = _maybe_extract_vector(v)
            if vec:
                return vec
        return None
    if isinstance(obj, (list, tuple)):
        if obj and all(isinstance(x, (int, float)) for x in obj):
            return [float(x) for x in obj]
        for v in obj:
            vec = _maybe_extract_vector(v)
            if vec:
                return vec
        return None
    return None


def _normalize(vec: list[float]) -> Optional[list[float]]:
    if not vec:
        return None
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 0.0:
        return None
    return [v / norm for v in vec]


def _load_embedding_index() -> list[tuple[str, list[float]]]:
    """Load and cache normalized task embedding vectors.

    Returns empty list if torch or embedding files are unavailable.
    """
    global _EMBEDDING_INDEX, _EMBEDDING_INDEX_READY
    if _EMBEDDING_INDEX_READY:
        return _EMBEDDING_INDEX or []
    _EMBEDDING_INDEX_READY = True
    _EMBEDDING_INDEX = []

    if not TASK_EMBEDDINGS_DIR.exists():
        return _EMBEDDING_INDEX
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        return _EMBEDDING_INDEX

    for path in sorted(TASK_EMBEDDINGS_DIR.glob("*_task_token_embed.pt")):
        task_id = path.stem.replace("_task_token_embed", "")
        try:
            task_id = canonical_test2_task_id(task_id)
        except ValueError:
            pass
        try:
            obj = torch.load(path, map_location="cpu")
        except Exception:
            continue
        vec = _maybe_extract_vector(obj)
        if not vec:
            continue
        nvec = _normalize(vec)
        if not nvec:
            continue
        _EMBEDDING_INDEX.append((task_id, nvec))
    return _EMBEDDING_INDEX


def _nearest_tasks_by_embedding(
    query_task_id: str,
    *,
    exclude_task_id: str,
    k: int,
) -> list[tuple[str, float]]:
    """Return nearest task ids by cosine distance (smaller is better)."""
    idx = _load_embedding_index()
    if not idx:
        return []
    qvec: Optional[list[float]] = None
    try:
        query_task_id = canonical_test2_task_id(query_task_id)
    except ValueError:
        pass
    try:
        exclude_task_id = canonical_test2_task_id(exclude_task_id)
    except ValueError:
        pass
    for tid, vec in idx:
        if tid == query_task_id:
            qvec = vec
            break
    if qvec is None:
        return []

    scored: list[tuple[str, float]] = []
    qn = len(qvec)
    for tid, vec in idx:
        if tid == exclude_task_id:
            continue
        if len(vec) != qn:
            continue
        cos = sum(a * b for a, b in zip(qvec, vec))
        dist = 1.0 - cos
        scored.append((tid, dist))
    scored.sort(key=lambda x: (x[1], x[0]))
    return scored[: max(1, k)]


def _load_train_pairs_for_task_id(task_id: str) -> List[GridPair]:
    """Return canonical training pairs for *task_id* (ARC-AGI-1 hash or ``test2_t<n>``).

    Returns an empty list if neither source resolves. Used by the embedding
    retrieval fast path so Test2 task ids returned by the embedding index can be
    materialized into instance-level pairs alongside ARC1 hits.
    """
    try:
        parse_test2_task_num(task_id)
        is_test2 = True
    except ValueError:
        is_test2 = False
    if is_test2:
        try:
            from framework.tasks.test2_dataset import load_test2_task
            task = load_test2_task(task_id)
        except Exception:
            return []
        return list(task.train_pairs)
    try:
        train, _, _ = _load_arc_original(task_id)
        return list(train)
    except Exception:
        return []


def _iter_arc_gen_stable_task_ids() -> List[str]:
    if not ARC_GEN_STABLE_ZIP.exists():
        return []
    with zipfile.ZipFile(ARC_GEN_STABLE_ZIP) as zf:
        return sorted(
            Path(n).stem for n in zf.namelist() if n.endswith(".json") and not n.startswith("__")
        )


def _collect_dynamic_pairs(
    exclude_task_id: str,
    rng: Optional[random.Random],
    *,
    max_pairs: int,
) -> List[Tuple[str, str, str, GridPair]]:
    """Return (source, task_id, detail, pair) for ARC-GEN dynamic samples."""
    canonical_exclude = exclude_task_id
    try:
        canonical_exclude = canonical_test2_task_id(exclude_task_id)
    except ValueError:
        pass
    tids = sorted(
        p.stem for p in ARC_ORIGINAL_DIR.glob("*.json") if p.stem != exclude_task_id
    )
    if rng is not None:
        rng.shuffle(tids)
    out: List[Tuple[str, str, str, GridPair]] = []
    for tid in tids:
        if len(out) >= max_pairs:
            break
        gen = _make_arc_gen_generator(tid)
        if gen is None:
            continue
        try:
            if rng is not None:
                pair = gen(1, rng=rng)[0]
            else:
                pair = gen(1)[0]
        except Exception:
            continue
        out.append(("arc_gen_dynamic", tid, f"dynamic[{len(out)}]", pair))
    if TEST2_DIR.is_dir() and len(out) < max_pairs:
        for task in iter_test2_tasks():
            if len(out) >= max_pairs:
                break
            if task.task_id == canonical_exclude:
                continue
            if task.arc_gen_generator is None:
                continue
            try:
                if rng is not None:
                    pair = task.arc_gen_generator(1, rng=rng)[0]  # type: ignore[misc]
                else:
                    pair = task.arc_gen_generator(1)[0]  # type: ignore[misc]
            except Exception:
                continue
            out.append(("test2_dynamic", task.task_id, f"dynamic[{len(out)}]", pair))
    return out


def _iter_pool_rows(
    exclude_task_id: str,
    rng: Optional[random.Random],
    *,
    num_dynamic: int,
) -> Iterator[Tuple[str, str, str, GridPair]]:
    canonical_exclude = exclude_task_id
    try:
        canonical_exclude = canonical_test2_task_id(exclude_task_id)
    except ValueError:
        pass
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

    if TEST2_DIR.is_dir():
        for task in iter_test2_tasks():
            if task.task_id == canonical_exclude:
                continue
            for i, p in enumerate(task.train_pairs):
                yield ("test2_train", task.task_id, f"train[{i}]", p)


def find_nearest_alternative_instances(
    query: GridPair,
    exclude_task_id: str,
    rng: Optional[random.Random],
    *,
    criterion: DistanceCriterion = "edit_distance",
    k: int = 3,
    num_dynamic: int = 50,
    fill: int = -1,
) -> List[NeighborInstance]:
    """Return *k* closest instances using configured distance criterion.

    Pass ``rng=None`` for a **deterministic** pool: ARC-GEN dynamic rows use the
    default generator sequence (no shuffle of task ids, no random seeds per draw).
    """
    if criterion == "task_embedding":
        # Fast path: task-level nearest neighbors in embedding space.
        # Pool covers both ARC-AGI-1 train tasks and Test2 tasks: ARC1 ids load
        # via :func:`_load_arc_original`; Test2 ids (``test2_t<n>``) load via
        # :func:`load_test2_task` so retrieval can mix both sources.
        nearest = _nearest_tasks_by_embedding(
            query_task_id=exclude_task_id,
            exclude_task_id=exclude_task_id,
            k=max(k * 6, 12),
        )
        if nearest:
            picked: List[NeighborInstance] = []
            seen: set[
                tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]
            ] = set()
            for tid, tdist in nearest:
                train_pairs = _load_train_pairs_for_task_id(tid)
                if not train_pairs:
                    continue
                for i, pair in enumerate(train_pairs):
                    key = (
                        tuple(tuple(row) for row in pair.input),
                        tuple(tuple(row) for row in pair.output),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    picked.append(
                        NeighborInstance(
                            source="task_embedding",
                            ref_task_id=tid,
                            detail=f"embed_train[{i}]",
                            pair=pair,
                            cell_edit_distance_in=-1,
                            cell_edit_distance_out=-1,
                            normalized_distance_in=tdist,
                            normalized_distance_out=tdist,
                            normalized_distance_avg=tdist,
                        )
                    )
                    if len(picked) >= k:
                        return picked
            # If embeddings were present but yielded too few candidates, continue
            # with edit-distance fallback below.

    # Original instance-level edit-distance ranking.
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
