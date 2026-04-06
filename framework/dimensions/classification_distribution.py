"""Build 5-way distribution classification items (ARC-GEN, corruption, flips, retrieval, mismatch).

Weights favor **same-task instance mismatch** most, then **corrupted verifier**, then
**color-flip** and **correct dynamic** at similar rates, then **retrieval** (nearest
alternative pair) least. Retrieval uses rank-0 once and rank-1 on a second draw; a
third retrieval is resampled to another category.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

from framework.corruption.cross_dynamic_mismatch import (
    find_cross_dynamic_mismatch,
    synthetic_input_borrowed_output_pair,
)
from framework.corruption.golf_ast import GolfSource, load_and_corrupt_golf_verifier
from framework.corruption.nearest_inputs import find_nearest_alternative_instances
from framework.corruption.verifier_ast import load_and_corrupt_re_arc_verifier
from framework.grids import GridPair, connected_component_color_flips_on_pair, is_equal_grid, sample_flip_count_favor_one
from framework.tasks.arc_dataset import load_task
from framework.tasks.base import ArcTask, Verifier

Category = Literal[
    "instance_mismatch",
    "corrupted_verifier",
    "color_flip",
    "dynamic_correct",
    "retrieval",
]

VerifierSlot = Literal["re_arc", "google", "keymoon", "neurips", "custom"]

# Relative sampling weights (sum = 1); normalized when drawing a category.
_CATEGORY_WEIGHTS: dict[Category, float] = {
    "instance_mismatch": 0.28,
    "corrupted_verifier": 0.24,
    "color_flip": 0.18,
    "dynamic_correct": 0.18,
    "retrieval": 0.12,
}


def _sample_category(rng: random.Random, allowed: Sequence[Category]) -> Category:
    cats = list(allowed)
    w = [_CATEGORY_WEIGHTS[c] for c in cats]
    (c,) = rng.choices(cats, weights=w, k=1)
    return c


def verifier_matches_train_test_stable_dynamic50(task: ArcTask, v: Verifier) -> bool:
    """True if *v* matches train, test, ARC-GEN stable (if any), and 50 dynamic pairs."""
    for p in task.train_pairs:
        try:
            out = v(copy.deepcopy(p.input))
        except Exception:
            return False
        if not is_equal_grid(out, p.output):
            return False
    n_test = min(len(task.test_inputs), len(task.test_outputs))
    for i in range(n_test):
        try:
            out = v(copy.deepcopy(task.test_inputs[i]))
        except Exception:
            return False
        if not is_equal_grid(out, task.test_outputs[i]):
            return False
    if task.arc_gen_synthetic_pairs:
        for p in task.arc_gen_synthetic_pairs:
            try:
                out = v(copy.deepcopy(p.input))
            except Exception:
                return False
            if not is_equal_grid(out, p.output):
                return False
    if task.arc_gen_generator is None:
        return False
    try:
        dyn = task.arc_gen_generator(50)
    except Exception:
        return False
    for p in dyn:
        try:
            out = v(copy.deepcopy(p.input))
        except Exception:
            return False
        if not is_equal_grid(out, p.output):
            return False
    return True


def select_verifier_for_task(task: ArcTask) -> Tuple[VerifierSlot, Verifier] | None:
    """First verifier in priority order that passes train / test / stable / dynamic50."""
    candidates: List[Tuple[VerifierSlot, Optional[Verifier]]] = [
        ("re_arc", task.verifier),
        ("google", task.secondary_verifier),
        ("keymoon", task.tertiary_verifier),
        ("neurips", task.quaternary_verifier),
        ("custom", task.quinary_verifier),
    ]
    for name, fn in candidates:
        if fn is None:
            continue
        if verifier_matches_train_test_stable_dynamic50(task, fn):
            return name, fn
    return None


def _sample_demo_pair_for_query(task: ArcTask, rng: random.Random) -> GridPair:
    opts: List[GridPair] = []
    opts.extend(task.train_pairs)
    if task.arc_gen_synthetic_pairs:
        opts.extend(task.arc_gen_synthetic_pairs)
    if task.arc_gen_generator:
        try:
            opts.extend(task.arc_gen_generator(50))
        except Exception:
            pass
    if not opts:
        raise ValueError("No demonstration pairs for retrieval query.")
    return copy.deepcopy(rng.choice(opts))


def _try_corrupted_pair(task_id: str, task: ArcTask, rng: random.Random) -> GridPair | None:
    picked = select_verifier_for_task(task)
    if picked is None:
        return None
    vname, _v = picked

    for _ in range(8):
        try:
            pair = copy.deepcopy(task.arc_gen_generator(1)[0])  # type: ignore[misc]
            inp = copy.deepcopy(pair.input)
        except Exception:
            continue

        if vname == "re_arc":
            try:
                corrupt_fn, _, _ = load_and_corrupt_re_arc_verifier(
                    task_id,
                    rng=rng,
                    sample_input=inp,
                    max_attempts=80,
                    max_normalized_cell_edit_distance=0.70,
                )
                bad = corrupt_fn(copy.deepcopy(inp))
                return GridPair(inp, bad)
            except Exception:
                continue

        if vname in ("google", "keymoon", "neurips"):
            src: GolfSource = vname  # type: ignore[assignment]
            try:
                corrupt_fn, _, _ = load_and_corrupt_golf_verifier(
                    task_id,
                    src,
                    rng=rng,
                    sample_input=inp,
                    max_attempts=80,
                    max_normalized_cell_edit_distance=0.70,
                )
                bad = corrupt_fn(copy.deepcopy(inp))
                return GridPair(inp, bad)
            except Exception:
                continue

        # custom: no AST hook; try re_arc then golf corruptions for a wrong output.
        if vname == "custom":
            try:
                corrupt_fn, _, _ = load_and_corrupt_re_arc_verifier(
                    task_id,
                    rng=rng,
                    sample_input=inp,
                    max_attempts=80,
                    max_normalized_cell_edit_distance=0.70,
                )
                bad = corrupt_fn(copy.deepcopy(inp))
                return GridPair(inp, bad)
            except Exception:
                pass
            for src in ("google", "keymoon", "neurips"):
                gs: GolfSource = src  # type: ignore[assignment]
                try:
                    corrupt_fn, _, _ = load_and_corrupt_golf_verifier(
                        task_id,
                        gs,
                        rng=rng,
                        sample_input=inp,
                        max_attempts=80,
                        max_normalized_cell_edit_distance=0.70,
                    )
                    bad = corrupt_fn(copy.deepcopy(inp))
                    return GridPair(inp, bad)
                except Exception:
                    continue
    return None


def _try_color_flip_pair(task: ArcTask, rng: random.Random) -> GridPair | None:
    opts: List[GridPair] = []
    if task.arc_gen_synthetic_pairs:
        opts.extend(task.arc_gen_synthetic_pairs)
    if task.arc_gen_generator:
        try:
            opts.extend(task.arc_gen_generator(50))
        except Exception:
            pass
    if not opts:
        return None
    base = copy.deepcopy(rng.choice(opts))
    n_ops = sample_flip_count_favor_one(rng, max_flips=5)
    flipped, _ = connected_component_color_flips_on_pair(base, rng, num_ops=n_ops)
    return flipped


def _try_instance_mismatch(
    task_id: str,
    task: ArcTask,
    *,
    cross_pool: int,
) -> GridPair | None:
    for _ in range(6):
        try:
            anchor = copy.deepcopy(task.arc_gen_generator(1)[0])  # type: ignore[misc]
            m = find_cross_dynamic_mismatch(task_id, anchor, pool_size=cross_pool)
        except Exception:
            m = None
        if m is None:
            continue
        return synthetic_input_borrowed_output_pair(m)
    return None


@dataclass(frozen=True)
class DistributionClassificationItem:
    pair: GridPair
    """True iff this pair is from the task's true distribution (positives)."""

    same_distribution: bool
    category: Category
    meta: dict[str, object]


@dataclass(frozen=True)
class DistributionClassificationBatch:
    task_id: str
    demonstrations: List[GridPair]
    items: List[DistributionClassificationItem]


def _draw_category(
    rng: random.Random,
    *,
    retrieval_count: int,
) -> Category:
    allowed: List[Category] = list(_CATEGORY_WEIGHTS.keys())
    if retrieval_count >= 2:
        allowed = [c for c in allowed if c != "retrieval"]
    return _sample_category(rng, allowed)


def _build_one_item(
    task_id: str,
    task: ArcTask,
    rng: random.Random,
    cat: Category,
    *,
    retrieval_rank: int,
    neighbor_dynamic: int,
    cross_pool: int,
) -> Tuple[Optional[GridPair], bool, dict[str, object], bool]:
    """Returns (pair, label, meta, used_retrieval_slot)."""
    meta: dict[str, object] = {"category": cat}

    if cat == "dynamic_correct":
        try:
            pair = copy.deepcopy(task.arc_gen_generator(1)[0])  # type: ignore[misc]
        except Exception:
            return None, False, meta, False
        meta["kind"] = "arc_gen_dynamic_correct"
        return pair, True, meta, False

    if cat == "instance_mismatch":
        pair = _try_instance_mismatch(task_id, task, cross_pool=cross_pool)
        if pair is None:
            return None, False, meta, False
        meta["kind"] = "same_task_instance_mismatch"
        return pair, False, meta, False

    if cat == "color_flip":
        pair = _try_color_flip_pair(task, rng)
        if pair is None:
            return None, False, meta, False
        meta["kind"] = "color_flip"
        return pair, False, meta, False

    if cat == "corrupted_verifier":
        pair = _try_corrupted_pair(task_id, task, rng)
        if pair is None:
            return None, False, meta, False
        meta["kind"] = "corrupted_verifier"
        return pair, False, meta, False

    # retrieval
    try:
        query = _sample_demo_pair_for_query(task, rng)
        neigh = find_nearest_alternative_instances(
            query,
            task_id,
            rng,
            k=3,
            num_dynamic=neighbor_dynamic,
        )
        if retrieval_rank >= len(neigh):
            return None, False, meta, False
        n = neigh[retrieval_rank]
        pair = copy.deepcopy(n.pair)
        meta["kind"] = "retrieval_nearest"
        meta["rank"] = retrieval_rank
        meta["query_task"] = n.ref_task_id
        return pair, False, meta, True
    except Exception:
        return None, False, meta, False


def sample_distribution_classification_batch(
    task_id: str,
    rng: random.Random,
    *,
    neighbor_dynamic: int = 50,
    cross_pool: int = 50,
) -> DistributionClassificationBatch:
    """Sample 5 labeled pairs for one task using weighted categories."""
    task = load_task(task_id)
    if task.arc_gen_generator is None:
        raise ValueError(f"Task {task_id} has no ARC-GEN dynamic generator.")
    if not task.train_pairs:
        raise ValueError(f"Task {task_id} has no train pairs.")

    demonstrations = list(task.train_pairs)
    items: List[DistributionClassificationItem] = []
    retrieval_count = 0

    for slot in range(5):
        item: Optional[DistributionClassificationItem] = None
        for _attempt in range(36):
            cat = _draw_category(rng, retrieval_count=retrieval_count)
            if cat == "retrieval" and retrieval_count >= 2:
                continue
            rank = retrieval_count if cat == "retrieval" else 0
            pair, label, meta, used_retrieval = _build_one_item(
                task_id,
                task,
                rng,
                cat,
                retrieval_rank=rank,
                neighbor_dynamic=neighbor_dynamic,
                cross_pool=cross_pool,
            )
            meta["slot"] = slot
            if pair is None:
                continue
            if used_retrieval:
                retrieval_count += 1
            item = DistributionClassificationItem(
                pair=pair,
                same_distribution=label,
                category=cat,
                meta=meta,
            )
            break

        if item is None:
            pair = copy.deepcopy(task.arc_gen_generator(1)[0])  # type: ignore[misc]
            item = DistributionClassificationItem(
                pair=pair,
                same_distribution=True,
                category="dynamic_correct",
                meta={"slot": slot, "kind": "fallback_dynamic_correct"},
            )
        items.append(item)

    rng.shuffle(items)
    return DistributionClassificationBatch(
        task_id=task_id,
        demonstrations=demonstrations,
        items=items,
    )
