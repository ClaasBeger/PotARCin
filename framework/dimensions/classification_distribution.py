"""Build 5-way distribution classification items (ARC-GEN, corruption, flips, retrieval, mismatch).

Includes an H-ARC human-error corruption source (same task, incorrect outputs).

Only ``dynamic_correct`` is a positive label; its sampling mass is set so that across
five independent draws, **E[# correct] ≈ 1.5** (i.e. P(dynamic_correct) = 0.30).
Other category masses are scaled to keep relative ratios and sum to 0.70.
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
from framework.corruption.h_arc import load_h_arc_incorrect_outputs_for_task
from framework.corruption.nearest_inputs import find_nearest_alternative_instances
from framework.corruption.verifier_ast import load_and_corrupt_re_arc_verifier
from framework.grids import (
    GridPair,
    assert_int_grid_pair,
    connected_component_color_flips_on_pair,
    is_equal_grid,
    sample_flip_count_favor_one,
)
from framework.tasks.arc_dataset import load_task
from framework.tasks.base import ArcTask, Verifier
from framework.verifier_selection import select_verifier_for_task

Category = Literal[
    "instance_mismatch",
    "corrupted_verifier",
    "h_arc_incorrect",
    "color_flip",
    "dynamic_correct",
    "retrieval",
]

VerifierSlot = Literal["re_arc", "google", "keymoon", "neurips", "custom"]

# Relative sampling weights (sum = 1). P(dynamic_correct)=0.30 => E[# correct in 5] = 1.5.
# Non-positive categories keep the same relative mix as before, scaled by 0.70/0.84.
_NON_POSITIVE_SCALE = 0.70 / 0.84  # free mass after fixing P(dynamic_correct)=0.30
_CATEGORY_WEIGHTS: dict[Category, float] = {
    "h_arc_incorrect": 0.24 * _NON_POSITIVE_SCALE,
    "instance_mismatch": 0.24 * _NON_POSITIVE_SCALE,
    "corrupted_verifier": 0.18 * _NON_POSITIVE_SCALE,
    "dynamic_correct": 0.30,
    "retrieval": 0.10 * _NON_POSITIVE_SCALE,
    "color_flip": 0.08 * _NON_POSITIVE_SCALE,
}

# If a sampled corruption/category cannot yield a valid candidate quickly,
# stop trying that type and resample category.
_MAX_SAME_TYPE_TRIES = 10


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
            if not is_equal_grid(out, p.output):
                return False
        except Exception:
            return False
    n_test = min(len(task.test_inputs), len(task.test_outputs))
    for i in range(n_test):
        try:
            out = v(copy.deepcopy(task.test_inputs[i]))
            if not is_equal_grid(out, task.test_outputs[i]):
                return False
        except Exception:
            return False
    if task.arc_gen_synthetic_pairs:
        for p in task.arc_gen_synthetic_pairs:
            try:
                out = v(copy.deepcopy(p.input))
                if not is_equal_grid(out, p.output):
                    return False
            except Exception:
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
            if not is_equal_grid(out, p.output):
                return False
        except Exception:
            return False
    return True


def valid_verifier_slots_for_task(task: ArcTask) -> List[VerifierSlot]:
    """Every verifier slot that passes :func:`verifier_matches_train_test_stable_dynamic50`.

    Used for offline CSV generation. Runtime selection uses
    :func:`framework.verifier_selection.select_verifier_for_task` (CSV order, optional
    per-instance dynamic pairs).
    """
    candidates: List[Tuple[VerifierSlot, Optional[Verifier]]] = [
        ("re_arc", task.verifier),
        ("google", task.secondary_verifier),
        ("keymoon", task.tertiary_verifier),
        ("neurips", task.quaternary_verifier),
        ("custom", task.quinary_verifier),
    ]
    out: List[VerifierSlot] = []
    for name, fn in candidates:
        if fn is None:
            continue
        if verifier_matches_train_test_stable_dynamic50(task, fn):
            out.append(name)
    return out


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
    if task.arc_gen_generator is None:
        return None
    for _ in range(8):
        try:
            pair = copy.deepcopy(task.arc_gen_generator(1)[0])  # type: ignore[misc]
            inp = copy.deepcopy(pair.input)
        except Exception:
            continue

        picked = select_verifier_for_task(task, dynamic_pairs=[pair])
        if picked is None:
            continue
        vname, _v = picked

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
                pair = GridPair(inp, bad)
                try:
                    assert_int_grid_pair(pair, name="corrupted_verifier pair")
                except Exception:
                    continue
                return pair
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
                pair = GridPair(inp, bad)
                try:
                    assert_int_grid_pair(pair, name="corrupted_verifier pair")
                except Exception:
                    continue
                return pair
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
                pair = GridPair(inp, bad)
                try:
                    assert_int_grid_pair(pair, name="corrupted_verifier pair")
                except Exception:
                    continue
                return pair
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
                    pair = GridPair(inp, bad)
                    try:
                        assert_int_grid_pair(pair, name="corrupted_verifier pair")
                    except Exception:
                        continue
                    return pair
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
    try:
        assert_int_grid_pair(flipped, name="color_flip pair")
    except Exception:
        return None
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


def _try_h_arc_pair(task: ArcTask, rng: random.Random) -> GridPair | None:
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
    try:
        assert_int_grid_pair(pair, name="h_arc_incorrect pair")
    except Exception:
        return None
    return pair


@dataclass(frozen=True)
class DistributionClassificationItem:
    pair: GridPair
    """True iff this pair is from the task's true distribution (positives)."""

    same_distribution: bool
    category: Category
    meta: dict[str, object]


def _grid_pairs_equal(a: GridPair, b: GridPair) -> bool:
    return is_equal_grid(a.input, b.input) and is_equal_grid(a.output, b.output)


def _pair_collides_with_items(
    pair: GridPair, items: Sequence[DistributionClassificationItem]
) -> bool:
    return any(_grid_pairs_equal(pair, it.pair) for it in items)


def _input_collides_with_items(
    pair: GridPair, items: Sequence[DistributionClassificationItem]
) -> bool:
    return any(is_equal_grid(pair.input, it.pair.input) for it in items)


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


def _pair_matches_task_rule(task: ArcTask, pair: GridPair, verifier: Verifier | None = None) -> bool:
    """True when the pair is accepted by the selected task verifier.

    Used as a safety gate so negative/corrupted classification items are truly
    inconsistent with the rule.
    """
    vf = verifier
    if vf is None:
        picked = select_verifier_for_task(task)
        if picked is None:
            return False
        _slot, vf = picked
    try:
        out = vf(copy.deepcopy(pair.input))
    except Exception:
        return False
    return is_equal_grid(out, pair.output)


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

    if cat == "h_arc_incorrect":
        pair = _try_h_arc_pair(task, rng)
        if pair is None:
            return None, False, meta, False
        meta["kind"] = "h_arc_incorrect_submission"
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
    # Guard: demonstrations should always be valid integer grids; fail fast if not.
    for p in demonstrations:
        assert_int_grid_pair(p, name="demonstration pair")
    picked_verifier = select_verifier_for_task(task)
    task_verifier = picked_verifier[1] if picked_verifier is not None else None
    items: List[DistributionClassificationItem] = []
    retrieval_count = 0

    for slot in range(5):
        item: Optional[DistributionClassificationItem] = None
        for _attempt in range(12):
            cat = _draw_category(rng, retrieval_count=retrieval_count)
            if cat == "retrieval" and retrieval_count >= 2:
                continue
            # Keep retrying within the same category so collision re-samples preserve type.
            for _same_cat_try in range(_MAX_SAME_TYPE_TRIES):
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
                # Hard guard: never let non-int artifacts into prompts / logs.
                try:
                    assert_int_grid_pair(pair, name=f"classification item slot {slot}")
                except Exception:
                    continue
                # Ensure corruptions are truly corrupt: every negative candidate
                # must fail verifier consistency, otherwise resample.
                if not label and _pair_matches_task_rule(task, pair, verifier=task_verifier):
                    continue
                if _pair_collides_with_items(pair, items):
                    continue
                # Additional uniqueness constraint: do not allow repeated inputs
                # within the same 5-item batch, regardless of output/category.
                if _input_collides_with_items(pair, items):
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
            if item is not None:
                break

        if item is None:
            fb_pair: GridPair | None = None
            for _ in range(240):
                try:
                    cand = copy.deepcopy(task.arc_gen_generator(1)[0])  # type: ignore[misc]
                except Exception:
                    continue
                try:
                    assert_int_grid_pair(cand, name=f"fallback classification item slot {slot}")
                except Exception:
                    continue
                # Fallback emits dynamic_correct positives; keep this explicit.
                if not _pair_matches_task_rule(task, cand, verifier=task_verifier):
                    continue
                if not _pair_collides_with_items(cand, items):
                    if _input_collides_with_items(cand, items):
                        continue
                    fb_pair = cand
                    break
            # Stronger fallback for strict uniqueness constraints:
            # sample larger dynamic pools and pick the first non-colliding input/output.
            if fb_pair is None:
                for _ in range(24):
                    try:
                        pool = task.arc_gen_generator(200)  # type: ignore[misc]
                    except Exception:
                        continue
                    for cand0 in pool:
                        try:
                            cand = copy.deepcopy(cand0)
                            assert_int_grid_pair(
                                cand, name=f"fallback-pool classification item slot {slot}"
                            )
                        except Exception:
                            continue
                        if _pair_collides_with_items(cand, items):
                            continue
                        if _input_collides_with_items(cand, items):
                            continue
                        fb_pair = cand
                        break
                    if fb_pair is not None:
                        break
            if fb_pair is None:
                raise RuntimeError(
                    f"{task_id}: could not sample a unique classification pair for slot {slot} "
                    f"after exhausting category attempts and dynamic fallbacks."
                )
            item = DistributionClassificationItem(
                pair=fb_pair,
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
