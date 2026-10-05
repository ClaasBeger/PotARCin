"""Build 5-way distribution classification items (ARC-GEN, corruption, flips, retrieval, mismatch).

Includes **P-ARC** plausible-wrong outputs (PotARC hand-authored pairs on Test2; H-ARC CSV pool on ARC1).

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
from framework.corruption.nearest_inputs import find_nearest_alternative_instances
from framework.corruption.p_arc import sample_p_arc_pair
from framework.corruption.test2_verifier_ast import load_and_corrupt_test2_verifier
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
from framework.tasks.test2_dataset import parse_test2_task_num, test2_task_subdir
from framework.verifier_selection import (
    VerifierSlot,
    select_verifier_for_task,
    valid_verifiers_for_task,
)

Category = Literal[
    "instance_mismatch",
    "corrupted_verifier",
    "p_arc_incorrect",
    "color_flip",
    "dynamic_correct",
    "retrieval",
]

# Relative sampling weights (sum = 1).
_CATEGORY_WEIGHTS: dict[Category, float] = {
    "p_arc_incorrect": 0.235,
    "instance_mismatch": 0.215,
    "corrupted_verifier": 0.13,
    "dynamic_correct": 0.25,
    "retrieval": 0.10333333333333333,
    "color_flip": 0.06666666666666667,
}

# Test2: lower retrieval / color_flip mass → higher P-ARC + corrupted verifier (ARC defaults unchanged).
_CATEGORY_WEIGHTS_TEST2: dict[Category, float] = {
    "p_arc_incorrect": 0.265,
    "instance_mismatch": 0.215,
    "corrupted_verifier": 0.160,
    "dynamic_correct": 0.25,
    "retrieval": 0.07333333333333333,
    "color_flip": 0.03666666666666667,
}
_SECOND_RETRIEVAL_WEIGHT_FACTOR = 0.35


def _category_weights_for_task(task_id: str) -> dict[Category, float]:
    try:
        parse_test2_task_num(task_id)
        return dict(_CATEGORY_WEIGHTS_TEST2)
    except ValueError:
        return dict(_CATEGORY_WEIGHTS)

# If a sampled corruption/category cannot yield a valid candidate quickly,
# stop trying that type and resample category.
_MAX_SAME_TYPE_TRIES = 10

# Retrieval: at most two retrieval items per 5-item batch, and if two are used,
# they must come from different source tasks. Neighbor ranks ``retrieval_slot +
# attempt`` are capped at 2 (0→1→2) while building each retrieval slot.
# Instance mismatch: mismatch ranks 0→1→2 by sorted output distance. At most three tries.
_MAX_RANK_LADDER_TRIES = 3


def _sample_category(
    rng: random.Random, allowed: Sequence[Category], weights: dict[Category, float]
) -> Category:
    cats = list(allowed)
    w = [weights[c] for c in cats]
    (c,) = rng.choices(cats, weights=w, k=1)
    return c


# Official ARC examples whose stored output is wrong, by task id and 0-based train
# index. A verifier that implements the task's rule correctly cannot match them, so
# they are left out of the validity check.
KNOWN_INCORRECT_TRAIN_EXAMPLES: dict[str, frozenset[int]] = {
    "e5062a87": frozenset({1}),  # train example 2
}


def verifier_matches_train_test_stable_dynamic50(task: ArcTask, v: Verifier) -> bool:
    """True if *v* matches train, test, ARC-GEN stable (if any), and 50 dynamic pairs."""
    skip = KNOWN_INCORRECT_TRAIN_EXAMPLES.get(task.task_id, frozenset())
    for i, p in enumerate(task.train_pairs):
        if i in skip:
            continue
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


def _deterministic_demo_pair_for_query(task: ArcTask) -> GridPair:
    """Fixed query pair for retrieval: first training example (deterministic)."""
    if not task.train_pairs:
        raise ValueError("No training pairs for retrieval query.")
    return copy.deepcopy(task.train_pairs[0])


def _try_corrupted_pair(task_id: str, task: ArcTask, rng: random.Random) -> GridPair | None:
    try:
        tnum = parse_test2_task_num(task_id)
    except ValueError:
        tnum = None
    if tnum is not None:
        if task.verifier is None or task.arc_gen_generator is None:
            return None
        try:
            task_dir = test2_task_subdir(tnum)
        except FileNotFoundError:
            return None
        for _ in range(16):
            try:
                pair = copy.deepcopy(task.arc_gen_generator(1, rng=rng)[0])  # type: ignore[misc]
                inp = copy.deepcopy(pair.input)
            except Exception:
                continue
            try:
                cfn, _, _tag = load_and_corrupt_test2_verifier(
                    task_id,
                    tnum,
                    task_dir,
                    rng=rng,
                    gold_fn=task.verifier,
                    sample_input=inp,
                    max_attempts=36,
                    max_normalized_cell_edit_distance=None,
                )
                bad = cfn(copy.deepcopy(inp))
                out_pair = GridPair(inp, bad)
                assert_int_grid_pair(out_pair, name="corrupted_verifier pair")
                return out_pair
            except Exception:
                continue
        return None
    if task.arc_gen_generator is None:
        return None
    for _ in range(8):
        try:
            pair = copy.deepcopy(task.arc_gen_generator(1, rng=rng)[0])  # type: ignore[misc]
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
            opts.extend(task.arc_gen_generator(50, rng=rng))
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
    rng: random.Random,
    *,
    cross_pool: int,
    mismatch_rank: int,
) -> GridPair | None:
    """Same-task mismatch: random anchor instance; *mismatch_rank*-th best swap in pool."""
    if task.arc_gen_generator is None:
        return None
    try:
        anchor = copy.deepcopy(task.arc_gen_generator(1, rng=rng)[0])  # type: ignore[misc]
        m = find_cross_dynamic_mismatch(
            task_id,
            anchor,
            pool_size=cross_pool,
            rank_index=mismatch_rank,
            rng=rng,
        )
    except Exception:
        return None
    if m is None:
        return None
    return synthetic_input_borrowed_output_pair(m)


def _try_p_arc_pair(task: ArcTask, rng: random.Random) -> GridPair | None:
    pair = sample_p_arc_pair(task, rng)
    if pair is None:
        return None
    try:
        assert_int_grid_pair(pair, name="p_arc_incorrect pair")
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


def _is_identity_pair(pair: GridPair) -> bool:
    return is_equal_grid(pair.input, pair.output)


@dataclass(frozen=True)
class DistributionClassificationBatch:
    task_id: str
    demonstrations: List[GridPair]
    items: List[DistributionClassificationItem]


def _draw_category(
    rng: random.Random,
    *,
    retrieval_count: int,
    weights: dict[Category, float],
) -> Category:
    allowed: List[Category] = list(weights.keys())
    if retrieval_count >= 2:
        allowed = [c for c in allowed if c != "retrieval"]
        return _sample_category(rng, allowed, weights)
    if retrieval_count >= 1:
        w = [
            (weights[c] * _SECOND_RETRIEVAL_WEIGHT_FACTOR) if c == "retrieval" else weights[c]
            for c in allowed
        ]
        (picked,) = rng.choices(allowed, weights=w, k=1)
        return picked
    return _sample_category(rng, allowed, weights)


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
    try:
        return is_equal_grid(out, pair.output)
    except Exception:
        return False


def _negative_pair_verified_corrupt(task: ArcTask, pair: GridPair) -> bool:
    """True iff the task-level verifier disagrees with *pair*'s output on its input.

    Negatives must not satisfy the rule. We use :func:`select_verifier_for_task`
    **without** ``dynamic_pairs``: passing ``dynamic_pairs=[pair]`` would require
    a verifier that already agrees with *pair*, which no true negative satisfies.
    """
    picked = select_verifier_for_task(task)
    if picked is None:
        return False
    _slot, vf = picked
    try:
        out = vf(copy.deepcopy(pair.input))
    except Exception:
        return True
    try:
        return not is_equal_grid(out, pair.output)
    except Exception:
        return True


def _positive_pair_matches_selected_verifier(task: ArcTask, pair: GridPair) -> bool:
    """True when *pair* is consistent with a verifier chosen using ``dynamic_pairs=[pair]``."""
    picked = select_verifier_for_task(task, dynamic_pairs=[pair])
    if picked is None:
        return False
    slot, verifier = picked
    try:
        out = verifier(copy.deepcopy(pair.input))
    except Exception:
        # Validation fallback: selected verifier crashed, try other valid ones.
        for alt_slot, alt_verifier in valid_verifiers_for_task(task):
            if alt_slot == slot:
                continue
            try:
                out_alt = alt_verifier(copy.deepcopy(pair.input))
            except Exception:
                continue
            try:
                if is_equal_grid(out_alt, pair.output):
                    return True
            except Exception:
                continue
        return False
    try:
        return is_equal_grid(out, pair.output)
    except Exception:
        return False


def _build_one_item(
    task_id: str,
    task: ArcTask,
    rng: random.Random,
    cat: Category,
    *,
    retrieval_neighbor_index: int,
    retrieval_exclude_task_ids: Optional[set[str]],
    mismatch_rank: int,
    neighbor_dynamic: int,
    cross_pool: int,
) -> Tuple[Optional[GridPair], bool, dict[str, object], bool]:
    """Returns (pair, label, meta, used_retrieval_slot)."""
    meta: dict[str, object] = {"category": cat}

    if cat == "dynamic_correct":
        try:
            pair = copy.deepcopy(task.arc_gen_generator(1, rng=rng)[0])  # type: ignore[misc]
        except Exception:
            return None, False, meta, False
        meta["kind"] = "arc_gen_dynamic_correct"
        return pair, True, meta, False

    if cat == "instance_mismatch":
        pair = _try_instance_mismatch(
            task_id,
            task,
            rng,
            cross_pool=cross_pool,
            mismatch_rank=mismatch_rank,
        )
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
        try:
            parse_test2_task_num(task_id)
            meta["corruption_source"] = "test2_verifier_ast"
        except ValueError:
            meta["corruption_source"] = "ast_pruned_or_golf_verifier"
        return pair, False, meta, False

    if cat == "p_arc_incorrect":
        pair = _try_p_arc_pair(task, rng)
        if pair is None:
            return None, False, meta, False
        try:
            parse_test2_task_num(task.task_id)
            meta["p_arc_source"] = "test2_hand_json"
        except ValueError:
            meta["p_arc_source"] = "h_arc_csv"
        meta["kind"] = "p_arc_plausible_wrong"
        return pair, False, meta, False

    # retrieval (deterministic query + ranked neighbors; no RNG)
    try:
        query = _deterministic_demo_pair_for_query(task)
        neigh = find_nearest_alternative_instances(
            query,
            task_id,
            None,
            criterion="task_embedding",
            k=max(12, neighbor_dynamic),
            num_dynamic=neighbor_dynamic,
        )
        if retrieval_exclude_task_ids:
            neigh = [n for n in neigh if n.ref_task_id not in retrieval_exclude_task_ids]
        if retrieval_neighbor_index >= len(neigh):
            return None, False, meta, False
        n = neigh[retrieval_neighbor_index]
        pair = copy.deepcopy(n.pair)
        meta["kind"] = "retrieval_nearest"
        meta["rank"] = retrieval_neighbor_index
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
    cat_weights = _category_weights_for_task(task_id)
    items: List[DistributionClassificationItem] = []
    retrieval_count = 0
    retrieval_task_ids: set[str] = set()

    for slot in range(5):
        item: Optional[DistributionClassificationItem] = None
        for _attempt in range(12):
            cat = _draw_category(rng, retrieval_count=retrieval_count, weights=cat_weights)
            # Keep retrying within the same category so collision re-samples preserve type.
            _max_tries = (
                _MAX_RANK_LADDER_TRIES
                if cat in ("retrieval", "instance_mismatch")
                else _MAX_SAME_TYPE_TRIES
            )
            for _same_cat_try in range(_max_tries):
                if cat == "retrieval":
                    retrieval_neighbor_index = min(retrieval_count + _same_cat_try, 2)
                    mismatch_rank = 0
                elif cat == "instance_mismatch":
                    retrieval_neighbor_index = 0
                    mismatch_rank = min(_same_cat_try, 2)
                else:
                    retrieval_neighbor_index = 0
                    mismatch_rank = 0
                pair, label, meta, used_retrieval = _build_one_item(
                    task_id,
                    task,
                    rng,
                    cat,
                    retrieval_neighbor_index=retrieval_neighbor_index,
                    retrieval_exclude_task_ids=retrieval_task_ids,
                    mismatch_rank=mismatch_rank,
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
                # Skip degenerate cases where input equals output.
                if _is_identity_pair(pair):
                    continue
                # Every negative must disagree with a verifier selected for this pair.
                if not label:
                    if not _negative_pair_verified_corrupt(task, pair):
                        continue
                if _pair_collides_with_items(pair, items):
                    continue
                # Additional uniqueness constraint: do not allow repeated inputs
                # within the same 5-item batch, regardless of output/category.
                if _input_collides_with_items(pair, items):
                    continue
                if used_retrieval:
                    retrieval_count += 1
                    ref_tid = meta.get("query_task")
                    if isinstance(ref_tid, str):
                        retrieval_task_ids.add(ref_tid)
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
                    cand = copy.deepcopy(
                        task.arc_gen_generator(1, rng=rng)[0]
                    )  # type: ignore[misc]
                except Exception:
                    continue
                try:
                    assert_int_grid_pair(cand, name=f"fallback classification item slot {slot}")
                except Exception:
                    continue
                if _is_identity_pair(cand):
                    continue
                # Fallback emits dynamic_correct positives; must match selected verifier.
                if not _positive_pair_matches_selected_verifier(task, cand):
                    continue
                if not _pair_collides_with_items(cand, items):
                    if _input_collides_with_items(cand, items):
                        continue
                    fb_pair = cand
                    break
            # Stronger fallback for strict uniqueness constraints:
            # sample larger dynamic pools and pick a random non-colliding input/output.
            if fb_pair is None:
                for _ in range(24):
                    try:
                        pool = task.arc_gen_generator(200, rng=rng)  # type: ignore[misc]
                    except Exception:
                        continue
                    rng.shuffle(pool)
                    for cand0 in pool:
                        try:
                            cand = copy.deepcopy(cand0)
                            assert_int_grid_pair(
                                cand, name=f"fallback-pool classification item slot {slot}"
                            )
                        except Exception:
                            continue
                        if _is_identity_pair(cand):
                            continue
                        if _pair_collides_with_items(cand, items):
                            continue
                        if _input_collides_with_items(cand, items):
                            continue
                        if not _positive_pair_matches_selected_verifier(task, cand):
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
