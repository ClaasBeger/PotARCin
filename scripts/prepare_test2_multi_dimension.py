#!/usr/bin/env python3
"""Audit + preflight for Test2 multi-dimension evaluation.

Steps performed for each ``test2_t<n>`` task (n=1..50):

1. Standardize **P-ARC** hand-corrupt files (same naming as legacy corrupt JSON).
   Each ``Test2/t<n>_*`` subdirectory must contain exactly **3** files (either
   ``t<n>_corrupt<i>.json`` or ``t<n>_corrupted<i>.json``); each is rewritten
   in-place if it deviates from ``{"train": [], "test": [{...}]}`` (one test pair).

2. Verifier sanity check. Selects the task verifier via
   :func:`framework.verifier_selection.select_verifier_for_task` and asserts
   it reproduces every published train + test pair, plus
   ``--dynamic-samples`` (default 300) generator samples (seeded RNG). Also
   confirms the verifier *rejects* every P-ARC pair (wrong published output).

3. Retrieval lookup. For each Test2 task, dump the top-K nearest neighbors
   produced by :func:`framework.corruption.nearest_inputs.find_nearest_alternative_instances`
   in ``task_embedding`` mode (mixed pool: ARC-AGI-1 train tasks +
   other Test2 tasks). The dump is written next to the report and is the
   "lookup" the multi-dim eval consults at sampling time.

4. Distribution-classification end-to-end smoke test. Runs
   :func:`framework.dimensions.classification_distribution.sample_distribution_classification_batch`
   with a fixed seed. Records per-task category coverage and asserts every
   negative item is rejected by the task verifier (the multi-dim wiring
   requirement).

A combined JSON report is written; the script exits non-zero if any task
fails any check.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from framework.corruption.nearest_inputs import find_nearest_alternative_instances
from framework.dimensions.classification_distribution import (
    sample_distribution_classification_batch,
)
from framework.grids import GridPair, is_equal_grid
from framework.tasks.test2_dataset import (
    TEST2_DIR,
    canonical_test2_task_id,
    iter_test2_tasks,
    parse_test2_task_num,
    test2_task_subdir,
)
from framework.verifier_selection import select_verifier_for_task


def _task_subdir_for_audit(root: Path, n: int) -> Path:
    """Resolve per-task folder; use canonical layout helper when *root* is default Test2."""
    if root.resolve() == TEST2_DIR.resolve():
        return test2_task_subdir(n)
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith(f"t{n}_"):
            return p
    raise FileNotFoundError(f"Missing Test2 subdirectory for t{n} under {root}")


def _corrupted_paths(task_dir: Path, n: int) -> list[Path]:
    canonical = sorted(task_dir.glob(f"t{n}_corrupted*.json"))
    if canonical:
        return canonical
    return sorted(task_dir.glob(f"t{n}_corrupt*.json"))


def _standardize_corrupted(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    train = payload.get("train")
    test = payload.get("test")
    already = isinstance(train, list) and len(train) == 0 and isinstance(test, list) and len(test) == 1
    if already:
        return False

    chosen: dict | None = None
    if isinstance(test, list) and test:
        ex = test[0]
        if isinstance(ex, dict) and "input" in ex and "output" in ex:
            chosen = {"input": ex["input"], "output": ex["output"]}
    if chosen is None and isinstance(train, list):
        for ex in train:
            if isinstance(ex, dict) and "input" in ex and "output" in ex:
                chosen = {"input": ex["input"], "output": ex["output"]}
                break
    if chosen is None:
        raise ValueError(f"{path}: could not recover corrupted test pair")

    path.write_text(json.dumps({"train": [], "test": [chosen]}, separators=(",", ":")), encoding="utf-8")
    return True


def _build_retrieval_dump(
    task,
    *,
    k_per_query: int,
    neighbor_dynamic: int,
) -> dict:
    """Dump nearest neighbors for the deterministic query pair (train[0])."""
    if not task.train_pairs:
        return {"error": "no train pairs"}
    query = task.train_pairs[0]
    try:
        neigh = find_nearest_alternative_instances(
            query,
            task.task_id,
            None,
            criterion="task_embedding",
            k=k_per_query,
            num_dynamic=neighbor_dynamic,
        )
    except Exception as e:
        return {"error": f"retrieval failed: {e!r}"}

    sources_seen: set[str] = set()
    items: list[dict] = []
    for n in neigh:
        sources_seen.add("test2" if n.ref_task_id.startswith("test2_") else "arc1")
        items.append(
            {
                "ref_task_id": n.ref_task_id,
                "source": n.source,
                "detail": n.detail,
                "normalized_distance_avg": n.normalized_distance_avg,
            }
        )
    return {
        "query_train_idx": 0,
        "k": k_per_query,
        "sources": sorted(sources_seen),
        "neighbors": items,
    }


def _smoke_classification_batch(
    task_id: str,
    *,
    seed: int,
    select_verifier_fn,
) -> dict:
    """Sample one 5-way distribution-classification batch and assert the wirings."""
    rng = random.Random(seed)
    try:
        batch = sample_distribution_classification_batch(task_id, rng)
    except Exception as e:
        return {"ok": False, "error": f"sampler failed: {e!r}"}

    cats: dict[str, int] = {}
    for it in batch.items:
        cats[it.category] = cats.get(it.category, 0) + 1
    item_meta = [
        {
            "slot": it.meta.get("slot"),
            "category": it.category,
            "kind": it.meta.get("kind"),
            "same_distribution": it.same_distribution,
            "rank": it.meta.get("rank"),
            "query_task": it.meta.get("query_task"),
        }
        for it in batch.items
    ]

    # Wiring check: for every negative item, verifier must NOT reproduce the output.
    failed_negatives: list[dict] = []
    failed_positives: list[dict] = []
    picked = select_verifier_fn()
    if picked is None:
        return {"ok": False, "error": "no verifier", "categories": cats, "items": item_meta}
    _slot, verifier = picked
    for idx, it in enumerate(batch.items):
        try:
            out = verifier(it.pair.input)
            agrees = is_equal_grid(out, it.pair.output)
        except Exception:
            agrees = False
        if it.same_distribution and not agrees:
            failed_positives.append({"slot": idx, "category": it.category})
        if (not it.same_distribution) and agrees:
            failed_negatives.append({"slot": idx, "category": it.category})

    return {
        "ok": not failed_negatives and not failed_positives,
        "categories": cats,
        "items": item_meta,
        "failed_negatives": failed_negatives,
        "failed_positives": failed_positives,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test2-root", type=Path, default=TEST2_DIR)
    p.add_argument("--dynamic-samples", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--retrieval-k", type=int, default=12)
    p.add_argument("--retrieval-num-dynamic", type=int, default=0)
    p.add_argument(
        "--retrieval-out",
        type=Path,
        default=None,
        help="Path for retrieval lookup dump JSON (default: Test2/test2_retrieval_lookup.json)",
    )
    p.add_argument(
        "--skip-classification-smoke",
        action="store_true",
        help="Skip the 5-way distribution-classification end-to-end smoke test (faster).",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Path for the audit report (default: Test2/test2_multi_dimension_audit_report.json)",
    )
    args = p.parse_args()

    root = args.test2_root.resolve()
    rng = random.Random(args.seed)

    schema_fixes = 0
    corrupted_fail_checks = 0
    bad_tasks: list[str] = []
    per_task: list[dict] = []
    retrieval_lookup: dict[str, dict] = {}

    for task in iter_test2_tasks():
        tnum = parse_test2_task_num(task.task_id)
        task_dir = _task_subdir_for_audit(root, tnum)
        cpaths = _corrupted_paths(task_dir, tnum)
        if len(cpaths) != 3:
            bad_tasks.append(task.task_id)
            per_task.append({"task_id": task.task_id, "error": f"expected 3 corrupted files, found {len(cpaths)}"})
            continue
        for cp in cpaths:
            if _standardize_corrupted(cp):
                schema_fixes += 1

        picked = select_verifier_for_task(task)
        if picked is None:
            bad_tasks.append(task.task_id)
            per_task.append({"task_id": task.task_id, "error": "no verifier available"})
            continue
        slot, verifier = picked

        train_ok = True
        for tp in task.train_pairs:
            try:
                if not is_equal_grid(verifier(tp.input), tp.output):
                    train_ok = False
                    break
            except Exception:
                train_ok = False
                break
        test_ok = True
        for inp, out in zip(task.test_inputs, task.test_outputs):
            try:
                if not is_equal_grid(verifier(inp), out):
                    test_ok = False
                    break
            except Exception:
                test_ok = False
                break

        dyn_ok = True
        dyn_first_failure = None
        if task.arc_gen_generator is None:
            dyn_ok = False
        else:
            try:
                generated = task.arc_gen_generator(int(args.dynamic_samples), rng=rng)  # type: ignore[misc]
            except Exception as e:
                dyn_ok = False
                dyn_first_failure = f"generator raised: {e!r}"
                generated = []
            for i, gp in enumerate(generated):
                try:
                    matched = is_equal_grid(verifier(gp.input), gp.output)
                except Exception as e:
                    matched = False
                    dyn_first_failure = f"verifier raised on sample {i}: {e!r}"
                if not matched:
                    dyn_ok = False
                    if dyn_first_failure is None:
                        dyn_first_failure = f"verifier disagrees on sample {i}"
                    break

        corrupted_rejected = True
        for cp in cpaths:
            payload = json.loads(cp.read_text(encoding="utf-8"))
            ex = payload["test"][0]
            pair = GridPair(ex["input"], ex["output"])
            try:
                same = is_equal_grid(verifier(pair.input), pair.output)
            except Exception:
                same = False
            if same:
                corrupted_rejected = False
                break
        if corrupted_rejected:
            corrupted_fail_checks += 1

        retrieval = _build_retrieval_dump(
            task, k_per_query=args.retrieval_k, neighbor_dynamic=args.retrieval_num_dynamic
        )
        retrieval_lookup[canonical_test2_task_id(task.task_id)] = retrieval

        if args.skip_classification_smoke:
            class_smoke = {"skipped": True}
            class_ok = True
        else:
            class_smoke = _smoke_classification_batch(
                task.task_id, seed=args.seed, select_verifier_fn=lambda: select_verifier_for_task(task)
            )
            class_ok = bool(class_smoke.get("ok", False))

        ok = train_ok and test_ok and dyn_ok and corrupted_rejected and class_ok
        if not ok:
            bad_tasks.append(task.task_id)
        per_task.append(
            {
                "task_id": task.task_id,
                "verifier_slot": slot,
                "train_ok": train_ok,
                "test_ok": test_ok,
                "dynamic_ok": dyn_ok,
                "dynamic_failure": dyn_first_failure,
                "corrupted_rejected": corrupted_rejected,
                "retrieval": {
                    "k": retrieval.get("k"),
                    "sources": retrieval.get("sources"),
                    "neighbor_count": len(retrieval.get("neighbors", []) or []),
                    "error": retrieval.get("error"),
                },
                "classification_smoke": class_smoke,
            }
        )

    report = {
        "schema_fixes": schema_fixes,
        "task_count": len(per_task),
        "tasks_with_issues": bad_tasks,
        "corrupted_rejection_pass_count": corrupted_fail_checks,
        "dynamic_samples_per_task": int(args.dynamic_samples),
        "retrieval_k_per_task": int(args.retrieval_k),
        "tasks": per_task,
    }
    out = args.report or (root / "test2_multi_dimension_audit_report.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Audit report written to: {out}")

    retrieval_path = args.retrieval_out or (root / "test2_retrieval_lookup.json")
    retrieval_path.write_text(
        json.dumps(
            {
                "criterion": "task_embedding",
                "k_per_task": int(args.retrieval_k),
                "neighbor_dynamic": int(args.retrieval_num_dynamic),
                "by_task": retrieval_lookup,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Retrieval lookup written to: {retrieval_path}")

    if bad_tasks:
        raise SystemExit(f"Audit failed for {len(bad_tasks)} task(s): {bad_tasks}")


if __name__ == "__main__":
    main()
