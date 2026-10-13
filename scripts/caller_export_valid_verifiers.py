"""Export a CSV mapping each ARC task to verifier slots valid at inference time.

Uses ``valid_verifier_slots_for_task`` in
``framework.dimensions.classification_distribution`` (train, test, ARC-GEN stable,
and 50 dynamic ARC-GEN pairs). Run once offline; at inference,
``framework.verifier_selection.select_verifier_for_task`` reads this CSV (and may
re-check instance-specific dynamic pairs).

Example::

    python scripts/caller_export_valid_verifiers.py --out task_valid_verifiers.csv --num-tasks 400

A stricter, reproducible rebuild across all cores::

    python scripts/caller_export_valid_verifiers.py --dynamic-samples 500 --seed 0 --workers 16 --call-timeout 10
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor

from framework.dimensions.classification_distribution import valid_verifier_slots_for_task
from framework.tasks.arc_dataset import iter_tasks, load_task
from framework.tasks.base import TaskSource

_SLOT_COLUMNS: tuple[str, ...] = ("re_arc", "google", "keymoon", "neurips", "custom")


def _valid_slots(job: tuple[str, int, int | None, float | None]) -> tuple[list[str], float]:
    """Valid slots for one task and the seconds it took.

    Loads the task in the worker so nothing unpicklable crosses processes.
    """
    task_id, dynamic_samples, seed, call_timeout_s = job
    start = time.monotonic()
    valid = valid_verifier_slots_for_task(
        load_task(task_id), dynamic_samples=dynamic_samples, seed=seed, call_timeout_s=call_timeout_s
    )
    return valid, time.monotonic() - start


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a CSV of valid verifier slots per ARC training task."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT_DIR / "task_valid_verifiers.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=400,
        help="Maximum tasks to scan (0 = entire training split).",
    )
    parser.add_argument(
        "--dynamic-samples",
        type=int,
        default=50,
        help="Generator pairs each verifier must match (default 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the generator pairs, so every slot sees the same draws (default: unseeded).",
    )
    parser.add_argument(
        "--call-timeout",
        type=float,
        default=None,
        help="Seconds a single verifier call may take before it counts as a mismatch (default: no limit).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Processes checking tasks in parallel (default 1).",
    )
    args = parser.parse_args()

    fieldnames = [
        "task_id",
        "valid_verifier_slots",
        "selected_verifier_slot",
        "num_valid",
        *_SLOT_COLUMNS,
    ]

    limit = args.num_tasks if args.num_tasks > 0 else None

    # Keep an existing CSV's choices where they still hold. The dynamic check samples
    # generator pairs, so a borderline slot can pass one rebuild and fail the next: a
    # rebuild only removes the previously listed slots that now fail, in their previous
    # order, and a hand-picked canonical slot is kept while it stays valid.
    previous: dict[str, dict[str, str]] = {}
    if args.out.is_file():
        with args.out.open(newline="", encoding="utf-8") as f:
            for old in csv.DictReader(f):
                previous[old["task_id"]] = old

    task_ids = [task.task_id for task in iter_tasks(split="train", source=TaskSource.ORIGINAL_ARC)]
    if limit is not None:
        task_ids = task_ids[:limit]
    jobs = [(task_id, args.dynamic_samples, args.seed, args.call_timeout) for task_id in task_ids]
    results: list[list[str]] = []
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        for task_id, (valid, seconds) in zip(
            task_ids, pool.map(_valid_slots, jobs, chunksize=1) if pool else map(_valid_slots, jobs)
        ):
            results.append(valid)
            if seconds > 60:
                print(f"{task_id} took {seconds:.0f}s", file=sys.stderr)
            if len(results) % 50 == 0:
                print(f"Processed {len(results)} tasks...", file=sys.stderr)
    finally:
        if pool:
            pool.shutdown()

    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for task_id, valid in zip(task_ids, results):
            old = previous.get(task_id)
            if old is not None:
                old_valid = [x for x in (old.get("valid_verifier_slots") or "").split(";") if x]
                still_valid = [x for x in old_valid if x in valid]
                if still_valid:
                    valid = still_valid
            kept = (old or {}).get("selected_verifier_slot", "").strip()
            selected = kept if kept in valid else (valid[0] if valid else "")
            row = {
                "task_id": task_id,
                "valid_verifier_slots": ";".join(valid),
                "selected_verifier_slot": selected,
                "num_valid": len(valid),
            }
            for slot in _SLOT_COLUMNS:
                row[slot] = 1 if slot in valid else 0
            writer.writerow(row)

    print(f"Wrote {args.out.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
