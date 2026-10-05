"""Export a CSV mapping each ARC task to verifier slots valid at inference time.

Uses ``valid_verifier_slots_for_task`` in
``framework.dimensions.classification_distribution`` (train, test, ARC-GEN stable,
and 50 dynamic ARC-GEN pairs). Run once offline; at inference,
``framework.verifier_selection.select_verifier_for_task`` reads this CSV (and may
re-check instance-specific dynamic pairs).

Example::

    python scripts/caller_export_valid_verifiers.py --out task_valid_verifiers.csv --num-tasks 400
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import csv

from framework.dimensions.classification_distribution import valid_verifier_slots_for_task
from framework.tasks.arc_dataset import iter_tasks
from framework.tasks.base import TaskSource

_SLOT_COLUMNS: tuple[str, ...] = ("re_arc", "google", "keymoon", "neurips", "custom")


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
    args = parser.parse_args()

    fieldnames = [
        "task_id",
        "valid_verifier_slots",
        "selected_verifier_slot",
        "num_valid",
        *_SLOT_COLUMNS,
    ]

    limit = args.num_tasks if args.num_tasks > 0 else None

    # Keep canonical choices made by hand in an existing CSV, as long as the slot
    # is still valid; otherwise the first valid slot becomes the canonical one.
    previous: dict[str, str] = {}
    if args.out.is_file():
        with args.out.open(newline="", encoding="utf-8") as f:
            for old in csv.DictReader(f):
                previous[old["task_id"]] = (old.get("selected_verifier_slot") or "").strip()

    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, task in enumerate(
            iter_tasks(split="train", source=TaskSource.ORIGINAL_ARC)
        ):
            if limit is not None and idx >= limit:
                break
            valid = valid_verifier_slots_for_task(task)
            kept = previous.get(task.task_id, "")
            selected = kept if kept in valid else (valid[0] if valid else "")
            row = {
                "task_id": task.task_id,
                "valid_verifier_slots": ";".join(valid),
                "selected_verifier_slot": selected,
                "num_valid": len(valid),
            }
            for slot in _SLOT_COLUMNS:
                row[slot] = 1 if slot in valid else 0
            writer.writerow(row)
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1} tasks...", file=sys.stderr)

    print(f"Wrote {args.out.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
