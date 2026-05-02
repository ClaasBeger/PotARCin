#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from framework.grids import GridPair, is_equal_grid
from framework.tasks.test2_dataset import TEST2_DIR, iter_test2_tasks, parse_test2_task_num
from framework.verifier_selection import select_verifier_for_task


def _task_subdir(root: Path, n: int) -> Path:
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith(f"t{n}_"):
            return p
    raise FileNotFoundError(f"Missing Test2 subdirectory for t{n}")


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


def main() -> None:
    p = argparse.ArgumentParser(description="Standardize and audit Test2 for multi-dimension evaluation.")
    p.add_argument("--test2-root", type=Path, default=TEST2_DIR)
    p.add_argument("--dynamic-samples", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args()

    root = args.test2_root.resolve()
    rng = random.Random(args.seed)

    schema_fixes = 0
    corrupted_fail_checks = 0
    bad_tasks: list[str] = []
    per_task: list[dict] = []

    for task in iter_test2_tasks():
        tnum = parse_test2_task_num(task.task_id)
        task_dir = _task_subdir(root, tnum)
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
        _slot, verifier = picked

        train_ok = True
        for p in task.train_pairs:
            try:
                if not is_equal_grid(verifier(p.input), p.output):
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
        if task.arc_gen_generator is None:
            dyn_ok = False
        else:
            try:
                generated = task.arc_gen_generator(int(args.dynamic_samples), rng=rng)  # type: ignore[misc]
            except Exception:
                dyn_ok = False
                generated = []
            for p in generated:
                try:
                    matched = is_equal_grid(verifier(p.input), p.output)
                except Exception:
                    matched = False
                if not matched:
                    dyn_ok = False
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

        ok = train_ok and test_ok and dyn_ok and corrupted_rejected
        if not ok:
            bad_tasks.append(task.task_id)
        per_task.append(
            {
                "task_id": task.task_id,
                "train_ok": train_ok,
                "test_ok": test_ok,
                "dynamic_ok": dyn_ok,
                "corrupted_rejected": corrupted_rejected,
            }
        )

    report = {
        "schema_fixes": schema_fixes,
        "task_count": len(per_task),
        "tasks_with_issues": bad_tasks,
        "corrupted_rejection_pass_count": corrupted_fail_checks,
        "tasks": per_task,
    }
    out = args.report or (root / "test2_multi_dimension_audit_report.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Audit report written to: {out}")
    if bad_tasks:
        raise SystemExit(f"Audit failed for {len(bad_tasks)} task(s)")


if __name__ == "__main__":
    main()
