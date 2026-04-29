from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
ARC1_TRAIN_DIR = ROOT_DIR / "external" / "arc_original_train"
RUN1_SUMMARY = (
    ROOT_DIR
    / "evaluation"
    / "multi_dimension_eval_runs"
    / "GPT54"
    / "Run1"
    / "Evaluation_Run_2026-04-14_gpt54_low"
    / "run_all_400_unified_strict_accuracy.json"
)
RUN2_SUMMARY = (
    ROOT_DIR
    / "evaluation"
    / "multi_dimension_eval_runs"
    / "GPT54"
    / "Run2"
    / "Evaluation_Run_2026-04-14_gpt54_low_run2"
    / "run_all_400_unified_strict_accuracy.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _target_bucket_counts(
    *,
    n: int,
    both_total: int,
    run1_only_total: int,
    run2_only_total: int,
    neither_total: int,
) -> tuple[int, int, int, int]:
    total = both_total + run1_only_total + run2_only_total + neither_total
    if total <= 0:
        raise ValueError("No tasks available for bucket allocation.")

    # Largest remainder method for deterministic proportional quotas.
    quotas = [
        ("both", n * both_total / total),
        ("run1_only", n * run1_only_total / total),
        ("run2_only", n * run2_only_total / total),
        ("neither", n * neither_total / total),
    ]
    base = {k: int(v) for k, v in quotas}
    rem = n - sum(base.values())
    order = sorted(quotas, key=lambda kv: (kv[1] - int(kv[1])), reverse=True)
    for i in range(rem):
        base[order[i][0]] += 1
    return base["both"], base["run1_only"], base["run2_only"], base["neither"]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build a 50-task ARC1 subset matched to GPT-5.4 full-dataset pass profile."
    )
    p.add_argument("--size", type=int, default=50, help="Subset size.")
    p.add_argument("--seed", type=int, default=77, help="Sampling seed.")
    p.add_argument(
        "--out",
        default="Test2/ARC1_subsets/gpt54_perf_matched_subset_50.json",
        help="Output JSON path.",
    )
    args = p.parse_args()

    if args.size <= 0:
        raise SystemExit("--size must be > 0")
    if not ARC1_TRAIN_DIR.exists():
        raise SystemExit(f"Missing ARC1 training directory: {ARC1_TRAIN_DIR}")

    run1 = _load_json(RUN1_SUMMARY)
    run2 = _load_json(RUN2_SUMMARY)

    all_task_ids = sorted(p.stem for p in ARC1_TRAIN_DIR.glob("*.json"))
    all_set = set(all_task_ids)
    run1_pass = set(run1.get("full_pass_task_ids", []))
    run2_pass = set(run2.get("full_pass_task_ids", []))

    both = sorted((run1_pass & run2_pass) & all_set)
    run1_only = sorted((run1_pass - run2_pass) & all_set)
    run2_only = sorted((run2_pass - run1_pass) & all_set)
    neither = sorted((all_set - run1_pass - run2_pass))

    if args.size > len(all_task_ids):
        raise SystemExit(f"--size={args.size} exceeds available tasks={len(all_task_ids)}")

    n_both, n_r1, n_r2, n_neither = _target_bucket_counts(
        n=args.size,
        both_total=len(both),
        run1_only_total=len(run1_only),
        run2_only_total=len(run2_only),
        neither_total=len(neither),
    )

    if n_both > len(both) or n_r1 > len(run1_only) or n_r2 > len(run2_only) or n_neither > len(neither):
        raise SystemExit("Requested quota exceeds available tasks in one or more buckets.")

    rng = random.Random(args.seed)
    selected = []
    selected.extend(rng.sample(both, n_both))
    selected.extend(rng.sample(run1_only, n_r1))
    selected.extend(rng.sample(run2_only, n_r2))
    selected.extend(rng.sample(neither, n_neither))
    rng.shuffle(selected)

    selected_set = set(selected)
    selected_run1 = len(selected_set & run1_pass)
    selected_run2 = len(selected_set & run2_pass)
    selected_expected_avg = ((selected_run1 + selected_run2) / 2) / args.size

    out = {
        "selection_goal": "50-task ARC-AGI-1 training subset with GPT-5.4 full-dataset-like performance",
        "selection_policy": {
            "method": "stratified sampling by GPT-5.4 full-task pass buckets",
            "buckets": {
                "both_runs_pass": "task in full_pass_task_ids for Run1 and Run2",
                "run1_only_pass": "task in Run1 full_pass_task_ids but not Run2",
                "run2_only_pass": "task in Run2 full_pass_task_ids but not Run1",
                "neither_pass": "task in neither run full_pass_task_ids",
            },
            "quota_rule": "proportional to full ARC1 training-set bucket frequencies (largest-remainder rounding)",
            "seed": args.seed,
        },
        "source_runs": {
            "run1_tag": run1.get("run_tag"),
            "run2_tag": run2.get("run_tag"),
            "run1_full_task_accuracy": run1.get("full_task_accuracy"),
            "run2_full_task_accuracy": run2.get("full_task_accuracy"),
        },
        "population_counts": {
            "total_tasks": len(all_task_ids),
            "both_runs_pass": len(both),
            "run1_only_pass": len(run1_only),
            "run2_only_pass": len(run2_only),
            "neither_pass": len(neither),
        },
        "subset_counts": {
            "selected": args.size,
            "both_runs_pass": n_both,
            "run1_only_pass": n_r1,
            "run2_only_pass": n_r2,
            "neither_pass": n_neither,
            "selected_run1_pass_count": selected_run1,
            "selected_run2_pass_count": selected_run2,
            "selected_run1_accuracy": selected_run1 / args.size,
            "selected_run2_accuracy": selected_run2 / args.size,
            "selected_expected_average_accuracy": selected_expected_avg,
        },
        "selected_task_ids": sorted(selected),
    }

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT_DIR / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()

