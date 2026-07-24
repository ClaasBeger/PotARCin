#!/usr/bin/env python3
"""Fill missing full 5-dim repeats for the GPT-5.4 same-seed stress run.

Re-runs exactly the deficit slots (43) with the original per-task seeds into the
same run tag. No skip: each fill is a real full evaluation.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from framework.eval_paths import dimension_out_dirs, multi_run_summary_dir
from scripts.run_all_dimensions_eval import run_all_dimensions_for_task
from scripts.run_seed_repeat_stress_subset import _load_task_seed_map_from_source_run

RUN_TAG = "Evaluation_Run_gpt54_subset20_repeat10_same_request"
SOURCE_TAG = "Evaluation_Run_2026-04-14_gpt54_low"
MODEL = "gpt-5.4"
DIMS = [
    "definition",
    "classification_distribution",
    "constrained_generation",
    "editing",
    "inversion",
]
# Exact deficits vs the original 20×10 summary (157 valid → need 43 more).
DEFICITS = {
    "6430c8c4": 3,
    "673ef223": 10,
    "6773b310": 10,
    "681b3aeb": 10,
    "77fdfe62": 10,
}


def main() -> None:
    seed_map = _load_task_seed_map_from_source_run(model=MODEL, run_tag=SOURCE_TAG)
    out_dirs = dimension_out_dirs(root_dir=ROOT_DIR, model=MODEL, run_tag=RUN_TAG)
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_path = (
        multi_run_summary_dir(root_dir=ROOT_DIR, model=MODEL, run_tag=RUN_TAG)
        / f"seed_repeat_fill_{ts}.json"
    )
    summary = {
        "mode": "subset_seed_repeat_fill_to_200",
        "created_at": ts,
        "run_tag": RUN_TAG,
        "source_run_tag": SOURCE_TAG,
        "model": MODEL,
        "deficits": DEFICITS,
        "runs": [],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    total = sum(DEFICITS.values())
    done = 0
    for task_id, n in DEFICITS.items():
        seed = int(seed_map[task_id])
        for i in range(n):
            done += 1
            print(f"[{done}/{total}] {task_id} fill {i+1}/{n} seed={seed}", flush=True)
            rep = run_all_dimensions_for_task(
                task_id=task_id,
                model=MODEL,
                provider=None,
                reasoning_effort="low",
                tools_enabled=False,
                seed=seed,
                definition_test_idx=0,
                dimension_retries=2,
                retry_backoff_s=1.5,
                out_dirs=out_dirs,
                skip_completed_dimensions=False,
                parallel_dimensions=1,
                classification_serial=True,
            )
            ok = all(bool((rep.get("dimensions") or {}).get(d, {}).get("ok")) for d in DIMS)
            print(f"  -> full_valid={ok}", flush=True)
            rep["repeat_index"] = i + 1
            rep["full_valid"] = ok
            summary["runs"].append(rep)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    n_ok = sum(1 for r in summary["runs"] if r.get("full_valid"))
    print(f"Fill summary: {summary_path}")
    print(f"New full_valid runs: {n_ok}/{total} (target combined 157+{n_ok}= {157+n_ok}/200)")


if __name__ == "__main__":
    main()
