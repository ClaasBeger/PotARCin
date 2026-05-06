#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from framework.eval_paths import dimension_out_dirs, multi_run_summary_dir
from scripts.run_all_dimensions_eval import run_all_dimensions_for_task


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_task_seed_map_from_source_run(*, model: str, run_tag: str) -> dict[str, int]:
    run_dir = multi_run_summary_dir(root_dir=ROOT_DIR, model=model, run_tag=run_tag)
    unified_path = run_dir / "run_all_400_unified.json"
    if not unified_path.is_file():
        raise SystemExit(f"Could not find source run file: {unified_path}")

    payload = _read_json(unified_path)
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise SystemExit(f"Invalid source run payload (missing runs): {unified_path}")

    out: dict[str, int] = {}
    for r in runs:
        if not isinstance(r, dict):
            continue
        tid = r.get("task_id")
        sd = r.get("seed")
        if isinstance(tid, str) and isinstance(sd, int):
            out[tid] = int(sd)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Repeat the same selected ARC benchmark tasks across multiple seeds to isolate "
            "model stochasticity (default: GPT-5.4, low, same selected tasks repeated 10 times)."
        )
    )
    p.add_argument(
        "--subset-json",
        type=str,
        default="Test2/ARC1_subsets/gpt54_nontrivial_success_subset_20_manual.json",
        help="Path (absolute or repo-relative) to JSON containing selected_task_ids.",
    )
    p.add_argument(
        "--run-tag",
        type=str,
        default="Evaluation_Run_gpt54_subset20_seedrepeat77",
        help="Run tag used for per-dimension output folders and summary location.",
    )
    p.add_argument("--model", type=str, default="gpt-5.4")
    p.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["openai", "anthropic", "google", "together", "openrouter"],
    )
    p.add_argument("--reasoning-effort", type=str, default="low")
    p.add_argument("--tools-enabled", action="store_true")
    p.add_argument("--num-repeats", type=int, default=10, help="Number of repeated evaluations per task (default: 10).")
    p.add_argument(
        "--source-run-tag",
        type=str,
        default="Evaluation_Run_2026-04-14_gpt54_low",
        help=(
            "Original benchmark run-tag to copy per-task seeds from "
            "(for request-level parity with the initial run)."
        ),
    )
    p.add_argument("--dimension-retries", type=int, default=2)
    p.add_argument("--retry-backoff-s", type=float, default=1.5)
    p.add_argument("--parallel-dimensions", type=int, default=1)
    p.add_argument("--classification-serial", action="store_true")
    p.add_argument(
        "--skip-completed-dimensions",
        action="store_true",
        help="If set, reuse existing artifacts when run_all_dimensions_eval determines a dim is complete.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    subset_path = Path(args.subset_json)
    if not subset_path.is_absolute():
        subset_path = (ROOT_DIR / subset_path).resolve()
    if not subset_path.is_file():
        raise SystemExit(f"subset json not found: {subset_path}")

    subset_payload = _read_json(subset_path)
    task_ids = subset_payload.get("selected_task_ids")
    if not isinstance(task_ids, list) or not all(isinstance(t, str) for t in task_ids):
        raise SystemExit(f"subset-json must contain selected_task_ids: {subset_path}")

    if args.num_repeats <= 0:
        raise SystemExit("--num-repeats must be >= 1")

    source_seed_map = _load_task_seed_map_from_source_run(model=args.model, run_tag=str(args.source_run_tag))
    missing = [tid for tid in task_ids if tid not in source_seed_map]
    if missing:
        preview = ", ".join(missing[:8])
        more = f" (+{len(missing)-8} more)" if len(missing) > 8 else ""
        raise SystemExit(
            "Some selected tasks are missing seeds in source run "
            f"'{args.source_run_tag}': {preview}{more}"
        )

    out_dirs = dimension_out_dirs(root_dir=ROOT_DIR, model=args.model, run_tag=args.run_tag)
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_path = (
        multi_run_summary_dir(root_dir=ROOT_DIR, model=args.model, run_tag=args.run_tag)
        / f"seed_repeat_stress_{ts}.json"
    )

    summary: dict[str, Any] = {
        "mode": "subset_seed_repeat_stress",
        "created_at": ts,
        "subset_json": str(subset_path),
        "task_count": len(task_ids),
        "run_tag": args.run_tag,
        "model": args.model,
        "provider": args.provider,
        "reasoning_effort": args.reasoning_effort,
        "tools_enabled": bool(args.tools_enabled),
        "seed_mode": "fixed_per_task_from_source_run",
        "source_run_tag": str(args.source_run_tag),
        "num_repeats": int(args.num_repeats),
        "task_seed_map": {tid: int(source_seed_map[tid]) for tid in task_ids},
        "dimension_retries": int(args.dimension_retries),
        "parallel_dimensions": int(args.parallel_dimensions),
        "classification_serial": bool(args.classification_serial),
        "runs": [],
    }
    _write_json(summary_path, summary)

    try:
        for ti, task_id in enumerate(task_ids):
            fixed_seed = int(source_seed_map[task_id])
            for ri in range(int(args.num_repeats)):
                print(
                    f"[task {ti+1}/{len(task_ids)}] {task_id}  "
                    f"[repeat {ri+1}/{int(args.num_repeats)}] [seed {fixed_seed}]"
                )
                rep = run_all_dimensions_for_task(
                    task_id=task_id,
                    model=args.model,
                    provider=args.provider,
                    reasoning_effort=args.reasoning_effort,
                    tools_enabled=bool(args.tools_enabled),
                    seed=fixed_seed,
                    definition_test_idx=0,
                    dimension_retries=int(args.dimension_retries),
                    retry_backoff_s=float(args.retry_backoff_s),
                    out_dirs=out_dirs,
                    skip_completed_dimensions=bool(args.skip_completed_dimensions),
                    parallel_dimensions=int(args.parallel_dimensions),
                    classification_serial=bool(args.classification_serial),
                    stitch_prior_work_context=False,
                    stitched_context_max_chars=12000,
                )
                rep["repeat_index"] = int(ri) + 1
                summary["runs"].append(rep)
                _write_json(summary_path, summary)
    except KeyboardInterrupt:
        _write_json(summary_path, summary)
        print(f"Interrupted; partial summary written: {summary_path}")
        raise SystemExit(130)

    print(f"Summary written: {summary_path}")


if __name__ == "__main__":
    main()

