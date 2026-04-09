from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from framework.dimensions.classification import (
    run_distribution_classification_api_evaluation_once,
)
from framework.dimensions.constrained_generation import (
    run_constrained_generation_api_evaluation_once,
)
from framework.dimensions.definition import run_definition_api_evaluation
from framework.dimensions.editing import run_editing_api_evaluation_once
from framework.dimensions.inversion import run_inversion_api_evaluation_once
from framework.tasks.arc_dataset import iter_tasks, load_task
from framework.tasks.base import TaskSource


_DIMENSION_ORDER = (
    "definition",
    "classification_distribution",
    "constrained_generation",
    "editing",
    "inversion",
)


def _is_transient_error(msg: str) -> bool:
    s = msg.lower()
    transient_markers = (
        "timed out",
        "timeout",
        "temporarily unavailable",
        "temporary failure",
        "connection reset",
        "connection aborted",
        "connection refused",
        "service unavailable",
        "too many requests",
        "rate limit",
        "502",
        "503",
        "504",
    )
    return any(tok in s for tok in transient_markers)


def run_all_dimensions_for_task(
    *,
    task_id: str,
    model: str = "gpt-5.4",
    reasoning_effort: str = "low",
    tools_enabled: bool = False,
    seed: int | None = None,
    definition_test_idx: int = 0,
    dimension_retries: int = 2,
    retry_backoff_s: float = 1.5,
) -> dict[str, Any]:
    """Run all five dimension evaluations for one task id.

    Returns a per-dimension report with success flag, result path and metrics/error.
    """
    # Independent per-dimension seeds avoid correlated sampling across dimensions.
    sys_rng = random.SystemRandom()
    if seed is None:
        dim_seeds = {
            "classification_distribution": int(sys_rng.randrange(1 << 31)),
            "editing": int(sys_rng.randrange(1 << 31)),
            "inversion": int(sys_rng.randrange(1 << 31)),
        }
    else:
        seeded = random.Random(seed)
        dim_seeds = {
            "classification_distribution": int(seeded.randrange(1 << 31)),
            "editing": int(seeded.randrange(1 << 31)),
            "inversion": int(seeded.randrange(1 << 31)),
        }

    report: dict[str, Any] = {
        "task_id": task_id,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "tools_enabled": tools_enabled,
        "seed": seed,
        "dimension_seeds": dim_seeds,
        "dimensions": {},
    }

    task = load_task(task_id)

    def _run(name: str, fn: Callable[[], dict[str, Any]]) -> None:
        attempts = max(1, int(dimension_retries) + 1)
        last_err = ""
        for attempt in range(1, attempts + 1):
            try:
                res = fn()
                report["dimensions"][name] = {
                    "ok": True,
                    "result_path": res.get("result_path"),
                    "metrics": res.get("metrics"),
                    "attempts": attempt,
                }
                return
            except Exception as e:
                last_err = str(e)
                if attempt >= attempts or not _is_transient_error(last_err):
                    break
                time.sleep(max(0.0, retry_backoff_s) * attempt)
        report["dimensions"][name] = {
            "ok": False,
            "error": last_err,
            "attempts": attempts,
        }

    _run(
        "definition",
        lambda: run_definition_api_evaluation(
            task=task,
            test_idx=definition_test_idx,
            model=model,
            api_endpoint="responses",
            reasoning_effort=reasoning_effort,
            tools_enabled=tools_enabled,
        ),
    )
    _run(
        "classification_distribution",
        lambda: run_distribution_classification_api_evaluation_once(
            task_id=task_id,
            model=model,
            reasoning_effort=reasoning_effort,
            tools_enabled=tools_enabled,
            seed=dim_seeds["classification_distribution"],
        ),
    )
    _run(
        "constrained_generation",
        lambda: run_constrained_generation_api_evaluation_once(
            task_id=task_id,
            model=model,
            reasoning_effort=reasoning_effort,
            tools_enabled=tools_enabled,
        ),
    )
    _run(
        "editing",
        lambda: run_editing_api_evaluation_once(
            task=task,
            model=model,
            reasoning_effort=reasoning_effort,
            tools_enabled=tools_enabled,
            seed=dim_seeds["editing"],
        ),
    )
    _run(
        "inversion",
        lambda: run_inversion_api_evaluation_once(
            task_id=task_id,
            model=model,
            reasoning_effort=reasoning_effort,
            tools_enabled=tools_enabled,
            seed=dim_seeds["inversion"],
        ),
    )
    return report


def _first_n_task_ids(n: int) -> list[str]:
    ids: list[str] = []
    for idx, task in enumerate(iter_tasks(split="train", source=TaskSource.ORIGINAL_ARC)):
        if idx >= n:
            break
        ids.append(task.task_id)
    return ids


def _write_run_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _extract_primary_metric(metrics: Any) -> float | None:
    if not isinstance(metrics, dict):
        return None
    for key in ("success_rate", "accuracy", "exact_match_rate"):
        val = metrics.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _build_overview(payload: dict[str, Any]) -> dict[str, Any]:
    runs = payload.get("runs", [])
    n_tasks = len(runs) if isinstance(runs, list) else 0
    by_dim: dict[str, dict[str, Any]] = {}
    for dim in _DIMENSION_ORDER:
        by_dim[dim] = {
            "ok_tasks": 0,
            "fail_tasks": 0,
            "ok_rate": 0.0,
            "mean_primary_metric": None,
            "metric_count": 0,
        }
    for run in runs if isinstance(runs, list) else []:
        dims = run.get("dimensions", {}) if isinstance(run, dict) else {}
        for dim in _DIMENSION_ORDER:
            entry = dims.get(dim, {}) if isinstance(dims, dict) else {}
            ok = bool(entry.get("ok"))
            if ok:
                by_dim[dim]["ok_tasks"] += 1
            else:
                by_dim[dim]["fail_tasks"] += 1
            m = _extract_primary_metric(entry.get("metrics"))
            if m is not None:
                cur = by_dim[dim].get("mean_primary_metric")
                if cur is None:
                    by_dim[dim]["mean_primary_metric"] = m
                else:
                    by_dim[dim]["mean_primary_metric"] = float(cur) + m
                by_dim[dim]["metric_count"] += 1
    for dim in _DIMENSION_ORDER:
        d = by_dim[dim]
        d["ok_rate"] = (d["ok_tasks"] / n_tasks) if n_tasks else 0.0
        if d["metric_count"] > 0 and d["mean_primary_metric"] is not None:
            d["mean_primary_metric"] = d["mean_primary_metric"] / d["metric_count"]
    return {
        "created_at": payload.get("created_at"),
        "task_count": n_tasks,
        "by_dimension": by_dim,
    }


def _write_overview_files(summary_path: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    overview = _build_overview(payload)
    json_path = summary_path.with_name(summary_path.stem + "_overview.json")
    md_path = summary_path.with_name(summary_path.stem + "_overview.md")
    json_path.write_text(json.dumps(overview, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Multi-Dimension Run Overview")
    lines.append("")
    lines.append(f"- Created: {overview.get('created_at')}")
    lines.append(f"- Tasks: {overview.get('task_count')}")
    lines.append("")
    lines.append("| Dimension | OK tasks | Failed tasks | OK rate | Mean primary metric |")
    lines.append("|---|---:|---:|---:|---:|")
    by_dim = overview.get("by_dimension", {})
    for dim in _DIMENSION_ORDER:
        d = by_dim.get(dim, {})
        mean_metric = d.get("mean_primary_metric")
        mean_str = "n/a" if mean_metric is None else f"{float(mean_metric):.4f}"
        lines.append(
            f"| {dim} | {int(d.get('ok_tasks', 0))} | {int(d.get('fail_tasks', 0))} | "
            f"{float(d.get('ok_rate', 0.0)):.2%} | {mean_str} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run GPT-based evaluations across all five dimensions."
    )
    p.add_argument("--task-id", default=None, help="Run all five dimensions for one task id.")
    p.add_argument(
        "--first-n",
        type=int,
        default=50,
        help="When --task-id is omitted: run first N ARC training tasks (default: 50).",
    )
    p.add_argument("--model", default="gpt-5.4", help="Model name")
    p.add_argument(
        "--reasoning-effort",
        default="low",
        help="Reasoning effort for responses API (e.g. low/medium/high)",
    )
    p.add_argument(
        "--tools-enabled",
        action="store_true",
        help="Enable code interpreter tool in API request",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed (default: None => true random per task/dimension)",
    )
    p.add_argument("--definition-test-idx", type=int, default=0, help="Definition test index")
    p.add_argument(
        "--dimension-retries",
        type=int,
        default=2,
        help="Retries per dimension on transient errors (default: 2)",
    )
    p.add_argument(
        "--retry-backoff-s",
        type=float,
        default=1.5,
        help="Base linear backoff in seconds between retries (default: 1.5)",
    )
    p.add_argument(
        "--summary-out",
        default=None,
        help="Optional output path for run summary JSON. Default: evaluation/multi_dimension_eval_runs/<timestamp>.json",
    )
    args = p.parse_args()

    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_path = (
        Path(args.summary_out)
        if args.summary_out
        else ROOT_DIR / "evaluation" / "multi_dimension_eval_runs" / f"run_{ts}.json"
    )

    if args.task_id:
        run = run_all_dimensions_for_task(
            task_id=args.task_id,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            tools_enabled=args.tools_enabled,
            seed=args.seed,
            definition_test_idx=args.definition_test_idx,
            dimension_retries=args.dimension_retries,
            retry_backoff_s=args.retry_backoff_s,
        )
        payload: dict[str, Any] = {
            "mode": "single_task",
            "created_at": ts,
            "task_count": 1,
            "runs": [run],
        }
        _write_run_summary(summary_path, payload)
        print(f"Summary saved: {summary_path}")
        return

    task_ids = _first_n_task_ids(args.first_n)
    runs: list[dict[str, Any]] = []
    for i, task_id in enumerate(task_ids):
        run_seed = (args.seed + i) if args.seed is not None else None
        seed_txt = "random" if run_seed is None else str(run_seed)
        print(f"[{i+1}/{len(task_ids)}] {task_id} (seed={seed_txt})")
        run = run_all_dimensions_for_task(
            task_id=task_id,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            tools_enabled=args.tools_enabled,
            seed=run_seed,
            definition_test_idx=args.definition_test_idx,
            dimension_retries=args.dimension_retries,
            retry_backoff_s=args.retry_backoff_s,
        )
        runs.append(run)
        ok_dims = sum(1 for d in run["dimensions"].values() if d.get("ok"))
        print(f"  -> completed {ok_dims}/5 dimensions")

    payload = {
        "mode": "first_n",
        "created_at": ts,
        "task_count": len(runs),
        "runs": runs,
    }
    _write_run_summary(summary_path, payload)
    print(f"Summary saved: {summary_path}")
    if len(runs) > 1:
        overview_json, overview_md = _write_overview_files(summary_path, payload)
        print(f"Overview saved: {overview_json}")
        print(f"Overview saved: {overview_md}")


if __name__ == "__main__":
    main()

