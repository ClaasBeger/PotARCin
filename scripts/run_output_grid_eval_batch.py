from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from framework.eval_paths import output_grid_run_dir
from framework.dimensions.output_grid_eval import run_output_grid_api_evaluation_once
from framework.tasks.arc_dataset import iter_tasks
from framework.tasks.base import TaskSource


def _is_transient_error(msg: str) -> bool:
    """HTTP / network errors that are worth retrying (aligned with run_all_dimensions_eval)."""
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


def _all_train_task_ids() -> list[str]:
    return [t.task_id for t in iter_tasks(split="train", source=TaskSource.ORIGINAL_ARC)]


def _load_task_ids_from_file(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    for ln in lines:
        s = ln.split("#", 1)[0].strip()
        if s:
            out.append(s)
    return out


def _has_logged_result(out_dir: Path, task_id: str) -> bool:
    """True if any prior per-task JSON exists for this task (same naming as run_output_grid_api_evaluation_once)."""
    if not out_dir.is_dir():
        return False
    return any(out_dir.glob(f"{task_id}_output_grid_eval_*.json"))


def _resolve_out_dir(out_dir: Path, run_tag: str | None, model: str) -> Path:
    if not out_dir.is_absolute():
        out_dir = ROOT_DIR / out_dir
    if run_tag:
        out_dir = output_grid_run_dir(root_dir=ROOT_DIR, model=model, run_tag=run_tag)
    return out_dir


def _run_one_task(
    *,
    task_id: str,
    test_idx: int,
    model: str,
    provider: str | None,
    reasoning_effort: str | None,
    tools_enabled: bool,
    max_train_demos: int | None,
    out_dir: Path,
    retries: int,
    retry_backoff_s: float,
) -> dict[str, Any]:
    """Call the API with retries on transient failures; returns a result row dict."""
    attempts = max(1, retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            res = run_output_grid_api_evaluation_once(
                task_id=task_id,
                test_idx=test_idx,
                model=model,
                provider=provider,
                reasoning_effort=reasoning_effort,
                tools_enabled=tools_enabled,
                max_train_demos=max_train_demos,
                out_dir=out_dir,
            )
            return {
                "task_id": task_id,
                "attempts": attempt,
                "result_path": res.get("result_path"),
                "evaluation_ok": res.get("evaluation_ok"),
                "exact_match": res.get("exact_match"),
                "metrics": res.get("metrics"),
                "error": res.get("error"),
            }
        except Exception as e:
            msg = str(e)
            if attempt < attempts and _is_transient_error(msg):
                time.sleep(retry_backoff_s * attempt)
                continue
            return {
                "task_id": task_id,
                "attempts": attempt,
                "result_path": None,
                "evaluation_ok": False,
                "exact_match": None,
                "metrics": None,
                "error": f"{type(e).__name__}: {e}",
            }
    raise RuntimeError("output grid batch: internal retry loop fell through")  # pragma: no cover


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run output-grid ARC evals over many tasks. Per-task JSONs are written under "
            "--out-dir (optionally scoped with --run-tag). "
            "With --skip-completed (default), tasks that already have a logged result file "
            "are skipped so you can re-run with --batch-size to continue a long job."
        )
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--tasks",
        nargs="+",
        metavar="TASK_ID",
        help="One or more ARC task ids (space-separated)",
    )
    src.add_argument(
        "--tasks-file",
        type=Path,
        help="Text file: one task id per line (# comments allowed)",
    )
    src.add_argument(
        "--all-train",
        action="store_true",
        help="All training tasks from external/arc_original_train/ (same order as iter_tasks)",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help=(
            "Optional run tag for scoped output. Files are written to "
            "evaluation/output_grid_eval_results/<ModelAlias>/<Run1|Run2>/<run_tag>/. "
            "Resume discovery only sees JSONs under this directory."
        ),
    )
    parser.add_argument(
        "--skip-completed",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "Skip task ids that already have any {task_id}_output_grid_eval_*.json under the "
            "effective output directory (default: skip)"
        ),
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="After filtering, drop the first N tasks (advanced ordering control)",
    )
    parser.add_argument(
        "--batch-size",
        "--limit",
        type=int,
        default=None,
        dest="batch_size",
        metavar="N",
        help=(
            "Run at most N tasks this invocation (after --skip-completed / --skip). "
            "Use this to process the benchmark in chunks; re-run the same command to continue."
        ),
    )
    parser.add_argument("--test-idx", type=int, default=0, help="Which test example per task")
    parser.add_argument("--model", default="gpt-5.4", help="Model name")
    parser.add_argument(
        "--provider",
        default=None,
        choices=["openai", "anthropic", "google"],
        help="LLM provider (default: infer from --model)",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="low",
        help="Reasoning effort for responses API (e.g. low/medium/high)",
    )
    parser.add_argument(
        "--tools-enabled",
        action="store_true",
        help="Enable code interpreter tool in API request",
    )
    parser.add_argument(
        "--max-train-demos",
        type=int,
        default=None,
        help="Optional cap on training demonstrations (default: all)",
    )
    parser.add_argument(
        "--out-dir",
        default="evaluation/output_grid_eval_results",
        type=Path,
        help="Base directory for per-task JSON and batch summaries",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Path for this invocation's batch summary JSON (default: <effective-out-dir>/batch_<timestamp>.json)",
    )
    parser.add_argument(
        "--delay-s",
        type=float,
        default=0.0,
        help=(
            "Optional pause between tasks (default: 0). Sequential calls can still hit "
            "per-minute API rate limits; a small delay (e.g. 0.2–1.0) may reduce 429 errors."
        ),
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Extra attempts per task on transient HTTP/network errors (default: 2 => up to 3 tries)",
    )
    parser.add_argument(
        "--retry-backoff-s",
        type=float,
        default=1.5,
        help="Linear backoff base in seconds: sleep backoff * attempt between retries",
    )
    parser.add_argument(
        "--continue-on-error",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Continue after a non-transient failure or failed parse (default: --continue-on-error)",
    )
    args = parser.parse_args()

    if args.tasks is not None:
        task_ids = list(args.tasks)
    elif args.tasks_file is not None:
        task_ids = _load_task_ids_from_file(args.tasks_file)
    else:
        task_ids = _all_train_task_ids()

    out_dir = _resolve_out_dir(args.out_dir, args.run_tag, args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    ordered_count = len(task_ids)
    skipped_completed = 0
    if args.skip_completed:
        filtered: list[str] = []
        for tid in task_ids:
            if _has_logged_result(out_dir, tid):
                skipped_completed += 1
            else:
                filtered.append(tid)
        task_ids = filtered

    if args.skip:
        task_ids = task_ids[args.skip :]
    if args.batch_size is not None:
        task_ids = task_ids[: args.batch_size]

    if not task_ids:
        print(
            f"No tasks to run (ordered={ordered_count}, skipped_already_logged={skipped_completed}, "
            f"after --skip/--batch-size). Nothing to do."
        )
        sys.exit(0)

    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_path = args.summary_json
    if summary_path is None:
        summary_path = out_dir / f"batch_{ts}.json"
    elif not summary_path.is_absolute():
        summary_path = ROOT_DIR / summary_path

    config: dict[str, Any] = {
        "test_idx": args.test_idx,
        "model": args.model,
        "provider": args.provider,
        "reasoning_effort": args.reasoning_effort,
        "tools_enabled": args.tools_enabled,
        "max_train_demos": args.max_train_demos,
        "out_dir": str(out_dir),
        "run_tag": args.run_tag,
        "skip_completed": args.skip_completed,
        "skip": args.skip,
        "batch_size": args.batch_size,
        "delay_s": args.delay_s,
        "retries": args.retries,
        "retry_backoff_s": args.retry_backoff_s,
    }

    print(
        f"Running {len(task_ids)} task(s); "
        f"skipped_already_logged={skipped_completed}, "
        f"effective_out_dir={out_dir}",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    parse_ok = 0
    exact_ok = 0
    failed = 0

    for i, task_id in enumerate(task_ids):
        print(f"[{i + 1}/{len(task_ids)}] {task_id} …", flush=True)
        row = _run_one_task(
            task_id=task_id,
            test_idx=args.test_idx,
            model=args.model,
            provider=args.provider,
            reasoning_effort=args.reasoning_effort,
            tools_enabled=args.tools_enabled,
            max_train_demos=args.max_train_demos,
            out_dir=out_dir,
            retries=args.retries,
            retry_backoff_s=args.retry_backoff_s,
        )
        results.append(row)
        if row.get("evaluation_ok"):
            parse_ok += 1
        if row.get("exact_match") is True:
            exact_ok += 1
        if not row.get("evaluation_ok"):
            failed += 1
            if not args.continue_on_error:
                break
        if args.delay_s > 0 and i + 1 < len(task_ids):
            time.sleep(args.delay_s)

    payload: dict[str, Any] = {
        "created_at": ts,
        "mode": "output_grid_batch",
        "run_tag": args.run_tag,
        "ordered_source_count": ordered_count,
        "skipped_already_logged": skipped_completed,
        "ran_task_count": len(task_ids),
        "completed_runs_recorded": len(results),
        "config": config,
        "aggregate": {
            "parse_ok": parse_ok,
            "exact_match": exact_ok,
            "failed_or_incomplete": failed,
        },
        "results": results,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Batch summary: {summary_path}")

    if failed:
        print(
            f"Finished with {failed} task(s) where evaluation_ok was false after retries.",
            file=sys.stderr,
        )
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
