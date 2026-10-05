#!/usr/bin/env python3
"""Run the pure output-grid ARC eval over the **ARC-AGI-1 evaluation split** (400 tasks).

Same prompt and parsing as the training-split runner
(``scripts/run_output_grid_eval_batch.py``); only the task source and the output
directory segment differ. Per-task JSONs land under::

    evaluation/output_grid_eval_results/<ModelAlias>/ARC1Eval/<Run1|Run2>/<run_tag>/

With ``--skip-completed`` (default) an already-logged task is skipped, so the
benchmark can be walked in chunks by re-running the same command with
``--batch-size``.

Kimi K2.5 via OpenRouter::

    python scripts/run_output_grid_eval_arc1_eval_batch.py \\
      --all-eval \\
      --model moonshotai/Kimi-K2.5:nitro \\
      --provider openrouter \\
      --reasoning-effort low \\
      --run-tag Evaluation_Run_kimi_arc1_eval_output_grid_run1 \\
      --batch-size 50

MiniMax M2.5 via OpenRouter::

    python scripts/run_output_grid_eval_arc1_eval_batch.py \\
      --all-eval \\
      --model minimax/minimax-m2.5:nitro \\
      --provider openrouter \\
      --reasoning-effort low \\
      --run-tag Evaluation_Run_minimax_arc1_eval_output_grid_run1 \\
      --batch-size 50
"""

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

from framework.dimensions.output_grid_eval import run_output_grid_api_evaluation
from framework.eval_paths import output_grid_run_dir
from framework.tasks.arc_eval_dataset import (
    arc_eval_task_ids,
    load_arc_eval_task,
    test_example_count,
)

DATASET_SEGMENT = "ARC1Eval"


def _is_transient_error(msg: str) -> bool:
    """HTTP / network errors that are worth retrying (aligned with the train-split runner)."""
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
        "did not contain any text content",
        "incompleteread",
        "incomplete read",
    )
    return any(tok in s for tok in transient_markers)


def _is_retryable_eval_failure(row: dict[str, Any]) -> bool:
    """Whether an evaluation result should be retried as likely transient/format issue."""
    if row.get("evaluation_ok") is True:
        return False
    # Output-grid runs commonly fail due to response formatting/parsing; retry those,
    # and conservatively retry any other evaluation_ok=false while budget permits.
    return True


def _load_task_ids_from_file(path: Path) -> list[str]:
    out: list[str] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = ln.split("#", 1)[0].strip()
        if s:
            out.append(s)
    return out


def _artifact_task_id(task_id: str, test_idx: int) -> str:
    """Per-artifact id; test example 0 keeps the bare task id (train-split parity)."""
    return task_id if test_idx == 0 else f"{task_id}_test{test_idx}"


def _has_logged_result(out_dir: Path, artifact_task_id: str) -> bool:
    if not out_dir.is_dir():
        return False
    for p in out_dir.glob(f"{artifact_task_id}_output_grid_eval_*.json"):
        # Failure artifacts do not count as completed.
        if "_output_grid_eval_error_" in p.name:
            continue
        return True
    return False


def _write_failure_artifact(
    *,
    out_dir: Path,
    task_id: str,
    artifact_task_id: str,
    test_idx: int,
    model: str,
    provider: str | None,
    reasoning_effort: str | None,
    tools_enabled: bool,
    max_train_demos: int | None,
    attempts: int,
    attempt_errors: list[dict[str, Any]],
) -> Path:
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = out_dir / f"{artifact_task_id}_output_grid_eval_error_{ts}.json"
    payload = {
        "task_id": artifact_task_id,
        "source_task_id": task_id,
        "dataset": "arc_agi_1_evaluation",
        "test_idx": test_idx,
        "model": model,
        "provider": provider,
        "reasoning_effort": reasoning_effort,
        "tools_enabled": tools_enabled,
        "max_train_demos": max_train_demos,
        "evaluation_ok": False,
        "exact_match": None,
        "metrics": None,
        "attempts": attempts,
        "attempt_errors": attempt_errors,
        "error": attempt_errors[-1]["error"] if attempt_errors else "unknown error",
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out_path


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
    retry_eval_failures: bool,
) -> dict[str, Any]:
    """Call the API with retries on transient failures; returns a result row dict."""
    artifact_task_id = _artifact_task_id(task_id, test_idx)
    attempts = max(1, retries + 1)
    attempt_errors: list[dict[str, Any]] = []
    for attempt in range(1, attempts + 1):
        try:
            task = load_arc_eval_task(task_id, task_id_override=artifact_task_id)
            res = run_output_grid_api_evaluation(
                task=task,
                test_idx=test_idx,
                model=model,
                provider=provider,
                reasoning_effort=reasoning_effort,
                tools_enabled=tools_enabled,
                max_train_demos=max_train_demos,
                out_dir=out_dir,
            )
            row = {
                "task_id": task_id,
                "artifact_task_id": artifact_task_id,
                "test_idx": test_idx,
                "attempts": attempt,
                "result_path": res.get("result_path"),
                "evaluation_ok": res.get("evaluation_ok"),
                "exact_match": res.get("exact_match"),
                "metrics": res.get("metrics"),
                "error": res.get("error"),
            }
            if (
                retry_eval_failures
                and attempt < attempts
                and _is_retryable_eval_failure(row)
            ):
                time.sleep(retry_backoff_s * attempt)
                continue
            return row
        except Exception as e:
            transient = _is_transient_error(str(e))
            attempt_errors.append(
                {
                    "attempt": attempt,
                    "error": f"{type(e).__name__}: {e}",
                    "transient": transient,
                }
            )
            if attempt < attempts and transient:
                time.sleep(retry_backoff_s * attempt)
                continue
            err_path = _write_failure_artifact(
                out_dir=out_dir,
                task_id=task_id,
                artifact_task_id=artifact_task_id,
                test_idx=test_idx,
                model=model,
                provider=provider,
                reasoning_effort=reasoning_effort,
                tools_enabled=tools_enabled,
                max_train_demos=max_train_demos,
                attempts=attempt,
                attempt_errors=attempt_errors,
            )
            return {
                "task_id": task_id,
                "artifact_task_id": artifact_task_id,
                "test_idx": test_idx,
                "attempts": attempt,
                "result_path": str(err_path),
                "evaluation_ok": False,
                "exact_match": None,
                "metrics": None,
                "error": f"{type(e).__name__}: {e}",
                "attempt_errors": attempt_errors,
            }
    raise RuntimeError("arc1-eval output-grid batch: internal retry loop fell through")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run output-grid ARC evals over the ARC-AGI-1 evaluation split. Per-task JSONs "
            "are written under evaluation/output_grid_eval_results/<ModelAlias>/ARC1Eval/"
            "<Run1|Run2>/<run_tag>/. With --skip-completed (default), tasks that already "
            "have a logged result are skipped, so re-running with --batch-size continues a "
            "long job."
        )
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--tasks",
        nargs="+",
        metavar="TASK_ID",
        help="One or more ARC-AGI-1 evaluation task ids (space-separated)",
    )
    src.add_argument(
        "--tasks-file",
        type=Path,
        help="Text file: one task id per line (# comments allowed)",
    )
    src.add_argument(
        "--all-eval",
        action="store_true",
        help="All 400 evaluation tasks, in sorted task-id order",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help=(
            "Optional run tag for scoped output. Files are written to "
            "evaluation/output_grid_eval_results/<ModelAlias>/ARC1Eval/<Run1|Run2>/<run_tag>/. "
            "Resume discovery only sees JSONs under this directory."
        ),
    )
    parser.add_argument(
        "--skip-completed",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "Skip tasks that already have a non-error {task_id}_output_grid_eval_*.json "
            "under the effective output directory (default: skip)"
        ),
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="After filtering, drop the first N work items (advanced ordering control)",
    )
    parser.add_argument(
        "--batch-size",
        "--limit",
        type=int,
        default=None,
        dest="batch_size",
        metavar="N",
        help=(
            "Run at most N work items this invocation (after --skip-completed / --skip). "
            "Re-run the same command to continue."
        ),
    )
    parser.add_argument(
        "--test-idx",
        type=int,
        default=0,
        help="Which test example per task (default: 0, matching the train-split runner)",
    )
    parser.add_argument(
        "--all-test-indices",
        action="store_true",
        help=(
            "Evaluate every test example of each task (19 of 400 eval tasks have 2). "
            "Artifacts for index > 0 are named <task_id>_test<idx>_output_grid_eval_*.json. "
            "Overrides --test-idx."
        ),
    )
    parser.add_argument("--model", required=True, help="Model name (e.g. moonshotai/Kimi-K2.5:nitro)")
    parser.add_argument(
        "--provider",
        default=None,
        choices=["openai", "anthropic", "google", "together", "openrouter"],
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
        default="evaluation/output_grid_eval_results_arc1_eval",
        type=Path,
        help="Base directory for per-task JSON and batch summaries (used when --run-tag is omitted)",
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
        help="Optional pause between calls (default: 0). A small delay may reduce 429 errors.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Extra attempts per work item on transient HTTP/network errors (default: 2 => up to 3 tries)",
    )
    parser.add_argument(
        "--retry-backoff-s",
        type=float,
        default=1.5,
        help="Linear backoff base in seconds: sleep backoff * attempt between retries",
    )
    parser.add_argument(
        "--retry-eval-failures",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "Retry work items when evaluation_ok is false (e.g. parse/format failures) "
            "within the same --retries budget (default: retry)"
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Continue after a non-transient failure or failed parse (default: --continue-on-error)",
    )
    args = parser.parse_args()

    known_ids = set(arc_eval_task_ids())
    if args.tasks is not None:
        task_ids = list(args.tasks)
    elif args.tasks_file is not None:
        task_ids = _load_task_ids_from_file(args.tasks_file)
    else:
        task_ids = sorted(known_ids)

    unknown = [t for t in task_ids if t not in known_ids]
    if unknown:
        raise SystemExit(
            f"Not in the ARC-AGI-1 evaluation split: {', '.join(unknown[:10])}"
            + (" …" if len(unknown) > 10 else "")
        )

    # Work items are (task_id, test_idx) pairs.
    work: list[tuple[str, int]] = []
    for tid in task_ids:
        if args.all_test_indices:
            work.extend((tid, i) for i in range(test_example_count(tid)))
        else:
            work.append((tid, args.test_idx))

    if args.run_tag:
        out_dir = output_grid_run_dir(
            root_dir=ROOT_DIR,
            model=args.model,
            run_tag=args.run_tag,
            dataset_segment=DATASET_SEGMENT,
        )
    else:
        out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ordered_count = len(work)
    skipped_completed = 0
    if args.skip_completed:
        filtered: list[tuple[str, int]] = []
        for tid, idx in work:
            if _has_logged_result(out_dir, _artifact_task_id(tid, idx)):
                skipped_completed += 1
            else:
                filtered.append((tid, idx))
        work = filtered

    if args.skip:
        work = work[args.skip :]
    if args.batch_size is not None:
        work = work[: args.batch_size]

    if not work:
        print(
            f"No work to run (ordered={ordered_count}, skipped_already_logged={skipped_completed}, "
            f"after --skip/--batch-size). Nothing to do."
        )
        sys.exit(0)

    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_path = args.summary_json
    if summary_path is None:
        summary_path = out_dir / f"batch_{ts}.json"
    elif not summary_path.is_absolute():
        summary_path = ROOT_DIR / summary_path

    print(
        f"Running ARC-AGI-1 eval output-grid on {len(work)} item(s); "
        f"skipped_already_logged={skipped_completed}, effective_out_dir={out_dir}",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    parse_ok = 0
    exact_ok = 0
    failed = 0

    for i, (task_id, test_idx) in enumerate(work):
        label = _artifact_task_id(task_id, test_idx)
        print(f"[{i + 1}/{len(work)}] {label} …", flush=True)
        row = _run_one_task(
            task_id=task_id,
            test_idx=test_idx,
            model=args.model,
            provider=args.provider,
            reasoning_effort=args.reasoning_effort,
            tools_enabled=args.tools_enabled,
            max_train_demos=args.max_train_demos,
            out_dir=out_dir,
            retries=args.retries,
            retry_backoff_s=args.retry_backoff_s,
            retry_eval_failures=args.retry_eval_failures,
        )
        results.append(row)
        if row.get("evaluation_ok"):
            parse_ok += 1
            print(
                f"  -> ok attempt={row.get('attempts')} "
                f"exact_match={row.get('exact_match')} "
                f"logged={row.get('result_path')}",
                flush=True,
            )
        else:
            failed += 1
            print(
                f"  -> FAILED attempts={row.get('attempts')} "
                f"error={row.get('error')} "
                f"logged={row.get('result_path')}",
                flush=True,
            )
            if not args.continue_on_error:
                break
        if row.get("exact_match") is True:
            exact_ok += 1
        if args.delay_s > 0 and i + 1 < len(work):
            time.sleep(args.delay_s)

    payload: dict[str, Any] = {
        "created_at": ts,
        "mode": "output_grid_batch_arc1_eval",
        "dataset": "arc_agi_1_evaluation",
        "run_tag": args.run_tag,
        "ordered_source_count": ordered_count,
        "skipped_already_logged": skipped_completed,
        "ran_task_count": len(work),
        "completed_runs_recorded": len(results),
        "config": {
            "test_idx": args.test_idx,
            "all_test_indices": args.all_test_indices,
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
            "retry_eval_failures": args.retry_eval_failures,
        },
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
            f"Finished with {failed} item(s) where evaluation_ok was false after retries.",
            file=sys.stderr,
        )
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
