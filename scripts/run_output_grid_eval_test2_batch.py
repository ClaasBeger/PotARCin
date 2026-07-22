from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from framework.dimensions.output_grid_eval import run_output_grid_api_evaluation
from framework.eval_paths import model_alias, run_bucket
from framework.grids import GridPair, is_int_grid
from framework.tasks.base import ArcTask


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
        "did not contain any text content",
        "incompleteread",
        "incomplete read",
    )
    return any(tok in s for tok in transient_markers)


def _parse_task_num(task_id: str) -> int:
    m = re.fullmatch(r"t(\d+)", task_id.strip().lower())
    if not m:
        raise ValueError(f"Invalid Test2 task id: {task_id!r} (expected like t1, t2, ..., t50)")
    return int(m.group(1))


def _load_test2_task(task_id: str, test2_root: Path) -> ArcTask:
    n = _parse_task_num(task_id)
    p = test2_root / f"t{n}.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing Test2 task file: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if "train" not in raw or "test" not in raw:
        raise ValueError(f"{p} must contain keys 'train' and 'test'")

    train_pairs: list[GridPair] = []
    for i, ex in enumerate(raw["train"]):
        inp = ex.get("input")
        out = ex.get("output")
        if not is_int_grid(inp) or not is_int_grid(out):
            raise ValueError(f"{p} train[{i}] has invalid grid(s)")
        train_pairs.append(GridPair(input=inp, output=out))

    test_inputs: list[list[list[int]]] = []
    test_outputs: list[list[list[int]]] = []
    for i, ex in enumerate(raw["test"]):
        inp = ex.get("input")
        out = ex.get("output")
        if not is_int_grid(inp) or not is_int_grid(out):
            raise ValueError(f"{p} test[{i}] has invalid grid(s)")
        test_inputs.append(inp)
        test_outputs.append(out)

    return ArcTask(
        task_id=f"test2_t{n}",
        train_pairs=train_pairs,
        test_inputs=test_inputs,
        test_outputs=test_outputs,
    )


def _has_logged_result(out_dir: Path, task_id: str) -> bool:
    # Do not treat failure artifacts (`..._output_grid_eval_error_...json`) as completed.
    for p in out_dir.glob(f"{task_id}_output_grid_eval_*.json"):
        if "_output_grid_eval_error_" in p.name:
            continue
        return True
    return False


def _test2_run_dir(*, root_dir: Path, model: str, run_tag: str) -> Path:
    return (
        root_dir
        / "evaluation"
        / "output_grid_eval_results"
        / model_alias(model)
        / "Test2"
        / run_bucket(run_tag)
        / run_tag
    )


def _artifact_task_id(task_id: str) -> str:
    return f"test2_t{_parse_task_num(task_id)}"


def _write_failure_artifact(
    *,
    out_dir: Path,
    task_id: str,
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
    artifact_task_id = _artifact_task_id(task_id)
    out_path = out_dir / f"{artifact_task_id}_output_grid_eval_error_{ts}.json"
    payload = {
        "task_id": artifact_task_id,
        "source_task_id": task_id,
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
    test2_root: Path,
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
    attempts = max(1, retries + 1)
    attempt_errors: list[dict[str, Any]] = []
    for attempt in range(1, attempts + 1):
        try:
            task = _load_test2_task(task_id, test2_root)
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
            transient = _is_transient_error(msg)
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
                "attempts": attempt,
                "result_path": str(err_path),
                "evaluation_ok": False,
                "exact_match": None,
                "metrics": None,
                "error": f"{type(e).__name__}: {e}",
                "attempt_errors": attempt_errors,
            }
    raise RuntimeError("test2 output-grid batch: internal retry loop fell through")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run output-grid correctness evaluation on Test2 tasks (t1..t50)."
    )
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--tasks", nargs="+", metavar="TASK_ID", help="Task ids like t1 t2 ... t50")
    src.add_argument("--all-test2", action="store_true", help="Run all Test2 tasks t1..t50")

    parser.add_argument("--test2-root", type=Path, default=Path("Test2"), help="Path to Test2 root dir")
    parser.add_argument("--run-tag", default=None, help="Optional run tag for scoped output directory")
    parser.add_argument("--skip-completed", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--skip", type=int, default=0, help="Drop first N tasks after filtering")
    parser.add_argument("--batch-size", "--limit", type=int, default=None, dest="batch_size", metavar="N")
    parser.add_argument("--test-idx", type=int, default=0, help="Which test example per task")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument(
        "--provider",
        default=None,
        choices=["openai", "anthropic", "google", "together", "openrouter"],
    )
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--tools-enabled", action="store_true")
    parser.add_argument("--max-train-demos", type=int, default=None)
    parser.add_argument(
        "--out-dir",
        default="evaluation/output_grid_eval_results_test2",
        type=Path,
        help="Base directory for per-task JSON and batch summaries (when --run-tag omitted)",
    )
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--delay-s", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-backoff-s", type=float, default=1.5)
    parser.add_argument("--continue-on-error", default=True, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    if args.tasks is not None:
        task_ids = [t.lower() for t in args.tasks]
    else:
        task_ids = [f"t{i}" for i in range(1, 51)]
    for tid in task_ids:
        _parse_task_num(tid)

    test2_root = args.test2_root if args.test2_root.is_absolute() else ROOT_DIR / args.test2_root
    if not test2_root.exists():
        raise SystemExit(f"Missing Test2 root: {test2_root}")

    if args.run_tag:
        out_dir = _test2_run_dir(root_dir=ROOT_DIR, model=args.model, run_tag=args.run_tag)
    else:
        out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ordered_count = len(task_ids)
    skipped_completed = 0
    if args.skip_completed:
        filtered: list[str] = []
        for tid in task_ids:
            artifact_tid = _artifact_task_id(tid)
            if _has_logged_result(out_dir, artifact_tid):
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

    print(
        f"Running Test2 output-grid eval on {len(task_ids)} task(s); "
        f"skipped_already_logged={skipped_completed}, effective_out_dir={out_dir}",
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
            test2_root=test2_root,
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
            print(
                f"  -> FAILED attempt={row.get('attempts')} "
                f"error={row.get('error')} "
                f"logged={row.get('result_path')}",
                flush=True,
            )
            if not args.continue_on_error:
                break
        else:
            print(
                f"  -> ok attempt={row.get('attempts')} "
                f"exact_match={row.get('exact_match')} "
                f"logged={row.get('result_path')}",
                flush=True,
            )
        if args.delay_s > 0 and i + 1 < len(task_ids):
            time.sleep(args.delay_s)

    payload: dict[str, Any] = {
        "created_at": ts,
        "mode": "output_grid_batch_test2",
        "run_tag": args.run_tag,
        "ordered_source_count": ordered_count,
        "skipped_already_logged": skipped_completed,
        "ran_task_count": len(task_ids),
        "completed_runs_recorded": len(results),
        "config": {
            "test2_root": str(test2_root),
            "test_idx": args.test_idx,
            "model": args.model,
            "provider": args.provider,
            "reasoning_effort": args.reasoning_effort,
            "tools_enabled": args.tools_enabled,
            "max_train_demos": args.max_train_demos,
            "out_dir": str(out_dir),
            "skip_completed": args.skip_completed,
            "skip": args.skip,
            "batch_size": args.batch_size,
            "delay_s": args.delay_s,
            "retries": args.retries,
            "retry_backoff_s": args.retry_backoff_s,
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
            f"Finished with {failed} task(s) where evaluation_ok was false after retries.",
            file=sys.stderr,
        )
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
