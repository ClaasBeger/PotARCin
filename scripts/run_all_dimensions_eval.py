from __future__ import annotations

import argparse
import concurrent.futures
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
from framework.eval_paths import dimension_out_dirs, multi_run_summary_dir
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


def _is_claude_run(model: str, provider: str | None) -> bool:
    p = (provider or "").strip().lower()
    m = (model or "").strip().lower()
    if p == "anthropic":
        return True
    return m.startswith("claude") or m.startswith("anthropic/claude")


def run_all_dimensions_for_task(
    *,
    task_id: str,
    model: str = "gpt-5.4",
    provider: str | None = None,
    reasoning_effort: str = "low",
    tools_enabled: bool = False,
    seed: int | None = None,
    definition_test_idx: int = 0,
    dimension_retries: int = 2,
    retry_backoff_s: float = 1.5,
    out_dirs: dict[str, Path] | None = None,
    skip_completed_dimensions: bool | None = None,
    parallel_dimensions: int = 1,
    classification_serial: bool = False,
    stitch_prior_work_context: bool = False,
    stitched_context_max_chars: int = 12000,
) -> dict[str, Any]:
    """Run all five dimension evaluations for one task id.

    When ``out_dirs`` is set (run-tag layout) and ``skip_completed_dimensions`` is
    not overridden to False, dimensions that already have a durable artifact under
    ``out_dirs`` are not re-called: latest JSON per dimension is re-used if it is
    "complete" (same rules as batch task discovery — see ``_has_any_result_for_task``).

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
        "provider": provider,
        "reasoning_effort": reasoning_effort,
        "tools_enabled": tools_enabled,
        "seed": seed,
        "dimension_seeds": dim_seeds,
        "stitch_prior_work_context": stitch_prior_work_context,
        "stitched_context_max_chars": stitched_context_max_chars,
        "dimensions": {},
    }

    if skip_completed_dimensions is None:
        skip_completed_dimensions = bool(out_dirs)

    task = load_task(task_id)

    def _trim_text(s: Any, limit: int) -> str:
        if not isinstance(s, str):
            return ""
        t = s.strip()
        if limit <= 0 or len(t) <= limit:
            return t
        return t[:limit] + "\n[TRUNCATED]"

    def _build_prior_work_context(prior_payloads: dict[str, dict[str, Any]], limit: int) -> str:
        per_dim_limit = max(500, limit // 3) if limit > 0 else 0
        blocks: list[str] = []
        for dim in _DIMENSION_ORDER:
            payload = prior_payloads.get(dim)
            if not isinstance(payload, dict):
                continue
            prompt_txt = _trim_text(payload.get("prompt"), per_dim_limit)
            raw_txt = _trim_text(payload.get("raw_response"), per_dim_limit)
            if not prompt_txt and not raw_txt:
                continue
            part: list[str] = [f"Prior dimension: {dim}"]
            if prompt_txt:
                part.append("Prompt:")
                part.append(prompt_txt)
            if raw_txt:
                part.append("Answer:")
                part.append(raw_txt)
            blocks.append("\n".join(part))
        if not blocks:
            return ""
        merged = "\n\n".join(blocks)
        return _trim_text(merged, limit)

    def _report_from_dimension_result(dim_payload: dict[str, Any]) -> dict[str, Any]:
        ok = bool(dim_payload.get("evaluation_ok"))
        if ok:
            return {
                "ok": True,
                "result_path": dim_payload.get("result_path"),
                "metrics": dim_payload.get("metrics"),
                "attempts": 1,
            }
        return {
            "ok": False,
            "error": str(dim_payload.get("error") or "evaluation_ok is false"),
            "result_path": dim_payload.get("result_path"),
            "metrics": dim_payload.get("metrics"),
            "attempts": 1,
        }

    def _run(name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        attempts = max(1, int(dimension_retries) + 1)
        last_err = ""
        last_result_path: str | None = None
        last_metrics: Any = None
        last_attempt = 0
        for attempt in range(1, attempts + 1):
            last_attempt = attempt
            try:
                res = fn()
                if isinstance(res, dict) and res.get("evaluation_ok") is False:
                    last_err = str(res.get("error") or "evaluation_ok is false")
                    last_result_path = res.get("result_path") if isinstance(res.get("result_path"), str) else None
                    last_metrics = res.get("metrics")
                    # Retry transient API failures even when dimensions report them as
                    # evaluation_ok=false payloads (not raised exceptions).
                    if attempt < attempts and _is_transient_error(last_err):
                        time.sleep(max(0.0, retry_backoff_s) * attempt)
                        continue
                    return {
                        "ok": False,
                        "error": last_err,
                        "result_path": last_result_path,
                        "metrics": last_metrics,
                        "attempts": attempt,
                    }
                return {
                    "ok": True,
                    "result_path": res.get("result_path"),
                    "metrics": res.get("metrics"),
                    "attempts": attempt,
                }
            except Exception as e:
                last_err = str(e)
                if attempt >= attempts or not _is_transient_error(last_err):
                    break
                time.sleep(max(0.0, retry_backoff_s) * attempt)
        return {
            "ok": False,
            "error": last_err,
            "result_path": last_result_path,
            "metrics": last_metrics,
            "attempts": last_attempt,
        }

    if stitch_prior_work_context:
        prior_payloads: dict[str, dict[str, Any]] = {}

        def _execute_dimension(
            *,
            dim_name: str,
            out_key: str,
            suffix: str,
            fn_with_context: Callable[[str | None], dict[str, Any]],
            prior_context: str | None,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            if skip_completed_dimensions and out_dirs:
                latest_path, payload = _read_latest_dimension_json(out_dirs[out_key], task_id, suffix)
                if latest_path is not None and payload is not None and _payload_indicates_dimension_complete(payload):
                    rep = _report_from_saved_dimension(payload, latest_path)
                    return payload, rep

            attempts = max(1, int(dimension_retries) + 1)
            last_payload: dict[str, Any] | None = None
            last_error = ""
            for attempt in range(1, attempts + 1):
                try:
                    payload = fn_with_context(prior_context)
                    last_payload = payload if isinstance(payload, dict) else {"evaluation_ok": False}
                    if bool(last_payload.get("evaluation_ok")):
                        rep = _report_from_dimension_result(last_payload)
                        rep["attempts"] = attempt
                        return last_payload, rep
                    last_error = str(last_payload.get("error") or "evaluation_ok is false")
                    if attempt < attempts and _is_transient_error(last_error):
                        time.sleep(max(0.0, retry_backoff_s) * attempt)
                        continue
                    rep = _report_from_dimension_result(last_payload)
                    rep["attempts"] = attempt
                    return last_payload, rep
                except Exception as e:
                    last_error = str(e)
                    if attempt >= attempts or not _is_transient_error(last_error):
                        break
                    time.sleep(max(0.0, retry_backoff_s) * attempt)

            fallback = last_payload or {
                "evaluation_ok": False,
                "error": last_error,
                "metrics": None,
                "result_path": None,
            }
            rep = _report_from_dimension_result(fallback)
            rep["attempts"] = attempts
            return fallback, rep

        stitched_jobs: list[tuple[str, str, str, Callable[[str | None], dict[str, Any]]]] = [
            (
                "definition",
                "definition",
                "definition_eval",
                lambda _ctx: run_definition_api_evaluation(
                    task=task,
                    test_idx=definition_test_idx,
                    model=model,
                    provider=provider,
                    api_endpoint="responses",
                    reasoning_effort=reasoning_effort,
                    tools_enabled=tools_enabled,
                    out_dir=out_dirs["definition"] if out_dirs else "evaluation/definition_eval_results",
                ),
            ),
            (
                "classification_distribution",
                "classification_distribution",
                "distribution_classification_eval",
                lambda ctx: run_distribution_classification_api_evaluation_once(
                    task_id=task_id,
                    model=model,
                    provider=provider,
                    reasoning_effort=reasoning_effort,
                    tools_enabled=tools_enabled,
                    seed=dim_seeds["classification_distribution"],
                    prior_work_context=ctx,
                    out_dir=(
                        out_dirs["classification_distribution"]
                        if out_dirs
                        else "evaluation/classification_eval_results"
                    ),
                ),
            ),
            (
                "constrained_generation",
                "constrained_generation",
                "constrained_generation_eval",
                lambda ctx: run_editing_api_evaluation_once(
                    task=task,
                    model=model,
                    provider=provider,
                    reasoning_effort=reasoning_effort,
                    tools_enabled=tools_enabled,
                    seed=dim_seeds["editing"],
                    prior_work_context=ctx,
                    out_dir=out_dirs["editing"] if out_dirs else "evaluation/editing_eval_results",
                ),
            ),
            (
                "constrained_generation",
                "constrained_generation",
                "constrained_generation_eval",
                lambda ctx: run_constrained_generation_api_evaluation_once(
                    task_id=task_id,
                    model=model,
                    provider=provider,
                    reasoning_effort=reasoning_effort,
                    tools_enabled=tools_enabled,
                    prior_work_context=ctx,
                    out_dir=(
                        out_dirs["constrained_generation"]
                        if out_dirs
                        else "evaluation/constrained_generation_eval_results"
                    ),
                ),
            ),
            (
                "inversion",
                "inversion",
                "inversion_eval",
                lambda ctx: run_inversion_api_evaluation_once(
                    task_id=task_id,
                    model=model,
                    provider=provider,
                    reasoning_effort=reasoning_effort,
                    tools_enabled=tools_enabled,
                    seed=dim_seeds["inversion"],
                    prior_work_context=ctx,
                    out_dir=out_dirs["inversion"] if out_dirs else "evaluation/inversion_eval_results",
                ),
            ),
        ]

        for dim_name, out_key, suffix, fn_with_context in stitched_jobs:
            ctx = None if dim_name == "definition" else _build_prior_work_context(
                prior_payloads, stitched_context_max_chars
            )
            payload, rep = _execute_dimension(
                dim_name=dim_name,
                out_key=out_key,
                suffix=suffix,
                fn_with_context=fn_with_context,
                prior_context=ctx,
            )
            report["dimensions"][dim_name] = rep
            prior_payloads[dim_name] = payload

        return report

    def _maybe_rerun_dimension(
        dim_name: str,
        out_dir_key: str,
        file_suffix: str,
        fn: Callable[[], dict[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        if skip_completed_dimensions and out_dirs:
            latest_path, payload = _read_latest_dimension_json(
                out_dirs[out_dir_key], task_id, file_suffix
            )
            if latest_path is not None and (
                payload is None or _payload_indicates_dimension_complete(payload)
            ):
                return dim_name, _report_from_saved_dimension(payload, latest_path)
        return dim_name, _run(dim_name, fn)

    jobs: list[tuple[str, str, str, Callable[[], dict[str, Any]]]] = [
        (
            "definition",
            "definition",
            "definition_eval",
            lambda: run_definition_api_evaluation(
                task=task,
                test_idx=definition_test_idx,
                model=model,
                provider=provider,
                api_endpoint="responses",
                reasoning_effort=reasoning_effort,
                tools_enabled=tools_enabled,
                out_dir=out_dirs["definition"] if out_dirs else "evaluation/definition_eval_results",
            ),
        ),
        (
            "classification_distribution",
            "classification_distribution",
            "distribution_classification_eval",
            lambda: run_distribution_classification_api_evaluation_once(
                task_id=task_id,
                model=model,
                provider=provider,
                reasoning_effort=reasoning_effort,
                tools_enabled=tools_enabled,
                seed=dim_seeds["classification_distribution"],
                out_dir=(
                    out_dirs["classification_distribution"]
                    if out_dirs
                    else "evaluation/classification_eval_results"
                ),
            ),
        ),
        (
            "constrained_generation",
            "constrained_generation",
            "constrained_generation_eval",
            lambda: run_constrained_generation_api_evaluation_once(
                task_id=task_id,
                model=model,
                provider=provider,
                reasoning_effort=reasoning_effort,
                tools_enabled=tools_enabled,
                out_dir=(
                    out_dirs["constrained_generation"]
                    if out_dirs
                    else "evaluation/constrained_generation_eval_results"
                ),
            ),
        ),
        (
            "editing",
            "editing",
            "editing_eval",
            lambda: run_editing_api_evaluation_once(
                task=task,
                model=model,
                provider=provider,
                reasoning_effort=reasoning_effort,
                tools_enabled=tools_enabled,
                seed=dim_seeds["editing"],
                out_dir=out_dirs["editing"] if out_dirs else "evaluation/editing_eval_results",
            ),
        ),
        (
            "inversion",
            "inversion",
            "inversion_eval",
            lambda: run_inversion_api_evaluation_once(
                task_id=task_id,
                model=model,
                provider=provider,
                reasoning_effort=reasoning_effort,
                tools_enabled=tools_enabled,
                seed=dim_seeds["inversion"],
                out_dir=out_dirs["inversion"] if out_dirs else "evaluation/inversion_eval_results",
            ),
        ),
    ]

    workers = max(1, int(parallel_dimensions))

    def _run_job_batch(
        batch_jobs: list[tuple[str, str, str, Callable[[], dict[str, Any]]]],
        *,
        batch_workers: int,
    ) -> None:
        if not batch_jobs:
            return
        if batch_workers <= 1 or len(batch_jobs) == 1:
            for dim_name, out_key, suffix, fn in batch_jobs:
                k, v = _maybe_rerun_dimension(dim_name, out_key, suffix, fn)
                report["dimensions"][k] = v
            return
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(batch_workers, len(batch_jobs))
        ) as pool:
            futures = [
                pool.submit(_maybe_rerun_dimension, dim_name, out_key, suffix, fn)
                for dim_name, out_key, suffix, fn in batch_jobs
            ]
            for fut in concurrent.futures.as_completed(futures):
                k, v = fut.result()
                report["dimensions"][k] = v

    if classification_serial:
        jobs_by_name = {j[0]: j for j in jobs}
        # User-selected schedule:
        #   1) definition + editing in parallel
        #   2) constrained_generation + inversion in parallel
        #   3) classification_distribution alone
        _run_job_batch(
            [jobs_by_name["definition"], jobs_by_name["editing"]],
            batch_workers=workers,
        )
        _run_job_batch(
            [jobs_by_name["constrained_generation"], jobs_by_name["inversion"]],
            batch_workers=workers,
        )
        _run_job_batch(
            [jobs_by_name["classification_distribution"]],
            batch_workers=1,
        )
    else:
        _run_job_batch(jobs, batch_workers=workers)
    return report


def _first_n_task_ids(n: int) -> list[str]:
    ids: list[str] = []
    for idx, task in enumerate(iter_tasks(split="train", source=TaskSource.ORIGINAL_ARC)):
        if idx >= n:
            break
        ids.append(task.task_id)
    return ids


def _all_task_ids() -> list[str]:
    return [t.task_id for t in iter_tasks(split="train", source=TaskSource.ORIGINAL_ARC)]


def _task_ids_from_subset_file(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = payload.get("selected_task_ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"Subset file {path} must contain non-empty selected_task_ids list.")
    out: list[str] = []
    for x in ids:
        if not isinstance(x, str) or not x.strip():
            raise ValueError(f"Subset file {path} contains invalid task id: {x!r}")
        out.append(x.strip())
    # Keep file order while deduplicating.
    deduped = list(dict.fromkeys(out))
    return deduped


def _read_latest_dimension_json(
    dir_path: Path, task_id: str, suffix: str
) -> tuple[Path | None, dict[str, Any] | None]:
    """Latest JSON path and payload for ``task_id`` in ``dir_path``, or ``(None, None)``."""
    if not dir_path.is_dir():
        return None, None
    matches = sorted(dir_path.glob(f"{task_id}_{suffix}_*.json"))
    if not matches:
        return None, None
    latest = matches[-1]
    try:
        return latest, json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return latest, None


def _payload_indicates_dimension_complete(payload: dict[str, Any]) -> bool:
    """True when this dimension should not be re-executed (matches batch discovery)."""
    ok = payload.get("evaluation_ok")
    if ok is True:
        return True
    if ok is None:
        return True
    raw = payload.get("raw_response")
    if isinstance(raw, str) and raw.strip():
        return True
    if raw is not None:
        return True
    return False


def _report_from_saved_dimension(
    payload: dict[str, Any] | None, latest_path: Path
) -> dict[str, Any]:
    """Build a per-dimension report entry from an on-disk artifact (no new API call)."""
    path_s = str(latest_path)
    if payload is None:
        return {
            "ok": True,
            "result_path": path_s,
            "metrics": None,
            "attempts": 0,
            "skipped_rerun": True,
        }
    if payload.get("evaluation_ok") is False:
        return {
            "ok": False,
            "error": str(payload.get("error") or "evaluation_ok is false"),
            "result_path": path_s,
            "metrics": payload.get("metrics"),
            "attempts": 0,
            "skipped_rerun": True,
        }
    return {
        "ok": True,
        "result_path": path_s,
        "metrics": payload.get("metrics"),
        "attempts": 0,
        "skipped_rerun": True,
    }


def _has_any_result_for_task(dir_path: Path, task_id: str, suffix: str) -> bool:
    latest, payload = _read_latest_dimension_json(dir_path, task_id, suffix)
    if latest is None:
        return False
    if payload is None:
        # Legacy/corrupt files: preserve backward compatibility by counting
        # presence as complete instead of hard-failing discovery.
        return True
    return _payload_indicates_dimension_complete(payload)


def _completed_tasks_for_run_tag(all_task_ids: list[str], out_dirs: dict[str, Path]) -> set[str]:
    completed: set[str] = set()
    for task_id in all_task_ids:
        if (
            _has_any_result_for_task(out_dirs["definition"], task_id, "definition_eval")
            and _has_any_result_for_task(
                out_dirs["classification_distribution"], task_id, "distribution_classification_eval"
            )
            and _has_any_result_for_task(
                out_dirs["constrained_generation"], task_id, "constrained_generation_eval"
            )
            and _has_any_result_for_task(out_dirs["editing"], task_id, "editing_eval")
            and _has_any_result_for_task(out_dirs["inversion"], task_id, "inversion_eval")
        ):
            completed.add(task_id)
    return completed


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
        "--batch-size",
        type=int,
        default=50,
        help=(
            "When --task-id is omitted: run next N pending ARC training tasks "
            "(default: 50), determined from run-tag output directories."
        ),
    )
    p.add_argument("--model", default="gpt-5.4", help="Model name")
    p.add_argument(
        "--provider",
        default=None,
        choices=["openai", "anthropic", "google", "together", "openrouter"],
        help="LLM provider (default: infer from --model)",
    )
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
        "--stitch-prior-work-context",
        action="store_true",
        help=(
            "Sequentially stitch prior dimension prompt/answer text into each subsequent "
            "non-definition dimension prompt for the same task."
        ),
    )
    p.add_argument(
        "--stitched-context-max-chars",
        type=int,
        default=12000,
        help="Max characters of stitched prior-work context passed to each prompt.",
    )
    p.add_argument(
        "--parallel-dimensions",
        type=int,
        default=1,
        help=(
            "Run dimensions concurrently per task (default: 1 = sequential). "
            "Values like 2-3 can substantially reduce wall time for slow providers."
        ),
    )
    p.add_argument(
        "--classification-serial",
        action="store_true",
        help=(
            "Run classification_distribution alone while allowing other dimensions "
            "to run in parallel batches (definition+editing, constrained+inversion)."
        ),
    )
    p.add_argument(
        "--parallel-tasks",
        type=int,
        default=1,
        help=(
            "Run pending tasks concurrently in batch mode (default: 1 = sequential). "
            "Use small values first to avoid provider rate-limit spikes."
        ),
    )
    p.add_argument(
        "--summary-out",
        default=None,
        help="Optional output path for run summary JSON. Default: evaluation/multi_dimension_eval_runs/<timestamp>.json",
    )
    p.add_argument(
        "--task-subset-file",
        default=None,
        help=(
            "Optional JSON file with selected_task_ids (e.g. Test2/ARC1_subsets/*.json). "
            "In batch mode, only those task ids are considered."
        ),
    )
    p.add_argument(
        "--run-tag",
        default=None,
        help=(
            "Optional run tag to isolate outputs. When set, per-dimension JSONs are "
            "written under evaluation/*_eval_results/<ModelAlias>/<Run1|Run2>/<run_tag>/, and "
            "summary defaults to "
            "evaluation/multi_dimension_eval_runs/<ModelAlias>/<Run1|Run2>/<run_tag>/run_<timestamp>.json"
        ),
    )
    p.add_argument(
        "--force-all-dimensions",
        action="store_true",
        help=(
            "With --run-tag, re-call the model for every dimension even when a complete "
            "artifact already exists (default: skip completed dimensions and only re-run failures)."
        ),
    )
    p.add_argument(
        "--seed-random",
        action="store_true",
        help=(
            "Generate one random base seed (printed + stored in summary) so randomness "
            "is reproducible across future model runs."
        ),
    )
    args = p.parse_args()
    if args.seed is not None and args.seed_random:
        raise SystemExit("Use either --seed or --seed-random, not both.")
    # For Claude, use fewer retries by default to reduce long tail stalls.
    # Explicit --dimension-retries from CLI always wins.
    if "--dimension-retries" not in sys.argv and _is_claude_run(args.model, args.provider):
        args.dimension_retries = 1

    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.seed is not None:
        base_seed: int | None = int(args.seed)
        seed_mode = "fixed"
    elif args.seed_random:
        base_seed = int(random.SystemRandom().randrange(1 << 31))
        seed_mode = "random_generated"
    else:
        base_seed = None
        seed_mode = "none"

    if args.run_tag:
        out_dirs = dimension_out_dirs(root_dir=ROOT_DIR, model=args.model, run_tag=args.run_tag)
    else:
        out_dirs = {}

    summary_path = (
        Path(args.summary_out)
        if args.summary_out
        else (
            (
                multi_run_summary_dir(root_dir=ROOT_DIR, model=args.model, run_tag=args.run_tag)
                / f"run_{ts}.json"
            )
            if args.run_tag
            else ROOT_DIR / "evaluation" / "multi_dimension_eval_runs" / f"run_{ts}.json"
        )
    )
    events_path = summary_path.with_name(summary_path.stem + "_events.jsonl")

    skip_completed = bool(out_dirs) and not args.force_all_dimensions

    def _append_event(event: dict[str, Any]) -> None:
        event_rec = {"ts": dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), **event}
        events_path.parent.mkdir(parents=True, exist_ok=True)
        with events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event_rec, ensure_ascii=False) + "\n")

    def _checkpoint_payload(
        *,
        runs_so_far: list[dict[str, Any]],
        in_progress_task_id: str | None = None,
        interrupted: bool = False,
    ) -> dict[str, Any]:
        return {
            "mode": "batch_pending",
            "created_at": ts,
            "task_count": len(runs_so_far),
            "run_tag": args.run_tag,
            "seed_mode": seed_mode,
            "base_seed": base_seed,
            "batch_size": args.batch_size,
            "completed_before_count": len(completed_before),
            "last_completed_task_id_before": last_completed_task_id,
            "interrupted": interrupted,
            "in_progress_task_id": in_progress_task_id,
            "runs": runs_so_far,
        }

    if args.task_id:
        run = run_all_dimensions_for_task(
            task_id=args.task_id,
            model=args.model,
            provider=args.provider,
            reasoning_effort=args.reasoning_effort,
            tools_enabled=args.tools_enabled,
            seed=base_seed,
            definition_test_idx=args.definition_test_idx,
            dimension_retries=args.dimension_retries,
            retry_backoff_s=args.retry_backoff_s,
            parallel_dimensions=args.parallel_dimensions,
            classification_serial=args.classification_serial,
            stitch_prior_work_context=bool(args.stitch_prior_work_context),
            stitched_context_max_chars=int(args.stitched_context_max_chars),
            out_dirs=out_dirs or None,
            skip_completed_dimensions=skip_completed,
        )
        payload: dict[str, Any] = {
            "mode": "single_task",
            "created_at": ts,
            "task_count": 1,
            "run_tag": args.run_tag,
            "seed_mode": seed_mode,
            "base_seed": base_seed,
            "runs": [run],
        }
        _write_run_summary(summary_path, payload)
        print(f"Summary saved: {summary_path}")
        return

    if not args.run_tag:
        raise SystemExit("--run-tag is required for batched mode so completed tasks can be discovered.")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be >= 1")

    if args.task_subset_file:
        subset_path = Path(args.task_subset_file)
        if not subset_path.is_absolute():
            subset_path = ROOT_DIR / subset_path
        all_task_ids = _task_ids_from_subset_file(subset_path)
        # Keep only ids that exist in ARC1 training split.
        canonical_ids = set(_all_task_ids())
        all_task_ids = [tid for tid in all_task_ids if tid in canonical_ids]
        if not all_task_ids:
            raise SystemExit(
                f"--task-subset-file resolved to zero valid ARC1 task ids: {subset_path}"
            )
    else:
        all_task_ids = _all_task_ids()
    completed_before = _completed_tasks_for_run_tag(all_task_ids, out_dirs)
    pending_task_ids = [tid for tid in all_task_ids if tid not in completed_before]
    task_ids = pending_task_ids[: args.batch_size]

    last_completed_task_id = None
    for tid in all_task_ids:
        if tid in completed_before:
            last_completed_task_id = tid

    print(
        f"Run tag '{args.run_tag}': completed={len(completed_before)}, "
        f"pending={len(pending_task_ids)}, batch_size={args.batch_size}"
    )
    if last_completed_task_id is not None:
        print(f"Last completed task (dataset order): {last_completed_task_id}")
    else:
        print("Last completed task (dataset order): none")
    if not task_ids:
        print("No pending tasks left for this run tag; nothing to do.")
        return

    task_index: dict[str, int] = {tid: i for i, tid in enumerate(all_task_ids)}
    runs: list[dict[str, Any]] = []
    if base_seed is not None:
        print(f"Using base seed: {base_seed} (mode={seed_mode})")
    else:
        print("Using non-reproducible random seeds (mode=none)")
    def _run_one(index: int, tid: str) -> tuple[int, str, int | None, dict[str, Any]]:
        run_seed = (base_seed + task_index[tid]) if base_seed is not None else None
        run = run_all_dimensions_for_task(
            task_id=tid,
            model=args.model,
            provider=args.provider,
            reasoning_effort=args.reasoning_effort,
            tools_enabled=args.tools_enabled,
            seed=run_seed,
            definition_test_idx=args.definition_test_idx,
            dimension_retries=args.dimension_retries,
            retry_backoff_s=args.retry_backoff_s,
            parallel_dimensions=args.parallel_dimensions,
            classification_serial=args.classification_serial,
            stitch_prior_work_context=bool(args.stitch_prior_work_context),
            stitched_context_max_chars=int(args.stitched_context_max_chars),
            out_dirs=out_dirs or None,
            skip_completed_dimensions=skip_completed,
        )
        return index, tid, run_seed, run

    task_workers = max(1, int(args.parallel_tasks))
    current_task_id: str | None = None
    try:
        if task_workers == 1:
            for i, task_id in enumerate(task_ids):
                current_task_id = task_id
                run_seed = (base_seed + task_index[task_id]) if base_seed is not None else None
                seed_txt = "random" if run_seed is None else str(run_seed)
                print(f"[{i+1}/{len(task_ids)}] {task_id} (seed={seed_txt})")
                _append_event({"event": "task_started", "task_id": task_id, "index": i + 1, "seed": run_seed})
                _, _, _, run = _run_one(i, task_id)
                runs.append(run)
                ok_dims = sum(1 for d in run["dimensions"].values() if d.get("ok"))
                skipped = sum(1 for d in run["dimensions"].values() if d.get("skipped_rerun"))
                ran = 5 - skipped
                print(f"  -> ok {ok_dims}/5 dimensions ({ran} re-run, {skipped} reused from disk)")
                _append_event(
                    {
                        "event": "task_completed",
                        "task_id": task_id,
                        "index": i + 1,
                        "ok_dimensions": ok_dims,
                        "rerun_dimensions": ran,
                        "reused_dimensions": skipped,
                        "dimension_errors": {
                            dim: d.get("error")
                            for dim, d in run.get("dimensions", {}).items()
                            if not d.get("ok") and d.get("error")
                        },
                    }
                )
                _write_run_summary(
                    summary_path,
                    _checkpoint_payload(runs_so_far=runs, in_progress_task_id=None, interrupted=False),
                )
                current_task_id = None
        else:
            print(
                f"Running up to {min(task_workers, len(task_ids))} tasks in parallel "
                f"(parallel-tasks={task_workers})"
            )
            indexed_runs: dict[int, tuple[str, int | None, dict[str, Any]]] = {}
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(task_workers, len(task_ids))
            ) as pool:
                futures = {}
                for i, task_id in enumerate(task_ids):
                    run_seed = (base_seed + task_index[task_id]) if base_seed is not None else None
                    _append_event({"event": "task_started", "task_id": task_id, "index": i + 1, "seed": run_seed})
                    futures[pool.submit(_run_one, i, task_id)] = (i, task_id)
                for fut in concurrent.futures.as_completed(futures):
                    i, task_id = futures[fut]
                    try:
                        idx, tid, run_seed, run = fut.result()
                    except Exception as e:
                        run_seed = (base_seed + task_index[task_id]) if base_seed is not None else None
                        run = {
                            "task_id": task_id,
                            "model": args.model,
                            "provider": args.provider,
                            "reasoning_effort": args.reasoning_effort,
                            "tools_enabled": bool(args.tools_enabled),
                            "seed": run_seed,
                            "dimensions": {
                                dim: {"ok": False, "error": str(e), "attempts": 0}
                                for dim in _DIMENSION_ORDER
                            },
                        }
                        idx, tid = i, task_id
                    indexed_runs[idx] = (tid, run_seed, run)
                    seed_txt = "random" if run_seed is None else str(run_seed)
                    ok_dims = sum(1 for d in run["dimensions"].values() if d.get("ok"))
                    skipped = sum(1 for d in run["dimensions"].values() if d.get("skipped_rerun"))
                    ran = 5 - skipped
                    print(f"[{idx+1}/{len(task_ids)}] {tid} (seed={seed_txt})")
                    print(f"  -> ok {ok_dims}/5 dimensions ({ran} re-run, {skipped} reused from disk)")
                    _append_event(
                        {
                            "event": "task_completed",
                            "task_id": tid,
                            "index": idx + 1,
                            "ok_dimensions": ok_dims,
                            "rerun_dimensions": ran,
                            "reused_dimensions": skipped,
                            "dimension_errors": {
                                dim: d.get("error")
                                for dim, d in run.get("dimensions", {}).items()
                                if not d.get("ok") and d.get("error")
                            },
                        }
                    )
                    partial_runs = [indexed_runs[k][2] for k in sorted(indexed_runs.keys())]
                    _write_run_summary(
                        summary_path,
                        _checkpoint_payload(
                            runs_so_far=partial_runs, in_progress_task_id=None, interrupted=False
                        ),
                    )
            for i in range(len(task_ids)):
                runs.append(indexed_runs[i][2])
    except KeyboardInterrupt:
        _append_event(
            {
                "event": "run_interrupted",
                "in_progress_task_id": current_task_id,
                "completed_tasks_in_this_batch": len(runs),
            }
        )
        _write_run_summary(
            summary_path,
            _checkpoint_payload(
                runs_so_far=runs, in_progress_task_id=current_task_id, interrupted=True
            ),
        )
        print(f"Interrupted. Partial summary saved: {summary_path}")
        raise

    payload = {
        "mode": "batch_pending",
        "created_at": ts,
        "task_count": len(runs),
        "run_tag": args.run_tag,
        "seed_mode": seed_mode,
        "base_seed": base_seed,
        "batch_size": args.batch_size,
        "completed_before_count": len(completed_before),
        "last_completed_task_id_before": last_completed_task_id,
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

