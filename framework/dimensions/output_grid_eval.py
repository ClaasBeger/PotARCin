from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any, Optional, Sequence

from framework.dimensions.inversion import parse_inversion_prediction
from framework.grids import Grid, GridPair, is_equal_grid
from framework.llm_api import request_text_completion, resolve_provider_config
from framework.tasks.arc_dataset import load_task
from framework.tasks.base import ArcTask


def _grid_to_str(grid: Grid) -> str:
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def build_output_grid_prompt(
    *,
    demonstrations: Sequence[GridPair],
    test_input: Grid,
) -> str:
    """Prompt: train demos + one test input; model returns a list-of-lists output grid (inversion format)."""
    lines: list[str] = []
    lines.append(
        "Find the common rule that maps an input grid to an output grid, given the examples below."
    )
    for idx, ex in enumerate(demonstrations, 1):
        lines.append(f"\nExample {idx}")
        lines.append("Input:")
        lines.append(_grid_to_str(ex.input))
        lines.append("Output:")
        lines.append(_grid_to_str(ex.output))
    lines.append(
        "\nBelow is a test input grid. Predict the corresponding output grid by applying the rule you "
        "found."
    )
    lines.append("Test Input:")
    lines.append(_grid_to_str(test_input))
    lines.append(
        "\nYour final answer must be ONLY a single Python list-of-lists grid for that output:\n"
        "[[...], ...]\n"
        "Use integer grid cells 0-9. Output nothing else."
    )
    return "\n".join(lines)


def _parse_trailing_space_separated_grid(text: str) -> Grid:
    """Parse the last contiguous block of space-separated integer rows."""
    s = text.strip()
    if "```" in s:
        m = re.search(r"```(?:\w*\n)?(.*?)```", s, re.DOTALL)
        if m:
            inner = m.group(1).strip()
            if inner:
                s = inner
    lines = [ln.rstrip() for ln in s.splitlines()]
    rows_rev: list[list[int]] = []
    for ln in reversed(lines):
        if not ln.strip():
            if rows_rev:
                break
            continue
        parts = ln.split()
        if not parts:
            continue
        try:
            row = [int(x) for x in parts]
        except ValueError:
            if rows_rev:
                break
            continue
        rows_rev.append(row)
    if not rows_rev:
        raise ValueError("no space-separated grid rows found")
    rows_rev.reverse()
    w = len(rows_rev[0])
    if any(len(r) != w for r in rows_rev):
        raise ValueError("jagged grid in space-separated output")
    return rows_rev


def parse_output_grid_response(text: str) -> Grid:
    """Primary: Python/JSON list-of-lists (same as inversion). Fallback: space-separated rows."""
    try:
        return parse_inversion_prediction(text)
    except ValueError:
        return _parse_trailing_space_separated_grid(text)


def _openai_responses_completion(
    *,
    prompt: str,
    model: str,
    provider: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    reasoning_effort: Optional[str] = None,
    tools_enabled: bool = False,
    timeout_s: int = 180,
) -> str:
    text, _provider, _base = request_text_completion(
        prompt=prompt,
        model=model,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
        tools_enabled=tools_enabled,
        timeout_s=timeout_s,
    )
    return text


def _run_output_grid_api_evaluation_for_task(
    *,
    task: ArcTask,
    test_idx: int = 0,
    model: str = "gpt-5.4",
    provider: str | None = None,
    reasoning_effort: Optional[str] = "low",
    tools_enabled: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_train_demos: Optional[int] = None,
    out_dir: str | Path = "evaluation/output_grid_eval_results",
) -> dict[str, Any]:
    """ARC-style eval: demos + one test input -> model predicts output grid -> exact match vs GT.

    After a model response is received, a JSON file is always written (including
    ``raw_response``). Parse failures set ``evaluation_ok`` false; on success,
    ``exact_match`` compares the parsed grid to ``task.test_outputs[test_idx]``.
    """
    tid = task.task_id
    if not task.train_pairs:
        raise ValueError(f"Task {tid} has no training pairs.")
    if test_idx < 0 or test_idx >= len(task.test_inputs) or test_idx >= len(task.test_outputs):
        raise ValueError(f"Invalid test_idx={test_idx} for task {tid}.")

    demos = list(task.train_pairs)
    if max_train_demos is not None and max_train_demos > 0:
        demos = demos[:max_train_demos]

    test_in = task.test_inputs[test_idx]
    expected = task.test_outputs[test_idx]
    prompt = build_output_grid_prompt(demonstrations=demos, test_input=test_in)

    resolved_provider, _effective_key, effective_base = resolve_provider_config(
        model=model,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
    )
    raw = _openai_responses_completion(
        prompt=prompt,
        model=model,
        provider=resolved_provider,
        api_key=api_key,
        base_url=effective_base,
        reasoning_effort=reasoning_effort,
        tools_enabled=tools_enabled,
    )

    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{tid}_output_grid_eval_{ts}.json"

    result: dict[str, Any] = {
        "task_id": tid,
        "test_idx": test_idx,
        "model": model,
        "provider": resolved_provider,
        "base_url": effective_base,
        "reasoning_effort": reasoning_effort,
        "tools_enabled": tools_enabled,
        "max_train_demos": max_train_demos,
        "prompt": prompt,
        "raw_response": raw,
        "ground_truth": expected,
        "evaluation_ok": False,
    }

    pred: Grid | None = None
    parse_error: str | None = None
    try:
        pred = parse_output_grid_response(raw)
    except Exception as e:
        parse_error = str(e)

    if pred is None:
        result["prediction"] = None
        result["error"] = parse_error or "parse failed"
        result["exact_match"] = None
        result["metrics"] = {"exact_match": 0.0}
    else:
        result["prediction"] = pred
        match = is_equal_grid(pred, expected)
        result["exact_match"] = match
        result["metrics"] = {"exact_match": 1.0 if match else 0.0}
        result["evaluation_ok"] = True

    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    result["result_path"] = str(out_path)
    return result


def run_output_grid_api_evaluation_once(
    *,
    task_id: str,
    test_idx: int = 0,
    model: str = "gpt-5.4",
    provider: str | None = None,
    reasoning_effort: Optional[str] = "low",
    tools_enabled: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_train_demos: Optional[int] = None,
    out_dir: str | Path = "evaluation/output_grid_eval_results",
) -> dict[str, Any]:
    """Load *task_id* from the dataset, then run :func:`_run_output_grid_api_evaluation_for_task`."""
    task = load_task(task_id)
    return _run_output_grid_api_evaluation_for_task(
        task=task,
        test_idx=test_idx,
        model=model,
        provider=provider,
        reasoning_effort=reasoning_effort,
        tools_enabled=tools_enabled,
        api_key=api_key,
        base_url=base_url,
        max_train_demos=max_train_demos,
        out_dir=out_dir,
    )


def run_output_grid_api_evaluation(
    *,
    task: ArcTask,
    test_idx: int = 0,
    model: str = "gpt-5.4",
    provider: str | None = None,
    reasoning_effort: Optional[str] = "low",
    tools_enabled: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_train_demos: Optional[int] = None,
    out_dir: str | Path = "evaluation/output_grid_eval_results",
) -> dict[str, Any]:
    """Same as :func:`run_output_grid_api_evaluation_once` but accepts a loaded :class:`ArcTask`."""
    return _run_output_grid_api_evaluation_for_task(
        task=task,
        test_idx=test_idx,
        model=model,
        provider=provider,
        reasoning_effort=reasoning_effort,
        tools_enabled=tools_enabled,
        api_key=api_key,
        base_url=base_url,
        max_train_demos=max_train_demos,
        out_dir=out_dir,
    )
