from __future__ import annotations

import copy
import datetime as dt
import json
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple
from urllib import error, request

from framework.dimensions.base import DimensionConfig, DimensionEvaluator, DimensionInstance, DimensionResult
from framework.grids import Grid, GridPair, is_equal_grid
from framework.tasks.base import ArcTask


def _grid_to_str(grid: Grid) -> str:
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def build_rule_text_prompt(
    puzzle_json: dict,
    test_idx: int,
    model: str = "gpt-5.4",
    max_train_demos: Optional[int] = 3,
) -> str:
    """Build a prompt that asks for a natural-language transformation rule.

    The returned rule should be reusable later to transform new grids *without*
    providing demonstrations again.
    """
    train = puzzle_json["train"]
    if max_train_demos is not None:
        train = train[:max_train_demos]

    lines: list[str] = []
    lines.append(
        "Find the common rule that maps an input grid to an output grid, "
        "given the examples below."
    )
    lines.append(
        "Write a natural-language rule (concise but unambiguous) that "
        "describes how to transform the input grid into the output grid."
    )
    lines.append(
        "This rule will be reused later on new grids without demonstrations."
    )

    for idx, ex in enumerate(train, 1):
        lines.append(f"\nExample {idx}:\n")
        lines.append("Input:")
        lines.append(_grid_to_str(ex["input"]))
        lines.append("Output:")
        lines.append(_grid_to_str(ex["output"]))

    lines.append("\nBelow is a test input grid.")
    test_grid = puzzle_json["test"][test_idx]["input"]
    lines.append("Input:")
    lines.append(_grid_to_str(test_grid))

    lines.append(
        "\nReturn ONLY the natural-language rule text. "
        "Do not include markdown, code fences, or any additional commentary."
    )

    return "\n".join(lines)


def build_apply_rule_prompt(rule_text: str, inputs: Sequence[Grid], *, label: str) -> str:
    """Build a prompt that applies a previously saved rule to multiple grids."""
    lines: list[str] = []
    lines.append("You are given a transformation rule as natural language.")
    lines.append("Apply it to each input grid and produce the corresponding output grids.")
    lines.append("")
    lines.append("RULE:")
    lines.append(rule_text)
    lines.append("")
    lines.append(f"{label}:")
    for i, grid in enumerate(inputs):
        lines.append(f"\nInput {i}:")
        lines.append(_grid_to_str(grid))

    lines.append(
        "\nReturn ONLY valid JSON with this schema:\n"
        '{ "outputs": [ <output_grid_0>, <output_grid_1>, ... ] }\n'
        "where each <output_grid_k> is a list of lists of integers "
        "with the same dimensions as the expected output."
    )
    return "\n".join(lines)


def _openai_responses_completion(
    *,
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    reasoning_effort: Optional[str] = None,
    tools_enabled: bool = False,
    timeout_s: int = 180,
) -> str:
    """Send a prompt to an OpenAI-compatible `/responses` endpoint."""
    url = base_url.rstrip("/") + "/responses"
    payload: dict[str, Any] = {"model": model, "input": prompt}
    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}
    if tools_enabled:
        payload["tools"] = [{"type": "code_interpreter", "container": {"type": "auto"}}]

    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as e:
        body_txt = e.read().decode("utf-8", errors="replace")
        raise ValueError(f"Responses API error {e.code}: {body_txt}") from e

    parsed = json.loads(raw)
    out_text = parsed.get("output_text")
    if isinstance(out_text, str) and out_text.strip():
        return out_text

    parts: list[str] = []
    for item in parsed.get("output", []):
        for c in item.get("content", []):
            txt = c.get("text")
            if isinstance(txt, str) and txt.strip():
                parts.append(txt)
    return "\n".join(parts).strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract and parse a JSON object from model output."""
    s = text.strip()
    # Remove code fences if present.
    if "```" in s:
        s = s.replace("```json", "```").replace("```", "")
        s = s.strip()
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Fallback: locate the first { ... last } substring.
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not locate JSON object in model output.")
    sub = s[start : end + 1]
    parsed2 = json.loads(sub)
    if not isinstance(parsed2, dict):
        raise ValueError("Parsed JSON was not an object.")
    return parsed2


def run_rule_definition_api_evaluation(
    *,
    task: ArcTask,
    test_idx: int = 0,
    model: str = "gpt-5.4",
    reasoning_effort: Optional[str] = None,
    tools_enabled: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    out_dir: str | Path = "evaluation/definition_rule_eval_results",
    dynamic_n: int = 50,
    max_train_demos: Optional[int] = 3,
) -> dict[str, Any]:
    """Two-stage Definition-by-rule pipeline:

    1) Infer a natural-language rule from demos + one test input.
    2) Re-apply the saved rule to train/test/stable/dynamic grids *without*
       demonstrations, and evaluate output grids against ground truth.
    """
    if not task.train_pairs:
        raise ValueError(f"Task {task.task_id} has no training pairs.")
    if test_idx < 0 or test_idx >= len(task.test_inputs) or test_idx >= len(task.test_outputs):
        raise ValueError(f"Invalid test_idx={test_idx} for task {task.task_id}.")

    puzzle_json = {
        "train": [{"input": p.input, "output": p.output} for p in task.train_pairs],
        "test": [{"input": task.test_inputs[test_idx]}],
    }
    prompt_rule = build_rule_text_prompt(
        puzzle_json=puzzle_json,
        test_idx=0,
        model=model,
        max_train_demos=max_train_demos,
    )

    effective_key = api_key or os.environ.get("OPENAI_API_KEY")
    effective_base = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not effective_key:
        raise ValueError("No API key provided. Set `api_key` or OPENAI_API_KEY.")

    # Stage 1: rule inference.
    raw_rule_response = _openai_responses_completion(
        prompt=prompt_rule,
        model=model,
        api_key=effective_key,
        base_url=effective_base,
        reasoning_effort=reasoning_effort,
        tools_enabled=tools_enabled,
    )
    rule_text = raw_rule_response.strip()

    # Build evaluation sets.
    eval_sets: dict[str, dict[str, list[Grid]]] = {}
    eval_sets["train"] = {
        "inputs": [p.input for p in task.train_pairs],
        "outputs": [p.output for p in task.train_pairs],
    }
    eval_sets["test"] = {
        "inputs": [task.test_inputs[test_idx]],
        "outputs": [task.test_outputs[test_idx]],
    }
    eval_sets["stable"] = {
        "inputs": [p.input for p in (task.arc_gen_synthetic_pairs or [])],
        "outputs": [p.output for p in (task.arc_gen_synthetic_pairs or [])],
    }
    if task.arc_gen_generator is not None:
        dyn = task.arc_gen_generator(dynamic_n)
        eval_sets["dynamic_n"] = {
            "inputs": [p.input for p in dyn],
            "outputs": [p.output for p in dyn],
        }
    else:
        eval_sets["dynamic_n"] = {"inputs": [], "outputs": []}

    eval_sets["re_arc_stable"] = {
        "inputs": [p.input for p in (task.re_arc_synthetic_pairs or [])],
        "outputs": [p.output for p in (task.re_arc_synthetic_pairs or [])],
    }
    if task.re_arc_generator is not None:
        dyn = task.re_arc_generator(dynamic_n)
        eval_sets["re_arc_dynamic_n"] = {
            "inputs": [p.input for p in dyn],
            "outputs": [p.output for p in dyn],
        }
    else:
        eval_sets["re_arc_dynamic_n"] = {"inputs": [], "outputs": []}

    result: dict[str, Any] = {
        "task_id": task.task_id,
        "test_idx": test_idx,
        "model": model,
        "base_url": effective_base,
        "prompt_rule": prompt_rule,
        "raw_rule_response": raw_rule_response,
        "rule_text": rule_text,
        "metrics": {},
        "first_failures": {},
        "split_raw_apply_responses": {},
    }

    for split_name, payload in eval_sets.items():
        inputs = payload["inputs"]
        expected = payload["outputs"]
        total = len(inputs)
        ok = 0
        failure: Optional[dict[str, Any]] = None

        if total == 0:
            result["metrics"][split_name] = {"ok": 0, "total": 0, "success_rate": 0.0}
            result["first_failures"][split_name] = None
            continue

        prompt_apply = build_apply_rule_prompt(rule_text, inputs, label=f"{split_name} inputs")
        raw_apply = _openai_responses_completion(
            prompt=prompt_apply,
            model=model,
            api_key=effective_key,
            base_url=effective_base,
            reasoning_effort=reasoning_effort,
            tools_enabled=tools_enabled,
        )
        result["split_raw_apply_responses"][split_name] = raw_apply

        try:
            parsed = _extract_json_object(raw_apply)
            outputs = parsed["outputs"]
        except Exception as e:
            failure = {"error": f"Could not parse outputs JSON: {e}"}
            outputs = None

        if outputs is not None:
            if not isinstance(outputs, list) or len(outputs) != total:
                if failure is None:
                    failure = {
                        "error": f"Model returned outputs of unexpected shape: len(outputs)={len(outputs) if isinstance(outputs, list) else 'n/a'} expected={total}"
                    }
            else:
                for i in range(total):
                    got = outputs[i]
                    exp = expected[i]
                    if is_equal_grid(got, exp):
                        ok += 1
                    elif failure is None:
                        failure = {"index": i, "input": inputs[i], "expected": exp, "got": got}

        result["metrics"][split_name] = {
            "ok": ok,
            "total": total,
            "success_rate": (ok / total) if total > 0 else 0.0,
        }
        result["first_failures"][split_name] = failure

    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{task.task_id}_definition_rule_eval_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["result_path"] = str(out_path)
    return result


