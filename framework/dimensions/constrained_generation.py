from __future__ import annotations

import ast
import copy
import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from framework.verifier_selection import select_verifier_for_task, valid_verifiers_for_task
from framework.dimensions.base import (
    DimensionConfig,
    DimensionEvaluator,
    DimensionInstance,
    DimensionResult,
)
from framework.grids import Grid, GridPair, is_equal_grid
from framework.llm_api import request_text_completion, resolve_provider_config
from framework.tasks.arc_dataset import load_task
from framework.tasks.base import ArcTask, DimensionName


GeneratedPair = Tuple[Grid, Grid]


@dataclass
class ConstrainedGenerationInstance(DimensionInstance[list[GridPair], None]):
    """Training examples; model must generate a new (input, output) pair."""

    pass


@dataclass
class ConstrainedGenerationMetrics:
    success_rate: float
    rejected_as_demo_copy: int = 0
    rejected_no_change: int = 0
    rejected_verifier_missing: int = 0
    rejected_verifier_mismatch: int = 0
    noted_arc_gen_stable_exact_match: int = 0


def _grid_to_str(grid: Grid) -> str:
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def build_constrained_generation_prompt(*, demonstrations: Sequence[GridPair]) -> str:
    lines: list[str] = []
    lines.append(
        "Find the common rule that maps an input grid to an output grid given the examples below."
    )
    lines.append("")
    for idx, ex in enumerate(demonstrations, 1):
        lines.append(f"Example {idx}:")
        lines.append("Input:")
        lines.append(_grid_to_str(ex.input))
        lines.append("Output:")
        lines.append(_grid_to_str(ex.output))
        lines.append("")
    lines.append(
        "Generate one NEW input/output pair that follows the same transformation rule.\n"
        "- Do NOT copy any demonstration pair exactly.\n"
        "- The generated input and output must be different (non-trivial transformation).\n"
        "- Your pair must be consistent with the same underlying rule."
    )
    lines.append(
        "\nYour final answer must be ONLY a single Python dict with exactly these keys:\n"
        "{\"input\": [[...], ...], \"output\": [[...], ...]}\n"
        "Use integer grid cells 0-9. Output nothing else."
    )
    return "\n".join(lines)


def _is_grid_like(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    if not all(isinstance(row, list) and row for row in value):
        return False
    return all(
        all(isinstance(cell, int) or (isinstance(cell, str) and cell.strip().isdigit()) for cell in row)
        for row in value
    )


def _is_prediction_dict(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    inp = value.get("input")
    out = value.get("output")
    return _is_grid_like(inp) and _is_grid_like(out)


def _extract_first_braced_object(text: str) -> str:
    s = text.strip()
    if "```" in s:
        s = re.sub(r"```(?:python|json)?\s*", "", s, count=1)
        s = s.replace("```", "").strip()

    # Scan balanced braces and choose the first parseable object with
    # {"input": <grid>, "output": <grid>} shape. Robust to leading prose.
    starts = [i for i, ch in enumerate(s) if ch == "{"]
    candidates: list[str] = []
    for start in starts:
        depth = 0
        for idx in range(start, len(s)):
            ch = s[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start : idx + 1]
                    try:
                        parsed = ast.literal_eval(candidate)
                    except Exception:
                        try:
                            parsed = json.loads(candidate)
                        except Exception:
                            continue
                    if _is_prediction_dict(parsed):
                        candidates.append(candidate)
                    break
            if depth < 0:
                break
    if not candidates:
        raise ValueError("Could not locate prediction dict in model output.")
    # Prefer the last valid object so explicit model self-corrections win.
    return candidates[-1]


def parse_constrained_generation_prediction(text: str) -> GeneratedPair:
    raw_obj = _extract_first_braced_object(text)
    obj: Any
    try:
        obj = ast.literal_eval(raw_obj)
    except Exception:
        obj = json.loads(raw_obj)
    if not isinstance(obj, dict):
        raise ValueError("Prediction must be a dict.")
    inp = obj.get("input")
    out = obj.get("output")
    if not isinstance(inp, list) or not isinstance(out, list):
        raise ValueError("Prediction dict must have list fields `input` and `output`.")
    def _to_int_grid(g: Any) -> Grid:
        if not isinstance(g, list):
            raise ValueError("Grid must be a list-of-lists.")
        out_rows: Grid = []
        for row in g:
            if not isinstance(row, list):
                raise ValueError("Grid rows must be lists.")
            out_row: list[int] = []
            for cell in row:
                if isinstance(cell, int):
                    out_row.append(cell)
                elif isinstance(cell, str) and cell.strip().isdigit():
                    out_row.append(int(cell.strip()))
                else:
                    raise ValueError("Grid cells must be integers (or numeric strings).")
            out_rows.append(out_row)
        return out_rows

    return _to_int_grid(inp), _to_int_grid(out)


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


class ConstrainedGenerationEvaluator(
    DimensionEvaluator[list[GridPair], None, GeneratedPair, ConstrainedGenerationMetrics]
):
    """Constrained-generation evaluator with verifier-based checks."""

    def __init__(self, config: DimensionConfig | None = None) -> None:
        self.config = config or DimensionConfig(
            name=DimensionName.CONSTRAINED_GENERATION
        )

    def generate_instances(
        self,
        tasks: Iterable[ArcTask],
    ) -> List[ConstrainedGenerationInstance]:
        instances: List[ConstrainedGenerationInstance] = []
        for task in tasks:
            if not task.train_pairs:
                continue
            instances.append(
                ConstrainedGenerationInstance(
                    task_id=task.task_id,
                    input=task.train_pairs,
                    target=None,
                )
            )
        return instances

    def score_predictions(
        self,
        instances: Sequence[ConstrainedGenerationInstance],
        predictions: Sequence[GeneratedPair],
    ) -> DimensionResult[ConstrainedGenerationMetrics]:
        if len(instances) != len(predictions):
            raise ValueError("instances and predictions must have the same length")
        if not instances:
            return DimensionResult(
                dimension=DimensionName.CONSTRAINED_GENERATION,
                metrics=ConstrainedGenerationMetrics(success_rate=0.0),
            )

        ok = 0
        rej_demo = 0
        rej_no_change = 0
        rej_no_verifier = 0
        rej_mismatch = 0
        noted_arc_gen_stable = 0

        for inst, (gen_in, gen_out) in zip(instances, predictions):
            # Rule 1: must not copy any demonstration pair exactly.
            copied_demo = any(
                is_equal_grid(gen_in, p.input) and is_equal_grid(gen_out, p.output)
                for p in inst.input
            )
            if copied_demo:
                rej_demo += 1
                continue

            # Rule 2: there must be a non-trivial transformation.
            if is_equal_grid(gen_in, gen_out):
                rej_no_change += 1
                continue

            # Rule 3: output must match the selected strict verifier on generated input.
            try:
                task = load_task(inst.task_id)
            except Exception:
                rej_no_verifier += 1
                continue

            # Note-only rail: exact copy of ARC-GEN stable pair (not rejected).
            if task.arc_gen_synthetic_pairs:
                if any(
                    is_equal_grid(gen_in, p.input) and is_equal_grid(gen_out, p.output)
                    for p in task.arc_gen_synthetic_pairs
                ):
                    noted_arc_gen_stable += 1

            picked = select_verifier_for_task(task)
            if picked is None:
                rej_no_verifier += 1
                continue
            _slot, verifier = picked
            try:
                expected = verifier(copy.deepcopy(gen_in))
            except Exception:
                # Validation fallback: selected verifier crashed, try other valid ones.
                matched_other = False
                for alt_slot, alt_verifier in valid_verifiers_for_task(task):
                    if alt_slot == _slot:
                        continue
                    try:
                        expected_alt = alt_verifier(copy.deepcopy(gen_in))
                    except Exception:
                        continue
                    try:
                        if is_equal_grid(expected_alt, gen_out):
                            matched_other = True
                            break
                    except Exception:
                        continue
                if matched_other:
                    ok += 1
                else:
                    rej_mismatch += 1
                continue
            try:
                matches = is_equal_grid(expected, gen_out)
            except Exception:
                matches = False
            if matches:
                ok += 1
            else:
                rej_mismatch += 1

        n = len(instances)
        metrics = ConstrainedGenerationMetrics(
            success_rate=ok / n,
            rejected_as_demo_copy=rej_demo,
            rejected_no_change=rej_no_change,
            rejected_verifier_missing=rej_no_verifier,
            rejected_verifier_mismatch=rej_mismatch,
            noted_arc_gen_stable_exact_match=noted_arc_gen_stable,
        )
        return DimensionResult(
            dimension=DimensionName.CONSTRAINED_GENERATION,
            metrics=metrics,
        )


def _cg_zero_metrics() -> dict[str, Any]:
    return {
        "success_rate": 0.0,
        "rejected_as_demo_copy": 0,
        "rejected_no_change": 0,
        "rejected_verifier_missing": 0,
        "rejected_verifier_mismatch": 0,
        "noted_arc_gen_stable_exact_match": 0,
    }


def run_constrained_generation_api_evaluation_once(
    *,
    task_id: str,
    model: str = "gpt-5.4",
    provider: str | None = None,
    reasoning_effort: Optional[str] = "low",
    tools_enabled: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    out_dir: str | Path = "evaluation/constrained_generation_eval_results",
) -> dict[str, Any]:
    """Call the model, parse, score, and write JSON under *out_dir*.

    After a non-empty model response is received, the result file is **always**
    written (including ``raw_response``), even when parsing or scoring fails.
    The returned dict includes ``evaluation_ok`` (bool) so batch runners can
    record ``result_path`` on failure without losing the transcript.
    """
    task = load_task(task_id)
    if not task.train_pairs:
        raise ValueError(f"Task {task_id} has no training pairs.")

    prompt = build_constrained_generation_prompt(demonstrations=task.train_pairs)
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
    out_path = out_root / f"{task_id}_constrained_generation_eval_{ts}.json"

    result: dict[str, Any] = {
        "task_id": task_id,
        "model": model,
        "provider": resolved_provider,
        "base_url": effective_base,
        "reasoning_effort": reasoning_effort,
        "tools_enabled": tools_enabled,
        "prompt": prompt,
        "raw_response": raw,
        "evaluation_ok": False,
    }

    pred: GeneratedPair | None = None
    parse_error: str | None = None
    try:
        pred = parse_constrained_generation_prediction(raw)
    except Exception as e:
        parse_error = str(e)

    if pred is None:
        result["error"] = parse_error or "parse failed"
        result["prediction"] = None
        result["metrics"] = _cg_zero_metrics()
        out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        result["result_path"] = str(out_path)
        return result

    result["prediction"] = {"input": pred[0], "output": pred[1]}

    scoring_error: str | None = None
    try:
        evaluator = ConstrainedGenerationEvaluator()
        inst = ConstrainedGenerationInstance(task_id=task_id, input=task.train_pairs, target=None)
        scored = evaluator.score_predictions([inst], [pred])
        m = scored.metrics
        result["metrics"] = {
            "success_rate": m.success_rate,
            "rejected_as_demo_copy": m.rejected_as_demo_copy,
            "rejected_no_change": m.rejected_no_change,
            "rejected_verifier_missing": m.rejected_verifier_missing,
            "rejected_verifier_mismatch": m.rejected_verifier_mismatch,
            "noted_arc_gen_stable_exact_match": m.noted_arc_gen_stable_exact_match,
        }
        result["evaluation_ok"] = True
    except Exception as e:
        scoring_error = str(e)
        result["error"] = scoring_error
        result["metrics"] = _cg_zero_metrics()

    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    result["result_path"] = str(out_path)
    return result

