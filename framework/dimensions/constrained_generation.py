from __future__ import annotations

import ast
import copy
import datetime as dt
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple
from urllib import error, request

from framework.verifier_selection import select_verifier_for_task
from framework.dimensions.base import (
    DimensionConfig,
    DimensionEvaluator,
    DimensionInstance,
    DimensionResult,
)
from framework.grids import Grid, GridPair, is_equal_grid
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


def _extract_first_braced_object(text: str) -> str:
    s = text.strip()
    if "```" in s:
        s = re.sub(r"```(?:python|json)?\s*", "", s, count=1)
        s = s.replace("```", "").strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not locate dict object in model output.")
    return s[start : end + 1]


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
    return inp, out  # type: ignore[return-value]


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
                rej_mismatch += 1
                continue
            if is_equal_grid(expected, gen_out):
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


def run_constrained_generation_api_evaluation_once(
    *,
    task_id: str,
    model: str = "gpt-5.4",
    reasoning_effort: Optional[str] = "low",
    tools_enabled: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    out_dir: str | Path = "evaluation/constrained_generation_eval_results",
) -> dict[str, Any]:
    task = load_task(task_id)
    if not task.train_pairs:
        raise ValueError(f"Task {task_id} has no training pairs.")

    prompt = build_constrained_generation_prompt(demonstrations=task.train_pairs)
    effective_key = api_key or os.environ.get("OPENAI_API_KEY")
    effective_base = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not effective_key:
        raise ValueError("No API key provided. Set `api_key` or OPENAI_API_KEY.")

    raw = _openai_responses_completion(
        prompt=prompt,
        model=model,
        api_key=effective_key,
        base_url=effective_base,
        reasoning_effort=reasoning_effort,
        tools_enabled=tools_enabled,
    )
    pred = parse_constrained_generation_prediction(raw)

    evaluator = ConstrainedGenerationEvaluator()
    inst = ConstrainedGenerationInstance(task_id=task_id, input=task.train_pairs, target=None)
    scored = evaluator.score_predictions([inst], [pred])
    m = scored.metrics
    metrics = {
        "success_rate": m.success_rate,
        "rejected_as_demo_copy": m.rejected_as_demo_copy,
        "rejected_no_change": m.rejected_no_change,
        "rejected_verifier_missing": m.rejected_verifier_missing,
        "rejected_verifier_mismatch": m.rejected_verifier_mismatch,
        "noted_arc_gen_stable_exact_match": m.noted_arc_gen_stable_exact_match,
    }

    result: dict[str, Any] = {
        "task_id": task_id,
        "model": model,
        "base_url": effective_base,
        "reasoning_effort": reasoning_effort,
        "tools_enabled": tools_enabled,
        "prompt": prompt,
        "raw_response": raw,
        "prediction": {"input": pred[0], "output": pred[1]},
        "metrics": metrics,
    }
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{task_id}_constrained_generation_eval_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["result_path"] = str(out_path)
    return result

