from __future__ import annotations

import ast
import copy
import datetime as dt
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence
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
from framework.tasks.base import ArcTask, DimensionName, Verifier


@dataclass
class InversionPrompt:
    """Demonstrations plus a desired output grid to invert."""

    demonstrations: list[GridPair]
    desired_output: Grid
    # Optional known input from the sampled source pair (for exact-match short-circuit).
    reference_input: Grid | None = None


@dataclass
class InversionInstance(DimensionInstance[InversionPrompt, Grid]):
    """Given demos and desired output, predict an input grid."""

    verifier: Verifier | None = None


@dataclass
class InversionMetrics:
    success_rate: float


def _grid_to_str(grid: Grid) -> str:
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def build_inversion_prompt(
    *,
    demonstrations: Sequence[GridPair],
    desired_output: Grid,
) -> str:
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
    lines.append("Now infer an input grid that would produce the following output under the same rule.")
    lines.append("Desired output:")
    lines.append(_grid_to_str(desired_output))
    lines.append(
        "\nYour final answer must be ONLY a single Python list-of-lists grid:\n"
        "[[...], ...]\n"
        "Use integer grid cells 0-9. Output nothing else."
    )
    return "\n".join(lines)


def _extract_first_bracketed_list(text: str) -> str:
    s = text.strip()
    if "```" in s:
        s = re.sub(r"```(?:python|json)?\s*", "", s, count=1)
        s = s.replace("```", "").strip()
    start = s.find("[")
    end = s.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not locate list grid in model output.")
    return s[start : end + 1]


def parse_inversion_prediction(text: str) -> Grid:
    raw_obj = _extract_first_bracketed_list(text)
    obj: Any
    try:
        obj = ast.literal_eval(raw_obj)
    except Exception:
        obj = json.loads(raw_obj)
    if not isinstance(obj, list):
        raise ValueError("Prediction must be a list-of-lists grid.")
    return obj  # type: ignore[return-value]


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


class InversionEvaluator(
    DimensionEvaluator[InversionPrompt, Grid, Grid, InversionMetrics]
):
    """Evaluator for inversion: find an input that maps to a given output."""

    def __init__(self, config: DimensionConfig | None = None) -> None:
        self.config = config or DimensionConfig(name=DimensionName.INVERSION)

    def generate_instances(self, tasks: Iterable[ArcTask]) -> List[InversionInstance]:
        rng = random.Random(self.config.random_seed if self.config.random_seed is not None else 0)
        n_per = max(1, self.config.num_instances_per_task)
        instances: List[InversionInstance] = []
        for task in tasks:
            if not task.train_pairs:
                continue

            # Primary source: ARC-GEN dynamic pairs.
            sampled_pairs: list[GridPair] = []
            if task.arc_gen_generator is not None:
                try:
                    sampled_pairs = task.arc_gen_generator(n_per)
                except Exception:
                    sampled_pairs = []

            # Fallback for tasks without dynamic generator (keeps evaluator robust).
            if not sampled_pairs:
                sampled_pairs = [
                    GridPair(copy.deepcopy(test_input), copy.deepcopy(test_output))
                    for test_input, test_output in zip(task.test_inputs, task.test_outputs)
                ][:n_per]

            for pair in sampled_pairs:
                picked = select_verifier_for_task(task, dynamic_pairs=[pair])
                if picked is None:
                    continue
                _slot, verifier = picked
                instances.append(
                    InversionInstance(
                        task_id=task.task_id,
                        input=InversionPrompt(
                            demonstrations=list(task.train_pairs),
                            desired_output=copy.deepcopy(pair.output),
                            reference_input=copy.deepcopy(pair.input),
                        ),
                        target=copy.deepcopy(pair.input),
                        verifier=verifier,
                    )
                )
        return instances

    def score_predictions(
        self,
        instances: Sequence[InversionInstance],
        predictions: Sequence[Grid],
    ) -> DimensionResult[InversionMetrics]:
        if len(instances) != len(predictions):
            raise ValueError("instances and predictions must have the same length")
        if not instances:
            return DimensionResult(
                dimension=DimensionName.INVERSION,
                metrics=InversionMetrics(success_rate=0.0),
            )

        correct = 0
        for inst, pred in zip(instances, predictions):
            # Inversion should produce an input that maps to the desired output; a
            # no-change identity (input == output) is treated as invalid.
            if is_equal_grid(pred, inst.input.desired_output):
                continue
            # Step 1: exact match with sampled ARC-GEN input can terminate early.
            if is_equal_grid(pred, inst.target):
                correct += 1
                continue
            if inst.verifier is None:
                continue
            try:
                produced = inst.verifier(copy.deepcopy(pred))
            except Exception:
                continue
            if is_equal_grid(produced, inst.input.desired_output):
                correct += 1

        return DimensionResult(
            dimension=DimensionName.INVERSION,
            metrics=InversionMetrics(success_rate=correct / len(instances)),
        )


def run_inversion_api_evaluation_once(
    *,
    task_id: str,
    model: str = "gpt-5.4",
    reasoning_effort: Optional[str] = "low",
    tools_enabled: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    seed: int = 0,
    out_dir: str | Path = "evaluation/inversion_eval_results",
) -> dict[str, Any]:
    task = load_task(task_id)
    evaluator = InversionEvaluator(
        config=DimensionConfig(name=DimensionName.INVERSION, num_instances_per_task=1, random_seed=seed)
    )
    instances = evaluator.generate_instances([task])
    if not instances:
        raise ValueError(f"No inversion instance could be generated for task {task_id}.")
    inst = instances[0]
    prompt = build_inversion_prompt(
        demonstrations=inst.input.demonstrations,
        desired_output=inst.input.desired_output,
    )

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
    pred_input = parse_inversion_prediction(raw)
    scored = evaluator.score_predictions([inst], [pred_input]).metrics

    result: dict[str, Any] = {
        "task_id": task_id,
        "model": model,
        "base_url": effective_base,
        "reasoning_effort": reasoning_effort,
        "tools_enabled": tools_enabled,
        "seed": seed,
        "prompt": prompt,
        "raw_response": raw,
        "desired_output": inst.input.desired_output,
        "reference_input": inst.input.reference_input,
        "prediction_input": pred_input,
        "metrics": {"success_rate": scored.success_rate},
    }
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{task_id}_inversion_eval_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["result_path"] = str(out_path)
    return result
