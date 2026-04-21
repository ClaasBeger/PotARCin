from __future__ import annotations

import ast
import copy
import datetime as dt
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

from framework.verifier_selection import (
    VerifierSlot,
    select_verifier_for_task,
    valid_verifiers_for_task,
)
from framework.dimensions.base import (
    DimensionConfig,
    DimensionEvaluator,
    DimensionInstance,
    DimensionResult,
)
from framework.grids import Grid, GridPair, is_equal_grid
from framework.llm_api import request_text_completion, resolve_provider_config
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
    verifier_slot: VerifierSlot | None = None


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
    lines.append("Now infer an input grid that would produce the following output by applying the same rule.")
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

            # Primary source: ARC-GEN dynamic pairs (stochastic draws; seed via config).
            sampled_pairs: list[GridPair] = []
            if task.arc_gen_generator is not None:
                try:
                    sampled_pairs = task.arc_gen_generator(n_per, rng=rng)
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
                slot, verifier = picked
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
                        verifier_slot=slot,
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
            # Step 1: exact match with sampled ARC-GEN input can terminate early.
            if is_equal_grid(pred, inst.target):
                correct += 1
                continue
            # Inversion should produce an input that maps to the desired output; a
            # no-change identity (input == output) is usually invalid, except when
            # it is also the exact sampled reference input (handled above).
            if is_equal_grid(pred, inst.input.desired_output):
                continue
            if inst.verifier is None:
                continue
            try:
                produced = inst.verifier(copy.deepcopy(pred))
            except Exception:
                # Fallback: if selected verifier crashes, try other valid verifiers.
                try:
                    task = load_task(inst.task_id)
                except Exception:
                    continue
                matched_other = False
                for slot, verifier in valid_verifiers_for_task(task):
                    if inst.verifier_slot is not None and slot == inst.verifier_slot:
                        continue
                    try:
                        produced_alt = verifier(copy.deepcopy(pred))
                    except Exception:
                        continue
                    if is_equal_grid(produced_alt, inst.input.desired_output):
                        matched_other = True
                        break
                if matched_other:
                    correct += 1
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
    provider: str | None = None,
    reasoning_effort: Optional[str] = "low",
    tools_enabled: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    seed: int = 0,
    out_dir: str | Path = "evaluation/inversion_eval_results",
) -> dict[str, Any]:
    """After the model returns, ``raw_response`` is always written to the eval JSON."""
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
    out_path = out_root / f"{task_id}_inversion_eval_{ts}.json"

    result: dict[str, Any] = {
        "task_id": task_id,
        "model": model,
        "provider": resolved_provider,
        "base_url": effective_base,
        "reasoning_effort": reasoning_effort,
        "tools_enabled": tools_enabled,
        "seed": seed,
        "prompt": prompt,
        "raw_response": raw,
        "desired_output": inst.input.desired_output,
        "reference_input": inst.input.reference_input,
        "evaluation_ok": False,
        "prediction_input": None,
        "metrics": {"success_rate": 0.0},
    }
    try:
        pred_input = parse_inversion_prediction(raw)
        scored = evaluator.score_predictions([inst], [pred_input]).metrics
        result["prediction_input"] = pred_input
        result["metrics"] = {"success_rate": scored.success_rate}
        result["evaluation_ok"] = True
    except Exception as e:
        result["error"] = str(e)

    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    result["result_path"] = str(out_path)
    return result
