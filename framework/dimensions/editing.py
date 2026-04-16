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
from typing import Any, Iterable, List, Literal, Optional, Sequence
from urllib import error, request

from framework.corruption.golf_ast import GolfSource, load_and_corrupt_golf_verifier
from framework.corruption.h_arc import load_h_arc_incorrect_outputs_for_task
from framework.corruption.verifier_ast import load_and_corrupt_re_arc_verifier
from framework.dimensions.base import (
    DimensionConfig,
    DimensionEvaluator,
    DimensionInstance,
    DimensionResult,
)
from framework.verifier_selection import select_verifier_for_task, valid_verifiers_for_task
from framework.grids import (
    Grid,
    GridPair,
    assert_int_grid,
    assert_int_grid_pair,
    connected_component_color_flips_on_pair,
    is_equal_grid,
    normalized_cell_edit_distance_padded,
    sample_flip_count_favor_one,
)
from framework.tasks.base import ArcTask, DimensionName

EditingVariant = Literal["h_arc_incorrect", "corrupted_verifier", "ground_truth", "color_flip"]

# Noticeable differences with H-ARC most likely.
_EDITING_BASE_WEIGHTS: dict[Literal["h_arc_incorrect", "corrupted_verifier", "color_flip"], float] = {
    "h_arc_incorrect": 0.56,
    "corrupted_verifier": 0.26,
    "color_flip": 0.18,
}


@dataclass
class EditingInstance(
    DimensionInstance[tuple[list[GridPair], GridPair], GridPair],
):
    """Training examples plus a corrupted pair that must be repaired."""

    pass


@dataclass
class EditingMetrics:
    success_rate: float


def _grid_to_str(grid: Grid) -> str:
    assert_int_grid(grid, name="grid")
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def _same_shape(a: Grid, b: Grid) -> bool:
    if len(a) != len(b):
        return False
    if not a:
        return not b
    return all(len(ra) == len(rb) for ra, rb in zip(a, b))


def _shape_preserving_pair_edit(a: GridPair, b: GridPair) -> bool:
    """True when edit can be interpreted as value/color-only changes.

    We intentionally treat any same-shape cell-value edit as "color-only" for this
    guardrail override: geometry/size did not change, only cell values did.
    """
    return _same_shape(a.input, b.input) and _same_shape(a.output, b.output)


def build_editing_prompt(
    *,
    demonstrations: Sequence[GridPair],
    corrupted_pair: GridPair,
    include_true: bool,
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

    if include_true:
        lines.append(
            "The following candidate input/output pair may already be correct or may have been corrupted in some way. "
            "Your task is to return the corrected version: if the pair is already correct, return it unchanged; "
            "if it is corrupted, fix whatever is necessary so that it becomes consistent with the rule inferred from the training examples."
        )
    else:
        lines.append(
            "The following candidate input/output pair has been corrupted in some way. "
            "Your task is to return the corrected input/output pair that is consistent with the rule inferred from the training examples."
        )

    lines.append("")
    lines.append("Candidate pair:")
    lines.append("Input:")
    lines.append(_grid_to_str(corrupted_pair.input))
    lines.append("Output:")
    lines.append(_grid_to_str(corrupted_pair.output))
    lines.append("")

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


def parse_editing_prediction(text: str) -> GridPair:
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
    return GridPair(inp, out)  # type: ignore[arg-type]


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


def run_editing_api_evaluation_once(
    *,
    task: ArcTask,
    model: str = "gpt-5.4",
    reasoning_effort: Optional[str] = "low",
    tools_enabled: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    seed: int = 0,
    include_true: bool = False,
    max_norm_edit: float | None = 0.70,
    out_dir: str | Path = "evaluation/editing_eval_results",
) -> dict[str, Any]:
    """Run editing eval; after the model returns, ``raw_response`` is always saved."""
    rng = random.Random(seed)
    # Sampling can occasionally hit brittle corruption branches; re-sample on errors.
    last_sampling_error: Exception | None = None
    corrupted_pair: GridPair | None = None
    target_pair: GridPair | None = None
    source: EditingVariant | None = None
    for _ in range(16):
        try:
            corrupted_pair, target_pair, source = sample_editing_pair(task, rng, include_true=include_true)
            # Reject degenerate identity examples early.
            if is_equal_grid(corrupted_pair.input, corrupted_pair.output):
                continue
            if is_equal_grid(target_pair.input, target_pair.output):
                continue
            break
        except Exception as e:
            last_sampling_error = e
            continue
    if corrupted_pair is None or target_pair is None or source is None:
        msg = "Editing sampling failed after retries."
        if last_sampling_error is not None:
            msg = f"{msg} Last error: {last_sampling_error}"
        raise RuntimeError(msg)

    prompt = build_editing_prompt(
        demonstrations=task.train_pairs,
        corrupted_pair=corrupted_pair,
        include_true=include_true,
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

    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{task.task_id}_editing_eval_{ts}.json"

    result: dict[str, Any] = {
        "task_id": task.task_id,
        "model": model,
        "base_url": effective_base,
        "reasoning_effort": reasoning_effort,
        "tools_enabled": tools_enabled,
        "seed": seed,
        "include_true": include_true,
        "corruption_source": source,
        "prompt": prompt,
        "raw_response": raw,
        "corrupted_pair": {"input": corrupted_pair.input, "output": corrupted_pair.output},
        "target_pair": {"input": target_pair.input, "output": target_pair.output},
        "evaluation_ok": False,
        "prediction_pair": None,
        "guardrail": {"enabled": max_norm_edit is not None},
        "metrics": {"success": False},
    }
    try:
        pred_pair = parse_editing_prediction(raw)
        if is_equal_grid(pred_pair.input, pred_pair.output):
            result["prediction_pair"] = {"input": pred_pair.input, "output": pred_pair.output}
            result["guardrail"] = {
                "enabled": max_norm_edit is not None,
                "identity_rejected": True,
                "passed": False,
            }
            result["metrics"] = {"success": False}
            result["evaluation_ok"] = True
            out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
            result["result_path"] = str(out_path)
            return result

        rail: dict[str, Any] = {"enabled": max_norm_edit is not None}
        if max_norm_edit is not None:
            d_in = normalized_cell_edit_distance_padded(
                corrupted_pair.input,
                pred_pair.input,
            )
            d_out = normalized_cell_edit_distance_padded(
                corrupted_pair.output,
                pred_pair.output,
            )
            rail["max_norm_edit"] = max_norm_edit
            rail["normalized_distance_input"] = d_in
            rail["normalized_distance_output"] = d_out
            passed = bool(not (d_in > max_norm_edit and d_out > max_norm_edit))
            if (
                not passed
                and source == "color_flip"
                and _shape_preserving_pair_edit(corrupted_pair, pred_pair)
            ):
                passed = True
                rail["distance_override"] = "color_flip_shape_preserving_edit"
            rail["passed"] = passed
        else:
            rail["passed"] = True

        picked = select_verifier_for_task(
            task,
            dynamic_pairs=[GridPair(target_pair.input, target_pair.output)],
        )
        ok = False
        if rail["passed"] and picked is not None:
            _slot, v = picked
            try:
                ok = is_equal_grid(v(copy.deepcopy(pred_pair.input)), pred_pair.output)
            except Exception:
                ok = False
        elif rail["passed"]:
            ok = is_equal_grid(pred_pair.input, target_pair.input) and is_equal_grid(
                pred_pair.output, target_pair.output
            )

        result["prediction_pair"] = {"input": pred_pair.input, "output": pred_pair.output}
        result["guardrail"] = rail
        result["metrics"] = {"success": ok}
        result["evaluation_ok"] = True
    except Exception as e:
        result["error"] = str(e)

    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    result["result_path"] = str(out_path)
    return result


def _editing_variant_weights(include_true: bool) -> dict[EditingVariant, float]:
    if not include_true:
        return {
            "h_arc_incorrect": _EDITING_BASE_WEIGHTS["h_arc_incorrect"],
            "corrupted_verifier": _EDITING_BASE_WEIGHTS["corrupted_verifier"],
            "color_flip": _EDITING_BASE_WEIGHTS["color_flip"],
        }
    # include_true=True: ground truth is 3rd most likely (below corrupted_verifier, above color_flip).
    return {
        "h_arc_incorrect": 0.48,
        "corrupted_verifier": 0.22,
        "ground_truth": 0.18,
        "color_flip": 0.12,
    }


def _sample_editing_variant(rng: random.Random, *, include_true: bool) -> EditingVariant:
    wmap = _editing_variant_weights(include_true)
    cats = list(wmap.keys())
    w = [wmap[c] for c in cats]
    (picked,) = rng.choices(cats, weights=w, k=1)
    return picked


def _try_h_arc_incorrect(task: ArcTask, rng: random.Random) -> tuple[GridPair, GridPair] | None:
    if not task.test_inputs or not task.test_outputs:
        return None
    rows = load_h_arc_incorrect_outputs_for_task(task.task_id, exclude_evaluation=True)
    if not rows:
        return None
    weights = [max(1, r.count) for r in rows]
    (picked,) = rng.choices(rows, weights=weights, k=1)
    idx = rng.randrange(min(len(task.test_inputs), len(task.test_outputs)))
    inp = copy.deepcopy(task.test_inputs[idx])
    target = GridPair(copy.deepcopy(inp), copy.deepcopy(task.test_outputs[idx]))
    corrupted = GridPair(copy.deepcopy(inp), copy.deepcopy(picked.output_grid))
    return corrupted, target


def _try_corrupted_verifier(task: ArcTask, rng: random.Random) -> tuple[GridPair, GridPair] | None:
    if task.arc_gen_generator is None:
        return None
    for _ in range(10):
        try:
            gold = copy.deepcopy(task.arc_gen_generator(1)[0])
        except Exception:
            continue
        inp = copy.deepcopy(gold.input)
        picked = select_verifier_for_task(task, dynamic_pairs=[gold])
        if picked is None:
            continue
        slot, _fn = picked
        if slot == "re_arc":
            try:
                cfn, _, _ = load_and_corrupt_re_arc_verifier(
                    task.task_id,
                    rng=rng,
                    sample_input=inp,
                    max_attempts=80,
                    max_normalized_cell_edit_distance=0.70,
                )
                bad = cfn(copy.deepcopy(inp))
                return GridPair(inp, bad), GridPair(copy.deepcopy(inp), copy.deepcopy(gold.output))
            except Exception:
                continue
        if slot in ("google", "keymoon", "neurips"):
            gs: GolfSource = slot  # type: ignore[assignment]
            try:
                cfn, _, _ = load_and_corrupt_golf_verifier(
                    task.task_id,
                    gs,
                    rng=rng,
                    sample_input=inp,
                    max_attempts=80,
                    max_normalized_cell_edit_distance=0.70,
                )
                bad = cfn(copy.deepcopy(inp))
                return GridPair(inp, bad), GridPair(copy.deepcopy(inp), copy.deepcopy(gold.output))
            except Exception:
                continue
        if slot == "custom":
            # Fall back to existing corruption operators.
            try:
                cfn, _, _ = load_and_corrupt_re_arc_verifier(
                    task.task_id,
                    rng=rng,
                    sample_input=inp,
                    max_attempts=80,
                    max_normalized_cell_edit_distance=0.70,
                )
                bad = cfn(copy.deepcopy(inp))
                return GridPair(inp, bad), GridPair(copy.deepcopy(inp), copy.deepcopy(gold.output))
            except Exception:
                pass
            for src in ("google", "keymoon", "neurips"):
                gs2: GolfSource = src  # type: ignore[assignment]
                try:
                    cfn, _, _ = load_and_corrupt_golf_verifier(
                        task.task_id,
                        gs2,
                        rng=rng,
                        sample_input=inp,
                        max_attempts=80,
                        max_normalized_cell_edit_distance=0.70,
                    )
                    bad = cfn(copy.deepcopy(inp))
                    return GridPair(inp, bad), GridPair(copy.deepcopy(inp), copy.deepcopy(gold.output))
                except Exception:
                    continue
    return None


def _try_color_flip(task: ArcTask, rng: random.Random) -> tuple[GridPair, GridPair] | None:
    opts: list[GridPair] = []
    opts.extend(task.train_pairs)
    if task.arc_gen_synthetic_pairs:
        opts.extend(task.arc_gen_synthetic_pairs)
    if task.arc_gen_generator is not None:
        try:
            opts.extend(task.arc_gen_generator(30))
        except Exception:
            pass
    if not opts:
        return None
    target = copy.deepcopy(rng.choice(opts))
    n_ops = sample_flip_count_favor_one(rng, max_flips=5)
    corrupted, _ = connected_component_color_flips_on_pair(copy.deepcopy(target), rng, num_ops=n_ops)
    return corrupted, target


def _corrupted_pair_fails_gold_verifier(
    task: ArcTask, corrupted: GridPair, gold: GridPair
) -> bool:
    """True iff the verifier chosen with *gold* in context disagrees with *corrupted*.

    Editing negatives must not be accidentally consistent with the task rule.
    """
    picked = select_verifier_for_task(task, dynamic_pairs=[gold])
    if picked is None:
        return False
    _slot, v = picked
    try:
        out = v(copy.deepcopy(corrupted.input))
    except Exception:
        return True
    return not is_equal_grid(out, corrupted.output)


def _try_ground_truth_pair(task: ArcTask, rng: random.Random) -> tuple[GridPair, GridPair] | None:
    opts: list[GridPair] = []
    opts.extend(task.train_pairs)
    if task.arc_gen_synthetic_pairs:
        opts.extend(task.arc_gen_synthetic_pairs)
    if task.arc_gen_generator is not None:
        try:
            opts.extend(task.arc_gen_generator(30))
        except Exception:
            pass
    if not opts:
        return None
    target = copy.deepcopy(rng.choice(opts))
    return copy.deepcopy(target), target


def sample_editing_pair(
    task: ArcTask,
    rng: random.Random,
    *,
    include_true: bool = False,
) -> tuple[GridPair, GridPair, EditingVariant]:
    """Return (corrupted_pair, corrected_pair, source)."""
    for _ in range(120):
        try:
            src = _sample_editing_variant(rng, include_true=include_true)
            picked: tuple[GridPair, GridPair] | None = None
            if src == "h_arc_incorrect":
                picked = _try_h_arc_incorrect(task, rng)
            elif src == "corrupted_verifier":
                picked = _try_corrupted_verifier(task, rng)
            elif src == "ground_truth":
                picked = _try_ground_truth_pair(task, rng)
            else:
                picked = _try_color_flip(task, rng)
            if picked is None:
                continue
            corrupted, target = picked
            assert_int_grid_pair(corrupted, name="editing corrupted pair")
            assert_int_grid_pair(target, name="editing target pair")
            if src != "ground_truth":
                if not _corrupted_pair_fails_gold_verifier(task, corrupted, target):
                    continue
            return corrupted, target, src
        except Exception:
            # Keep sampling; one broken branch should not fail the entire dimension.
            continue

    # Last-resort fallback: color flip on any available valid base pair.
    bases: list[GridPair] = []
    bases.extend(task.train_pairs)
    if task.arc_gen_synthetic_pairs:
        bases.extend(task.arc_gen_synthetic_pairs)
    if task.arc_gen_generator is not None:
        try:
            bases.extend(task.arc_gen_generator(30))
        except Exception:
            pass
    for _ in range(120):
        if not bases:
            break
        try:
            base = copy.deepcopy(rng.choice(bases))
            assert_int_grid_pair(base, name="editing fallback base pair")
            n_ops = sample_flip_count_favor_one(rng, max_flips=5)
            corrupted, _ = connected_component_color_flips_on_pair(copy.deepcopy(base), rng, num_ops=n_ops)
            assert_int_grid_pair(corrupted, name="editing fallback corrupted pair")
            if not _corrupted_pair_fails_gold_verifier(task, corrupted, base):
                continue
            return corrupted, base, "color_flip"
        except Exception:
            continue
    raise RuntimeError(f"{task.task_id}: could not sample a valid editing pair after retries.")


class EditingEvaluator(
    DimensionEvaluator[
        tuple[list[GridPair], GridPair],
        GridPair,
        GridPair,
        EditingMetrics,
    ]
):
    """Editing evaluator with weighted corruption sources and verifier-based scoring."""

    def __init__(self, config: DimensionConfig | None = None, *, include_true: bool = False) -> None:
        self.config = config or DimensionConfig(name=DimensionName.EDITING)
        self.include_true = include_true

    def generate_instances(
        self,
        tasks: Iterable[ArcTask],
    ) -> List[EditingInstance]:
        rng = random.Random(self.config.random_seed if self.config.random_seed is not None else 0)
        n_per = max(1, self.config.num_instances_per_task)
        instances: List[EditingInstance] = []
        for task in tasks:
            if not task.train_pairs:
                continue
            for _ in range(n_per):
                corrupted_pair, target_pair, _source = sample_editing_pair(
                    task,
                    rng,
                    include_true=self.include_true,
                )
                instances.append(
                    EditingInstance(
                        task_id=task.task_id,
                        input=(list(task.train_pairs), corrupted_pair),
                        target=target_pair,
                    )
                )
        return instances

    def score_predictions(
        self,
        instances: Sequence[EditingInstance],
        predictions: Sequence[GridPair],
    ) -> DimensionResult[EditingMetrics]:
        if len(instances) != len(predictions):
            raise ValueError("instances and predictions must have the same length")
        if not instances:
            return DimensionResult(
                dimension=DimensionName.EDITING,
                metrics=EditingMetrics(success_rate=0.0),
            )

        from framework.tasks.arc_dataset import load_task

        ok = 0
        total = 0
        for inst, pred in zip(instances, predictions):
            total += 1
            # Verify with task-specific best verifier, with target fallback.
            try:
                task = load_task(inst.task_id)
            except Exception:
                task = None
            if task is not None:
                ref = inst.target
                picked = select_verifier_for_task(
                    task,
                    dynamic_pairs=[GridPair(ref.input, ref.output)],
                )
                if picked is not None:
                    _slot, v = picked
                    try:
                        expected = v(copy.deepcopy(pred.input))
                        if is_equal_grid(expected, pred.output):
                            ok += 1
                        continue
                    except Exception:
                        # Validation fallback: selected verifier crashed, try others.
                        matched_other = False
                        for alt_slot, alt_v in valid_verifiers_for_task(task):
                            if alt_slot == _slot:
                                continue
                            try:
                                expected_alt = alt_v(copy.deepcopy(pred.input))
                            except Exception:
                                continue
                            if is_equal_grid(expected_alt, pred.output):
                                matched_other = True
                                break
                        if matched_other:
                            ok += 1
                            continue
            # Fallback when no verifier available.
            if is_equal_grid(pred.input, inst.target.input) and is_equal_grid(pred.output, inst.target.output):
                ok += 1

        metrics = EditingMetrics(success_rate=(ok / total) if total else 0.0)
        return DimensionResult(
            dimension=DimensionName.EDITING,
            metrics=metrics,
        )

