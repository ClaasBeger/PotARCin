from __future__ import annotations

import copy
import datetime as dt
import json
import random
from pathlib import Path
from typing import Any, Optional

from framework.corruption.golf_ast import GolfSource, load_and_corrupt_golf_verifier
from framework.corruption.verifier_ast import load_and_corrupt_re_arc_verifier
from framework.dimensions.definition import (
    _extract_python_script,
    evaluate_definition_script_on_task,
)
from framework.grids import GridPair
from framework.llm_api import request_text_completion
from framework.tasks.base import ArcTask
from framework.verifier_selection import select_verifier_for_task


def _grid_to_str(grid: list[list[int]]) -> str:
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def _build_verifier_repair_prompt(
    *,
    task: ArcTask,
    corrupted_verifier_source: str,
    verifier_slot: str,
) -> str:
    lines: list[str] = []
    lines.append(
        "Find the common rule that maps an input grid to an output grid from the demonstrations."
    )
    lines.append(
        "Then repair the corrupted verifier code so it correctly instantiates that same rule."
    )
    lines.append("")
    for idx, ex in enumerate(task.train_pairs, 1):
        lines.append(f"Example {idx}:")
        lines.append("Input:")
        lines.append(_grid_to_str(ex.input))
        lines.append("Output:")
        lines.append(_grid_to_str(ex.output))
        lines.append("")
    lines.append(f"Corrupted verifier source (origin slot: {verifier_slot}):")
    lines.append("```python")
    lines.append(corrupted_verifier_source)
    lines.append("```")
    lines.append("")
    lines.append(
        "Return a repaired Python script that defines exactly one callable entrypoint:"
    )
    lines.append("def solve(grid):")
    lines.append("It must accept list[list[int]] and return list[list[int]].")
    lines.append("Output only the Python script, with no explanation or markdown.")
    return "\n".join(lines)


def _sample_corrupted_verifier_source(
    *,
    task: ArcTask,
    rng: random.Random,
    max_norm_edit: float,
) -> tuple[str, str, int, list[list[int]], list[list[int]]]:
    if task.arc_gen_generator is None:
        raise ValueError(f"Task {task.task_id} has no ARC-GEN dynamic generator.")

    last_error: Exception | None = None
    for _ in range(16):
        try:
            gold = copy.deepcopy(task.arc_gen_generator(1, rng=rng)[0])
        except TypeError:
            gold = copy.deepcopy(task.arc_gen_generator(1)[0])
        except Exception as e:
            last_error = e
            continue

        picked = select_verifier_for_task(task, dynamic_pairs=[gold])
        if picked is None:
            continue
        slot, _verifier = picked
        inp = copy.deepcopy(gold.input)
        out = copy.deepcopy(gold.output)

        try:
            if slot == "re_arc":
                _cfn, src, idx = load_and_corrupt_re_arc_verifier(
                    task.task_id,
                    rng=rng,
                    sample_input=inp,
                    max_attempts=120,
                    max_normalized_cell_edit_distance=max_norm_edit,
                )
                return src, slot, idx, inp, out

            if slot in ("google", "keymoon", "neurips"):
                gs: GolfSource = slot  # type: ignore[assignment]
                _cfn, src, idx = load_and_corrupt_golf_verifier(
                    task.task_id,
                    gs,
                    rng=rng,
                    sample_input=inp,
                    max_attempts=120,
                    max_normalized_cell_edit_distance=max_norm_edit,
                )
                return src, slot, idx, inp, out

            if slot == "custom":
                try:
                    _cfn, src, idx = load_and_corrupt_re_arc_verifier(
                        task.task_id,
                        rng=rng,
                        sample_input=inp,
                        max_attempts=120,
                        max_normalized_cell_edit_distance=max_norm_edit,
                    )
                    return src, "custom->re_arc", idx, inp, out
                except Exception:
                    pass
                for src_slot in ("google", "keymoon", "neurips"):
                    gs2: GolfSource = src_slot  # type: ignore[assignment]
                    try:
                        _cfn, src, idx = load_and_corrupt_golf_verifier(
                            task.task_id,
                            gs2,
                            rng=rng,
                            sample_input=inp,
                            max_attempts=120,
                            max_normalized_cell_edit_distance=max_norm_edit,
                        )
                        return src, f"custom->{src_slot}", idx, inp, out
                    except Exception:
                        continue
        except Exception as e:
            last_error = e
            continue

    if last_error is not None:
        raise RuntimeError(f"Could not sample corrupted verifier source: {last_error}")
    raise RuntimeError("Could not sample corrupted verifier source.")


def run_verifier_repair_api_evaluation_once(
    *,
    task: ArcTask,
    model: str = "gpt-5.4",
    provider: str | None = None,
    reasoning_effort: Optional[str] = "low",
    tools_enabled: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    seed: int = 0,
    max_norm_edit: float = 0.70,
    test_idx: int = 0,
    arc_gen_dynamic_n: int = 50,
    re_arc_dynamic_n: int = 50,
    out_dir: str | Path = "evaluation/verifier_repair_experiment_results",
) -> dict[str, Any]:
    rng = random.Random(seed)
    src, slot, corruption_index, sample_input, sample_output = _sample_corrupted_verifier_source(
        task=task,
        rng=rng,
        max_norm_edit=max_norm_edit,
    )
    prompt = _build_verifier_repair_prompt(
        task=task,
        corrupted_verifier_source=src,
        verifier_slot=slot,
    )
    raw, resolved_provider, effective_base = request_text_completion(
        prompt=prompt,
        model=model,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
        tools_enabled=tools_enabled,
    )
    script = _extract_python_script(raw)

    result: dict[str, Any] = {
        "task_id": task.task_id,
        "model": model,
        "provider": resolved_provider,
        "base_url": effective_base,
        "reasoning_effort": reasoning_effort,
        "tools_enabled": tools_enabled,
        "seed": seed,
        "max_norm_edit": max_norm_edit,
        "test_idx": test_idx,
        "arc_gen_dynamic_n": arc_gen_dynamic_n,
        "re_arc_dynamic_n": re_arc_dynamic_n,
        "prompt": prompt,
        "raw_response": raw,
        "script": script,
        "corrupted_verifier_slot": slot,
        "corrupted_verifier_index": corruption_index,
        "corrupted_verifier_source": src,
        "corruption_sample_pair": {"input": sample_input, "output": sample_output},
        "script_loaded": False,
        "script_load_error": None,
        "metrics": {},
        "first_failures": {},
        "evaluation_ok": False,
    }

    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{task.task_id}_verifier_repair_experiment_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    try:
        eval_result = evaluate_definition_script_on_task(
            task=task,
            script=script,
            test_idx=test_idx,
            arc_gen_dynamic_n=arc_gen_dynamic_n,
            arc_gen_dynamic_rng=rng,
            re_arc_dynamic_n=re_arc_dynamic_n,
        )
        result["script_loaded"] = eval_result["script_loaded"]
        result["script_load_error"] = eval_result["script_load_error"]
        result["metrics"] = eval_result["metrics"]
        result["first_failures"] = eval_result["first_failures"]
        if eval_result.get("error_code") == "excessive-timeouts":
            result["error"] = str(eval_result.get("error") or "excessive-timeouts")
            result["evaluation_ok"] = False
        else:
            result["evaluation_ok"] = True
    except Exception as e:
        result["error"] = str(e)
        result["script_load_error"] = str(e)

    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    result["result_path"] = str(out_path)
    return result

