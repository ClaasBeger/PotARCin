from __future__ import annotations

import copy
import datetime as dt
import json
import multiprocessing as mp
import os
import signal
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence
from urllib import error, request

from framework.dimensions.base import (
    DimensionConfig,
    DimensionEvaluator,
    DimensionInstance,
    DimensionResult,
)
from framework.grids import Grid, GridPair, is_equal_grid
from framework.llm_api import request_text_completion, resolve_provider_config
from framework.tasks.base import ArcTask, DimensionName, Verifier


# In the definition dimension, the prediction is typically a Python program
# (as a string) or a callable. For now, we keep this abstract as `str`.
ProgramText = str
_SOLVE_CALL_TIMEOUT_S = 600


def _call_with_timeout(fn: Callable[[Grid], Grid], grid: Grid, timeout_s: int = _SOLVE_CALL_TIMEOUT_S) -> Grid:
    """Run solve(grid) with a hard timeout to avoid hangs."""
    if timeout_s <= 0:
        return fn(grid)

    if os.name != "posix":
        # Fallback for non-POSIX environments.
        return fn(grid)

    class _SolveTimeout(Exception):
        pass

    def _handler(_signum: int, _frame: Any) -> None:
        raise _SolveTimeout(f"solve() timed out after {timeout_s}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_s)
    try:
        return fn(grid)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _run_script_worker(script: str, grid: Grid, q: mp.Queue) -> None:
    """Child process worker: load script, run solve, and report result/error."""
    try:
        fn = _load_script_callable(script)
        got = fn(grid)
        q.put({"ok": True, "got": got})
    except Exception as e:
        q.put({"ok": False, "error": str(e)})


def _call_script_with_hard_timeout(script: str, grid: Grid, timeout_s: int = _SOLVE_CALL_TIMEOUT_S) -> Grid:
    """Run generated script in a separate process with a hard wall-clock timeout."""
    if timeout_s <= 0:
        fn = _load_script_callable(script)
        return fn(grid)
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_run_script_worker, args=(script, grid, q))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2)
        raise TimeoutError(f"solve() timed out after {timeout_s}s")
    if q.empty():
        raise RuntimeError("solve() subprocess exited without returning a result")
    msg = q.get()
    if not isinstance(msg, dict):
        raise RuntimeError("solve() subprocess returned malformed payload")
    if msg.get("ok") is True:
        return msg.get("got")
    raise RuntimeError(str(msg.get("error") or "unknown solve() subprocess error"))

def _grid_to_str(grid: Grid) -> str:
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def build_written_prompt(
    puzzle_json: dict,
    test_idx: int,
    model: str = "o3",
) -> str:
    """Build a written prompt for the definition dimension.

    The model sees demonstrations and one test input, then must produce a
    Python program that captures the underlying rule for general inputs.
    """
    lines: list[str] = []

    lines.append(
        "Find the common rule that maps an input grid to an output grid, "
        "given the examples below. Generate a python script that takes a "
        "grid as input and transforms it according to the this rule. "
        "Anticipate that this program will be applied to several grids, "
        "not only the shown test grid."
    )
    lines.append(
        "\nYour script must define exactly one callable entrypoint:\n"
        "def solve(grid):\n"
        "where `grid` is passed as a list of lists of integers (list[list[int]]) "
        "and the function returns the transformed grid in the same format."
    )

    for idx, ex in enumerate(puzzle_json["train"], 1):
        lines.append(f"\nExample {idx}:\n")
        lines.append("Input:")
        lines.append(_grid_to_str(ex["input"]))
        lines.append("Output:")
        lines.append(_grid_to_str(ex["output"]))

    lines.append("\nBelow is a test input grid.\n")
    lines.append("Input:")
    test_grid = puzzle_json["test"][test_idx]["input"]
    lines.append(_grid_to_str(test_grid))

    lines.append(
        "\nYour final answer should just be the python script itself, no other text or markdown."
    )

    return "\n".join(lines)


def _openai_compatible_chat_completion(
    *,
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float = 0.0,
    timeout_s: int = 120,
) -> str:
    """Send a prompt to an OpenAI-compatible `/chat/completions` endpoint."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
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
        raise ValueError(
            f"Responses API error {e.code}: {body_txt}"
        ) from e
    parsed = json.loads(raw)
    return parsed["choices"][0]["message"]["content"]


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
    """Send a prompt to provider-native completion endpoint."""
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


def _extract_python_script(text: str) -> str:
    """Extract Python code, preferring fenced ```python blocks if present."""
    s = text.strip()
    if "```" not in s:
        return s
    chunks = s.split("```")
    # Fenced blocks are at odd indices.
    for i in range(1, len(chunks), 2):
        block = chunks[i].strip()
        if block.startswith("python"):
            return block[len("python") :].lstrip()
    for i in range(1, len(chunks), 2):
        block = chunks[i].strip()
        if block:
            return block
    return s


def _load_script_callable(script: str) -> Callable[[Grid], Grid]:
    """Load script and return a callable transformer."""
    global_ns: dict[str, Any] = {
        "__builtins__": __builtins__,
        "copy": copy,
    }
    local_ns: dict[str, Any] = {}
    exec(script, global_ns, local_ns)

    candidates = [
        "solve",
        "transform",
        "predict",
        "apply_rule",
        "f",
    ]
    for name in candidates:
        fn = local_ns.get(name) or global_ns.get(name)
        if callable(fn):
            return fn

    for scope in (local_ns, global_ns):
        for _name, fn in scope.items():
            if callable(fn):
                return fn
    raise ValueError("No callable function found in generated script.")


def evaluate_definition_script_on_task(
    *,
    task: ArcTask,
    script: str,
    test_idx: int = 0,
) -> dict[str, Any]:
    """Evaluate a solved definition script across all configured splits."""
    if not task.train_pairs:
        raise ValueError(f"Task {task.task_id} has no training pairs.")
    if test_idx < 0 or test_idx >= len(task.test_inputs) or test_idx >= len(task.test_outputs):
        raise ValueError(f"Invalid test_idx={test_idx} for task {task.task_id}.")

    eval_sets: dict[str, list[dict[str, Grid]]] = {
        "train": [{"input": p.input, "output": p.output} for p in task.train_pairs],
        "test": [{"input": task.test_inputs[test_idx], "output": task.test_outputs[test_idx]}],
        "stable": [
            {"input": p.input, "output": p.output}
            for p in (task.arc_gen_synthetic_pairs or [])
        ],
        "dynamic50": [],
        "re_arc_stable": [
            {"input": p.input, "output": p.output}
            for p in (task.re_arc_synthetic_pairs or [])
        ],
        "re_arc_dynamic50": [],
    }
    if task.arc_gen_generator is not None:
        dyn = task.arc_gen_generator(50)
        eval_sets["dynamic50"] = [{"input": p.input, "output": p.output} for p in dyn]
    if task.re_arc_generator is not None:
        dyn = task.re_arc_generator(50)
        eval_sets["re_arc_dynamic50"] = [{"input": p.input, "output": p.output} for p in dyn]

    out: dict[str, Any] = {
        "script_loaded": False,
        "script_load_error": None,
        "metrics": {},
        "first_failures": {},
    }
    try:
        fn = _load_script_callable(script)
        out["script_loaded"] = True
    except Exception as e:
        out["script_load_error"] = f"{e}\n{traceback.format_exc()}"
        fn = None

    for split_name, pairs in eval_sets.items():
        total = len(pairs)
        ok = 0
        failure: Optional[dict[str, Any]] = None
        if fn is not None:
            for i, ex in enumerate(pairs):
                try:
                    # Use a hard process-level timeout to avoid hangs that ignore SIGALRM.
                    got = _call_script_with_hard_timeout(script, copy.deepcopy(ex["input"]))
                    if is_equal_grid(got, ex["output"]):
                        ok += 1
                    elif failure is None:
                        failure = {
                            "index": i,
                            "input": ex["input"],
                            "expected": ex["output"],
                            "got": got,
                        }
                except Exception as e:
                    if failure is None:
                        failure = {"index": i, "error": str(e)}
        out["metrics"][split_name] = {
            "ok": ok,
            "total": total,
            "success_rate": (ok / total) if total > 0 else 0.0,
        }
        out["first_failures"][split_name] = failure
    return out


def run_definition_api_evaluation(
    *,
    task: ArcTask,
    test_idx: int = 0,
    model: str = "o3",
    provider: str | None = None,
    reasoning_effort: Optional[str] = None,
    tools_enabled: bool = False,
    api_endpoint: str = "responses",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    out_dir: str | Path = "evaluation/definition_eval_results",
) -> dict[str, Any]:
    """End-to-end Definition evaluation: prompt -> model -> script -> scoring.

    After the model returns, ``raw_response`` is always written to the eval JSON,
    even if script evaluation crashes (``evaluation_ok`` false, ``error`` set).

    Evaluation checks the generated script on:
    - all train pairs
    - selected test example (by test_idx)
    - all ARC-GEN stable pairs
    - 50 ARC-GEN dynamic pairs
    """
    if not task.train_pairs:
        raise ValueError(f"Task {task.task_id} has no training pairs.")
    if test_idx < 0 or test_idx >= len(task.test_inputs) or test_idx >= len(task.test_outputs):
        raise ValueError(f"Invalid test_idx={test_idx} for task {task.task_id}.")

    puzzle_json = {
        "train": [{"input": p.input, "output": p.output} for p in task.train_pairs],
        "test": [{"input": task.test_inputs[test_idx]}],
    }
    prompt = build_written_prompt(
        puzzle_json=puzzle_json,
        test_idx=0,
        model=model,
    )

    resolved_provider: str | None = None
    effective_base: str | None = None
    if api_endpoint == "responses":
        raw_response, resolved_provider, effective_base = request_text_completion(
            prompt=prompt,
            model=model,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
            tools_enabled=tools_enabled,
        )
    elif api_endpoint == "chat_completions":
        resolved_provider, effective_key, effective_base = resolve_provider_config(
            model=model,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
        )
        if resolved_provider != "openai":
            raise ValueError("api_endpoint='chat_completions' is only supported for provider='openai'.")
        raw_response = _openai_compatible_chat_completion(
            prompt=prompt,
            model=model,
            api_key=effective_key,
            base_url=effective_base,
        )
    else:
        raise ValueError(
            f"Unsupported api_endpoint={api_endpoint}. "
            "Use 'responses' or 'chat_completions'."
        )
    script = _extract_python_script(raw_response)

    result: dict[str, Any] = {
        "task_id": task.task_id,
        "test_idx": test_idx,
        "model": model,
        "provider": resolved_provider,
        "base_url": effective_base,
        "api_endpoint": api_endpoint,
        "reasoning_effort": reasoning_effort,
        "tools_enabled": tools_enabled,
        "prompt": prompt,
        "raw_response": raw_response,
        "script": script,
        "script_loaded": False,
        "script_load_error": None,
        "metrics": {},
        "first_failures": {},
        "evaluation_ok": False,
    }
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{task.task_id}_definition_eval_{ts}.json"
    # Persist immediately after model response/script extraction so interrupted
    # script evaluation still leaves a recoverable artifact on disk.
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    try:
        eval_result = evaluate_definition_script_on_task(
            task=task,
            script=script,
            test_idx=test_idx,
        )
        result["script_loaded"] = eval_result["script_loaded"]
        result["script_load_error"] = eval_result["script_load_error"]
        result["metrics"] = eval_result["metrics"]
        result["first_failures"] = eval_result["first_failures"]
        result["evaluation_ok"] = True
    except Exception as e:
        result["error"] = str(e)
        result["script_load_error"] = str(e)

    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    result["result_path"] = str(out_path)
    return result


@dataclass
class DefinitionInstance(DimensionInstance[list[GridPair], list[Grid]]):
    """Training pairs plus held-out test inputs."""

    pass


@dataclass
class DefinitionMetrics:
    """Placeholder for definition-dimension metrics."""

    success_rate: float


class DefinitionEvaluator(
    DimensionEvaluator[list[GridPair], list[Grid], ProgramText, DefinitionMetrics]
):
    """Skeleton evaluator for the definition dimension."""

    def __init__(self, config: DimensionConfig | None = None) -> None:
        self.config = config or DimensionConfig(name=DimensionName.DEFINITION)

    def generate_instances(self, tasks: Iterable[ArcTask]) -> List[DefinitionInstance]:
        instances: List[DefinitionInstance] = []
        for task in tasks:
            instances.append(
                DefinitionInstance(
                    task_id=task.task_id,
                    input=task.train_pairs,
                    target=task.test_inputs,
                )
            )
        return instances

    def score_predictions(
        self,
        instances: Sequence[DefinitionInstance],
        predictions: Sequence[ProgramText],
    ) -> DimensionResult[DefinitionMetrics]:
        if len(instances) != len(predictions):
            raise ValueError("instances and predictions must have the same length")

        # Placeholder scoring: no real execution, just a dummy metric.
        metrics = DefinitionMetrics(success_rate=0.0)
        return DimensionResult(dimension=DimensionName.DEFINITION, metrics=metrics)

