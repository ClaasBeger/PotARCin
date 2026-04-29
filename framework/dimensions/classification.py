from __future__ import annotations

import ast
import datetime as dt
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

from framework.dimensions.base import (
    DimensionConfig,
    DimensionEvaluator,
    DimensionInstance,
    DimensionResult,
)
from framework.dimensions.classification_distribution import (
    sample_distribution_classification_batch,
)
from framework.grids import Grid, GridPair, assert_int_grid
from framework.llm_api import request_text_completion, resolve_provider_config
from framework.tasks.base import ArcTask, DimensionName
from framework.tasks.arc_dataset import load_task, iter_tasks


Label = bool


@dataclass
class ClassificationCandidate:
    """Candidate example to classify as same-rule or not."""

    input: Grid | None
    output: Grid


@dataclass
class ClassificationInstance(DimensionInstance[tuple[list[GridPair], ClassificationCandidate], Label]):
    """Training examples plus a candidate pair and a ground-truth label."""

    pass


@dataclass
class ClassificationMetrics:
    accuracy: float


@dataclass
class DistributionClassificationMetrics:
    """Aggregated over all instance×position labels."""

    accuracy: float
    exact_match_rate: float


@dataclass
class DistributionClassificationInstance(
    DimensionInstance[tuple[list[GridPair], list[GridPair]], list[bool]],
):
    """Demonstrations plus five candidate pairs (unlabeled in `input`); five ground-truth booleans."""

    pass


class ClassificationEvaluator(
    DimensionEvaluator[
        tuple[list[GridPair], ClassificationCandidate],
        Label,
        Label,
        ClassificationMetrics,
    ]
):
    """Skeleton evaluator for the classification dimension."""

    def __init__(self, config: DimensionConfig | None = None) -> None:
        self.config = config or DimensionConfig(name=DimensionName.CLASSIFICATION)

    def generate_instances(
        self,
        tasks: Iterable[ArcTask],
    ) -> List[ClassificationInstance]:
        # Placeholder: no real negative sampling yet, just a trivial instance per task.
        instances: List[ClassificationInstance] = []
        for task in tasks:
            if not task.train_pairs:
                continue
            first_pair = task.train_pairs[0]
            candidate = ClassificationCandidate(
                input=first_pair.input,
                output=first_pair.output,
            )
            instances.append(
                ClassificationInstance(
                    task_id=task.task_id,
                    input=([first_pair], candidate),
                    target=True,
                )
            )
        return instances

    def score_predictions(
        self,
        instances: Sequence[ClassificationInstance],
        predictions: Sequence[Label],
    ) -> DimensionResult[ClassificationMetrics]:
        if len(instances) != len(predictions):
            raise ValueError("instances and predictions must have the same length")
        if not instances:
            metrics = ClassificationMetrics(accuracy=0.0)
            return DimensionResult(
                dimension=DimensionName.CLASSIFICATION,
                metrics=metrics,
            )

        correct = 0
        for inst, pred in zip(instances, predictions):
            if inst.target == pred:
                correct += 1
        accuracy = correct / len(instances)
        metrics = ClassificationMetrics(accuracy=accuracy)
        return DimensionResult(
            dimension=DimensionName.CLASSIFICATION,
            metrics=metrics,
        )


class DistributionClassificationEvaluator(
    DimensionEvaluator[
        tuple[list[GridPair], list[GridPair]],
        list[bool],
        Sequence[Union[bool, int]],
        DistributionClassificationMetrics,
    ],
):
    """Five-way distribution match: same underlying rule / generator vs not."""

    def __init__(self, config: DimensionConfig | None = None) -> None:
        self.config = config or DimensionConfig(name=DimensionName.CLASSIFICATION)

    def generate_instances(
        self,
        tasks: Iterable[ArcTask],
    ) -> List[DistributionClassificationInstance]:
        seed = self.config.random_seed
        rng = random.Random(seed if seed is not None else 0)
        instances: List[DistributionClassificationInstance] = []
        n_per = max(1, self.config.num_instances_per_task)
        for task in tasks:
            if task.arc_gen_generator is None or not task.train_pairs:
                continue
            for _ in range(n_per):
                try:
                    batch = sample_distribution_classification_batch(task.task_id, rng)
                except (ValueError, OSError, KeyError):
                    continue
                demos = batch.demonstrations
                candidates = [it.pair for it in batch.items]
                target = [it.same_distribution for it in batch.items]
                if len(candidates) != 5 or len(target) != 5:
                    continue
                instances.append(
                    DistributionClassificationInstance(
                        task_id=task.task_id,
                        input=(demos, candidates),
                        target=target,
                    )
                )
        return instances

    def score_predictions(
        self,
        instances: Sequence[DistributionClassificationInstance],
        predictions: Sequence[Sequence[Union[bool, int]]],
    ) -> DimensionResult[DistributionClassificationMetrics]:
        if len(instances) != len(predictions):
            raise ValueError("instances and predictions must have the same length")
        if not instances:
            m = DistributionClassificationMetrics(accuracy=0.0, exact_match_rate=0.0)
            return DimensionResult(dimension=DimensionName.CLASSIFICATION, metrics=m)

        total = 0
        correct = 0
        exact = 0
        for inst, pred in zip(instances, predictions):
            if len(inst.target) != 5:
                raise ValueError("Each instance must have 5 target labels.")
            norm = _normalize_five_binary_labels(list(pred))
            if len(norm) != 5:
                raise ValueError("Each prediction must have 5 labels.")
            all_ok = True
            for a, b in zip(inst.target, norm):
                total += 1
                if bool(a) == bool(b):
                    correct += 1
                else:
                    all_ok = False
            if all_ok:
                exact += 1

        n = len(instances)
        metrics = DistributionClassificationMetrics(
            accuracy=correct / total if total else 0.0,
            exact_match_rate=exact / n,
        )
        return DimensionResult(dimension=DimensionName.CLASSIFICATION, metrics=metrics)


def _normalize_five_binary_labels(pred: Sequence[Union[bool, int]]) -> list[bool]:
    if len(pred) != 5:
        raise ValueError("Expected length 5.")
    out: list[bool] = []
    for x in pred:
        if isinstance(x, bool):
            out.append(x)
        elif x in (0, 1):
            out.append(bool(x))
        else:
            raise ValueError(f"Invalid label {x!r}; use 0/1 or booleans.")
    return out


def _grid_to_str(grid: Grid) -> str:
    assert_int_grid(grid, name="grid")
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def build_classification_prompt(
    *,
    demonstrations: Sequence[GridPair],
    candidates: Sequence[Tuple[GridPair, bool]],
) -> str:
    """Build a classification prompt.

    The model sees demonstrations and several candidate input/output pairs.
    It must judge for each candidate whether it follows the same underlying rule.
    """
    lines: list[str] = []
    lines.append(
        "Find the common rule that maps an input grid to an output grid, "
        "given the examples below."
    )
    lines.append(
        "You will then be given multiple additional input/output pairs. "
        "For each pair, decide whether it is consistent with the same rule."
    )
    lines.append("")

    for idx, ex in enumerate(demonstrations, 1):
        lines.append(f"Example {idx}:")
        lines.append("Input:")
        lines.append(_grid_to_str(ex.input))
        lines.append("Output:")
        lines.append(_grid_to_str(ex.output))
        lines.append("")

    lines.append("Candidates:")
    for i, (pair, _label) in enumerate(candidates):
        lines.append(f"\nCandidate {i}:")
        lines.append("Input:")
        lines.append(_grid_to_str(pair.input))
        lines.append("Output:")
        lines.append(_grid_to_str(pair.output))

    lines.append(
        "\nReturn ONLY valid JSON with this schema:\n"
        '{ "labels": [true_or_false_for_candidate_0, true_or_false_for_candidate_1, ...] }\n'
        "where `true` means the candidate is consistent with the demonstrations' rule."
    )
    return "\n".join(lines)


def build_distribution_classification_prompt(
    *,
    demonstrations: Sequence[GridPair],
    candidates: Sequence[GridPair],
    prior_work_context: str | None = None,
) -> str:
    """Five test pairs: model outputs only a length-5 Python list of 0/1."""
    lines: list[str] = []
    lines.append(
        "Find the common rule that maps an input grid to an output grid given the examples below. "
        "You will see five additional input/output pairs (test items 0–4). "
        "For each test item, decide whether that pair could have been produced by the "
        "same underlying rule and data distribution as the training examples "
        "(1 = same distribution / consistent, 0 = not)."
    )
    lines.append("")

    for idx, ex in enumerate(demonstrations, 1):
        lines.append(f"Training example {idx}:")
        lines.append("Input:")
        lines.append(_grid_to_str(ex.input))
        lines.append("Output:")
        lines.append(_grid_to_str(ex.output))
        lines.append("")

    if prior_work_context:
        lines.append("Here is some work on this task you have performed on adjacent requests.")
        lines.append(prior_work_context)
        lines.append("")

    lines.append("Test pairs (evaluate each in order, index 0 … 4):")
    for i, pair in enumerate(candidates):
        lines.append(f"\nTest {i}:")
        lines.append("Input:")
        lines.append(_grid_to_str(pair.input))
        lines.append("Output:")
        lines.append(_grid_to_str(pair.output))

    lines.append(
        "\nYour final answer must be only a single Python list of exactly five integers, each 0 or 1, "
        "in order for test 0 through test 4, for example [1, 0, 1, 1, 0]. "
        "Output nothing else: no other text, no markdown code fences, and no explanation."
    )
    return "\n".join(lines)


def _parse_space_separated_grid(block: str) -> Grid:
    rows: list[list[int]] = []
    for line in block.split("\n"):
        line_st = line.strip()
        if not line_st:
            continue
        rows.append([int(x) for x in line_st.split()])
    if not rows:
        raise ValueError("empty grid in prompt fragment")
    w = len(rows[0])
    if any(len(r) != w for r in rows):
        raise ValueError("jagged grid in prompt fragment")
    return rows


def parse_distribution_classification_test_pairs_from_prompt(prompt: str) -> list[GridPair]:
    """Rebuild the five test pairs from a saved distribution-classification prompt.

    Matches :func:`build_distribution_classification_prompt` (``Test 0``…``Test 4`` blocks
    between the ``Test pairs`` header and ``Your final answer must``), including prompts
    that use a Unicode ellipsis (``…``) in the header line.
    """
    m = re.search(r"Test pairs \(evaluate each in order, index 0 .{1,5}4\):\s*", prompt)
    if not m:
        raise ValueError("distribution classification prompt: missing 'Test pairs' header")
    tail = "\n\nYour final answer must be"
    k = prompt.find(tail, m.end())
    if k < 0:
        raise ValueError("distribution classification prompt: missing answer tail marker")
    block = prompt[m.end() : k].strip()
    chunks = re.split(r"(?=^Test \d+:\s*$)", block, flags=re.MULTILINE)
    indexed: list[tuple[int, GridPair]] = []
    for ch in chunks:
        ch_st = ch.strip()
        if not ch_st:
            continue
        mm = re.match(
            r"^Test (\d+):\s*\nInput:\s*\n(.*?)\nOutput:\s*\n(.*)$",
            ch_st,
            re.DOTALL,
        )
        if not mm:
            raise ValueError(f"distribution classification prompt: bad test block: {ch_st[:80]!r}")
        idx = int(mm.group(1))
        inp = _parse_space_separated_grid(mm.group(2).strip())
        out = _parse_space_separated_grid(mm.group(3).strip())
        indexed.append((idx, GridPair(inp, out)))
    indexed.sort(key=lambda t: t[0])
    if len(indexed) != 5 or [t[0] for t in indexed] != list(range(5)):
        raise ValueError(f"distribution classification prompt: expected Test 0–4, got {indexed!r}")
    return [pair for _i, pair in indexed]


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


def _extract_json_object(text: str) -> dict[str, Any]:
    s = text.strip()
    if "```" in s:
        s = s.replace("```json", "```").replace("```", "").strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not locate JSON object in model output.")
    return json.loads(s[start : end + 1])


def parse_distribution_classification_prediction(text: str, *, n: int = 5) -> list[bool]:
    """Parse five binary labels from a Python ``[0,1,...]`` list (only expected model output)."""
    s = text.strip()
    if "```" in s:
        s = re.sub(r"```(?:python)?\s*", "", s, count=1)
        s = s.replace("```", "").strip()

    for line in s.splitlines():
        t = line.strip()
        if t.startswith("[") and t.endswith("]"):
            try:
                v = ast.literal_eval(t)
                if isinstance(v, list) and len(v) == n:
                    return _normalize_five_binary_labels(v)
            except Exception:
                pass

    try:
        v = ast.literal_eval(s)
        if isinstance(v, list) and len(v) == n:
            return _normalize_five_binary_labels(v)
    except Exception:
        pass

    try:
        v = json.loads(s)
        if isinstance(v, list) and len(v) == n:
            return _normalize_five_binary_labels(v)
    except Exception:
        pass

    for m in re.finditer(r"\[[^\[\]]+\]", s):
        try:
            v = ast.literal_eval(m.group(0))
            if isinstance(v, list) and len(v) == n:
                return _normalize_five_binary_labels(v)
        except Exception:
            pass

    raise ValueError("Could not parse five binary labels from model output.")


def sample_classification_candidates_from_arc_gen_dynamic(
    *,
    task_id: str,
    n_pos: int = 3,
    n_neg: int = 3,
    seed: int = 0,
    corrupt_fn: Optional[Callable[[GridPair, str], GridPair]] = None,
) -> List[Tuple[GridPair, bool, dict[str, Any]]]:
    """Sample candidates for classification from ARC-GEN dynamic generators.

    - Positives: ARC-GEN dynamic pairs from the same task.
    - Negatives: ARC-GEN dynamic pairs from other tasks.
    - corrupt_fn: optional hook applied to some samples (kept as no-op for now).
    """
    rng = random.Random(seed)
    task = load_task(task_id)
    if task.arc_gen_generator is None:
        raise ValueError(f"Task {task_id} has no ARC-GEN generator.")

    candidates: List[Tuple[GridPair, bool, dict[str, Any]]] = []

    pos_pairs = task.arc_gen_generator(n_pos, rng=rng)
    for p in pos_pairs:
        meta = {"source_task_id": task_id, "label": True, "corruption": None}
        candidates.append((p, True, meta))

    other_ids = [t.task_id for t in iter_tasks(split="train") if t.task_id != task_id]
    rng.shuffle(other_ids)
    neg_added = 0
    for oid in other_ids:
        if neg_added >= n_neg:
            break
        ot = load_task(oid)
        if ot.arc_gen_generator is None:
            continue
        try:
            p = ot.arc_gen_generator(1, rng=rng)[0]
        except Exception:
            continue
        meta = {"source_task_id": oid, "label": False, "corruption": None}
        candidates.append((p, False, meta))
        neg_added += 1

    # Placeholder corruption stage (disabled until specified).
    # We'll keep the hook so you can plug in your corruption rule later.
    if corrupt_fn is not None:
        # Example policy (currently none applied): leave as-is.
        pass

    rng.shuffle(candidates)
    return candidates


def run_classification_api_evaluation_once(
    *,
    task_id: str,
    model: str = "gpt-5.4",
    provider: str | None = None,
    reasoning_effort: Optional[str] = "low",
    tools_enabled: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    n_pos: int = 3,
    n_neg: int = 3,
    seed: int = 0,
    out_dir: str | Path = "evaluation/classification_eval_results",
) -> dict[str, Any]:
    """One-shot classification eval: sample candidates -> prompt -> model -> score.

    After the model returns, ``raw_response`` is always persisted; parse/score
    failures set ``evaluation_ok`` false and ``error`` but still write JSON.
    """
    task = load_task(task_id)
    if not task.train_pairs:
        raise ValueError(f"Task {task_id} has no training pairs.")

    demos = task.train_pairs
    sampled = sample_classification_candidates_from_arc_gen_dynamic(
        task_id=task_id,
        n_pos=n_pos,
        n_neg=n_neg,
        seed=seed,
        corrupt_fn=None,
    )
    candidates_for_prompt: List[Tuple[GridPair, bool]] = [(p, lbl) for (p, lbl, _m) in sampled]
    prompt = build_classification_prompt(demonstrations=demos, candidates=candidates_for_prompt)

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

    gt = [bool(lbl) for (_p, lbl, _m) in sampled]
    n = len(gt)

    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{task_id}_classification_eval_{ts}.json"

    result: dict[str, Any] = {
        "task_id": task_id,
        "model": model,
        "provider": resolved_provider,
        "base_url": effective_base,
        "reasoning_effort": reasoning_effort,
        "tools_enabled": tools_enabled,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "seed": seed,
        "prompt": prompt,
        "raw_response": raw,
        "ground_truth": gt,
        "candidates": [
            {
                "input": p.input,
                "output": p.output,
                "label": lbl,
                "meta": meta,
            }
            for (p, lbl, meta) in sampled
        ],
        "evaluation_ok": False,
        "parsed": None,
        "predictions": None,
        "metrics": {"accuracy": 0.0, "correct": 0, "total": n},
    }
    try:
        parsed = _extract_json_object(raw)
        labels = parsed.get("labels")
        if not isinstance(labels, list):
            raise ValueError("Model response JSON must contain a list field `labels`.")
        if len(labels) != len(sampled):
            raise ValueError(f"Expected {len(sampled)} labels, got {len(labels)}.")
        pred = [bool(x) for x in labels]
        correct = sum(int(a == b) for a, b in zip(gt, pred))
        accuracy = correct / n if n else 0.0
        result["parsed"] = parsed
        result["predictions"] = pred
        result["metrics"] = {"accuracy": accuracy, "correct": correct, "total": n}
        result["evaluation_ok"] = True
    except Exception as e:
        result["error"] = str(e)

    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    result["result_path"] = str(out_path)
    return result


def run_distribution_classification_api_evaluation_once(
    *,
    task_id: str,
    model: str = "gpt-5.4",
    provider: str | None = None,
    reasoning_effort: Optional[str] = "low",
    tools_enabled: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    seed: int = 0,
    prior_work_context: str | None = None,
    out_dir: str | Path = "evaluation/classification_eval_results",
) -> dict[str, Any]:
    """Sample a 5-way distribution batch, prompt the model, parse ``[0/1]×5``, score.

    After the model returns, ``raw_response`` is always persisted; parse failures
    set ``evaluation_ok`` false and ``error`` but still write JSON.
    """
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{task_id}_distribution_classification_eval_{ts}.json"

    # Keep a durable artifact from the start so failures can still be audited.
    result: dict[str, Any] = {
        "task_id": task_id,
        "model": model,
        "provider": provider,
        "base_url": base_url,
        "reasoning_effort": reasoning_effort,
        "tools_enabled": tools_enabled,
        "seed": seed,
        "prompt": None,
        "raw_response": None,
        "test_pairs": [],
        "ground_truth": [],
        "item_meta": [],
        "evaluation_ok": False,
        "predictions": None,
        "metrics": {
            "accuracy": 0.0,
            "correct": 0,
            "total": 0,
            "exact_match": 0,
        },
        "stage": "init",
    }

    def _persist() -> None:
        out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    try:
        task = load_task(task_id)
        if not task.train_pairs:
            raise ValueError(f"Task {task_id} has no training pairs.")
        if task.arc_gen_generator is None:
            raise ValueError(f"Task {task_id} has no ARC-GEN dynamic generator.")

        result["stage"] = "sampling"
        rng = random.Random(seed)
        batch = sample_distribution_classification_batch(task_id, rng)
        demos = batch.demonstrations
        candidates = [it.pair for it in batch.items]
        meta = [{"category": it.category, "meta": it.meta} for it in batch.items]
        gt = [it.same_distribution for it in batch.items]

        prompt = build_distribution_classification_prompt(
            demonstrations=demos,
            candidates=candidates,
            prior_work_context=prior_work_context,
        )
        result["prompt"] = prompt
        result["prior_work_context"] = prior_work_context
        result["test_pairs"] = [{"input": p.input, "output": p.output} for p in candidates]
        result["ground_truth"] = gt
        result["item_meta"] = meta
        result["metrics"]["total"] = len(gt)

        resolved_provider, _effective_key, effective_base = resolve_provider_config(
            model=model,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
        )
        result["stage"] = "api_call"
        raw = _openai_responses_completion(
            prompt=prompt,
            model=model,
            provider=resolved_provider,
            api_key=api_key,
            base_url=effective_base,
            reasoning_effort=reasoning_effort,
            tools_enabled=tools_enabled,
        )
        result["provider"] = resolved_provider
        result["base_url"] = effective_base
        # Step 1: get model output and persist immediately before parse/score.
        result["raw_response"] = raw
        result["stage"] = "raw_recorded"
        _persist()

        # Step 2: parse and score independently from the API call stage.
        result["stage"] = "parsing"
        pred = parse_distribution_classification_prediction(raw)

        result["stage"] = "scoring"
        n_gt = len(gt)
        correct = sum(int(bool(a) == bool(b)) for a, b in zip(gt, pred))
        accuracy = correct / n_gt if n_gt else 0.0
        result["predictions"] = pred
        result["metrics"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": n_gt,
            "exact_match": int(correct == n_gt),
        }
        result["evaluation_ok"] = True
        result["stage"] = "done"
    except Exception as e:
        result["error"] = str(e)
        if result.get("stage") != "raw_recorded":
            result["stage"] = f"{result.get('stage', 'unknown')}_failed"

    _persist()
    result["result_path"] = str(out_path)
    return result

