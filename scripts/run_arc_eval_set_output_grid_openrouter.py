#!/usr/bin/env python3
"""Compute ARC-AGI-1 *evaluation-set* output-grid exact-match via OpenRouter.

Uses the vendored evaluation JSONs under:
  external/VARC/VARC/raw_data/ARC-AGI/data/evaluation/

Default model: meta-llama/llama-4-maverick (OpenRouter).
Requires OPENROUTER_API_KEY.

Example:
  python scripts/run_arc_eval_set_output_grid_openrouter.py
  python scripts/run_arc_eval_set_output_grid_openrouter.py --limit 10 --skip-completed
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from framework.dimensions.output_grid_eval import (  # noqa: E402
    build_output_grid_prompt,
    parse_output_grid_response,
)
from framework.grids import GridPair, is_equal_grid  # noqa: E402
from framework.llm_api import request_text_completion  # noqa: E402

DEFAULT_EVAL_DIR = (
    ROOT_DIR / "external" / "VARC" / "VARC" / "raw_data" / "ARC-AGI" / "data" / "evaluation"
)
DEFAULT_MODEL = "meta-llama/llama-4-maverick"
DEFAULT_OUT_DIR = (
    ROOT_DIR
    / "evaluation"
    / "output_grid_eval_results"
    / "Llama4Maverick"
    / "ARC_AGI1_evaluation"
    / "output_grid_eval_set"
)


def _load_eval_task(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "train" not in data or "test" not in data:
        raise ValueError(f"{path} missing train/test")
    train_pairs = [
        GridPair(input=ex["input"], output=ex["output"]) for ex in data["train"]
    ]
    test = data["test"]
    if not test:
        raise ValueError(f"{path} has empty test list")
    return {
        "task_id": path.stem,
        "train_pairs": train_pairs,
        "test_input": test[0]["input"],
        "test_output": test[0]["output"],
    }


def _list_eval_tasks(eval_dir: Path) -> list[Path]:
    paths = sorted(eval_dir.glob("*.json"))
    if not paths:
        raise SystemExit(f"No evaluation task JSONs in {eval_dir}")
    return paths


def _has_logged_result(out_dir: Path, task_id: str) -> bool:
    return any(out_dir.glob(f"{task_id}_output_grid_eval_*.json"))


def _run_one(
    *,
    task: dict[str, Any],
    model: str,
    provider: str,
    reasoning_effort: str | None,
    out_dir: Path,
    timeout_s: int,
) -> dict[str, Any]:
    tid = task["task_id"]
    prompt = build_output_grid_prompt(
        demonstrations=task["train_pairs"],
        test_input=task["test_input"],
    )
    raw, resolved_provider, base_url = request_text_completion(
        prompt=prompt,
        model=model,
        provider=provider,
        reasoning_effort=reasoning_effort,
        tools_enabled=False,
        timeout_s=timeout_s,
    )
    expected = task["test_output"]
    result: dict[str, Any] = {
        "task_id": tid,
        "split": "arc_agi1_evaluation",
        "test_idx": 0,
        "model": model,
        "provider": resolved_provider,
        "base_url": base_url,
        "reasoning_effort": reasoning_effort,
        "prompt": prompt,
        "raw_response": raw,
        "ground_truth": expected,
        "evaluation_ok": False,
        "exact_match": None,
        "prediction": None,
        "metrics": {"exact_match": 0.0},
    }
    try:
        pred = parse_output_grid_response(raw)
        result["prediction"] = pred
        match = bool(is_equal_grid(pred, expected))
        result["exact_match"] = match
        result["metrics"] = {"exact_match": 1.0 if match else 0.0}
        result["evaluation_ok"] = True
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tid}_output_grid_eval_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    result["result_path"] = str(out_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "ARC-AGI-1 evaluation-set output-grid accuracy via OpenRouter "
            f"(default model: {DEFAULT_MODEL})."
        )
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=DEFAULT_EVAL_DIR,
        help="Directory of ARC-AGI evaluation task JSONs",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="Optional reasoning effort (default: omit / provider default)",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Only first N tasks")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N tasks")
    parser.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip tasks that already have a result JSON in --out-dir (default: on)",
    )
    parser.add_argument("--delay-s", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Extra attempts on exception (default: 1 => up to 2 tries)",
    )
    args = parser.parse_args()

    eval_dir = args.eval_dir if args.eval_dir.is_absolute() else ROOT_DIR / args.eval_dir
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT_DIR / args.out_dir
    paths = _list_eval_tasks(eval_dir)
    if args.skip:
        paths = paths[args.skip :]
    if args.limit is not None:
        paths = paths[: max(0, args.limit)]

    if args.skip_completed:
        before = len(paths)
        paths = [p for p in paths if not _has_logged_result(out_dir, p.stem)]
        print(f"Tasks to run: {len(paths)} (skipped_already_logged={before - len(paths)})")
    else:
        print(f"Tasks to run: {len(paths)}")

    if not paths:
        print("Nothing to do.")
        return

    n_exact = 0
    n_ok = 0
    n_fail = 0
    rows: list[dict[str, Any]] = []

    for i, path in enumerate(paths, 1):
        task = _load_eval_task(path)
        tid = task["task_id"]
        print(f"[{i}/{len(paths)}] {tid} …", flush=True)
        last_err: Exception | None = None
        result: dict[str, Any] | None = None
        attempts = max(1, args.retries + 1)
        for attempt in range(1, attempts + 1):
            try:
                result = _run_one(
                    task=task,
                    model=args.model,
                    provider=args.provider,
                    reasoning_effort=args.reasoning_effort,
                    out_dir=out_dir,
                    timeout_s=args.timeout_s,
                )
                last_err = None
                break
            except Exception as e:
                last_err = e
                print(f"  attempt {attempt}/{attempts} failed: {type(e).__name__}: {e}")
                if attempt < attempts:
                    time.sleep(min(8.0, attempt * 1.5))
        if result is None:
            n_fail += 1
            rows.append(
                {
                    "task_id": tid,
                    "evaluation_ok": False,
                    "exact_match": None,
                    "error": f"{type(last_err).__name__}: {last_err}",
                }
            )
            print(f"  -> FAILED {last_err}")
        else:
            rows.append(
                {
                    "task_id": tid,
                    "evaluation_ok": result.get("evaluation_ok"),
                    "exact_match": result.get("exact_match"),
                    "result_path": result.get("result_path"),
                    "error": result.get("error"),
                }
            )
            if result.get("evaluation_ok"):
                n_ok += 1
            else:
                n_fail += 1
            if result.get("exact_match") is True:
                n_exact += 1
            print(
                f"  -> ok={result.get('evaluation_ok')} exact={result.get('exact_match')} "
                f"logged={result.get('result_path')}"
            )
        if args.delay_s > 0 and i < len(paths):
            time.sleep(args.delay_s)

    n = len(rows)
    summary = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "eval_dir": str(eval_dir),
        "out_dir": str(out_dir),
        "model": args.model,
        "provider": args.provider,
        "n_tasks": n,
        "n_evaluation_ok": n_ok,
        "n_parse_or_api_fail": n_fail,
        "n_exact_match": n_exact,
        "exact_match_rate_among_ran": (n_exact / n) if n else None,
        "exact_match_rate_among_ok": (n_exact / n_ok) if n_ok else None,
        "results": rows,
    }
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_path = out_dir / f"summary_{ts}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"exact_match: {n_exact}/{n}" + (f" ({100 * n_exact / n:.1f}%)" if n else ""))
    print(f"evaluation_ok (parsed): {n_ok}/{n}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
