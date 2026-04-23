"""Run one Distribution (5-way) Classification model eval — mirrors ``run_definition_eval_once.py``.

Uses the prompt from ``framework.dimensions.classification.build_distribution_classification_prompt``
(via ``run_distribution_classification_api_evaluation_once``).

From repo root (requires matching provider key, e.g. ``OPENAI_API_KEY``)::

    python scripts/run_distribution_classification_eval_once.py --task-id a85d4709 --seed 0

Optional: same API flags as definition eval (``--reasoning-effort``, ``--tools-enabled``, ``--out-dir``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from framework.dimensions.classification import run_distribution_classification_api_evaluation_once


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one Distribution Classification (5-way) model eval for a single task."
    )
    parser.add_argument("--task-id", default="a85d4709", help="ARC task id")
    parser.add_argument("--model", default="gpt-5.4", help="Model name")
    parser.add_argument(
        "--provider",
        default=None,
        choices=["openai", "anthropic", "google", "together"],
        help="LLM provider (default: infer from --model)",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="low",
        help="Reasoning effort for responses API (e.g. low/medium/high)",
    )
    parser.add_argument(
        "--tools-enabled",
        action="store_true",
        help="Enable code interpreter tool in API request",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for sampling the 5 pairs")
    parser.add_argument(
        "--out-dir",
        default="evaluation/classification_eval_results",
        help="Directory for saved evaluation JSON",
    )
    args = parser.parse_args()

    result = run_distribution_classification_api_evaluation_once(
        task_id=args.task_id,
        model=args.model,
        provider=args.provider,
        reasoning_effort=args.reasoning_effort,
        tools_enabled=args.tools_enabled,
        seed=args.seed,
        out_dir=args.out_dir,
    )

    print(f"saved: {result['result_path']}")
    if not result.get("evaluation_ok", True):
        print("evaluation_ok: false")
        if result.get("error"):
            print("error:", result["error"])
        sys.exit(1)
    print("metrics:")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
