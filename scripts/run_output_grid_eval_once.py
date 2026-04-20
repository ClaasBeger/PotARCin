from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from framework.dimensions.output_grid_eval import run_output_grid_api_evaluation_once


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one direct output-grid ARC eval (demos + test input -> predicted grid vs GT)."
    )
    parser.add_argument("--task-id", default="a85d4709", help="ARC task id")
    parser.add_argument("--test-idx", type=int, default=0, help="Which test example to use")
    parser.add_argument("--model", default="gpt-5.4", help="Model name")
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
    parser.add_argument(
        "--max-train-demos",
        type=int,
        default=None,
        help="Optional cap on number of training demonstrations (default: all)",
    )
    parser.add_argument(
        "--out-dir",
        default="evaluation/output_grid_eval_results",
        help="Directory for saved evaluation JSON",
    )
    args = parser.parse_args()

    result = run_output_grid_api_evaluation_once(
        task_id=args.task_id,
        test_idx=args.test_idx,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        tools_enabled=args.tools_enabled,
        max_train_demos=args.max_train_demos,
        out_dir=args.out_dir,
    )

    print(f"saved: {result['result_path']}")
    print("exact_match:", result.get("exact_match"))
    print("evaluation_ok:", result.get("evaluation_ok"))
    if result.get("error"):
        print("error:", result["error"])
    if not result.get("evaluation_ok", True):
        sys.exit(1)
    print("metrics:")
    print(json.dumps(result.get("metrics", {}), indent=2))


if __name__ == "__main__":
    main()
