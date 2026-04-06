from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from framework.dimensions.definition import run_definition_api_evaluation
from framework.tasks.arc_dataset import load_task


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one Definition-dimension model eval for a single task."
    )
    parser.add_argument("--task-id", default="a85d4709", help="ARC task id")
    parser.add_argument("--test-idx", type=int, default=0, help="Test input index")
    parser.add_argument("--model", default="gpt-5.4", help="Model name")
    parser.add_argument(
        "--api-endpoint",
        default="responses",
        choices=["responses", "chat_completions"],
        help="OpenAI-compatible endpoint type",
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
    parser.add_argument(
        "--out-dir",
        default="evaluation/definition_eval_results",
        help="Directory for saved evaluation JSON",
    )
    args = parser.parse_args()

    task = load_task(args.task_id)
    result = run_definition_api_evaluation(
        task=task,
        test_idx=args.test_idx,
        model=args.model,
        api_endpoint=args.api_endpoint,
        reasoning_effort=args.reasoning_effort,
        tools_enabled=args.tools_enabled,
        out_dir=args.out_dir,
    )

    print(f"saved: {result['result_path']}")
    print("metrics:")
    print(json.dumps(result["metrics"], indent=2))
    if not result.get("script_loaded", False):
        print("script_load_error:")
        print(result.get("script_load_error"))


if __name__ == "__main__":
    main()
