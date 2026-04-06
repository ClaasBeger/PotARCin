from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from framework.dimensions.editing import run_editing_api_evaluation_once
from framework.tasks.arc_dataset import load_task


def main() -> None:
    p = argparse.ArgumentParser(description="Run one Editing-dimension model eval for a single task.")
    p.add_argument("--task-id", default="a85d4709", help="ARC task id")
    p.add_argument("--model", default="gpt-5.4", help="Model name")
    p.add_argument("--reasoning-effort", default="low", help="Reasoning effort for responses API")
    p.add_argument("--tools-enabled", action="store_true", help="Enable code interpreter tool")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for corruption sampling")
    p.add_argument("--include-true", action="store_true", help="Allow sampling an already-correct pair")
    p.add_argument(
        "--out-dir",
        default="evaluation/editing_eval_results",
        help="Directory for saved evaluation JSON",
    )
    args = p.parse_args()

    task = load_task(args.task_id)
    res = run_editing_api_evaluation_once(
        task=task,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        tools_enabled=args.tools_enabled,
        seed=args.seed,
        include_true=args.include_true,
        out_dir=args.out_dir,
    )
    print(f"saved: {res['result_path']}")
    print("metrics:")
    print(json.dumps(res["metrics"], indent=2))


if __name__ == "__main__":
    main()

