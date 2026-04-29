from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from framework.dimensions.verifier_repair_experiment import run_verifier_repair_api_evaluation_once
from framework.tasks.arc_dataset import load_task


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run one standalone verifier-repair experiment for a single ARC task."
    )
    p.add_argument("--task-id", default="a85d4709", help="ARC task id")
    p.add_argument("--test-idx", type=int, default=0, help="Test input index for evaluation")
    p.add_argument("--model", default="gpt-5.4", help="Model name")
    p.add_argument(
        "--provider",
        default=None,
        choices=["openai", "anthropic", "google", "together", "openrouter"],
        help="LLM provider (default: infer from --model)",
    )
    p.add_argument("--reasoning-effort", default="low", help="Reasoning effort for responses API")
    p.add_argument("--tools-enabled", action="store_true", help="Enable code interpreter tool")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument(
        "--max-norm-edit",
        type=float,
        default=0.70,
        help="Max normalized padded output drift allowed when sampling corruption.",
    )
    p.add_argument(
        "--arc-gen-dynamic-n",
        type=int,
        default=50,
        help="Number of ARC-GEN dynamic samples used during evaluation.",
    )
    p.add_argument(
        "--re-arc-dynamic-n",
        type=int,
        default=50,
        help="Number of re_arc dynamic samples used during evaluation.",
    )
    p.add_argument(
        "--out-dir",
        default="evaluation/verifier_repair_experiment_results",
        help="Directory for saved experiment JSON",
    )
    args = p.parse_args()

    task = load_task(args.task_id)
    res = run_verifier_repair_api_evaluation_once(
        task=task,
        model=args.model,
        provider=args.provider,
        reasoning_effort=args.reasoning_effort,
        tools_enabled=args.tools_enabled,
        seed=args.seed,
        max_norm_edit=args.max_norm_edit,
        test_idx=args.test_idx,
        arc_gen_dynamic_n=args.arc_gen_dynamic_n,
        re_arc_dynamic_n=args.re_arc_dynamic_n,
        out_dir=args.out_dir,
    )
    print(f"saved: {res['result_path']}")
    if not res.get("evaluation_ok", True):
        print("evaluation_ok: false")
        if res.get("error"):
            print("error:", res["error"])
        sys.exit(1)
    print("metrics:")
    print(json.dumps(res["metrics"], indent=2))
    if not res.get("script_loaded", False):
        print("script_load_error:")
        print(res.get("script_load_error"))


if __name__ == "__main__":
    main()

