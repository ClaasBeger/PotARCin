from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from framework.dimensions.classification import run_classification_api_evaluation_once


def main() -> None:
    p = argparse.ArgumentParser(description="Run one Classification eval call.")
    p.add_argument("--task-id", default="a85d4709")
    p.add_argument("--model", default="gpt-5.4")
    p.add_argument(
        "--provider",
        default=None,
        choices=["openai", "anthropic", "google"],
        help="LLM provider (default: infer from --model)",
    )
    p.add_argument("--reasoning-effort", default="low")
    p.add_argument("--tools-enabled", action="store_true")
    p.add_argument("--n-pos", type=int, default=3)
    p.add_argument("--n-neg", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="evaluation/classification_eval_results")
    args = p.parse_args()

    res = run_classification_api_evaluation_once(
        task_id=args.task_id,
        model=args.model,
        provider=args.provider,
        reasoning_effort=args.reasoning_effort,
        tools_enabled=args.tools_enabled,
        n_pos=args.n_pos,
        n_neg=args.n_neg,
        seed=args.seed,
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


if __name__ == "__main__":
    main()

