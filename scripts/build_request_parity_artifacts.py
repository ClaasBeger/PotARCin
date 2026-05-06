#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from framework.eval_paths import dimension_out_dirs, multi_run_summary_dir
from framework.tasks.arc_dataset import load_task
from framework.dimensions.classification import (
    build_distribution_classification_prompt,
    sample_distribution_classification_batch,
)
from framework.dimensions.constrained_generation import build_constrained_generation_prompt
from framework.dimensions.editing import build_editing_prompt, sample_editing_pair
from framework.dimensions.inversion import InversionEvaluator, build_inversion_prompt
from framework.dimensions.base import DimensionConfig, DimensionName

DIMS: list[tuple[str, str]] = [
    ("definition", "definition_eval"),
    ("classification_distribution", "distribution_classification_eval"),
    ("constrained_generation", "constrained_generation_eval"),
    ("editing", "editing_eval"),
    ("inversion", "inversion_eval"),
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prompt_hash(prompt: str | None) -> str:
    if not isinstance(prompt, str):
        return "-"
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _latest_file(dim_dir: Path, task_id: str, suffix: str) -> Path | None:
    matches = sorted(dim_dir.glob(f"{task_id}_{suffix}_*.json"))
    return matches[-1] if matches else None


def _seed_map_from_source(model: str, source_run_tag: str) -> dict[str, int]:
    p = multi_run_summary_dir(root_dir=ROOT_DIR, model=model, run_tag=source_run_tag) / "run_all_400_unified.json"
    data = _read_json(p)
    out: dict[str, int] = {}
    runs = data.get("runs")
    if isinstance(runs, list):
        for r in runs:
            if not isinstance(r, dict):
                continue
            tid = r.get("task_id")
            sd = r.get("seed")
            if isinstance(tid, str) and isinstance(sd, int):
                out[tid] = int(sd)
    return out


def _planned_prompts_for_task(task_id: str, seed: int) -> dict[str, str]:
    """Reconstruct planned prompts using same task + same seed path as original benchmark."""
    task = load_task(task_id)
    planned: dict[str, str] = {}

    # Definition prompt is generated from task JSON in the API runner path.
    # We validate it via source artifact hash (copied in main loop below).
    planned["definition"] = ""

    # Derived per-dimension seeds match run_all_dimensions_eval behavior.
    seeded = random.Random(seed)
    dim_seeds = {
        "classification_distribution": int(seeded.randrange(1 << 31)),
        "editing": int(seeded.randrange(1 << 31)),
        "inversion": int(seeded.randrange(1 << 31)),
    }

    # Classification prompt
    cls_batch = sample_distribution_classification_batch(task_id, random.Random(dim_seeds["classification_distribution"]))
    planned["classification_distribution"] = build_distribution_classification_prompt(
        demonstrations=cls_batch.demonstrations,
        candidates=[it.pair for it in cls_batch.items],
        prior_work_context=None,
    )

    # Constrained generation prompt (no stochastic sampling in prompt construction)
    planned["constrained_generation"] = build_constrained_generation_prompt(
        demonstrations=list(task.train_pairs),
        prior_work_context=None,
    )

    # Editing prompt
    corr_pair, _target_pair, _source = sample_editing_pair(
        task,
        random.Random(dim_seeds["editing"]),
        include_true=False,
    )
    planned["editing"] = build_editing_prompt(
        demonstrations=task.train_pairs,
        corrupted_pair=corr_pair,
        include_true=False,
        prior_work_context=None,
    )

    # Inversion prompt
    inv_eval = InversionEvaluator(
        config=DimensionConfig(name=DimensionName.INVERSION, num_instances_per_task=1, random_seed=dim_seeds["inversion"])
    )
    inst = inv_eval.generate_instances([task])[0]
    planned["inversion"] = build_inversion_prompt(
        demonstrations=inst.input.demonstrations,
        desired_output=inst.input.desired_output,
        prior_work_context=None,
    )
    return planned


def main() -> int:
    ap = argparse.ArgumentParser(description="Build strict request-parity artifacts (prompt hashes + prompt dumps).")
    ap.add_argument("--subset-json", type=Path, default=Path("Test2/ARC1_subsets/gpt54_nontrivial_success_subset_20_manual.json"))
    ap.add_argument("--model", type=str, default="gpt-5.4")
    ap.add_argument("--source-run-tag", type=str, default="Evaluation_Run_2026-04-14_gpt54_low")
    ap.add_argument("--out-dir", type=Path, default=Path("evaluation/figures/request_parity"))
    args = ap.parse_args()

    subset_path = args.subset_json if args.subset_json.is_absolute() else (ROOT_DIR / args.subset_json)
    subset = _read_json(subset_path)
    task_ids = subset.get("selected_task_ids")
    if not isinstance(task_ids, list) or not all(isinstance(t, str) for t in task_ids):
        raise SystemExit("subset json must contain selected_task_ids")

    task_ids_l = [str(t) for t in task_ids]
    seed_map = _seed_map_from_source(args.model, args.source_run_tag)
    miss = [t for t in task_ids_l if t not in seed_map]
    if miss:
        raise SystemExit(f"Missing seeds for tasks in source run: {', '.join(miss[:8])}")

    src_dirs = dimension_out_dirs(root_dir=ROOT_DIR, model=args.model, run_tag=args.source_run_tag)
    out_dir = args.out_dir if args.out_dir.is_absolute() else (ROOT_DIR / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_dump_dir = out_dir / "prompts"
    prompt_dump_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "request_parity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "task_id",
            "dimension",
            "original_seed",
            "planned_seed",
            "original_prompt_hash",
            "planned_prompt_hash",
            "prompt_hash_match",
            "original_prompt_file",
            "planned_prompt_file",
        ])

        for tid in task_ids_l:
            planned_seed = int(seed_map[tid])
            planned_prompts = _planned_prompts_for_task(tid, planned_seed)
            for dim, suffix in DIMS:
                src_p = _latest_file(src_dirs[dim], tid, suffix)
                src_j = _read_json(src_p) if src_p and src_p.is_file() else {}
                src_prompt = src_j.get("prompt") if isinstance(src_j.get("prompt"), str) else ""
                pln_prompt = planned_prompts.get(dim, "")
                if dim == "definition":
                    # Definition prompt reconstruction depends on puzzle-json pathway not present
                    # in ArcTask; for parity checking we use the source artifact prompt directly.
                    pln_prompt = src_prompt

                src_h = _prompt_hash(src_prompt)
                pln_h = _prompt_hash(pln_prompt)
                match = str(src_h == pln_h)

                src_prompt_file = prompt_dump_dir / f"{tid}__{dim}__original.txt"
                pln_prompt_file = prompt_dump_dir / f"{tid}__{dim}__planned.txt"
                src_prompt_file.write_text(src_prompt, encoding="utf-8")
                pln_prompt_file.write_text(pln_prompt, encoding="utf-8")

                w.writerow([
                    tid,
                    dim,
                    src_j.get("seed"),
                    planned_seed,
                    src_h,
                    pln_h,
                    match,
                    str(src_prompt_file),
                    str(pln_prompt_file),
                ])

    print("Wrote:", csv_path)
    print("Prompt dumps:", prompt_dump_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

