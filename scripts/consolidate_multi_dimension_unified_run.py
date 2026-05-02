"""Merge per-dimension eval JSONs into canonical unified run files (400 tasks, 8×50 batches).

Reads latest ``<task_id>_*.json`` under each dimension directory from ``dimension_out_dirs``,
in canonical ARC train order, and writes:

- ``run_all_400_unified.json``
- ``run_batch{1..8}_unified_50.json``
- per-batch strict summaries (``*_strict_accuracy.json``, etc.) and ``*_overview.md``
- ``run_all_400_unified_strict_accuracy.json`` and ``run_all_400_unified_strict_accuracy_by_batch.json``

Typical usage (from repo root)::

    python scripts/consolidate_multi_dimension_unified_run.py \\
        --run-tag Evaluation_Run_minimax_m25_test \\
        --model minimax/minimax-m2.5:nitro

Then build HTML (fast in-process default)::

    for i in $(seq 1 8); do
      python scripts/build_test_run_unified_visualizations.py \\
        --run-json evaluation/multi_dimension_eval_runs/MiniMaxM25/Run1/Evaluation_Run_minimax_m25_test/run_batch${i}_unified_50.json \\
        --out-dir Demonstrations/Evaluation_Run_minimax_m25_test_batch${i}
    done
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
_mpl_cfg = ROOT_DIR / ".mplconfig"
_mpl_cfg.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cfg))
os.environ.setdefault("MPLBACKEND", "Agg")

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from framework.eval_paths import dimension_out_dirs, multi_run_summary_dir
from framework.tasks.arc_dataset import iter_tasks
from framework.tasks.base import TaskSource


def _latest_payload(task_id: str, dim: str, dim_dirs: dict[str, Path]) -> tuple[str | None, dict]:
    d = dim_dirs[dim]
    matches = sorted(d.glob(f"{task_id}_*.json"))
    if not matches:
        return None, {}
    p = matches[-1]
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return str(p.resolve()), {}
    return str(p.resolve()), payload if isinstance(payload, dict) else {}


def _has_model_response(payload: dict) -> bool:
    rr = payload.get("raw_response")
    return isinstance(rr, str) and bool(rr.strip())


def _dim_report(task_id: str, dim: str, dim_dirs: dict[str, Path]) -> tuple[dict | None, dict]:
    rp, payload = _latest_payload(task_id, dim, dim_dirs)
    if rp is None:
        return None, {}
    eval_ok = bool(payload.get("evaluation_ok"))
    model_response_present = _has_model_response(payload)
    ok = bool(eval_ok or model_response_present)
    rep: dict = {
        "ok": ok,
        "result_path": rp,
        "metrics": payload.get("metrics"),
        "attempts": 0,
        "skipped_rerun": True,
    }
    if (not eval_ok) and model_response_present:
        rep["evaluation_error"] = str(payload.get("error") or "evaluation_ok is false")
    elif not ok:
        rep["error"] = str(payload.get("error") or "missing or invalid payload")
    return rep, payload


def _strict_definition_pass(run: dict) -> bool:
    d = (((run.get("dimensions") or {}).get("definition")) or {})
    m = d.get("metrics") or {}
    for k in ("train", "test", "stable", "dynamic50"):
        mm = m.get(k)
        if not isinstance(mm, dict):
            return False
        if mm.get("ok") != mm.get("total"):
            return False
    return True


def _strict_classification_pass(run: dict) -> bool:
    d = (((run.get("dimensions") or {}).get("classification_distribution")) or {})
    m = d.get("metrics") or {}
    return m.get("exact_match") == 1


def _strict_cg_pass(run: dict) -> bool:
    d = (((run.get("dimensions") or {}).get("constrained_generation")) or {})
    m = d.get("metrics") or {}
    return m.get("success_rate") == 1.0


def _strict_edit_pass(run: dict) -> bool:
    d = (((run.get("dimensions") or {}).get("editing")) or {})
    m = d.get("metrics") or {}
    return m.get("success") is True


def _strict_inv_pass(run: dict) -> bool:
    d = (((run.get("dimensions") or {}).get("inversion")) or {})
    m = d.get("metrics") or {}
    return m.get("success_rate") == 1.0


def _strict_dim_counts(subruns: list[dict]) -> dict[str, int]:
    return {
        "definition": sum(1 for r in subruns if _strict_definition_pass(r)),
        "classification_distribution": sum(1 for r in subruns if _strict_classification_pass(r)),
        "constrained_generation": sum(1 for r in subruns if _strict_cg_pass(r)),
        "editing": sum(1 for r in subruns if _strict_edit_pass(r)),
        "inversion": sum(1 for r in subruns if _strict_inv_pass(r)),
    }


def _strict_full_pass(run: dict) -> bool:
    return (
        _strict_definition_pass(run)
        and _strict_classification_pass(run)
        and _strict_cg_pass(run)
        and _strict_edit_pass(run)
        and _strict_inv_pass(run)
    )


def consolidate(*, root_dir: Path, model: str, run_tag: str) -> None:
    dim_dirs = dimension_out_dirs(root_dir=root_dir, model=model, run_tag=run_tag)
    run_dir = multi_run_summary_dir(root_dir=root_dir, model=model, run_tag=run_tag)
    run_dir.mkdir(parents=True, exist_ok=True)

    task_ids = [t.task_id for t in iter_tasks(split="train", source=TaskSource.ORIGINAL_ARC)]
    if len(task_ids) != 400:
        raise SystemExit(f"Expected 400 ARC train tasks, got {len(task_ids)}")

    runs: list[dict] = []
    missing: list[str] = []
    for tid in task_ids:
        dims: dict = {}
        sample_payload: dict | None = None
        complete = True
        for dim in (
            "definition",
            "classification_distribution",
            "constrained_generation",
            "editing",
            "inversion",
        ):
            out = _dim_report(tid, dim, dim_dirs)
            if out[0] is None:
                complete = False
                break
            rep, payload = out
            dims[dim] = rep
            if sample_payload is None and isinstance(payload, dict) and payload:
                sample_payload = payload
        if not complete:
            missing.append(tid)
            continue
        sp = sample_payload or {}
        runs.append(
            {
                "task_id": tid,
                "model": sp.get("model", model),
                "provider": sp.get("provider", "openrouter"),
                "reasoning_effort": sp.get("reasoning_effort", "low"),
                "tools_enabled": bool(sp.get("tools_enabled", False)),
                "seed": None,
                "dimension_seeds": {},
                "dimensions": dims,
            }
        )

    if missing:
        preview = ", ".join(missing[:20])
        more = f" (+{len(missing) - 20} more)" if len(missing) > 20 else ""
        raise SystemExit(f"Missing dimension artifacts for {len(missing)} task(s): {preview}{more}")

    created_at = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    all_payload = {
        "mode": "batch_unified",
        "created_at": created_at,
        "task_count": len(runs),
        "run_tag": run_tag,
        "seed_mode": "fixed",
        "base_seed": 77,
        "batch_size": 400,
        "batch_index": 0,
        "batch_slice": "train[0:400]",
        "completed_before_count": 0,
        "last_completed_task_id_before": "none",
        "merged_from_sources": ["filled_from_latest_dimension_artifacts"],
        "runs": runs,
    }
    (run_dir / "run_all_400_unified.json").write_text(json.dumps(all_payload, indent=2), encoding="utf-8")

    by_batch: list[dict] = []
    source_files: list[str] = []
    for i in range(8):
        lo = i * 50
        hi = lo + 50
        sub = runs[lo:hi]
        bidx = i + 1
        bj = {
            "mode": "batch_unified",
            "created_at": created_at,
            "task_count": len(sub),
            "run_tag": run_tag,
            "seed_mode": "fixed",
            "base_seed": 77,
            "batch_size": 50,
            "batch_index": bidx,
            "batch_slice": f"train[{lo}:{hi}]",
            "completed_before_count": lo,
            "last_completed_task_id_before": "none" if lo == 0 else task_ids[lo - 1],
            "merged_from_sources": ["filled_from_latest_dimension_artifacts"],
            "runs": sub,
        }
        batch_name = f"run_batch{bidx}_unified_50.json"
        (run_dir / batch_name).write_text(json.dumps(bj, indent=2), encoding="utf-8")
        source_files.append(batch_name)

        full_pass_ids = [r["task_id"] for r in sub if _strict_full_pass(r)]
        dim_counts = _strict_dim_counts(sub)
        acc = len(full_pass_ids) / len(sub) if sub else 0.0
        strict_payload = {
            "run_tag": run_tag,
            "batch_index": bidx,
            "task_count": len(sub),
            "strict_full_task_pass_count": len(full_pass_ids),
            "strict_full_task_accuracy": acc,
            "strict_full_task_accuracy_percent": round(acc * 100, 2),
            "strict_by_dimension_pass_count": dim_counts,
            "created_at": created_at,
        }
        for nm in (
            f"run_batch{bidx}_unified_50_strict_accuracy.json",
            f"run_batch{bidx}_unified_50_strict_full_task_accuracy.json",
            f"run_batch{bidx}_unified_50_full_task_accuracy_all5.json",
        ):
            (run_dir / nm).write_text(json.dumps(strict_payload, indent=2), encoding="utf-8")

        md = (
            f"# Batch {bidx} Strict Full-Task Accuracy\n\n"
            f"- Run tag: {run_tag}\n"
            f"- Tasks: {len(sub)}\n"
            f"- Strict full-task pass: {len(full_pass_ids)}/{len(sub)} ({acc * 100:.2f}%)\n\n"
            f"## Dimension pass counts (strict)\n"
            f"- definition: {dim_counts['definition']}/{len(sub)} ({(dim_counts['definition'] / len(sub)) * 100:.2f}%)\n"
            f"- classification_distribution: {dim_counts['classification_distribution']}/{len(sub)} "
            f"({(dim_counts['classification_distribution'] / len(sub)) * 100:.2f}%)\n"
            f"- constrained_generation: {dim_counts['constrained_generation']}/{len(sub)} "
            f"({(dim_counts['constrained_generation'] / len(sub)) * 100:.2f}%)\n"
            f"- editing: {dim_counts['editing']}/{len(sub)} ({(dim_counts['editing'] / len(sub)) * 100:.2f}%)\n"
            f"- inversion: {dim_counts['inversion']}/{len(sub)} ({(dim_counts['inversion'] / len(sub)) * 100:.2f}%)\n"
        )
        (run_dir / f"run_batch{bidx}_unified_50_overview.md").write_text(md, encoding="utf-8")

        by_batch.append(
            {
                "batch_index": bidx,
                "strict_full_task_pass_count": len(full_pass_ids),
                "task_count": len(sub),
                "strict_full_task_accuracy": acc,
            }
        )

    full_pass_ids = [r["task_id"] for r in runs if _strict_full_pass(r)]
    dim_counts_all = _strict_dim_counts(runs)
    overall_acc = len(full_pass_ids) / len(runs)
    overall = {
        "run_tag": run_tag,
        "merged_at": created_at,
        "canonical_task_count": len(runs),
        "source_run_files_in_merge_order": source_files,
        "definition_strict_pass": dim_counts_all["definition"],
        "classification_exact_pass": dim_counts_all["classification_distribution"],
        "constrained_generation_pass": dim_counts_all["constrained_generation"],
        "editing_pass": dim_counts_all["editing"],
        "inversion_pass": dim_counts_all["inversion"],
        "full_task_pass": len(full_pass_ids),
        "full_task_accuracy": overall_acc,
        "full_task_accuracy_percent": round(overall_acc * 100, 2),
        "strict_full_task_accuracy": overall_acc,
        "strict_full_task_accuracy_percent": round(overall_acc * 100, 2),
        "full_pass_task_ids": full_pass_ids,
    }
    (run_dir / "run_all_400_unified_strict_accuracy.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )
    (run_dir / "run_all_400_unified_strict_accuracy_by_batch.json").write_text(
        json.dumps({"run_tag": run_tag, "by_batch": by_batch, "created_at": created_at}, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote unified + strict summaries under {run_dir}")
    print(f"Overall strict full-task: {len(full_pass_ids)}/{len(runs)} ({overall_acc * 100:.2f}%)")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Merge per-dimension eval JSONs into run_all_400_unified + 8×50 batch files and strict summaries."
    )
    p.add_argument("--run-tag", required=True)
    p.add_argument("--model", required=True, help='Model id as passed to eval (e.g. minimax/minimax-m2.5:nitro)')
    p.add_argument(
        "--root",
        type=Path,
        default=ROOT_DIR,
        help="Repo root (default: parent of scripts/)",
    )
    args = p.parse_args()
    consolidate(root_dir=args.root.resolve(), model=args.model.strip(), run_tag=args.run_tag.strip())


if __name__ == "__main__":
    main()
