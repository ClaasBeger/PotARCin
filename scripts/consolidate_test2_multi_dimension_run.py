"""Build canonical Test2 multi-dimension summary JSONs from per-dimension eval artifacts.

Writes (under ``evaluation/multi_dimension_eval_runs/.../<run_tag>/``)::

  - run_canonical.json
  - run_canonical_overview.json / .md
  - run_canonical_strict_full_task.json
  - test2_run_performance_overview.md

Strict full-task rules match ``consolidate_multi_dimension_unified_run.py`` and other Test2
canonical runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from framework.eval_paths import dimension_out_dirs, multi_run_summary_dir
from framework.tasks.test2_dataset import iter_test2_tasks
import consolidate_multi_dimension_unified_run as cmu


def _build_overview_dict(payload: dict) -> dict:
    """Match run_all_dimensions_eval._build_overview (primary metrics only)."""
    runs = payload.get("runs", [])
    n_tasks = len(runs) if isinstance(runs, list) else 0
    order = (
        "definition",
        "classification_distribution",
        "constrained_generation",
        "editing",
        "inversion",
    )
    by_dim: dict = {}
    for dim in order:
        by_dim[dim] = {
            "ok_tasks": 0,
            "fail_tasks": 0,
            "ok_rate": 0.0,
            "mean_primary_metric": None,
            "metric_count": 0,
        }
    for run in runs if isinstance(runs, list) else []:
        dims = run.get("dimensions", {}) if isinstance(run, dict) else {}
        for dim in order:
            entry = dims.get(dim, {}) if isinstance(dims, dict) else {}
            ok = bool(entry.get("ok"))
            if ok:
                by_dim[dim]["ok_tasks"] += 1
            else:
                by_dim[dim]["fail_tasks"] += 1
            metrics = entry.get("metrics")
            val = None
            if isinstance(metrics, dict):
                for key in ("success_rate", "accuracy", "exact_match_rate"):
                    v = metrics.get(key)
                    if isinstance(v, (int, float)):
                        val = float(v)
                        break
            if val is not None:
                cur = by_dim[dim].get("mean_primary_metric")
                if cur is None:
                    by_dim[dim]["mean_primary_metric"] = val
                else:
                    by_dim[dim]["mean_primary_metric"] = float(cur) + val
                by_dim[dim]["metric_count"] += 1
    for dim in order:
        d = by_dim[dim]
        d["ok_rate"] = (d["ok_tasks"] / n_tasks) if n_tasks else 0.0
        if d["metric_count"] > 0 and d["mean_primary_metric"] is not None:
            d["mean_primary_metric"] = d["mean_primary_metric"] / d["metric_count"]
    return {
        "created_at": payload.get("created_at"),
        "task_count": n_tasks,
        "by_dimension": by_dim,
    }


def consolidate(*, root_dir: Path, model: str, run_tag: str) -> None:
    dim_dirs = dimension_out_dirs(
        root_dir=root_dir, model=model, run_tag=run_tag, dataset_segment="Test2"
    )
    run_dir = multi_run_summary_dir(
        root_dir=root_dir, model=model, run_tag=run_tag, dataset_segment="Test2"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    task_ids = [t.task_id for t in iter_test2_tasks()]
    if len(task_ids) != 50:
        raise SystemExit(f"Expected 50 Test2 tasks, got {len(task_ids)}")

    runs: list[dict] = []
    missing: list[str] = []
    base_seed_hint: int | None = None
    for tid in task_ids:
        dims: dict = {}
        payloads: dict[str, dict] = {}
        complete = True
        for dim in (
            "definition",
            "classification_distribution",
            "constrained_generation",
            "editing",
            "inversion",
        ):
            rep, payload = cmu._dim_report(tid, dim, dim_dirs)
            if rep is None:
                complete = False
                break
            dims[dim] = rep
            if isinstance(payload, dict) and payload:
                payloads[dim] = payload
        if not complete:
            missing.append(tid)
            continue
        def_pl = payloads.get("definition") or {}
        cls_pl = payloads.get("classification_distribution") or {}
        edit_pl = payloads.get("editing") or {}
        inv_pl = payloads.get("inversion") or {}
        seed_val = def_pl.get("seed") or cls_pl.get("seed")
        if base_seed_hint is None and isinstance(seed_val, int):
            base_seed_hint = seed_val
        runs.append(
            {
                "task_id": tid,
                "model": def_pl.get("model") or cls_pl.get("model") or model,
                "provider": def_pl.get("provider") or cls_pl.get("provider"),
                "reasoning_effort": def_pl.get("reasoning_effort", "low"),
                "tools_enabled": bool(def_pl.get("tools_enabled", False)),
                "seed": seed_val,
                "dimension_seeds": {
                    "classification_distribution": cls_pl.get("seed"),
                    "editing": edit_pl.get("seed"),
                    "inversion": inv_pl.get("seed"),
                },
                "stitch_prior_work_context": False,
                "stitched_context_max_chars": 12000,
                "dimensions": dims,
            }
        )

    if missing:
        preview = ", ".join(missing[:15])
        more = f" (+{len(missing) - 15} more)" if len(missing) > 15 else ""
        raise SystemExit(f"Missing dimension artifacts for {len(missing)} task(s): {preview}{more}")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "mode": "completed",
        "created_at": ts,
        "task_count": 50,
        "run_tag": run_tag,
        "task_source": "test2",
        "seed_mode": "fixed",
        "base_seed": base_seed_hint,
        "batch_size": 50,
        "completed_before_count": 0,
        "last_completed_task_id_before": None,
        "runs": runs,
        "merged_from": ["consolidate_test2_multi_dimension_run.py"],
    }

    (run_dir / "run_canonical.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    overview_payload = dict(payload)
    overview_payload["created_at"] = "canonical"
    ov = _build_overview_dict(overview_payload)
    (run_dir / "run_canonical_overview.json").write_text(json.dumps(ov, indent=2), encoding="utf-8")

    full_pass_ids = [r["task_id"] for r in runs if cmu._strict_full_pass(r)]
    dim_counts = cmu._strict_dim_counts(runs)
    strict_doc = {
        "task_count": 50,
        "strict_full_task_pass_count": len(full_pass_ids),
        "strict_full_task_accuracy": len(full_pass_ids) / 50,
        "strict_pass_task_ids": full_pass_ids,
        "strict_fail_task_ids": [r["task_id"] for r in runs if r["task_id"] not in full_pass_ids],
        "per_dimension_strict_pass_count": dim_counts,
    }
    (run_dir / "run_canonical_strict_full_task.json").write_text(
        json.dumps(strict_doc, indent=2), encoding="utf-8"
    )

    # Markdown overview (dimension OK table + strict summary)
    lines: list[str] = []
    lines.append(f"# Test2 multi-dimension — `{run_tag}`")
    lines.append("")
    lines.append(f"- **Model:** `{payload['runs'][0].get('model', model)}`")
    lines.append(f"- **Tasks:** 50 · **Strict full-task pass:** {len(full_pass_ids)}/50 ({100 * len(full_pass_ids) / 50:.2f}%)")
    lines.append("")
    lines.append("## Per-dimension (API ok)")
    lines.append("")
    lines.append("| Dimension | OK | Fail | OK rate | Mean primary metric |")
    lines.append("|---|---:|---:|---:|---:|")
    order = (
        "definition",
        "classification_distribution",
        "constrained_generation",
        "editing",
        "inversion",
    )
    byd = ov.get("by_dimension", {})
    for dim in order:
        d = byd.get(dim, {})
        mm = d.get("mean_primary_metric")
        mm_s = "n/a" if mm is None else f"{float(mm):.4f}"
        lines.append(
            f"| `{dim}` | {int(d.get('ok_tasks', 0))} | {int(d.get('fail_tasks', 0))} | "
            f"{float(d.get('ok_rate', 0)):.2%} | {mm_s} |"
        )
    lines.append("")
    lines.append("## Strict full-task (all five dimensions pass strict checks)")
    lines.append("")
    for k, v in dim_counts.items():
        lines.append(f"- **{k}:** {v}/50")
    lines.append("")
    lines.append(f"- **Combined strict pass:** {len(full_pass_ids)}/50")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `run_canonical.json` — merged run")
    lines.append("- `run_canonical_overview.json` — dimension OK / mean metrics")
    lines.append("- `run_canonical_strict_full_task.json` — strict counts and task lists")
    lines.append("")
    md_ov = "\n".join(lines) + "\n"
    (run_dir / "run_canonical_overview.md").write_text(md_ov, encoding="utf-8")

    perf = (
        f"# Test2 performance snapshot — `{run_tag}`\n\n"
        f"Generated: {ts}\n\n"
        f"## Strict full-task accuracy\n\n"
        f"- **{len(full_pass_ids)}/50** ({100 * len(full_pass_ids) / 50:.2f}%)\n\n"
        f"### Passing task IDs\n\n"
        f"{', '.join(full_pass_ids) if full_pass_ids else '*(none)*'}\n\n"
        f"### Per-dimension strict pass counts\n\n"
        + "\n".join(f"- `{k}`: {v}/50" for k, v in dim_counts.items())
        + "\n"
    )
    (run_dir / "test2_run_performance_overview.md").write_text(perf, encoding="utf-8")

    print(f"Wrote canonical summaries under {run_dir}")
    print(f"Strict full-task: {len(full_pass_ids)}/50 ({100 * len(full_pass_ids) / 50:.2f}%)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-tag", required=True)
    p.add_argument("--model", required=True, help="Model id as used in eval (e.g. minimax/minimax-m2.5:nitro)")
    p.add_argument("--root", type=Path, default=ROOT_DIR)
    args = p.parse_args()
    consolidate(root_dir=args.root.resolve(), model=args.model.strip(), run_tag=args.run_tag.strip())


if __name__ == "__main__":
    main()
