#!/usr/bin/env python3
"""Rebuild joint batch + summary JSON for a Test2 output-grid run from per-task artifacts.

Use when a run folder has complete `test2_t*_output_grid_eval_*.json` files but a stale or
partial `batch_*.json`, or when mirroring the `joint_batch_test2.json` / `joint_summary_test2.json`
pattern used for other models.
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


def _latest_per_task(run_dir: Path) -> dict[int, Path]:
    by_n: dict[int, list[Path]] = {}
    for p in run_dir.glob("test2_t*_output_grid_eval_*.json"):
        if "_output_grid_eval_error_" in p.name:
            continue
        name = p.name
        if not name.startswith("test2_t"):
            continue
        rest = name[len("test2_t") :]
        n_str = rest.split("_", 1)[0]
        if not n_str.isdigit():
            continue
        n = int(n_str)
        by_n.setdefault(n, []).append(p)
    out: dict[int, Path] = {}
    for n, paths in by_n.items():
        out[n] = max(paths, key=lambda x: x.name)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing test2_t*_output_grid_eval_*.json files",
    )
    p.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Override run_tag (default: run-dir name)",
    )
    args = p.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT_DIR / args.run_dir
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")

    run_tag = args.run_tag or run_dir.name
    latest = _latest_per_task(run_dir)
    missing = [n for n in range(1, 51) if n not in latest]
    if missing:
        raise SystemExit(f"Missing tasks (expected t1..t50): {missing[:20]}{'...' if len(missing) > 20 else ''}")

    sample = json.loads(latest[1].read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")

    results: list[dict] = []
    parse_ok = exact_ct = 0
    eval_fail = 0
    for n in range(1, 51):
        path = latest[n]
        data = json.loads(path.read_text(encoding="utf-8"))
        ev_ok = data.get("evaluation_ok") is True
        ex = data.get("exact_match") is True
        if ev_ok:
            parse_ok += 1
        else:
            eval_fail += 1
        if ex:
            exact_ct += 1
        metrics_ex = 1.0 if ex else 0.0
        results.append(
            {
                "task_id": f"t{n}",
                "attempts": 1,
                "result_path": str(path),
                "evaluation_ok": ev_ok,
                "exact_match": ex,
                "metrics": {"exact_match": metrics_ex},
                "error": None,
            }
        )

    batch = {
        "created_at": now,
        "mode": "output_grid_batch_test2",
        "run_tag": run_tag,
        "ordered_source_count": 50,
        "skipped_already_logged": 0,
        "ran_task_count": 50,
        "completed_runs_recorded": 50,
        "config": {
            "test2_root": str(ROOT_DIR / "Test2"),
            "test_idx": sample.get("test_idx", 0),
            "model": sample.get("model"),
            "provider": sample.get("provider"),
            "reasoning_effort": sample.get("reasoning_effort"),
            "tools_enabled": sample.get("tools_enabled", False),
            "max_train_demos": sample.get("max_train_demos"),
            "out_dir": str(run_dir),
            "note": "Joint summary rebuilt from latest per-task artifacts via consolidate_test2_output_grid_run.py",
        },
        "aggregate": {
            "parse_ok": parse_ok,
            "exact_match": exact_ct,
            "failed_or_incomplete": eval_fail,
        },
        "results": results,
    }

    summary = {
        "run_tag": run_tag,
        "model": sample.get("model"),
        "provider": sample.get("provider"),
        "suite": "Test2_output_grid",
        "task_count": 50,
        "evaluation_ok_count": parse_ok,
        "evaluation_ok_rate": parse_ok / 50,
        "exact_match_count": exact_ct,
        "exact_match_rate": exact_ct / 50,
        "parse_fail_count": 50 - parse_ok,
        "rebuilt_at": now,
    }

    joint_batch_path = run_dir / "joint_batch_test2.json"
    joint_summary_path = run_dir / "joint_summary_test2.json"
    joint_batch_path.write_text(json.dumps(batch, indent=2) + "\n", encoding="utf-8")
    joint_summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    overview_md = run_dir / "output_grid_test2_run_overview.md"
    overview_md.write_text(
        f"# Test2 output grid — `{run_tag}`\n\n"
        f"- **Model:** `{summary['model']}` ({summary['provider']})\n"
        f"- **Tasks:** 50 (Test2 t1–t50)\n"
        f"- **evaluation_ok:** {parse_ok}/50 ({100 * parse_ok / 50:.1f}%)\n"
        f"- **Exact match (test output):** {exact_ct}/50 ({100 * exact_ct / 50:.1f}%)\n"
        f"- **Artifacts:** `joint_batch_test2.json`, `joint_summary_test2.json`\n"
        f"- **Reports:** `output_grid_report_all.html`, `output_grid_report_mismatches.html`, `index.html`\n"
        f"- **Rebuilt:** {now} UTC\n",
        encoding="utf-8",
    )

    print(f"Wrote {joint_batch_path.relative_to(ROOT_DIR)}")
    print(f"Wrote {joint_summary_path.relative_to(ROOT_DIR)}")
    print(f"Wrote {overview_md.relative_to(ROOT_DIR)}")
    print(f"exact_match: {exact_ct}/50 ({100 * exact_ct / 50:.1f}%)")


if __name__ == "__main__":
    main()
