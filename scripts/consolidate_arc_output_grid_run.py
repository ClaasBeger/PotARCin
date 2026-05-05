#!/usr/bin/env python3
"""Consolidate ARC original-train output-grid artifacts into a standard run directory layout.

Writes:
  - Per-task JSON kept or moved under --dest (dedupe by latest mtime).
  - batch_arc_train_full.json (400 tasks, joint summary).
  - batch_arc_train_batch01.json … batch_arc_train_batch08.json (50 tasks each).
  - output_grid_run_overview.{json,md} (Claude-style overview + batch_summary_paths).
  - Archives prior batch_*.json snapshots and duplicate per-task files under
    archive_intermediate_runs/.
  - Optionally builds HTML reports via build_output_grid_eval_html.py and index.html.

Example::

    python scripts/consolidate_arc_output_grid_run.py \\
      --source evaluation/output_grid_eval_results/Gemini31Pro/Run1/gemini31pro_low_vanilla_run \\
      --dest evaluation/output_grid_eval_results/Gemini31Pro/Run1/Evaluation_Run_2026-04-25_gemini31pro_output_grid_run1 \\
      --run-tag Evaluation_Run_2026-04-25_gemini31pro_output_grid_run1

In-place (dest already holds per-task JSON)::

    python scripts/consolidate_arc_output_grid_run.py \\
      --dest evaluation/output_grid_eval_results/Gemini31Pro/Run2/Evaluation_Run_2026-04-30_gemini31pro_output_grid_run2 \\
      --run-tag Evaluation_Run_2026-04-30_gemini31pro_output_grid_run2 \\
      --in-place
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]

JOINT_NAME = "batch_arc_train_full.json"
BATCH_PREFIX = "batch_arc_train_batch"
BATCH_COUNT = 8
BATCH_SIZE = 50

DEFAULT_CANONICAL_BATCH = (
    ROOT_DIR
    / "evaluation"
    / "output_grid_eval_results"
    / "ClaudeO46"
    / "Run1"
    / "Evaluation_Run_2026-05-01_claude_opus46_output_grid_run1"
    / JOINT_NAME
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_task_order(canonical_batch: Path) -> list[str]:
    data = _load_json(canonical_batch)
    results = data.get("results")
    if not isinstance(results, list):
        raise SystemExit(f"No results[] in {canonical_batch}")
    ids = [str(r.get("task_id", "")) for r in results]
    if len(ids) != 400 or len(set(ids)) != 400:
        raise SystemExit(f"Expected 400 unique task ids in {canonical_batch}, got {len(ids)}")
    return ids


def _gather_task_files(search_roots: list[Path]) -> dict[str, list[Path]]:
    by_id: dict[str, list[Path]] = defaultdict(list)
    for root in search_roots:
        if not root.is_dir():
            continue
        for p in root.glob("*_output_grid_eval_*.json"):
            # Skip archived duplicates tree if nested (flat layout expected)
            if "archive_intermediate_runs" in p.parts:
                continue
            tid = p.name.split("_output_grid_eval_")[0]
            by_id[tid].append(p)
    return by_id


def _pick_winner(paths: list[Path]) -> tuple[Path, list[Path]]:
    if not paths:
        raise ValueError("empty paths")
    winner = max(paths, key=lambda p: p.stat().st_mtime)
    losers = [p for p in paths if p != winner]
    return winner, losers


def _result_row(*, task_id: str, result_path: Path) -> dict[str, Any]:
    data = _load_json(result_path)
    attempts = 1
    # Prefer parsed attempts field if present on artifact
    raw_att = data.get("attempts")
    if isinstance(raw_att, int) and raw_att >= 1:
        attempts = raw_att
    ev_ok = data.get("evaluation_ok")
    ex = data.get("exact_match")
    metrics = data.get("metrics")
    if metrics is None and isinstance(ex, bool):
        metrics = {"exact_match": 1.0 if ex else 0.0}
    return {
        "task_id": task_id,
        "attempts": attempts,
        "result_path": str(result_path.resolve()),
        "evaluation_ok": ev_ok,
        "exact_match": ex,
        "metrics": metrics,
        "error": data.get("error"),
    }


def _batch_payload(
    *,
    created_at: str,
    run_tag: str,
    ordered_count: int,
    skipped: int,
    task_rows: list[dict[str, Any]],
    dest: Path,
    sample_cfg: dict[str, Any],
    batch_index: int | None,
    batch_size: int,
) -> dict[str, Any]:
    parse_ok = sum(1 for r in task_rows if r.get("evaluation_ok") is True)
    exact_ok = sum(1 for r in task_rows if r.get("exact_match") is True)
    failed = sum(1 for r in task_rows if r.get("evaluation_ok") is not True)
    cfg = {
        "test_idx": sample_cfg.get("test_idx", 0),
        "model": sample_cfg.get("model"),
        "provider": sample_cfg.get("provider"),
        "reasoning_effort": sample_cfg.get("reasoning_effort"),
        "tools_enabled": sample_cfg.get("tools_enabled", False),
        "max_train_demos": sample_cfg.get("max_train_demos"),
        "out_dir": str(dest.resolve()),
        "run_tag": run_tag,
        "skip_completed": True,
        "skip": 0,
        "batch_size": batch_size,
        "delay_s": 0.0,
        "retries": sample_cfg.get("retries", 2),
        "retry_backoff_s": sample_cfg.get("retry_backoff_s", 1.5),
        "retry_eval_failures": sample_cfg.get("retry_eval_failures", True),
        "consolidated_arc_batch_index": batch_index,
        "note": "Summaries rebuilt by consolidate_arc_output_grid_run.py from per-task artifacts.",
    }
    return {
        "created_at": created_at,
        "mode": "output_grid_batch",
        "run_tag": run_tag,
        "ordered_source_count": ordered_count,
        "skipped_already_logged": skipped,
        "ran_task_count": len(task_rows),
        "completed_runs_recorded": len(task_rows),
        "config": cfg,
        "aggregate": {
            "parse_ok": parse_ok,
            "exact_match": exact_ok,
            "failed_or_incomplete": failed,
        },
        "results": task_rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dest", type=Path, required=True, help="Run directory (created if missing)")
    p.add_argument(
        "--source",
        type=Path,
        action="append",
        default=None,
        help="Source directory with per-task JSON (repeatable). Default: --dest when --in-place.",
    )
    p.add_argument("--run-tag", type=str, required=True)
    p.add_argument("--canonical-batch", type=Path, default=DEFAULT_CANONICAL_BATCH)
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Gather artifacts from --dest only (sources default to dest).",
    )
    p.add_argument(
        "--no-move-sources",
        action="store_true",
        help="Copy winners instead of moving when consolidating from another directory.",
    )
    p.add_argument(
        "--skip-html",
        action="store_true",
        help="Do not run build_output_grid_eval_html.py or write index.html.",
    )
    args = p.parse_args()

    dest = args.dest if args.dest.is_absolute() else ROOT_DIR / args.dest
    dest.mkdir(parents=True, exist_ok=True)
    arch = dest / "archive_intermediate_runs"
    arch.mkdir(parents=True, exist_ok=True)
    readme = arch / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Archived intermediate artifacts\n\n"
            "- Prior monolithic or partial `batch_*.json` snapshots from incremental runs.\n"
            "- Duplicate per-task JSON attempts (`duplicate_per_task_json/`), kept for debugging.\n\n"
            "Canonical summaries live next to this folder: `batch_arc_train_full.json`, "
            "`batch_arc_train_batch01.json` … `batch_arc_train_batch08.json`, plus "
            "`output_grid_run_overview.json`.\n",
            encoding="utf-8",
        )

    canonical_batch = (
        args.canonical_batch if args.canonical_batch.is_absolute() else ROOT_DIR / args.canonical_batch
    )
    order = _canonical_task_order(canonical_batch)

    if args.in_place:
        search_roots = [dest]
    elif args.source:
        search_roots = [s if s.is_absolute() else ROOT_DIR / s for s in args.source]
    else:
        search_roots = [dest]

    by_id = _gather_task_files(search_roots)
    missing = [tid for tid in order if tid not in by_id]
    if missing:
        raise SystemExit(f"Missing {len(missing)} task artifacts (showing up to 25): {missing[:25]}")

    dup_arch = arch / "duplicate_per_task_json"
    dup_arch.mkdir(parents=True, exist_ok=True)

    # Move archive old batch summaries (not our standard names)
    standard_batch_names = {JOINT_NAME} | {f"{BATCH_PREFIX}{i:02d}.json" for i in range(1, BATCH_COUNT + 1)}
    for root in {dest, *search_roots}:
        if not root.is_dir():
            continue
        for bp in root.glob("batch_*.json"):
            if bp.parent != root:
                continue
            if bp.name in standard_batch_names:
                continue
            target = arch / bp.name
            if target.exists():
                stem = bp.stem
                target = arch / f"{stem}_extra.json"
            shutil.move(str(bp), str(target))
            print(f"Archived intermediate batch → {target.relative_to(ROOT_DIR)}")

    winner_paths: dict[str, Path] = {}
    for tid in order:
        winner, losers = _pick_winner(by_id[tid])
        dest_path = dest / winner.name
        if winner.resolve() != dest_path.resolve():
            if args.no_move_sources:
                shutil.copy2(winner, dest_path)
            else:
                shutil.move(str(winner), str(dest_path))
            winner = dest_path
        else:
            winner = winner.resolve()
        winner_paths[tid] = Path(winner)
        for loser in losers:
            if loser.resolve() == winner_paths[tid].resolve():
                continue
            tgt = dup_arch / loser.name
            if tgt.exists():
                tgt = dup_arch / f"{loser.stem}_dup.json"
            shutil.move(str(loser), str(tgt))
            print(f"Archived duplicate artifact → {tgt.relative_to(ROOT_DIR)}")

    # Sample config from first task
    sample = _load_json(winner_paths[order[0]])

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    all_rows = [_result_row(task_id=tid, result_path=winner_paths[tid]) for tid in order]

    joint_path = dest / JOINT_NAME
    joint_path.write_text(
        json.dumps(
            _batch_payload(
                created_at=ts,
                run_tag=args.run_tag,
                ordered_count=400,
                skipped=0,
                task_rows=all_rows,
                dest=dest,
                sample_cfg=sample,
                batch_index=None,
                batch_size=400,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {joint_path.relative_to(ROOT_DIR)}")

    batch_paths: list[Path] = []
    for bi in range(BATCH_COUNT):
        chunk = all_rows[bi * BATCH_SIZE : (bi + 1) * BATCH_SIZE]
        if len(chunk) != BATCH_SIZE:
            raise SystemExit(f"Internal batch slice error: chunk {bi} len {len(chunk)}")
        bp = dest / f"{BATCH_PREFIX}{bi + 1:02d}.json"
        bp.write_text(
            json.dumps(
                _batch_payload(
                    created_at=ts,
                    run_tag=args.run_tag,
                    ordered_count=400,
                    skipped=0,
                    task_rows=chunk,
                    dest=dest,
                    sample_cfg=sample,
                    batch_index=bi + 1,
                    batch_size=BATCH_SIZE,
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        batch_paths.append(bp)
        print(f"Wrote {bp.relative_to(ROOT_DIR)}")

    exact_ok = sum(1 for r in all_rows if r.get("exact_match") is True)
    eval_false_ids = [r["task_id"] for r in all_rows if r.get("evaluation_ok") is not True]
    mismatch_ids = [r["task_id"] for r in all_rows if r.get("exact_match") is False]

    overview = {
        "created_at": iso,
        "run_tag": args.run_tag,
        "run_dir": str(dest.resolve()),
        "dataset": "ARC original train",
        "canonical_task_count": 400,
        "per_task_json_found": 400,
        "missing_task_ids": [],
        "all_evaluation_ok": len(eval_false_ids) == 0,
        "evaluation_ok_false_count": len(eval_false_ids),
        "evaluation_ok_false": eval_false_ids,
        "exact_match_true": exact_ok,
        "exact_match_false": 400 - exact_ok,
        "exact_match_accuracy": round(exact_ok / 400, 6),
        "exact_match_percent": round(100 * exact_ok / 400, 4),
        "exact_match_false_task_ids": mismatch_ids,
        "batch_summary_path": str(joint_path.resolve()),
        "batch_summary_paths": [str(p.resolve()) for p in batch_paths],
        "batch_size": BATCH_SIZE,
        "batch_count": BATCH_COUNT,
    }
    ov_json = dest / "output_grid_run_overview.json"
    ov_json.write_text(json.dumps(overview, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {ov_json.relative_to(ROOT_DIR)}")

    ov_md = dest / "output_grid_run_overview.md"
    mismatch_line = " ".join(mismatch_ids)
    ov_md.write_text(
        f"# Output grid run — performance overview\n\n"
        f"- **Run tag:** `{args.run_tag}`\n"
        f"- **Model:** `{sample.get('model')}` ({sample.get('provider')})\n"
        f"- **Tasks:** 400 (canonical ARC train order)\n"
        f"- **Generated:** {iso} (UTC)\n\n"
        f"## Parse / logging health\n\n"
        f"- **Per-task JSON on disk:** 400/400\n"
        f"- **Missing artifacts:** 0\n"
        f"- **All `evaluation_ok`:** **{'yes' if not eval_false_ids else 'no'}**\n\n"
        f"## Accuracy (test index {sample.get('test_idx', 0)})\n\n"
        f"- **Exact match:** {exact_ok}/400 (**{100 * exact_ok / 400:.2f}%**)\n"
        f"- **Not exact match:** {400 - exact_ok}/400\n\n"
        f"### Batch summaries (50 tasks each)\n\n"
        + "\n".join(f"- `{p.name}`" for p in batch_paths)
        + "\n\n"
        f"### Joint batch JSON\n\n"
        f"- `{JOINT_NAME}`\n\n"
        f"### Task ids with `exact_match: false`\n\n```\n{mismatch_line}\n```\n",
        encoding="utf-8",
    )
    print(f"Wrote {ov_md.relative_to(ROOT_DIR)}")

    if not args.skip_html:
        html_tool = ROOT_DIR / "scripts" / "build_output_grid_eval_html.py"
        out_all = dest / "output_grid_report_all.html"
        out_mm = dest / "output_grid_report_mismatches.html"
        subprocess.run(
            [sys.executable, str(html_tool), "--batch-json", str(joint_path), "--out", str(out_all)],
            check=True,
            cwd=str(ROOT_DIR),
        )
        subprocess.run(
            [
                sys.executable,
                str(html_tool),
                "--batch-json",
                str(joint_path),
                "--out",
                str(out_mm),
                "--only-mismatches",
            ],
            check=True,
            cwd=str(ROOT_DIR),
        )

        batch_links = []
        for bp in batch_paths:
            stem = bp.stem
            rep = dest / f"{stem}_report.html"
            mm_rep = dest / f"{stem}_report_mismatches.html"
            subprocess.run(
                [sys.executable, str(html_tool), "--batch-json", str(bp), "--out", str(rep)],
                check=True,
                cwd=str(ROOT_DIR),
            )
            subprocess.run(
                [
                    sys.executable,
                    str(html_tool),
                    "--batch-json",
                    str(bp),
                    "--out",
                    str(mm_rep),
                    "--only-mismatches",
                ],
                check=True,
                cwd=str(ROOT_DIR),
            )
            batch_links.append((bp.name, rep.name, mm_rep.name))

        idx = dest / "index.html"
        batch_rows = "".join(
            f"<li><code>{bn}</code>: <a href=\"{r}\">all</a> · <a href=\"{m}\">mismatches</a></li>\n"
            for bn, r, m in batch_links
        )
        idx.write_text(
            f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ARC output grid — {args.run_tag}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 1.5rem; max-width: 900px; }}
    h1 {{ font-size: 1.35rem; }}
    ul {{ line-height: 1.9; }}
    code {{ background: #f4f4f4; padding: 0.1rem 0.3rem; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>ARC train output grid evaluation</h1>
  <p>Run: <code>{args.run_tag}</code></p>
  <ul>
    <li><a href="output_grid_report_all.html">HTML report (all 400 tasks)</a></li>
    <li><a href="output_grid_report_mismatches.html">HTML report (mismatches only)</a></li>
    <li><a href="{JOINT_NAME}">Joint batch summary JSON (400 tasks)</a></li>
    <li><a href="output_grid_run_overview.json">Overview JSON</a></li>
    <li><a href="output_grid_run_overview.md">Overview Markdown</a></li>
  </ul>
  <h2>Batches of {BATCH_SIZE}</h2>
  <ul>
{batch_rows}
  </ul>
</body>
</html>
""",
            encoding="utf-8",
        )
        print(f"Wrote {idx.relative_to(ROOT_DIR)}")

    # Remove empty source dirs (only when single external source fully drained)
    if not args.in_place and args.source and len(args.source) == 1:
        src = args.source[0] if args.source[0].is_absolute() else ROOT_DIR / args.source[0]
        if src.is_dir() and src.resolve() != dest.resolve():
            leftovers = list(src.iterdir())
            if not leftovers:
                src.rmdir()
                print(f"Removed empty source dir {src.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
