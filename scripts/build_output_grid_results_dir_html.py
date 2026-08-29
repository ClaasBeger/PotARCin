#!/usr/bin/env python3
"""Build a simple GT-vs-prediction HTML report from a directory of output-grid JSONs.

Example:
  python scripts/build_output_grid_results_dir_html.py \\
    --results-dir evaluation/output_grid_eval_results/Llama4Maverick/ARC_AGI1_evaluation/output_grid_eval_set
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.build_output_grid_eval_html import (  # noqa: E402
    _page_html,
    _section_html,
)


def _latest_by_task(results_dir: Path) -> list[tuple[str, Path, dict[str, Any]]]:
    by: dict[str, list[Path]] = {}
    for p in results_dir.glob("*_output_grid_eval_*.json"):
        if "_error_" in p.name or p.name.startswith("summary_"):
            continue
        tid = p.name.split("_output_grid")[0]
        by.setdefault(tid, []).append(p)
    out: list[tuple[str, Path, dict[str, Any]]] = []
    for tid in sorted(by):
        p = max(by[tid], key=lambda x: x.stat().st_mtime)
        data = json.loads(p.read_text(encoding="utf-8"))
        out.append((tid, p, data))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HTML visualization: ground-truth test output vs model prediction."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing *_output_grid_eval_*.json files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output HTML path (default: <results-dir>/compare_gt_vs_pred.html)",
    )
    parser.add_argument(
        "--only-mismatches",
        action="store_true",
        help="Include only tasks where exact_match is false",
    )
    parser.add_argument(
        "--only-matches",
        action="store_true",
        help="Include only tasks where exact_match is true",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of task sections (after filters)",
    )
    parser.add_argument(
        "--raw-max-chars",
        type=int,
        default=800,
        help="Max chars of raw_response to embed (0 = omit)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir if args.results_dir.is_absolute() else ROOT_DIR / args.results_dir
    if not results_dir.is_dir():
        raise SystemExit(f"Not a directory: {results_dir}")

    out_path = args.out
    if out_path is None:
        out_path = results_dir / "compare_gt_vs_pred.html"
    elif not out_path.is_absolute():
        out_path = ROOT_DIR / out_path

    items = _latest_by_task(results_dir)
    n_exact = sum(1 for _, _, d in items if d.get("exact_match") is True)
    n_ok = sum(1 for _, _, d in items if d.get("evaluation_ok") is True)

    sections: list[str] = []
    for tid, path, data in items:
        exact = data.get("exact_match")
        if args.only_mismatches and exact is not False:
            continue
        if args.only_matches and exact is not True:
            continue
        gt = data.get("ground_truth")
        if not isinstance(gt, list):
            continue
        pred = data.get("prediction")
        raw = data.get("raw_response")
        raw_snippet = None
        if args.raw_max_chars > 0 and isinstance(raw, str) and raw.strip():
            raw_snippet = raw[: args.raw_max_chars]
            if len(raw) > args.raw_max_chars:
                raw_snippet += "\n…"
        sections.append(
            _section_html(
                task_id=tid,
                gt=gt,
                pred=pred if isinstance(pred, list) else None,
                exact_match=exact,
                eval_ok=data.get("evaluation_ok"),
                result_path=str(path),
                raw_snippet=raw_snippet,
            )
        )
        if args.limit is not None and len(sections) >= args.limit:
            break

    intro = (
        f"<p><b>{len(items)}</b> tasks with results · "
        f"exact_match <b>{n_exact}/{len(items)}</b> · "
        f"evaluation_ok <b>{n_ok}/{len(items)}</b> · "
        f"showing <b>{len(sections)}</b> section(s)</p>"
        f"<p class='muted'>Mismatch cells are highlighted with a pink inset border on the prediction.</p>"
    )
    title = f"GT vs prediction — {results_dir.name}"
    html = _page_html(title, intro + "\n" + "\n".join(sections))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
