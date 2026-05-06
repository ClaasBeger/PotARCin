#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from framework.eval_paths import dimension_out_dirs, multi_run_summary_dir

DIMENSIONS: list[tuple[str, str]] = [
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


def _latest_dim_file(dim_dir: Path, task_id: str, suffix: str) -> Path | None:
    matches = sorted(dim_dir.glob(f"{task_id}_{suffix}_*.json"))
    return matches[-1] if matches else None


def _seed_map_from_source_run(model: str, source_run_tag: str) -> dict[str, int]:
    unified = multi_run_summary_dir(root_dir=ROOT_DIR, model=model, run_tag=source_run_tag) / "run_all_400_unified.json"
    data = _read_json(unified)
    runs = data.get("runs")
    out: dict[str, int] = {}
    if isinstance(runs, list):
        for r in runs:
            if not isinstance(r, dict):
                continue
            tid = r.get("task_id")
            sd = r.get("seed")
            if isinstance(tid, str) and isinstance(sd, int):
                out[tid] = int(sd)
    return out


def _load_original_requests(
    *,
    task_ids: list[str],
    model: str,
    source_run_tag: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    dirs = dimension_out_dirs(root_dir=ROOT_DIR, model=model, run_tag=source_run_tag)
    out: dict[str, dict[str, dict[str, Any]]] = {}

    for tid in task_ids:
        out[tid] = {}
        for dim, suffix in DIMENSIONS:
            p = _latest_dim_file(dirs[dim], tid, suffix)
            if p is None:
                out[tid][dim] = {"status": "missing"}
                continue
            payload = _read_json(p)
            prompt = payload.get("prompt")
            if not isinstance(prompt, str):
                prompt = None
            out[tid][dim] = {
                "status": "ok",
                "file": str(p),
                "seed": payload.get("seed"),
                "prompt_hash": _prompt_hash(prompt),
                "prompt_preview": (prompt[:280] + " ...") if isinstance(prompt, str) and len(prompt) > 280 else (prompt or "-"),
                "arc_gen_dynamic_seed": payload.get("arc_gen_dynamic_seed"),
                "arc_gen_dynamic_n": payload.get("arc_gen_dynamic_n"),
                "definition_mode": payload.get("definition_mode"),
            }
    return out


def _build_html(
    *,
    subset_path: Path,
    source_run_tag: str,
    repeat_run_tag: str,
    model: str,
    repeats: int,
    task_ids: list[str],
    seed_map: dict[str, int],
    original: dict[str, dict[str, dict[str, Any]]],
) -> str:
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for tid in task_ids:
        base_seed = seed_map.get(tid, "missing")
        for dim, _suffix in DIMENSIONS:
            d = original.get(tid, {}).get(dim, {})
            status = d.get("status", "missing")
            file_s = html.escape(str(d.get("file", "-")))
            seed_s = html.escape(str(d.get("seed", "-")))
            ph = html.escape(str(d.get("prompt_hash", "-")))
            preview = html.escape(str(d.get("prompt_preview", "-")))
            def_extra = ""
            if dim == "definition":
                def_extra = (
                    f"<div><b>definition_mode:</b> {html.escape(str(d.get('definition_mode', '-')))}</div>"
                    f"<div><b>arc_gen_dynamic_n:</b> {html.escape(str(d.get('arc_gen_dynamic_n', '-')))}</div>"
                    f"<div><b>arc_gen_dynamic_seed:</b> {html.escape(str(d.get('arc_gen_dynamic_seed', '-')))}</div>"
                )

            if status != "ok":
                rows.append(
                    f"<tr><td>{tid}</td><td>{dim}</td><td colspan='2'><span class='warn'>Missing original artifact</span></td></tr>"
                )
                continue

            planned = (
                f"<div><b>repeat run-tag:</b> {html.escape(repeat_run_tag)}</div>"
                f"<div><b>repeats:</b> {repeats}</div>"
                f"<div><b>planned fixed seed per repeat:</b> {html.escape(str(base_seed))}</div>"
                f"<div><b>expected request parity:</b> same task + same seed + same eval pipeline</div>"
            )
            rows.append(
                "<tr>"
                f"<td>{tid}</td>"
                f"<td>{dim}</td>"
                "<td>"
                f"<div><b>seed:</b> {seed_s}</div>"
                f"<div><b>prompt_hash:</b> {ph}</div>"
                f"<div><b>artifact:</b> <code>{file_s}</code></div>"
                f"{def_extra}"
                f"<details><summary>Prompt preview</summary><pre>{preview}</pre></details>"
                "</td>"
                f"<td>{planned}</td>"
                "</tr>"
            )

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Request Inspection — Original vs Repeat Plan</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; margin: 18px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px; vertical-align: top; }}
    th {{ background: #f9fafb; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #f8fafc; padding: 8px; border: 1px solid #e5e7eb; }}
    .meta {{ margin-bottom: 12px; color: #374151; }}
    .warn {{ color: #b91c1c; font-weight: 600; }}
  </style>
</head>
<body>
  <h2>Request Inspection: Original Run vs Repeat-10 Plan</h2>
  <div class="meta">
    <div><b>Generated:</b> {html.escape(ts)}</div>
    <div><b>Model:</b> {html.escape(model)}</div>
    <div><b>Subset:</b> <code>{html.escape(str(subset_path))}</code></div>
    <div><b>Source run-tag:</b> {html.escape(source_run_tag)}</div>
    <div><b>Repeat run-tag:</b> {html.escape(repeat_run_tag)}</div>
    <div><b>Task count:</b> {len(task_ids)}</div>
  </div>
  <table>
    <thead>
      <tr>
        <th style="width: 10%;">Task</th>
        <th style="width: 10%;">Dimension</th>
        <th style="width: 40%;">Original Request Artifact</th>
        <th style="width: 40%;">Planned Repeat Request (no variation)</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Build HTML inspection of original vs repeat request parity.")
    ap.add_argument("--subset-json", type=Path, default=Path("Test2/ARC1_subsets/gpt54_nontrivial_success_subset_20_manual.json"))
    ap.add_argument("--model", type=str, default="gpt-5.4")
    ap.add_argument("--source-run-tag", type=str, default="Evaluation_Run_2026-04-14_gpt54_low")
    ap.add_argument("--repeat-run-tag", type=str, default="Evaluation_Run_gpt54_subset20_repeat10_same_request")
    ap.add_argument("--num-repeats", type=int, default=10)
    ap.add_argument("--output", type=Path, default=Path("evaluation/figures/gpt54_repeat_request_inspection.html"))
    args = ap.parse_args()

    subset_path = args.subset_json if args.subset_json.is_absolute() else (ROOT_DIR / args.subset_json)
    subset = _read_json(subset_path)
    task_ids = subset.get("selected_task_ids")
    if not isinstance(task_ids, list) or not all(isinstance(t, str) for t in task_ids):
        raise SystemExit("subset json must contain selected_task_ids")

    seed_map = _seed_map_from_source_run(args.model, args.source_run_tag)
    original = _load_original_requests(task_ids=task_ids, model=args.model, source_run_tag=args.source_run_tag)
    html_text = _build_html(
        subset_path=subset_path.resolve(),
        source_run_tag=args.source_run_tag,
        repeat_run_tag=args.repeat_run_tag,
        model=args.model,
        repeats=int(args.num_repeats),
        task_ids=task_ids,
        seed_map=seed_map,
        original=original,
    )
    out = args.output if args.output.is_absolute() else (ROOT_DIR / args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")
    print("Wrote:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

