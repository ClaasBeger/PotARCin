from __future__ import annotations

import argparse
import json
import sys
from html import escape
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from framework.grids import Grid

# Standard ARC colour palette
_SYMBOL_TO_COLOR: dict[int, str] = {
    0: "#000000",
    1: "#0074D9",
    2: "#FF4136",
    3: "#2ECC40",
    4: "#FFDC00",
    5: "#AAAAAA",
    6: "#F012BE",
    7: "#FF851B",
    8: "#7FDBFF",
    9: "#870C25",
}


def _cell_px(grid: Grid, *, max_px: int = 22) -> int:
    h = max(1, len(grid))
    w = max(1, len(grid[0]) if grid else 1)
    return int(max(8, min(max_px, 420 / max(h, w))))


def _mismatch_positions(gt: Grid, pred: Grid | None) -> tuple[set[tuple[int, int]] | None, str | None]:
    """Return (positions where pred differs from gt, or None if incomparable) and optional note."""
    if pred is None:
        return None, "No parsed prediction"
    if len(pred) != len(gt):
        return None, f"Shape mismatch: GT {len(gt)}×{len(gt[0]) if gt else 0} vs pred {len(pred)}×{len(pred[0]) if pred else 0}"
    w0 = len(gt[0]) if gt else 0
    bad: set[tuple[int, int]] = set()
    for r, row in enumerate(pred):
        if r >= len(gt) or len(row) != len(gt[r]) or len(gt[r]) != w0:
            return None, "Jagged or width mismatch vs ground truth"
        for c, v in enumerate(row):
            if int(v) != int(gt[r][c]):
                bad.add((r, c))
    return bad, None


def _grid_html(
    grid: Grid,
    title: str,
    *,
    highlight: set[tuple[int, int]] | None = None,
    max_px: int = 22,
) -> str:
    size = _cell_px(grid, max_px=max_px)
    h = len(grid)
    rows_html: list[str] = []
    for r in range(h):
        cells: list[str] = []
        for c in range(len(grid[r])):
            sym = int(grid[r][c])
            color = _SYMBOL_TO_COLOR.get(sym, "#000000")
            extra = ""
            if highlight is not None and (r, c) in highlight:
                extra = " box-shadow: inset 0 0 0 2px #ff0055;"
            cells.append(
                f"<div class='arc-cell' style='background:{color};width:{size}px;height:{size}px;{extra}'></div>"
            )
        rows_html.append("<div class='arc-row'>" + "".join(cells) + "</div>")
    return (
        f"<div class='panel'><div class='panel-title'>{escape(title)}</div>"
        f"<div class='arc-grid'>{''.join(rows_html)}</div></div>"
    )


def _section_html(
    *,
    task_id: str,
    gt: Grid,
    pred: Grid | None,
    exact_match: bool | None,
    eval_ok: bool | None,
    result_path: str | None,
    raw_snippet: str | None,
) -> str:
    highlights, shape_note = _mismatch_positions(gt, pred)
    status_parts: list[str] = []
    if exact_match is True:
        status_parts.append('<span class="ok">exact match</span>')
    elif exact_match is False:
        status_parts.append('<span class="bad">not exact match</span>')
    else:
        status_parts.append('<span class="muted">no exact_match flag</span>')
    if eval_ok is False:
        status_parts.append('<span class="bad">evaluation_ok false</span>')
    status = " · ".join(status_parts)
    path_line = f"<div class='path'>{escape(result_path or '')}</div>" if result_path else ""

    grids = '<div class="pair">'
    grids += _grid_html(gt, "Ground truth (test output)")
    if pred is not None and shape_note is None:
        grids += _grid_html(pred, "Model prediction", highlight=highlights)
    elif pred is not None and shape_note:
        grids += f"<div class='panel'><div class='panel-title'>Model prediction</div><p class='warn'>{escape(shape_note)}</p>"
        grids += _grid_html(pred, "Prediction (raw shape)", highlight=None)
        grids += "</div>"
    else:
        grids += f"<div class='panel'><div class='panel-title'>Model prediction</div><p class='warn'>{escape(shape_note or 'Missing prediction')}</p></div>"
    grids += "</div>"

    mismatch_count = len(highlights) if highlights is not None else None
    meta = ""
    if mismatch_count is not None and exact_match is False:
        meta = f"<p class='meta'>{mismatch_count} cell(s) differ from ground truth.</p>"
    elif shape_note:
        meta = f"<p class='meta warn'>{escape(shape_note)}</p>"

    raw_block = ""
    if raw_snippet:
        raw_block = f"<details class='raw'><summary>Raw model response (truncated)</summary><pre>{escape(raw_snippet)}</pre></details>"

    return f"""
<section class="task" id="task-{escape(task_id)}">
  <h2><code>{escape(task_id)}</code></h2>
  <p class="status">{status}</p>
  {path_line}
  {meta}
  {grids}
  {raw_block}
</section>
"""


def _page_html(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 1.25rem 1.5rem; max-width: 1200px; }}
    h1 {{ font-size: 1.35rem; }}
    h2 {{ font-size: 1.1rem; margin-top: 1.75rem; border-top: 1px solid #ddd; padding-top: 1rem; }}
    .ok {{ color: #0a7; font-weight: 600; }}
    .bad {{ color: #c12; font-weight: 600; }}
    .muted {{ color: #666; }}
    .path {{ font-size: 0.75rem; color: #555; word-break: break-all; margin: 0.25rem 0; }}
    .pair {{ display: flex; flex-wrap: wrap; gap: 1.5rem; align-items: flex-start; margin: 0.75rem 0; }}
    .panel {{ flex: 0 0 auto; }}
    .panel-title {{ font-weight: 600; font-size: 0.9rem; margin-bottom: 0.35rem; }}
    .arc-grid {{ display: inline-block; border: 1px solid rgba(120,120,120,0.75); background: #222; }}
    .arc-row {{ display: flex; }}
    .arc-cell {{
      box-sizing: border-box;
      border-right: 1px solid rgba(85,85,85,0.85);
      border-bottom: 1px solid rgba(85,85,85,0.85);
    }}
    .arc-row:last-child .arc-cell {{ border-bottom: 0; }}
    .arc-cell:last-child {{ border-right: 0; }}
    .meta {{ font-size: 0.9rem; margin: 0.35rem 0; }}
    .warn {{ color: #a50; }}
    details.raw {{ margin-top: 0.75rem; font-size: 0.85rem; }}
    details.raw pre {{ white-space: pre-wrap; word-break: break-word; background: #f4f4f4; padding: 0.75rem; border-radius: 6px; max-height: 200px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  {body}
</body>
</html>
"""


def _load_result(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an HTML report from an output-grid batch summary (GT vs prediction per task)."
    )
    parser.add_argument(
        "--batch-json",
        type=Path,
        required=True,
        help="Path to batch_*.json from run_output_grid_eval_batch.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output HTML path (default: next to batch JSON with _report.html suffix)",
    )
    parser.add_argument(
        "--raw-max-chars",
        type=int,
        default=1200,
        help="Max chars of raw_response to embed per task (0 = omit)",
    )
    parser.add_argument(
        "--only-mismatches",
        action="store_true",
        help="Include only tasks where exact_match is false",
    )
    args = parser.parse_args()

    batch_path = args.batch_json
    if not batch_path.is_absolute():
        batch_path = ROOT_DIR / batch_path
    if not batch_path.is_file():
        raise SystemExit(f"Not a file: {batch_path}")

    batch = _load_result(batch_path)
    results = batch.get("results")
    if not isinstance(results, list):
        raise SystemExit("batch JSON missing results[]")

    out_path = args.out
    if out_path is None:
        out_path = batch_path.with_name(batch_path.stem + "_report.html")
    elif not out_path.is_absolute():
        out_path = ROOT_DIR / out_path

    sections: list[str] = []
    for row in results:
        if not isinstance(row, dict):
            continue
        task_id = str(row.get("task_id", ""))
        rp = row.get("result_path")
        if not task_id or not isinstance(rp, str):
            continue
        rpath = Path(rp)
        if not rpath.is_file():
            sections.append(
                f"<section class='task'><h2><code>{escape(task_id)}</code></h2>"
                f"<p class='warn'>Missing result file: {escape(rp)}</p></section>"
            )
            continue
        data = _load_result(rpath)
        gt = data.get("ground_truth")
        pred = data.get("prediction")
        if not isinstance(gt, list):
            continue
        exact_match = row.get("exact_match") if "exact_match" in row else data.get("exact_match")
        eval_ok = row.get("evaluation_ok") if "evaluation_ok" in row else data.get("evaluation_ok")
        if args.only_mismatches and exact_match is not False:
            continue
        raw = data.get("raw_response")
        raw_snippet = None
        if args.raw_max_chars > 0 and isinstance(raw, str) and raw.strip():
            raw_snippet = raw[: args.raw_max_chars]
            if len(raw) > args.raw_max_chars:
                raw_snippet += "\n…"

        sections.append(
            _section_html(
                task_id=task_id,
                gt=gt,  # type: ignore[arg-type]
                pred=pred if isinstance(pred, list) else None,
                exact_match=exact_match,
                eval_ok=eval_ok,
                result_path=rp,
                raw_snippet=raw_snippet,
            )
        )

    summary_bits = []
    if isinstance(batch.get("aggregate"), dict):
        agg = batch["aggregate"]
        summary_bits.append(
            f"parse_ok≈{agg.get('parse_ok')}, exact_match count≈{agg.get('exact_match')}, "
            f"failed≈{agg.get('failed_or_incomplete')}"
        )
    summary_bits.append(f"{len(sections)} task section(s)")
    intro = "<p>" + escape(" · ".join(summary_bits)) + "</p>"

    title = f"Output grid eval — {batch.get('run_tag', batch_path.parent.name)}"
    html = _page_html(title, intro + "\n" + "\n".join(sections))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
