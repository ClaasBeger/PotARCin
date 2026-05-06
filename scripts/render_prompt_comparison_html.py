#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

GRID_LINE_RE = re.compile(r"^\s*\d+(?:\s+\d+)*\s*$")


def _is_grid_line(line: str) -> bool:
    return bool(GRID_LINE_RE.match(line))


def _render_grid(lines: list[str]) -> str:
    rows = [[int(x) for x in ln.strip().split()] for ln in lines if ln.strip()]
    if not rows:
        return ""
    cells = []
    for r in rows:
        tds = "".join(f"<td class='c{v}'>{v}</td>" for v in r)
        cells.append(f"<tr>{tds}</tr>")
    return f"<table class='grid'>{''.join(cells)}</table>"


def _render_prompt_with_grids(prompt: str) -> str:
    lines = prompt.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        if _is_grid_line(lines[i]):
            j = i
            block: list[str] = []
            while j < len(lines) and _is_grid_line(lines[j]):
                block.append(lines[j])
                j += 1
            out.append("<div class='grid-wrap'>")
            out.append(_render_grid(block))
            out.append("</div>")
            i = j
            continue
        line = lines[i]
        if line.strip():
            out.append(f"<div class='txt'>{html.escape(line)}</div>")
        else:
            out.append("<div class='sp'></div>")
        i += 1
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Render side-by-side prompt comparison HTML with grid visualization.")
    ap.add_argument(
        "--parity-csv",
        type=Path,
        default=Path("evaluation/figures/request_parity/request_parity.csv"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/figures/request_parity/prompt_comparison.html"),
    )
    args = ap.parse_args()

    csv_path = args.parity_csv if args.parity_csv.is_absolute() else (ROOT_DIR / args.parity_csv)
    out_path = args.output if args.output.is_absolute() else (ROOT_DIR / args.output)

    rows_html: list[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        for row in rd:
            task_id = row["task_id"]
            dim = row["dimension"]
            match = row.get("prompt_hash_match", "")
            op = Path(row["original_prompt_file"])
            pp = Path(row["planned_prompt_file"])
            op_txt = op.read_text(encoding="utf-8") if op.is_file() else ""
            pp_txt = pp.read_text(encoding="utf-8") if pp.is_file() else ""

            left = _render_prompt_with_grids(op_txt)
            right = _render_prompt_with_grids(pp_txt)

            rows_html.append(
                "<div class='card'>"
                f"<div class='head'><span><b>{html.escape(task_id)}</b> · {html.escape(dim)}</span>"
                f"<span class='match'>hash_match={html.escape(match)}</span></div>"
                "<div class='cols'>"
                f"<div class='col'><div class='lbl'>Original</div>{left}</div>"
                f"<div class='col'><div class='lbl'>Planned</div>{right}</div>"
                "</div></div>"
            )

    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Prompt Comparison</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; margin: 16px; background: #f8fafc; }}
    .card {{ background: white; border: 1px solid #d1d5db; border-radius: 8px; margin-bottom: 14px; overflow: hidden; }}
    .head {{ display:flex; justify-content:space-between; align-items:center; background:#f3f4f6; padding:8px 10px; font-size:13px; }}
    .match {{ color:#374151; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .cols {{ display:grid; grid-template-columns: 1fr 1fr; gap: 10px; padding: 10px; }}
    .col {{ border:1px solid #e5e7eb; border-radius:6px; padding:8px; background:#fcfcfd; overflow:auto; max-height:520px; }}
    .lbl {{ font-weight:600; margin-bottom:6px; color:#111827; }}
    .txt {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; line-height:1.35; color:#1f2937; white-space: pre-wrap; }}
    .sp {{ height:7px; }}
    .grid-wrap {{ margin: 6px 0 9px; }}
    table.grid {{ border-collapse: collapse; }}
    table.grid td {{ width: 19px; height: 19px; text-align:center; font-size:10px; border:1px solid #d1d5db; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .c0{{background:#ffffff}} .c1{{background:#1f77b4;color:#fff}} .c2{{background:#ff7f0e}}
    .c3{{background:#2ca02c;color:#fff}} .c4{{background:#d62728;color:#fff}} .c5{{background:#9467bd;color:#fff}}
    .c6{{background:#8c564b;color:#fff}} .c7{{background:#e377c2}} .c8{{background:#7f7f7f;color:#fff}}
    .c9{{background:#bcbd22}}
  </style>
</head>
<body>
  <h2>Original vs Planned Prompts (Grid Rendered)</h2>
  {''.join(rows_html)}
</body>
</html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    print("Wrote:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

