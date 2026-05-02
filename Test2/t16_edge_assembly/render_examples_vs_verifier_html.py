#!/usr/bin/env python3
"""Build t16_examples_vs_verifier.html from Test2/t16.json (do not edit that JSON here).

Runs ``transform_t16`` from this directory's ``generator`` (verifier logic) and compares
each author-labeled output to the verifier output. Re-run after changing the verifier:

    python Test2/t16_edge_assembly/render_examples_vs_verifier_html.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_T16_JSON = _ROOT / "Test2" / "t16.json"
_OUT_HTML = Path(__file__).resolve().parent / "t16_examples_vs_verifier.html"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generator import transform_t16  # noqa: E402


def _diff_mask(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    return [[0 if a[r][c] == b[r][c] else 9 for c in range(len(a[0]))] for r in range(len(a))]


def _count_diff(a: list[list[int]], b: list[list[int]]) -> int:
    return sum(1 for r in range(len(a)) for c in range(len(a[0])) if a[r][c] != b[r][c])


def main() -> None:
    payload = json.loads(_T16_JSON.read_text(encoding="utf-8"))
    train = payload["train"]
    test = payload["test"]
    cases: list[dict] = []

    for i, ex in enumerate(train):
        inp = ex["input"]
        author = ex["output"]
        ver = transform_t16([row[:] for row in inp])
        cases.append(
            {
                "kind": "train",
                "index": i,
                "input": inp,
                "author": author,
                "verifier": ver,
                "diff": _diff_mask(author, ver),
                "diff_n": _count_diff(author, ver),
            }
        )
    for i, ex in enumerate(test):
        inp = ex["input"]
        author = ex["output"]
        ver = transform_t16([row[:] for row in inp])
        cases.append(
            {
                "kind": "test",
                "index": i,
                "input": inp,
                "author": author,
                "verifier": ver,
                "diff": _diff_mask(author, ver),
                "diff_n": _count_diff(author, ver),
            }
        )

    data_json = json.dumps(cases, separators=(",", ":"))

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Test2 t16 — your examples vs verifier</title>
</head>
<body style="font-family:Segoe UI,sans-serif;background:#1a1a1a;color:#eee;max-width:1400px;margin:24px auto;">
  <h1>Test2 t16 — labeled examples vs <code>transform_t16</code></h1>
  <p style="opacity:0.9;">
    Grids come from <code>Test2/t16.json</code> only. Verifier column is computed by this task&rsquo;s
    <code>generator.transform_t16</code> at render time. Regenerate with
    <code>python Test2/t16_edge_assembly/render_examples_vs_verifier_html.py</code>.
  </p>
  <div id="summary" style="margin-bottom:14px;"></div>
  <div id="sections"></div>
  <script>
    const cases = {data_json};

    const palette = {{
      0: "#000000", 1: "#1E90FF", 2: "#FF4136", 3: "#2ECC40", 4: "#FFDC00",
      5: "#AAAAAA", 6: "#F39C12", 7: "#9B59B6", 8: "#00CED1", 9: "#FF69B4"
    }};

    function makeGrid(grid, title) {{
      const wrap = document.createElement("div");
      wrap.style.margin = "8px";
      const label = document.createElement("div");
      label.style.fontWeight = "600";
      label.style.marginBottom = "4px";
      label.textContent = title;
      wrap.appendChild(label);
      const box = document.createElement("div");
      for (let r = 0; r < grid.length; r++) {{
        const row = document.createElement("div");
        row.style.whiteSpace = "nowrap";
        for (let c = 0; c < grid[r].length; c++) {{
          const v = grid[r][c];
          const cell = document.createElement("div");
          cell.style.width = "16px";
          cell.style.height = "16px";
          cell.style.background = palette[v] || "#444";
          cell.style.border = "1px solid #555";
          cell.style.boxSizing = "border-box";
          cell.style.display = "inline-block";
          cell.style.verticalAlign = "top";
          row.appendChild(cell);
        }}
        box.appendChild(row);
      }}
      wrap.appendChild(box);
      return wrap;
    }}

    const root = document.getElementById("sections");
    let mism = 0;
    for (const ex of cases) {{
      if (ex.diff_n > 0) mism++;
      const section = document.createElement("section");
      section.style.border = "1px solid #444";
      section.style.padding = "10px";
      section.style.borderRadius = "8px";
      section.style.marginBottom = "14px";
      const h3 = document.createElement("h3");
      h3.style.margin = "4px 0 8px";
      const lab = ex.kind === "train" ? `Train ${{ex.index}}` : `Test ${{ex.index}}`;
      h3.textContent = lab + (ex.diff_n === 0 ? " — match" : ` — ${{ex.diff_n}} cells differ (author vs verifier)`);
      section.appendChild(h3);
      const row = document.createElement("div");
      row.style.display = "flex";
      row.style.flexWrap = "wrap";
      row.style.gap = "14px";
      row.style.alignItems = "flex-start";
      row.appendChild(makeGrid(ex.input, "Input"));
      row.appendChild(makeGrid(ex.author, "Your output (t16.json)"));
      row.appendChild(makeGrid(ex.verifier, "Verifier (transform_t16)"));
      row.appendChild(makeGrid(ex.diff, "Diff (pink = author ≠ verifier)"));
      section.appendChild(row);
      root.appendChild(section);
    }}
    document.getElementById("summary").textContent =
      mism === 0
        ? "All listed examples match the verifier."
        : `${{mism}} example(s) differ from the verifier.`;
  </script>
</body>
</html>
"""

    _OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote {_OUT_HTML.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
