"""
Generate 50 t50 samples, cross-verify each one, and write:
  • t50_samples_50.json   — {"train": [...50 pairs...]}
  • t50_samples_50.html   — visual preview
"""

from __future__ import annotations

import html as _html
import json
import random
from pathlib import Path

from generator import generate
from verifier  import verify_t50

# --------------------------------------------------------------------------- #
# Colour palette (matches the task's integer → RGB mapping)                   #
# --------------------------------------------------------------------------- #
_HEX = {
    0: "#000000",
    1: "#0069CF",   # blue
    2: "#FF3937",   # red
    3: "#00C543",   # green
    4: "#FFD731",   # yellow
    5: "#A0A0A0",   # gray
    6: "#F916B3",   # magenta
    7: "#FF7A2C",   # orange
    8: "#63D6FC",   # cyan
    9: "#820F23",   # brown
}


# --------------------------------------------------------------------------- #
# HTML helpers                                                                 #
# --------------------------------------------------------------------------- #

def _grid_html(grid: list, label: str, cell: int = 13) -> str:
    rows_html = []
    for row in grid:
        cells = "".join(
            f'<span style="display:inline-block;width:{cell}px;height:{cell}px;'
            f'background:{_HEX.get(v, "#000")};border:1px solid #2a2a2a;'
            f'box-sizing:border-box;vertical-align:top;"></span>'
            for v in row
        )
        rows_html.append(f"<div>{cells}</div>")
    escaped = _html.escape(label)
    return (
        f'<div style="margin:6px 12px 6px 0;">'
        f'<div style="font-size:12px;font-weight:600;color:#ccc;margin-bottom:3px;">'
        f'{escaped}</div>'
        + "".join(rows_html)
        + "</div>"
    )


def _sample_card(pair: dict, idx: int) -> str:
    inp_html = _grid_html(pair["input"],  f"Sample {idx} — input")
    out_html = _grid_html(pair["output"], f"Sample {idx} — output")
    return (
        "<section style='"
        "border:1px solid #3a3a3a;border-radius:8px;padding:10px 14px;"
        "margin-bottom:16px;background:#1e1e1e;'>"
        "<div style='display:flex;flex-wrap:wrap;align-items:flex-start;'>"
        + inp_html + out_html
        + "</div></section>"
    )


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    root = Path(__file__).resolve().parent

    rng = random.Random(42)
    samples: list[dict] = []

    for i in range(50):
        ex = generate(rng=rng)

        # Cross-verify: verifier must reproduce the generator's output exactly
        computed = verify_t50(ex["input"])
        if computed != ex["output"]:
            # Dump both grids for debugging before raising
            print(f"--- Mismatch on sample {i} ---")
            for r, (gen_row, ver_row) in enumerate(
                zip(ex["output"], computed)
            ):
                if gen_row != ver_row:
                    print(f"  row {r:2d}  gen={gen_row}")
                    print(f"         ver={ver_row}")
            raise RuntimeError(f"Verifier mismatch on sample {i}.")

        samples.append(ex)

    # ----- JSON ------------------------------------------------------------ #
    out_json = root / "t50_samples_50.json"
    out_json.write_text(
        json.dumps({"train": samples}, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Wrote {out_json}  ({out_json.stat().st_size:,} bytes)")

    # ----- HTML ------------------------------------------------------------ #
    rule_blurb = (
        "Each input contains a grid of gray nodes, each with an adjacent orange "
        "arrow indicating direction. One node is highlighted. Follow the arrows "
        "from the highlighted node until a cycle is detected. Draw the cycle as "
        "a polygon outline in the highlight colour; restore the start to gray."
    )
    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'/>",
        "<title>t50 — 50 samples</title></head>",
        "<body style='font-family:Segoe UI,Arial,sans-serif;background:#111;",
        "color:#ddd;max-width:1400px;margin:24px auto;padding:0 16px;'>",
        "<h1 style='color:#fff;'>t50 — 50 generated samples</h1>",
        f"<p style='max-width:700px;line-height:1.5;color:#aaa;'>{_html.escape(rule_blurb)}</p>",
    ]
    for i, s in enumerate(samples):
        parts.append(_sample_card(s, i))
    parts.append("</body></html>")

    out_html = root / "t50_samples_50.html"
    out_html.write_text("".join(parts), encoding="utf-8")
    print(f"Wrote {out_html}  ({out_html.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
