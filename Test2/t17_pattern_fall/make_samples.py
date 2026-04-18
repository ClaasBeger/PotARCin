"""Generate 50 t17 samples and write JSON + HTML preview."""

from __future__ import annotations

import html as html_module
import json
import random
from pathlib import Path

from generator import generate
from verifier import verify_t17

_COLORS = {
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


def _grid_to_html(grid: list[list[int]], title: str, cell_px: int = 14) -> str:
    rows: list[str] = []
    h = len(grid)
    w = len(grid[0]) if h else 0
    for r in range(h):
        cells: list[str] = []
        for c in range(w):
            sym = int(grid[r][c])
            color = _COLORS.get(sym, "#000000")
            cells.append(
                f'<div style="width:{cell_px}px;height:{cell_px}px;background:{color};'
                "border:1px solid #555;box-sizing:border-box;display:inline-block;"
                'vertical-align:top;"></div>'
            )
        rows.append("<div>" + "".join(cells) + "</div>")
    return (
        f'<div style="margin:8px;"><div style="font-weight:600;margin-bottom:4px;">'
        f"{html_module.escape(title)}</div>"
        + "".join(rows)
        + "</div>"
    )


def _pair_card(pair: dict[str, list[list[int]]], idx: int) -> str:
    return (
        "<section style='border:1px solid #444;padding:10px;border-radius:8px;margin-bottom:14px;'>"
        f"<h3 style='margin:4px 0 8px;'>Sample {idx}</h3>"
        "<div style='display:flex;flex-wrap:wrap;gap:10px;align-items:flex-start;'>"
        f"{_grid_to_html(pair['input'], 'Input')}"
        f"{_grid_to_html(pair['output'], 'Output')}"
        "</div></section>"
    )


def main() -> None:
    root = Path(__file__).resolve().parent
    out_json = root / "t17_samples_50.json"
    out_html = root / "t17_samples_50.html"

    rng = random.Random(0)
    samples: list[dict[str, list[list[int]]]] = []
    for _ in range(50):
        ex = generate(rng=rng)
        if verify_t17(ex["input"]) != ex["output"]:
            raise RuntimeError("Verifier mismatch on generated sample.")
        samples.append(ex)

    out_json.write_text(json.dumps({"train": samples}, indent=2), encoding="utf-8")

    parts: list[str] = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'/>",
        "<title>Test2/t17 50 samples</title></head>",
        "<body style='font-family:Segoe UI,sans-serif;background:#1a1a1a;color:#eee;max-width:1200px;margin:24px auto;'>",
        "<h1>Test2 t17 - 50 generated samples</h1>",
        "<p>Rule: one recurring shape appears in multiple vertical lanes, repeating down the grid. "
        "Some copies may be clipped at the top/bottom edges. Every copy shifts downward by the "
        "shape bounding height. This implies top-edge completion from off-grid repeats and bottom "
        "truncation for shapes shifted out of frame.</p>",
    ]
    for i, sample in enumerate(samples):
        parts.append(_pair_card(sample, i))
    parts.append("</body></html>")
    out_html.write_text("".join(parts), encoding="utf-8")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_html}")


if __name__ == "__main__":
    main()
