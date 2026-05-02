"""Generate 50 t16 samples and write JSON + HTML preview."""

from __future__ import annotations

import html as html_module
import json
import random
from pathlib import Path

from generator import generate
from verifier import verify_t16


def _generate_one(master: random.Random, ridge_chain_target: int | None) -> dict[str, list[list[int]]]:
    """``ridge_chain_target`` forces a perimeter diagonal singleton ridge of that length when possible."""
    phases: list[tuple[int | None, int]] = [(ridge_chain_target, 480)]
    if ridge_chain_target is not None:
        phases.append((None, 720))

    for tgt, budget in phases:
        for _ in range(budget):
            stream = random.Random(master.getrandbits(64))
            try:
                ex = generate(rng=stream, ridge_chain_target=tgt)
            except RuntimeError:
                continue
            if verify_t16(ex["input"]) != ex["output"]:
                raise RuntimeError("Verifier mismatch on generated sample.")
            return ex
    raise RuntimeError("failed to sample valid t16 instance for HTML pack")

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
    out_json = root / "t16_samples_50.json"
    out_html = root / "t16_samples_50.html"

    master = random.Random(0)
    # Stratify so the HTML pack reliably shows train-style ridges (rejection hides them otherwise).
    ridge_quota = ([3] * 28) + ([2] * 8) + ([4] * 5) + ([None] * 9)
    master.shuffle(ridge_quota)

    samples: list[dict[str, list[list[int]]]] = []
    for forced_ridge in ridge_quota:
        samples.append(_generate_one(master, forced_ridge))

    out_json.write_text(json.dumps({"train": samples}, indent=2), encoding="utf-8")

    parts: list[str] = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'/>",
        "<title>Test2/t16 50 samples</title></head>",
        "<body style='font-family:Segoe UI,sans-serif;background:#1a1a1a;color:#eee;max-width:1200px;margin:24px auto;'>",
        "<h1>Test2 t16 - 50 generated samples</h1>",
        "<p>King-linked colored orth-islands are one prototype each (including diagonal ties). "
        "Valid targets are rigid orthogonal assemblies under translation only; prototypes may be placed "
        "in any order when snapping onto the growing union. About half of this pack forces a perimeter "
        "diagonal ridge of length 3–4 monominos so those motifs show up despite downstream rejection. "
        "Orth-grey blobs matching an assembly recolor; task-colored cells clear. Each pair satisfies "
        "<code>verify_t16(input)==output</code>.</p>",
    ]
    for i, sample in enumerate(samples):
        parts.append(_pair_card(sample, i))
    parts.append("</body></html>")
    out_html.write_text("".join(parts), encoding="utf-8")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_html}")


if __name__ == "__main__":
    main()
