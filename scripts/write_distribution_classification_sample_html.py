"""Write a static HTML page showing one sampled 5-way distribution-classification case.

The **prompt** shown to models is built by
``framework.dimensions.classification.build_distribution_classification_prompt`` (same
wording as in API eval). Run from repo root::

    python scripts/write_distribution_classification_sample_html.py --task-id a85d4709 --seed 0

Output defaults to ``Demonstrations/classification_demos/<task_id>_distribution_classification_sample.html``.
"""

from __future__ import annotations

import argparse
import html as html_module
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from framework.dimensions.classification import build_distribution_classification_prompt
from framework.dimensions.classification_distribution import sample_distribution_classification_batch

_SYMBOL_TO_COLOR = {
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


def _grid_to_html(grid: list[list[int]], title: str, cell_px: int = 18) -> str:
    h = len(grid)
    w = len(grid[0]) if h else 0
    rows = []
    for r in range(h):
        cells = []
        for c in range(w):
            sym = int(grid[r][c])
            color = _SYMBOL_TO_COLOR.get(sym, "#000000")
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


def _pair_row(pair, label: str, cell_px: int = 18) -> str:
    return (
        f"<h3>{html_module.escape(label)}</h3>"
        f'<div style="display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start;">'
        f"{_grid_to_html(pair.input, 'Input', cell_px)}"
        f"{_grid_to_html(pair.output, 'Output', cell_px)}"
        f"</div>"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Write distribution classification sample HTML.")
    p.add_argument("--task-id", default="a85d4709")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out",
        default="",
        help="Output path (default: Demonstrations/classification_demos/<task>_...html)",
    )
    args = p.parse_args()

    rng = random.Random(args.seed)
    batch = sample_distribution_classification_batch(args.task_id, rng)
    demos = batch.demonstrations
    items = batch.items
    candidates = [it.pair for it in items]
    labels = [it.same_distribution for it in items]
    prompt = build_distribution_classification_prompt(
        demonstrations=demos,
        candidates=candidates,
    )

    parts: list[str] = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'/>",
        f"<title>Distribution classification — {html_module.escape(batch.task_id)}</title>",
        "</head><body style='font-family:Segoe UI,sans-serif;background:#1a1a1a;color:#eee;max-width:1100px;margin:24px auto;'>",
        f"<h1>Distribution classification (sample)</h1>",
        f"<p>Task <code>{html_module.escape(batch.task_id)}</code>, RNG seed <b>{args.seed}</b>. "
        "Each test row is one category (instance mismatch, corrupted verifier, color flip, "
        "correct ARC-GEN dynamic, or retrieval); order below is <b>after</b> shuffling.</p>",
        "<section style='border:1px solid #444;padding:16px;border-radius:8px;margin-bottom:24px;'>",
        "<h2>Ground-truth labels (1 = same distribution)</h2>",
        "<p><code>",
        html_module.escape(str([1 if x else 0 for x in labels])),
        "</code></p>",
        "<ul>",
    ]
    for i, it in enumerate(items):
        parts.append(
            f"<li><b>Test {i}</b> — category <code>{html_module.escape(it.category)}</code>, "
            f"same_distribution=<b>{it.same_distribution}</b>, meta: <code>{html_module.escape(repr(it.meta))}</code></li>"
        )
    parts.extend(["</ul>", "</section>"])

    parts.append(
        "<section style='border:1px solid #444;padding:16px;border-radius:8px;margin-bottom:24px;'>"
        "<h2>Training demonstrations</h2>"
    )
    for j, ex in enumerate(demos, 1):
        parts.append(_pair_row(ex, f"Training example {j}"))
    parts.append("</section>")

    parts.append(
        "<section style='border:1px solid #444;padding:16px;border-radius:8px;margin-bottom:24px;'>"
        "<h2>Five test pairs (candidate order)</h2>"
    )
    for i, it in enumerate(items):
        parts.append(_pair_row(it.pair, f"Test {i} — {it.category}"))
    parts.append("</section>")

    parts.append(
        "<section style='border:1px solid #444;padding:16px;border-radius:8px;margin-bottom:24px;'>"
        "<h2>Prompt text (for models)</h2>"
        "<p>Built by <code>framework.dimensions.classification.build_distribution_classification_prompt</code>.</p>"
        f"<pre style='white-space:pre-wrap;background:#111;padding:12px;border-radius:6px;overflow:auto;'>"
        f"{html_module.escape(prompt)}</pre>"
        "</section>"
    )

    parts.append("</body></html>")
    html = "".join(parts)

    out_path = Path(args.out) if args.out else (
        ROOT_DIR
        / "Demonstrations"
        / "classification_demos"
        / f"{batch.task_id}_distribution_classification_sample.html"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
