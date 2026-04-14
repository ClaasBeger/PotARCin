from __future__ import annotations

import argparse
import copy
import random
import sys
from html import escape
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from framework.dimensions.classification_distribution import sample_distribution_classification_batch
from framework.grids import Grid
from framework.tasks.arc_dataset import load_task
from framework.verifier_selection import select_verifier_for_task

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


def _grid_html(grid: Grid, cell_px: int = 14) -> str:
    rows: list[str] = []
    for row in grid:
        cells = "".join(
            f'<span class="cell" style="width:{cell_px}px;height:{cell_px}px;background:{_COLORS.get(int(v), "#000")}"></span>'
            for v in row
        )
        rows.append(f"<div class='row'>{cells}</div>")
    return "<div class='grid'>" + "".join(rows) + "</div>"


def _pair_html(title: str, inp: Grid, out: Grid) -> str:
    return (
        "<div class='card'>"
        f"<div class='title'>{escape(title)}</div>"
        f"<div class='io'><span>Input</span>{_grid_html(inp)}</div>"
        f"<div class='io'><span>Output</span>{_grid_html(out)}</div>"
        "</div>"
    )


def _note_card(title: str, text: str) -> str:
    return (
        "<div class='card'>"
        f"<div class='title'>{escape(title)}</div>"
        f"<p class='note'>{escape(text)}</p>"
        "</div>"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--preflight-index", type=int, default=None, help="Use preflight seed = 1000 + index")
    p.add_argument("--indices", default="1,4", help="Comma-separated candidate indices")
    p.add_argument("--out", default="")
    args = p.parse_args()

    idxs = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
    if args.seed is not None and args.preflight_index is not None:
        raise ValueError("Use either --seed or --preflight-index, not both.")
    seed = args.seed if args.seed is not None else (1000 + args.preflight_index if args.preflight_index is not None else 1000)
    rng = random.Random(seed)
    batch = sample_distribution_classification_batch(args.task_id, rng)
    task = load_task(args.task_id)
    picked = select_verifier_for_task(task)
    if picked is None:
        raise RuntimeError("No verifier selected for task.")
    slot, vf = picked

    sections: list[str] = []
    for i in idxs:
        it = batch.items[i]
        extra = ""
        right_card = ""
        if it.category == "retrieval":
            q = it.meta.get("query_task")
            r = it.meta.get("rank")
            extra = (
                "<p class='note'>This candidate is <b>retrieval</b>: it is sourced from a nearest alternative task instance "
                f"(query_task={escape(str(q))}, rank={escape(str(r))}), not produced by output corruption on this task.</p>"
            )
            right_card = _pair_html("Retrieved source pair (already non-corrupted)", it.pair.input, it.pair.output)
        else:
            extra = (
                "<p class='note'>Right card shows the clean non-corrupted output for the same input using the selected verifier.</p>"
            )
            try:
                clean_out = vf(copy.deepcopy(it.pair.input))
                right_card = _pair_html("Non-corrupted pair (same input, verifier output)", it.pair.input, clean_out)
            except Exception as e:
                right_card = _note_card(
                    "Non-corrupted pair unavailable",
                    f"Selected verifier could not process this input ({type(e).__name__}).",
                )
        sections.append(
            "<section>"
            f"<h2>Candidate {i} — {escape(it.category)} (label {1 if it.same_distribution else 0})</h2>"
            f"{extra}"
            "<div class='row-flex'>"
            + _pair_html("Sampled pair (possibly corrupted)", it.pair.input, it.pair.output)
            + right_card
            + "</div></section>"
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'/><title>{escape(args.task_id)} clean pair preview</title>
<style>
body{{background:#0f1115;color:#e8eaed;font-family:Segoe UI,sans-serif;margin:0}}
.wrap{{max-width:1200px;margin:0 auto;padding:24px}}
h1{{margin:0 0 10px}} h2{{color:#7fb3ff;font-size:18px}}
section{{background:#171a21;border:1px solid #2a3140;border-radius:10px;padding:12px 14px;margin:14px 0}}
.row-flex{{display:flex;gap:12px;flex-wrap:wrap}}
.card{{border:1px solid #2a3140;background:#12151c;border-radius:8px;padding:8px 10px}}
.title{{font-size:12px;color:#9aa0a6;margin-bottom:6px}}
.io{{display:flex;gap:8px;align-items:flex-start;margin-bottom:6px}}
.io span{{width:46px;color:#9aa0a6;font-size:12px;padding-top:2px}}
.row{{display:flex}} .cell{{display:inline-block;border:1px solid rgba(255,255,255,0.12)}}
.note{{color:#9aa0a6}}
</style></head><body><div class='wrap'>
<h1>Classification clean-pair preview — <code>{escape(args.task_id)}</code></h1>
<p class='note'>Seed={seed}, verifier slot={escape(slot)}, indices={escape(args.indices)}</p>
{''.join(sections)}
</div></body></html>"""

    out_path = (
        Path(args.out)
        if args.out
        else ROOT_DIR / "Demonstrations" / "preflight_first10" / f"{args.task_id}_clean_pairs.html"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
