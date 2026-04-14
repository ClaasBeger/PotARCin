"""Build HTML previews for first 10 tasks without model calls.

This renders the *inputs to be sent* for all dimensions (definition,
classification-distribution, constrained generation, editing, inversion),
using local sampling only.
"""

from __future__ import annotations

import json
import random
import sys
import time
from html import escape
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from framework.dimensions.classification import build_distribution_classification_prompt
from framework.dimensions.classification_distribution import sample_distribution_classification_batch
from framework.dimensions.constrained_generation import build_constrained_generation_prompt
from framework.dimensions.definition import build_written_prompt
from framework.dimensions.editing import build_editing_prompt, sample_editing_pair
from framework.dimensions.inversion import build_inversion_prompt
from framework.grids import Grid, GridPair
from framework.tasks.arc_dataset import iter_tasks, load_task
from framework.tasks.base import TaskSource

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


def _grid_html(grid: Grid, *, cell_px: int = 14) -> str:
    rows: list[str] = []
    for row in grid:
        cells = "".join(
            f'<span class="cell" style="width:{cell_px}px;height:{cell_px}px;background:{_SYMBOL_TO_COLOR.get(int(v), "#000")}"></span>'
            for v in row
        )
        rows.append(f"<div class='row'>{cells}</div>")
    return "<div class='grid'>" + "".join(rows) + "</div>"


def _pair_html(title: str, p: GridPair) -> str:
    return (
        "<div class='pair'>"
        f"<div class='pair-title'>{escape(title)}</div>"
        f"<div class='io'><span>Input</span>{_grid_html(p.input)}</div>"
        f"<div class='io'><span>Output</span>{_grid_html(p.output)}</div>"
        "</div>"
    )


def _page(task_id: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'/><title>{escape(task_id)} preflight</title>
<style>
body{{background:#0f1115;color:#e8eaed;font-family:Segoe UI,sans-serif;margin:0}}
.wrap{{max-width:1100px;margin:0 auto;padding:24px}}
section{{background:#171a21;border:1px solid #2a3140;border-radius:10px;padding:14px 16px;margin:14px 0}}
h1{{margin:0 0 6px}} h2{{margin:0 0 10px;color:#7fb3ff;font-size:18px}}
.pairs{{display:flex;flex-wrap:wrap;gap:10px}}
.pair{{border:1px solid #2a3140;border-radius:8px;padding:8px;background:#12151c}}
.pair-title{{font-size:12px;color:#9aa0a6;margin-bottom:6px}}
.io{{display:flex;gap:8px;align-items:flex-start;margin-bottom:6px}}
.io span{{width:44px;color:#9aa0a6;font-size:12px;padding-top:2px}}
.row{{display:flex}} .cell{{display:inline-block;border:1px solid rgba(255,255,255,0.1)}}
pre{{background:#0c0e12;border:1px solid #2a3140;border-radius:8px;padding:10px;white-space:pre-wrap;max-height:240px;overflow:auto}}
.note{{color:#9aa0a6;font-size:13px}}
a{{color:#7fb3ff}}
</style></head><body><div class='wrap'>{body}</div></body></html>"""


def _index(out_dir: Path, task_ids: list[str]) -> None:
    rows = "".join(
        f"<li><a href='{escape(t)}_preflight.html'>{escape(t)}</a></li>" for t in task_ids
    )
    html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'/><title>Preflight first 10</title></head>
<body style='font-family:Segoe UI,sans-serif;background:#0f1115;color:#e8eaed;padding:24px'>
<h1>First 10 task preflight pages</h1>
<p>No model calls were made. These pages show the prepared prompts/examples per dimension.</p>
<ol>{rows}</ol></body></html>"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    t0 = time.perf_counter()
    out_dir = ROOT_DIR / "Demonstrations" / "preflight_first10"
    out_dir.mkdir(parents=True, exist_ok=True)

    task_ids: list[str] = []
    for i, t in enumerate(iter_tasks(split="train", source=TaskSource.ORIGINAL_ARC)):
        if i >= 10:
            break
        task_ids.append(t.task_id)

    for i, tid in enumerate(task_ids):
        task = load_task(tid)
        rng = random.Random(1000 + i)

        demos = "".join(_pair_html(f"Train {j+1}", p) for j, p in enumerate(task.train_pairs))
        puzzle_json = {
            "train": [{"input": p.input, "output": p.output} for p in task.train_pairs],
            "test": [{"input": task.test_inputs[0], "output": task.test_outputs[0] if task.test_outputs else task.test_inputs[0]}],
        }
        def_prompt = build_written_prompt(puzzle_json=puzzle_json, test_idx=0)

        batch = sample_distribution_classification_batch(tid, rng)
        cls_prompt = build_distribution_classification_prompt(
            demonstrations=batch.demonstrations,
            candidates=[it.pair for it in batch.items],
        )
        cls_pairs = "".join(
            _pair_html(
                f"Candidate {k} — {it.category} ({'1' if it.same_distribution else '0'})",
                it.pair,
            )
            for k, it in enumerate(batch.items)
        )

        cg_prompt = build_constrained_generation_prompt(demonstrations=task.train_pairs)

        corrupted, target, source = sample_editing_pair(task, rng, include_true=False)
        ed_prompt = build_editing_prompt(
            demonstrations=task.train_pairs,
            corrupted_pair=corrupted,
            include_true=False,
        )

        inv_pair = task.arc_gen_generator(1, rng=rng)[0] if task.arc_gen_generator is not None else task.train_pairs[0]
        inv_prompt = build_inversion_prompt(
            demonstrations=task.train_pairs,
            desired_output=inv_pair.output,
        )

        body = f"""
<h1>Preflight task <code>{escape(tid)}</code></h1>
<p class='note'>Prepared locally without calling the model.</p>
<section><h2>Training demonstrations</h2><div class='pairs'>{demos}</div></section>
<section><h2>Definition prompt</h2><pre>{escape(def_prompt)}</pre></section>
<section><h2>Classification-distribution prompt + candidates</h2><pre>{escape(cls_prompt)}</pre><div class='pairs'>{cls_pairs}</div></section>
<section><h2>Constrained generation prompt</h2><pre>{escape(cg_prompt)}</pre></section>
<section><h2>Editing prompt + sampled pair ({escape(source)})</h2><pre>{escape(ed_prompt)}</pre><div class='pairs'>{_pair_html(f'Corrupted — {source}', corrupted)}{_pair_html('Target — ground_truth', target)}</div></section>
<section><h2>Inversion prompt + desired output / reference input</h2><pre>{escape(inv_prompt)}</pre><div class='pairs'>{_pair_html('Reference inversion pair', inv_pair)}</div></section>
"""
        (out_dir / f"{tid}_preflight.html").write_text(_page(tid, body), encoding="utf-8")

    _index(out_dir, task_ids)
    elapsed = time.perf_counter() - t0
    print(f"Wrote {len(task_ids)} preflight pages under {out_dir}")
    print(f"Elapsed wall time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
