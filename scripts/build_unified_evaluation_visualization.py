"""Build a single HTML page visualizing all five evaluation dimensions for one task.

Prefers saved eval JSON under ``evaluation/`` when present; otherwise samples from ARC-GEN / samplers.

Usage (from repo root)::

    python scripts/build_unified_evaluation_visualization.py --task-id a85d4709

Output defaults to ``Demonstrations/unified_evaluations/<task_id>_unified_evaluation.html``.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from html import escape
from pathlib import Path
from typing import Any, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from framework.dimensions.classification_distribution import (
    sample_distribution_classification_batch,
    select_verifier_for_task,
)
from framework.dimensions.editing import sample_editing_pair
from framework.dimensions.inversion import InversionEvaluator
from framework.dimensions.base import DimensionConfig
from framework.grids import Grid, GridPair
from framework.tasks.arc_dataset import load_task
from framework.tasks.base import DimensionName

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


def _grid_html(grid: Grid, *, cell_px: int = 16) -> str:
    h = len(grid)
    w = len(grid[0]) if h else 0
    rows: list[str] = []
    for r in range(h):
        cells = []
        for c in range(w):
            sym = int(grid[r][c])
            color = _SYMBOL_TO_COLOR.get(sym, "#000000")
            cells.append(
                f'<span class="cell" style="width:{cell_px}px;height:{cell_px}px;background:{color};"></span>'
            )
        rows.append('<div class="row">' + "".join(cells) + "</div>")
    return '<div class="grid">' + "".join(rows) + "</div>"


def _pair_block(title: str, pair: GridPair, *, border: str, cell_px: int = 16) -> str:
    title_html = f'<div class="pair-title">{escape(title)}</div>' if title.strip() else ""
    return (
        f'<div class="pair-card" style="border-color:{border};">'
        f"{title_html}"
        f'<div class="pair-io"><span class="iolabel">Input</span>{_grid_html(pair.input, cell_px=cell_px)}</div>'
        f'<div class="pair-io"><span class="iolabel">Output</span>{_grid_html(pair.output, cell_px=cell_px)}</div>'
        f"</div>"
    )


def _arrow(*, points_left: bool = False) -> str:
    ch = "←" if points_left else "→"
    return f'<div class="arrow" aria-hidden="true">{ch}</div>'


def _instruction_blurb(prompt: Optional[str], fallback: str) -> str:
    """First instructional paragraph from a full model prompt (before examples)."""
    if not prompt or not str(prompt).strip():
        return fallback
    s = str(prompt).strip()
    for marker in (
        "\n\nExample 1:",
        "\nTraining example 1:",
        "\n\nTraining example 1:",
        "\nExample 1:\n",
    ):
        if marker in s:
            s = s.split(marker)[0].strip()
            break
    lines_out: list[str] = []
    for line in s.split("\n"):
        t = line.strip()
        if t.startswith("Example ") and ":" in t[:24]:
            break
        if t.startswith("Training example ") and ":" in t[:32]:
            break
        lines_out.append(line)
    s = "\n".join(lines_out).strip()
    if len(s) > 720:
        s = s[:717].rstrip() + "…"
    return s if s else fallback


def _tail_after_examples(prompt: Optional[str]) -> str:
    """Instruction text after demos (before the model's answer format line), when present."""
    if not prompt or not str(prompt).strip():
        return ""
    p = str(prompt)
    end_candidates = ("\n\nYour final answer", "\nYour final answer must", "\n\nBelow is a test input")
    for start in (
        "Generate one NEW input/output pair",
        "The following candidate input/output pair",
        "Now infer an input grid",
        "You will see five additional",
    ):
        if start not in p:
            continue
        i = p.index(start)
        rest = p[i:]
        for em in end_candidates:
            j = rest.find(em)
            if j != -1:
                rest = rest[:j]
        # Avoid duplicating target grids in the summary text (shown below as figures).
        for cut in ("\n\nDesired output:", "\nDesired output:"):
            j = rest.find(cut)
            if j != -1:
                rest = rest[:j]
        return rest.strip()
    return ""


def _summary_from_prompt(
    prompt: Optional[str],
    fallback: str,
    *,
    with_post_example_tail: bool = False,
) -> str:
    base = _instruction_blurb(prompt, fallback)
    if with_post_example_tail and prompt:
        tail = _tail_after_examples(prompt)
        if tail:
            base = f"{base}\n\n{tail}".strip()
    if len(base) > 1400:
        base = base[:1397].rstrip() + "…"
    return base


def _section_summary(text: str) -> str:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    blocks = "".join(f"<p>{escape(p)}</p>" for p in paras) if paras else f"<p>{escape(text)}</p>"
    return f'<div class="section-summary">{blocks}</div>'


def _section_footer(inner_html: str) -> str:
    return f'<div class="section-footer">{inner_html}</div>'


def _single_grid_card(
    *,
    title: str,
    grid: Grid,
    border: str,
    cell_px: int,
    row_label: str,
) -> str:
    return (
        f'<div class="pair-card single-grid-card" style="border-color:{border};">'
        f'<div class="pair-title">{escape(title)}</div>'
        f'<div class="pair-io single-io"><span class="iolabel">{escape(row_label)}</span>'
        f"{_grid_html(grid, cell_px=cell_px)}</div></div>"
    )


def _constrained_gen_side_by_side(pair: GridPair, *, cell_px: int = 26) -> str:
    return (
        '<div class="cg-center">'
        '<div class="cg-side-by-side">'
        f'{_single_grid_card(title="Generated input", grid=pair.input, border="#5c9ded", cell_px=cell_px, row_label="Input")}'
        f'{_single_grid_card(title="Generated output", grid=pair.output, border="#5c9ded", cell_px=cell_px, row_label="Output")}'
        "</div></div>"
    )


def _latest_json(directory: Path, task_id: str) -> Optional[Path]:
    if not directory.is_dir():
        return None
    matches = sorted(directory.glob(f"{task_id}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _html_page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{escape(title)}</title>
<style>
:root {{
  --bg: #0f1115;
  --surface: #171a21;
  --border: #2a3140;
  --text: #e8eaed;
  --muted: #9aa0a6;
  --accent: #5c9ded;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: "Segoe UI", system-ui, -apple-system, Roboto, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.5;
}}
.wrap {{
  max-width: 1100px;
  margin: 0 auto;
  padding: 2.5rem 1.5rem 4rem;
}}
h1 {{
  font-weight: 600;
  font-size: 1.75rem;
  margin: 0 0 0.5rem;
  letter-spacing: -0.02em;
}}
.sub {{
  color: var(--muted);
  margin-bottom: 2rem;
  font-size: 0.95rem;
}}
section {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.5rem 1.75rem;
  margin-bottom: 2rem;
}}
section h2 {{
  margin: 0 0 1rem;
  font-size: 1.15rem;
  font-weight: 600;
  color: var(--accent);
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.75rem;
}}
.demo-grid {{
  display: flex;
  flex-wrap: wrap;
  gap: 1.25rem;
  justify-content: center;
}}
.pair-card {{
  border: 3px solid var(--border);
  border-radius: 10px;
  padding: 0.75rem 1rem;
  background: #12151c;
}}
.pair-title {{
  font-size: 0.8rem;
  color: var(--muted);
  margin-bottom: 0.5rem;
  text-align: center;
}}
.pair-io {{
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
  margin-bottom: 0.5rem;
}}
.iolabel {{
  font-size: 0.7rem;
  color: var(--muted);
  width: 3.2rem;
  flex-shrink: 0;
  padding-top: 0.35rem;
}}
.grid .row {{ display: flex; }}
.cell {{
  display: inline-block;
  border: 1px solid rgba(255,255,255,0.12);
  vertical-align: top;
}}
.class-row {{
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  justify-content: center;
}}
.class-item {{
  flex: 0 1 200px;
}}
.badge {{
  font-size: 0.7rem;
  color: var(--muted);
  text-align: center;
  margin-bottom: 0.35rem;
}}
.flow {{
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
  gap: 1rem;
}}
.arrow {{
  font-size: 2rem;
  color: var(--muted);
  font-weight: 300;
  user-select: none;
}}
pre.code {{
  background: #0c0e12;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem;
  overflow: auto;
  max-height: 420px;
  font-size: 0.8rem;
  line-height: 1.4;
}}
.note {{
  font-size: 0.85rem;
  color: var(--muted);
  margin-top: 0.75rem;
}}
.metrics {{
  font-size: 0.8rem;
  color: var(--muted);
  margin-top: 0.5rem;
}}
.section-summary {{
  margin: 0 0 1.25rem;
  font-size: 0.95rem;
  color: #c4c7ce;
  line-height: 1.55;
}}
.section-summary p {{
  margin: 0 0 0.65rem;
}}
.section-summary p:last-child {{
  margin-bottom: 0;
}}
.section-footer {{
  margin-top: 1.35rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border);
  font-size: 0.82rem;
  color: var(--muted);
}}
.section-footer .note {{
  margin-top: 0;
}}
.cg-center {{
  display: flex;
  justify-content: center;
  width: 100%;
}}
.cg-side-by-side {{
  display: flex;
  flex-wrap: wrap;
  align-items: stretch;
  justify-content: center;
  gap: 1.75rem;
  max-width: 100%;
}}
.single-grid-card .pair-io.single-io {{
  margin-bottom: 0;
}}
.single-grid-card {{
  min-width: min(100%, 280px);
  padding: 1rem 1.25rem;
}}
</style>
</head>
<body>
<div class="wrap">
{body}
</div>
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Unified evaluation visualization HTML.")
    p.add_argument("--task-id", default="a85d4709")
    p.add_argument(
        "--out",
        default="",
        help="Output HTML path (default: Demonstrations/unified_evaluations/<task>_unified_evaluation.html)",
    )
    args = p.parse_args()
    task_id = args.task_id
    task = load_task(task_id)

    eval_root = ROOT_DIR / "evaluation"
    def_json = _latest_json(eval_root / "definition_eval_results", task_id)
    cls_json = _latest_json(eval_root / "classification_eval_results", task_id)
    cg_json = _latest_json(eval_root / "constrained_generation_eval_results", task_id)
    ed_json = _latest_json(eval_root / "editing_eval_results", task_id)
    inv_json = _latest_json(eval_root / "inversion_eval_results", task_id)

    # --- Section: demos ---
    demo_blocks = []
    for i, ex in enumerate(task.train_pairs, 1):
        demo_blocks.append(_pair_block(f"Training {i}", ex, border="#2a3140"))
    demos_html = '<div class="demo-grid">' + "".join(demo_blocks) + "</div>"

    # --- Definition ---
    def_prompt: Optional[str] = None
    def_body = ""
    def_footer_inner = ""
    if def_json:
        data = _load_json(def_json)
        def_prompt = data.get("prompt") if isinstance(data.get("prompt"), str) else None
        script = data.get("script") or ""
        def_body = f'<pre class="code">{escape(script)}</pre>'
        def_footer_inner = f'<p class="note">Source: latest definition eval — <code>{escape(str(def_json.name))}</code>. Model: {escape(str(data.get("model", "")))}.</p>'
        m = data.get("metrics", {})
        if m:
            tr = (m.get("train") or {}).get("success_rate", "—")
            te = (m.get("test") or {}).get("success_rate", "—")
            d50 = (m.get("dynamic50") or {}).get("success_rate", "—")
            def_footer_inner += f"<div class=\"metrics\">Metrics snapshot: train {tr}, test {te}, dynamic50 {d50}</div>"
    else:
        picked = select_verifier_for_task(task)
        vhint = ""
        if picked:
            slot, _fn = picked
            vhint = f" A reference verifier is available (<code>{escape(slot)}</code>) that matches train/test/ARC-GEN checks."
        def_body = (
            '<p class="note">No definition eval JSON under <code>evaluation/definition_eval_results/</code>. '
            f"Run a definition eval to record a synthesized <code>solve</code> script.{vhint}</p>"
        )
        def_footer_inner = ""
    def_summary = _section_summary(
        _summary_from_prompt(
            def_prompt,
            "The model must infer a Python program: a single callable `solve(grid)` that maps each input grid to the correct output grid using the same rule as the demonstrations, and that behaves sensibly on unseen inputs.",
            with_post_example_tail=False,
        )
    )
    def_block = def_summary + def_body + (_section_footer(def_footer_inner) if def_footer_inner else "")

    # --- Classification: resample with seed from eval if possible ---
    cls_prompt: Optional[str] = None
    cls_seed = 0
    if cls_json:
        _cj = _load_json(cls_json)
        cls_seed = int(_cj.get("seed", 0))
        cls_prompt = _cj.get("prompt") if isinstance(_cj.get("prompt"), str) else None
    rng_cls = random.Random(cls_seed)
    cls_batch = sample_distribution_classification_batch(task_id, rng_cls)
    cls_items = list(cls_batch.items)
    gt = [it.same_distribution for it in cls_items]
    cls_cards = []
    for i, it in enumerate(cls_items):
        color = "#2ecc71" if gt[i] else "#e74c3c"
        cls_cards.append(
            f'<div class="class-item"><div class="badge">Test {i} ({escape(it.category)})</div>'
            f'{_pair_block("", it.pair, border=color)}</div>'
        )
    cls_html = '<div class="class-row">' + "".join(cls_cards) + "</div>"
    cls_summary = _section_summary(
        _summary_from_prompt(
            cls_prompt,
            "Given the training examples and five candidate input/output pairs, the model must judge whether each candidate could come from the same underlying rule and distribution as the demos (same distribution vs. not).",
            with_post_example_tail=False,
        )
    )
    cls_footer = _section_footer(
        f'<p class="note">Sampled with seed <b>{cls_seed}</b> (matches latest classification eval when that file exists). '
        f"Green border = same distribution; red = not.</p>"
    )
    cls_block = cls_summary + cls_html + cls_footer

    # --- Constrained generation ---
    cg_prompt: Optional[str] = None
    cg_pair: Optional[GridPair] = None
    cg_footer_inner = ""
    if cg_json:
        d = _load_json(cg_json)
        cg_prompt = d.get("prompt") if isinstance(d.get("prompt"), str) else None
        pred = d.get("prediction")
        if isinstance(pred, dict) and "input" in pred and "output" in pred:
            cg_pair = GridPair(pred["input"], pred["output"])
            cg_footer_inner = f'<p class="note">Source: constrained-generation eval — <code>{escape(cg_json.name)}</code>.</p>'
    if cg_pair is None and task.arc_gen_generator:
        try:
            cg_pair = copy.deepcopy(task.arc_gen_generator(1)[0])
            cg_footer_inner = '<p class="note">No CG eval JSON; showing one ARC-GEN dynamic pair (same distribution as the task).</p>'
        except Exception:
            cg_footer_inner = '<p class="note">Could not sample ARC-GEN dynamic pair.</p>'
    cg_summary = _section_summary(
        _summary_from_prompt(
            cg_prompt,
            "The model must invent one new input/output pair that follows the same transformation as the training examples: not a copy of a demo, with a non-trivial mapping, and consistent with the underlying rule.",
            with_post_example_tail=True,
        )
    )
    if cg_pair:
        cg_html = _constrained_gen_side_by_side(cg_pair, cell_px=28)
    else:
        cg_html = "<p class='note'>No pair to show.</p>"
    cg_block = cg_summary + cg_html + (_section_footer(cg_footer_inner) if cg_footer_inner else "")

    # --- Editing ---
    ed_prompt: Optional[str] = None
    ed_flow = ""
    ed_footer_inner = ""
    if ed_json:
        d = _load_json(ed_json)
        ed_prompt = d.get("prompt") if isinstance(d.get("prompt"), str) else None
        cp = d.get("corrupted_pair")
        tp = d.get("target_pair")
        if isinstance(cp, dict) and isinstance(tp, dict):
            c_pair = GridPair(cp["input"], cp["output"])
            t_pair = GridPair(tp["input"], tp["output"])
            ed_flow = (
                '<div class="flow">'
                + _pair_block("Corrupted (candidate)", c_pair, border="#e74c3c")
                + _arrow()
                + _pair_block("Corrected (ground truth)", t_pair, border="#2ecc71")
                + "</div>"
            )
            ed_footer_inner = f'<p class="note">Source: editing eval — <code>{escape(ed_json.name)}</code>.</p>'
    if not ed_flow:
        rng_ed = random.Random(0)
        corrupted, target, src = sample_editing_pair(task, rng_ed, include_true=True)
        ed_flow = (
            '<div class="flow">'
            + _pair_block(f"Corrupted ({src})", corrupted, border="#e74c3c")
            + _arrow()
            + _pair_block("Corrected (ground truth)", target, border="#2ecc71")
            + "</div>"
        )
        ed_footer_inner = "<p class='note'>No editing eval JSON; sampled editing pair with <code>include_true=True</code>.</p>"
    ed_summary = _section_summary(
        _summary_from_prompt(
            ed_prompt,
            "The model sees a corrupted input/output pair that does not match the rule inferred from the demos. It must return the corrected pair: the input and output that are consistent with that rule.",
            with_post_example_tail=True,
        )
    )
    ed_block = ed_summary + ed_flow + _section_footer(ed_footer_inner)

    # --- Inversion (display: input ← output; arrow points left toward the inferred input) ---
    inv_prompt: Optional[str] = None
    inv_flow = ""
    inv_footer_inner = ""
    inv_cell = 20
    if inv_json:
        d = _load_json(inv_json)
        inv_prompt = d.get("prompt") if isinstance(d.get("prompt"), str) else None
        des = d.get("desired_output")
        ref = d.get("reference_input")
        if des is not None and ref is not None:
            inv_flow = (
                '<div class="flow">'
                + _single_grid_card(
                    title="Corresponding input (reference)",
                    grid=ref,
                    border="#2ecc71",
                    cell_px=inv_cell,
                    row_label="Input",
                )
                + _arrow(points_left=True)
                + _single_grid_card(
                    title="Given output (target)",
                    grid=des,
                    border="#5c9ded",
                    cell_px=inv_cell,
                    row_label="Output",
                )
                + "</div>"
            )
            inv_footer_inner = f'<p class="note">Source: inversion eval — <code>{escape(inv_json.name)}</code>.</p>'
    if not inv_flow:
        ev = InversionEvaluator(
            config=DimensionConfig(name=DimensionName.INVERSION, num_instances_per_task=1, random_seed=0)
        )
        insts = ev.generate_instances([task])
        if insts:
            inst = insts[0]
            des = inst.input.desired_output
            ref = inst.target
            inv_flow = (
                '<div class="flow">'
                + _single_grid_card(
                    title="Corresponding input (reference)",
                    grid=ref,
                    border="#2ecc71",
                    cell_px=inv_cell,
                    row_label="Input",
                )
                + _arrow(points_left=True)
                + _single_grid_card(
                    title="Given output (target)",
                    grid=des,
                    border="#5c9ded",
                    cell_px=inv_cell,
                    row_label="Output",
                )
                + "</div>"
            )
            inv_footer_inner = "<p class='note'>No inversion eval JSON; sampled ARC-GEN dynamic instance (seed 0).</p>"
        else:
            inv_flow = "<p class='note'>Could not build inversion instance.</p>"
    inv_summary = _section_summary(
        _summary_from_prompt(
            inv_prompt,
            "The model sees the training examples and a target output grid. It must infer which input grid would map to that output under the same rule (inverse of the usual forward prediction).",
            with_post_example_tail=True,
        )
    )
    inv_block = inv_summary + inv_flow + (_section_footer(inv_footer_inner) if inv_footer_inner else "")

    body = f"""
<h1>Unified evaluation overview</h1>
<p class="sub">Task <code>{escape(task_id)}</code> — all five dimensions at a glance.</p>

<section>
<h2>1. Training demonstrations</h2>
{demos_html}
</section>

<section>
<h2>2. Definition</h2>
{def_block}
</section>

<section>
<h2>3. Classification (distribution)</h2>
{cls_block}
</section>

<section>
<h2>4. Constrained generation</h2>
{cg_block}
</section>

<section>
<h2>5. Editing</h2>
{ed_block}
</section>

<section>
<h2>6. Inversion</h2>
{inv_block}
</section>
"""

    out_path = Path(args.out) if args.out else (
        ROOT_DIR / "Demonstrations" / "unified_evaluations" / f"{task_id}_unified_evaluation.html"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = _html_page(f"Evaluations — {task_id}", body)
    out_path.write_text(html, encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
