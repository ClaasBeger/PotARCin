"""Build a single HTML page visualizing all five evaluation dimensions for one task.

Prefers saved eval JSON under ``evaluation/`` when present. Dimensions without a saved eval show
a short notice only (no synthetic or re-sampled “placeholder” grids).

Usage (from repo root)::

    python scripts/build_unified_evaluation_visualization.py --task-id a85d4709

Output defaults to ``Demonstrations/unified_evaluations/<task_id>_unified_evaluation.html``.
"""

from __future__ import annotations

import os
import argparse
import json
import random
import sys
from html import escape
from pathlib import Path
from typing import Any, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
# Matplotlib may be imported transitively; default ~/.matplotlib is often not writable in
# sandboxes/CI, forcing a fresh temp cache + font rebuild in every process.
_mpl_cfg = ROOT_DIR / ".mplconfig"
_mpl_cfg.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cfg))
os.environ.setdefault("MPLBACKEND", "Agg")

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from framework.dimensions.classification import parse_distribution_classification_test_pairs_from_prompt
from framework.dimensions.classification_distribution import (
    sample_distribution_classification_batch,
    select_verifier_for_task,
)
from framework.dimensions.constrained_generation import build_constrained_generation_prompt
from framework.dimensions.editing import sample_editing_pair
from framework.grids import Grid, GridPair
from framework.tasks.arc_dataset import load_task

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

_INVALID_CELL_BG = "#6a1b9a"


def _cell_background(cell: Any) -> str:
    """ARC cells are ints 0–9; corrupted or bad JSON may nest lists/tuples."""
    if isinstance(cell, bool):
        cell = int(cell)
    if isinstance(cell, int) and 0 <= cell <= 9:
        return _SYMBOL_TO_COLOR.get(cell, "#000000")
    return _INVALID_CELL_BG


def _grid_html(grid: Grid, *, cell_px: int = 16) -> str:
    try:
        h = len(grid)
        w = len(grid[0]) if h else 0
        rows: list[str] = []
        for r in range(h):
            cells = []
            row = grid[r]
            for c in range(w):
                color = _cell_background(row[c])
                cells.append(
                    f'<span class="cell" style="width:{cell_px}px;height:{cell_px}px;background:{color};"></span>'
                )
            rows.append('<div class="row">' + "".join(cells) + "</div>")
        return '<div class="grid">' + "".join(rows) + "</div>"
    except (TypeError, ValueError, IndexError) as e:
        return f'<p class="note">Invalid grid for display: {escape(str(e))}</p>'


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
    matches = sorted(
        directory.rglob(f"{task_id}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _latest_debug_seed(task_id: str) -> Optional[int]:
    """Try to recover the classification seed from latest debug_<task_id>.json."""
    root = ROOT_DIR / "evaluation" / "multi_dimension_eval_runs"
    if not root.is_dir():
        return None
    matches = sorted(
        root.rglob(f"debug_{task_id}.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in matches:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            runs = payload.get("runs")
            if not isinstance(runs, list):
                continue
            for run in runs:
                if not isinstance(run, dict):
                    continue
                if str(run.get("task_id")) != task_id:
                    continue
                dim_seeds = run.get("dimension_seeds")
                if not isinstance(dim_seeds, dict):
                    continue
                s = dim_seeds.get("classification_distribution")
                if isinstance(s, int):
                    return s
        except Exception:
            continue
    return None


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
.class-answers {{
  font-size: 0.72rem;
  text-align: center;
  margin-bottom: 0.45rem;
  color: #aeb4bf;
}}
.class-answers.class-mismatch {{
  color: #f5b041;
  font-weight: 500;
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
    p.add_argument("--definition-json", default="", help="Optional explicit definition eval JSON path")
    p.add_argument(
        "--classification-json", default="", help="Optional explicit classification eval JSON path"
    )
    p.add_argument(
        "--constrained-generation-json",
        default="",
        help="Optional explicit constrained generation eval JSON path",
    )
    p.add_argument("--editing-json", default="", help="Optional explicit editing eval JSON path")
    p.add_argument("--inversion-json", default="", help="Optional explicit inversion eval JSON path")
    p.add_argument(
        "--out",
        default="",
        help="Output HTML path (default: Demonstrations/unified_evaluations/<task>_unified_evaluation.html)",
    )
    args = p.parse_args()
    task_id = args.task_id
    task = load_task(task_id)

    eval_root = ROOT_DIR / "evaluation"
    def _explicit_or_latest(explicit: str, eval_dir: str) -> Optional[Path]:
        if explicit:
            pth = Path(explicit)
            if not pth.is_absolute():
                pth = ROOT_DIR / pth
            return pth if pth.is_file() else None
        return _latest_json(eval_root / eval_dir, task_id)

    def_json = _explicit_or_latest(args.definition_json, "definition_eval_results")
    cls_json = _explicit_or_latest(args.classification_json, "classification_eval_results")
    cg_json = _explicit_or_latest(args.constrained_generation_json, "constrained_generation_eval_results")
    ed_json = _explicit_or_latest(args.editing_json, "editing_eval_results")
    inv_json = _explicit_or_latest(args.inversion_json, "inversion_eval_results")

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
            parts: list[str] = []

            def _append_metric(key: str, label: str) -> None:
                mm = m.get(key)
                if not isinstance(mm, dict):
                    return
                sr = mm.get("success_rate")
                if sr is None:
                    return
                parts.append(f"{label} {sr}")

            # Stable ordering, while still surfacing non-default dynamic sample sizes
            # such as dynamic1000 / re_arc_dynamic1000 in stress runs.
            _append_metric("train", "train")
            _append_metric("test", "test")
            _append_metric("stable", "ARC-GEN stable")
            for key in sorted(k for k in m.keys() if str(k).startswith("dynamic")):
                _append_metric(str(key), str(key))
            _append_metric("re_arc_stable", "Re-ARC stable")
            for key in sorted(k for k in m.keys() if str(k).startswith("re_arc_dynamic")):
                _append_metric(str(key), str(key))
            if parts:
                def_footer_inner += (
                    f"<div class=\"metrics\">Metrics snapshot: {', '.join(parts)}</div>"
                )
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

    # --- Classification: exact test pairs + gold / model labels from latest eval JSON when present ---
    cls_prompt: Optional[str] = None
    cls_seed = 0
    cls_json_name = ""
    cls_pairs: Optional[list[GridPair]] = None
    cls_gt: list[bool] = []
    cls_pred: list[bool] = []
    cls_meta: list[Any] = []
    cls_raw = ""
    if cls_json:
        _cj = _load_json(cls_json)
        cls_seed = int(_cj.get("seed", 0))
        cls_prompt = _cj.get("prompt") if isinstance(_cj.get("prompt"), str) else None
        cls_json_name = cls_json.name
        _rr = _cj.get("raw_response")
        cls_raw = _rr if isinstance(_rr, str) else ""
        _gt = _cj.get("ground_truth")
        _pr = _cj.get("predictions")
        if isinstance(_gt, list):
            cls_gt = [bool(x) for x in _gt]
        if isinstance(_pr, list):
            cls_pred = [bool(x) for x in _pr]
        _im = _cj.get("item_meta")
        if isinstance(_im, list):
            cls_meta = _im
        _tp = _cj.get("test_pairs")
        if isinstance(_tp, list) and len(_tp) == 5:
            try:
                cls_pairs = [GridPair(d["input"], d["output"]) for d in _tp if isinstance(d, dict)]
                if len(cls_pairs) != 5:
                    cls_pairs = None
            except (KeyError, TypeError, ValueError):
                cls_pairs = None
        if cls_pairs is None and cls_prompt:
            try:
                cls_pairs = parse_distribution_classification_test_pairs_from_prompt(cls_prompt)
            except ValueError:
                cls_pairs = None

    cls_cards: list[str] = []
    if cls_pairs is not None and len(cls_pairs) == 5:
        for i, it in enumerate(cls_pairs):
            cat = ""
            if i < len(cls_meta) and isinstance(cls_meta[i], dict):
                cat = str(cls_meta[i].get("category", ""))
            g = cls_gt[i] if i < len(cls_gt) else None
            pr = cls_pred[i] if i < len(cls_pred) else None
            border = "#2ecc71" if g else "#e74c3c"
            if g is None:
                border = "#7f8c8d"
            try:
                pair_html = _pair_block("", it, border=border)
            except (TypeError, ValueError, IndexError) as e:
                pair_html = (
                    f'<p class="note">Could not render this pair (bad grid data): {escape(str(e))}</p>'
                )
            g_s = "1" if g else ("0" if g is False else "?")
            p_s = "1" if pr else ("0" if pr is False else "?")
            if g is not None and pr is not None:
                verdict = "✓ match" if g == pr else "✗ mismatch"
                mismatch_cls = "" if g == pr else " class-mismatch"
            else:
                verdict = ""
                mismatch_cls = ""
            badge = f"Test {i}" + (f" ({escape(cat)})" if cat else "")
            cls_cards.append(
                f'<div class="class-item"><div class="badge">{badge}</div>'
                f'<div class="class-answers{mismatch_cls}">Gold: <b>{g_s}</b> · Model: <b>{p_s}</b>'
                f"{(f' · {escape(verdict)}' if verdict else '')}</div>"
                f"{pair_html}</div>"
            )
        cls_html = '<div class="class-row">' + "".join(cls_cards) + "</div>"
        raw_pre = (
            f'<p class="note" style="margin-top:0.85rem">Raw model response:</p>'
            f'<pre class="code">{escape(cls_raw.strip())}</pre>'
            if cls_raw.strip()
            else ""
        )
        cls_footer = _section_footer(
            f'<p class="note">Source: <code>{escape(str(cls_json_name))}</code>. '
            f"Pairs come from the saved eval (<code>test_pairs</code> if present, otherwise parsed from "
            f"the logged <code>prompt</code>) — not re-sampled. Eval seed: <b>{cls_seed}</b>. "
            f"Card border: ground truth same-distribution (green = 1, red = 0).</p>"
            + raw_pre
        )
    elif cls_json:
        cls_html = (
            "<p class=\"note\">Could not read the five test pairs from this classification eval file "
            "(expected <code>test_pairs</code> or a parseable <code>prompt</code>). Re-sampling would "
            "not match the logged run, so nothing is shown here.</p>"
        )
        cls_footer = _section_footer(
            f'<p class="note">File: <code>{escape(str(cls_json_name))}</code>.</p>'
        )
    else:
        # Fallback: still show one classification sample with a placeholder prediction.
        fallback_seed = _latest_debug_seed(task_id)
        if fallback_seed is None:
            fallback_seed = 0
        cls_seed = fallback_seed
        try:
            batch = sample_distribution_classification_batch(task_id, random.Random(cls_seed))
            cls_pairs = [it.pair for it in batch.items]
            cls_gt = [bool(it.same_distribution) for it in batch.items]
            cls_pred = [not g for g in cls_gt]  # Placeholder: deliberately all incorrect.
            cls_meta = [{"category": it.category, "meta": it.meta} for it in batch.items]
            cls_cards = []
            for i, it in enumerate(cls_pairs):
                cat = ""
                if i < len(cls_meta) and isinstance(cls_meta[i], dict):
                    cat = str(cls_meta[i].get("category", ""))
                g = cls_gt[i]
                pr = cls_pred[i]
                border = "#2ecc71" if g else "#e74c3c"
                pair_html = _pair_block("", it, border=border)
                verdict = "✗ mismatch"
                badge = f"Test {i}" + (f" ({escape(cat)})" if cat else "")
                cls_cards.append(
                    f'<div class="class-item"><div class="badge">{badge}</div>'
                    f'<div class="class-answers class-mismatch">Gold: <b>{"1" if g else "0"}</b> · '
                    f'Model: <b>{"1" if pr else "0"}</b> · {escape(verdict)}</div>'
                    f"{pair_html}</div>"
                )
            cls_html = '<div class="class-row">' + "".join(cls_cards) + "</div>"
            cls_footer = _section_footer(
                "<p class='note'>No usable classification eval JSON was found for this task, so this section "
                "uses a deterministic sampled batch with a placeholder model response. "
                "Placeholder response policy: every label is intentionally incorrect "
                "(prediction = logical NOT of ground truth). "
                f"Sampling seed: <b>{cls_seed}</b>.</p>"
            )
        except Exception as e:
            cls_html = (
                "<p class='note'>No classification eval JSON for this task under "
                "<code>evaluation/classification_eval_results/</code>, and fallback sampling failed: "
                f"{escape(str(e))}</p>"
            )
            cls_footer = ""

    cls_summary = _section_summary(
        _summary_from_prompt(
            cls_prompt,
            "Given the training examples and five candidate input/output pairs, the model must judge whether each candidate could come from the same underlying rule and distribution as the demos (same distribution vs. not).",
            with_post_example_tail=False,
        )
    )
    cls_block = cls_summary + cls_html + cls_footer

    # --- Constrained generation ---
    cg_prompt: Optional[str] = None
    cg_pair: Optional[GridPair] = None
    cg_footer_inner = ""
    cg_success: Optional[bool] = None
    cg_metrics_line = ""
    if cg_json:
        d = _load_json(cg_json)
        cg_prompt = d.get("prompt") if isinstance(d.get("prompt"), str) else None
        pred = d.get("prediction")
        m = d.get("metrics")
        if isinstance(m, dict):
            sr = m.get("success_rate")
            if isinstance(sr, (int, float)):
                cg_success = float(sr) >= 0.999999
            metric_parts: list[str] = []
            for key, label in (
                ("success_rate", "success_rate"),
                ("rejected_as_demo_copy", "demo_copy"),
                ("rejected_no_change", "no_change"),
                ("rejected_verifier_missing", "verifier_missing"),
                ("rejected_verifier_mismatch", "verifier_mismatch"),
            ):
                if key in m:
                    metric_parts.append(f"{label}={m[key]}")
            if metric_parts:
                cg_metrics_line = ", ".join(metric_parts)
        if isinstance(pred, dict) and "input" in pred and "output" in pred:
            cg_pair = GridPair(pred["input"], pred["output"])
            cg_footer_inner = f'<p class="note">Source: constrained-generation eval — <code>{escape(cg_json.name)}</code>.</p>'
    elif not cg_json and task.train_pairs:
        cg_prompt = build_constrained_generation_prompt(demonstrations=task.train_pairs)
    cg_summary = _section_summary(
        _summary_from_prompt(
            cg_prompt,
            "The model must invent one new input/output pair that follows the same transformation as the training examples: not a copy of a demo, with a non-trivial mapping, and consistent with the underlying rule.",
            with_post_example_tail=True,
        )
    )
    if cg_pair:
        cg_border = "#5c9ded"
        cg_title = ""
        if cg_success is True:
            cg_border = "#2ecc71"
            cg_title = '<p class="note" style="text-align:center;margin:0 0 0.75rem">Model pair status: <b>valid under rule</b></p>'
        elif cg_success is False:
            cg_border = "#e74c3c"
            cg_title = '<p class="note" style="text-align:center;margin:0 0 0.75rem">Model pair status: <b>invalid under rule</b></p>'
        cg_html = cg_title + (
            '<div class="cg-center">'
            '<div class="cg-side-by-side">'
            f'{_single_grid_card(title="Generated input", grid=cg_pair.input, border=cg_border, cell_px=28, row_label="Input")}'
            f'{_single_grid_card(title="Generated output", grid=cg_pair.output, border=cg_border, cell_px=28, row_label="Output")}'
            "</div></div>"
        )
    else:
        if cg_json:
            d = _load_json(cg_json)
            err = d.get("error")
            err_txt = f" Error: <code>{escape(str(err))}</code>." if err else ""
            cg_html = (
                "<p class='note'>Constrained-generation eval JSON exists, but the latest result is invalid "
                "(no valid <code>prediction</code> could be parsed). "
                f"File: <code>{escape(cg_json.name)}</code>.{err_txt}</p>"
            )
        else:
            cg_html = (
                "<p class='note'>No constrained-generation eval JSON under "
                "<code>evaluation/constrained_generation_eval_results/</code>. "
                "Run a constrained-generation eval to record the model pair and metrics.</p>"
            )
    if cg_metrics_line:
        cg_footer_inner += f'<div class="metrics">Metrics snapshot: {escape(cg_metrics_line)}</div>'
    cg_block = cg_summary + cg_html + (_section_footer(cg_footer_inner) if cg_footer_inner else "")

    # --- Editing ---
    ed_prompt: Optional[str] = None
    ed_flow = ""
    ed_footer_inner = ""
    if ed_json:
        d = _load_json(ed_json)
        ed_prompt = d.get("prompt") if isinstance(d.get("prompt"), str) else None
        cp = d.get("corrupted_pair")
        pp = d.get("prediction_pair")
        m = d.get("metrics")
        success_flag: Optional[bool] = None
        if isinstance(m, dict) and isinstance(m.get("success"), bool):
            success_flag = bool(m.get("success"))
        if isinstance(cp, dict) and isinstance(pp, dict):
            c_pair = GridPair(cp["input"], cp["output"])
            p_pair = GridPair(pp["input"], pp["output"])
            pred_border = "#7f8c8d"
            pred_tag = ""
            if success_flag is True:
                pred_border = "#2ecc71"
                pred_tag = " (success)"
            elif success_flag is False:
                pred_border = "#e74c3c"
                pred_tag = " (failure)"
            ed_flow = (
                '<div class="flow">'
                + _pair_block("Corrupted (candidate)", c_pair, border="#e74c3c")
                + _arrow()
                + _pair_block(f"Model-corrected pair{pred_tag}", p_pair, border=pred_border)
                + "</div>"
            )
            ed_footer_inner = f'<p class="note">Source: editing eval — <code>{escape(ed_json.name)}</code>.</p>'
    if not ed_flow:
        if ed_json:
            d = _load_json(ed_json)
            err = d.get("error")
            err_txt = f" Error: <code>{escape(str(err))}</code>." if err else ""
            ed_flow = (
                "<p class='note'>Editing eval JSON exists, but the model output is invalid "
                "(prediction could not be parsed/validated). "
                f"File: <code>{escape(ed_json.name)}</code>.{err_txt}</p>"
            )
            ed_footer_inner = f'<p class="note">Source: editing eval — <code>{escape(ed_json.name)}</code>.</p>'
        else:
            ed_flow = "<p class='note'>No editing eval JSON with model output found for this task.</p>"
            ed_footer_inner = ""
    ed_summary = _section_summary(
        _summary_from_prompt(
            ed_prompt,
            "The model sees a corrupted input/output pair that does not match the rule inferred from the demos. It must return the corrected pair: the input and output that are consistent with that rule.",
            with_post_example_tail=True,
        )
    )
    ed_block = ed_summary + ed_flow + _section_footer(ed_footer_inner)

    # --- Inversion (display: model input ← output; omit reference input card) ---
    inv_prompt: Optional[str] = None
    inv_flow = ""
    inv_footer_inner = ""
    inv_cell = 20
    if inv_json:
        d = _load_json(inv_json)
        inv_prompt = d.get("prompt") if isinstance(d.get("prompt"), str) else None
        des = d.get("desired_output")
        ref = d.get("reference_input")
        pred = d.get("prediction_input")
        if des is not None and ref is not None and pred is not None:
            exact_ref_match = pred == ref
            inv_metrics = d.get("metrics")
            inv_success: Optional[bool] = None
            if isinstance(inv_metrics, dict):
                if isinstance(inv_metrics.get("success"), bool):
                    inv_success = bool(inv_metrics.get("success"))
                elif isinstance(inv_metrics.get("success_rate"), (int, float)):
                    inv_success = float(inv_metrics.get("success_rate")) >= 0.999999
            pred_border = "#7f8c8d"
            pred_title = "Model-generated input"
            if inv_success is True:
                pred_border = "#2ecc71"
                pred_title = "Model-generated input (valid under rule)"
            elif inv_success is False:
                pred_border = "#e74c3c"
                pred_title = "Model-generated input (invalid under rule)"
            inv_parts = [
                _single_grid_card(
                    title=pred_title,
                    grid=pred,
                    border=pred_border,
                    cell_px=inv_cell,
                    row_label="Input",
                ),
                _arrow(points_left=True),
                _single_grid_card(
                    title="Given output (target)",
                    grid=des,
                    border="#5c9ded",
                    cell_px=inv_cell,
                    row_label="Output",
                ),
            ]
            inv_flow = '<div class="flow">' + "".join(inv_parts) + "</div>"
            inv_footer_inner = (
                f'<p class="note">Source: inversion eval — <code>{escape(inv_json.name)}</code>. '
                f'Exact reference-input match: <b>{"yes" if exact_ref_match else "no"}</b>.</p>'
            )
    if not inv_flow:
        if inv_json:
            d = _load_json(inv_json)
            err = d.get("error")
            err_txt = f" Error: <code>{escape(str(err))}</code>." if err else ""
            inv_flow = (
                "<p class='note'>Inversion eval JSON exists, but the model output is invalid "
                "(required fields could not be parsed). "
                f"File: <code>{escape(inv_json.name)}</code>.{err_txt}</p>"
            )
            inv_footer_inner = f'<p class="note">Source: inversion eval — <code>{escape(inv_json.name)}</code>.</p>'
        else:
            inv_flow = (
                "<p class='note'>No inversion eval JSON with model output under "
                "<code>evaluation/inversion_eval_results/</code>. "
                "Run an inversion eval to record the prompt, reference input, and model prediction.</p>"
            )
            inv_footer_inner = ""
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
