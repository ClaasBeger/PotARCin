"""
Demo: AST-corrupt a verifier (re_arc ``verify_<task>`` or a golf ``taskNNN.py``)
and compare outputs on one ARC-GEN dynamic sample.

- **re_arc**: drop one middle assignment and rewire loads (see
  ``framework.corruption.verifier_ast``).
- **golf**: prune one binary op in a ``p = lambda g: ...`` (or single-return
  ``solve``) by replacing a :class:`ast.BinOp` with its left subtree.

Corruptions are resampled until the corrupted output differs from gold **and**
the **normalized padded cell edit distance** between gold and corrupt outputs is
at most a threshold (default ``0.70``). Padding handles differing grid sizes.

A second block finds the **three closest** pairs from other tasks vs a **random
demonstration pair** from this task (train, ARC-GEN stable, or up to 50 ARC-GEN
dynamic samples — not canonical test, which has no gold output here). Score =
average of normalized padded distances on **input** and **output**.

A third block samples a pair from ARC-GEN stable or dynamic and applies recolor
**attempts**: each attempt **first** picks a cell (black cells are less likely),
then takes its 4-connected same-color neighborhood — if it has at least two cells,
the **whole** neighborhood gets one new color; otherwise only that cell. New
colors are biased away from black.

A **fourth** block (**same-task instance mismatch**) fixes the **task**: it uses
the anchor ARC-GEN dynamic **input**, then draws many more dynamic pairs for that
**same** task and picks an **output** from a **different instance** (different
input grid) whose padded edit distance to the anchor **output** is minimal,
excluding grid-equal outputs.

Usage (from repo root)::

    python scripts/verifier_corruption_demo.py --task-id a85d4709 --verifier re_arc
    python scripts/verifier_corruption_demo.py --task-id a85d4709 --verifier golf-keymoon
    python scripts/verifier_corruption_demo.py --task-id a85d4709 --verifier both
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from html import escape
from pathlib import Path
from typing import Literal

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from framework.corruption.golf_ast import (
    GolfSource,
    load_and_corrupt_golf_verifier,
    uncorrupted_golf_verifier,
)
from framework.corruption.cross_dynamic_mismatch import (
    CrossDynamicMismatch,
    find_cross_dynamic_mismatch,
    synthetic_input_borrowed_output_pair,
)
from framework.corruption.nearest_inputs import NeighborInstance, find_nearest_alternative_instances
from framework.corruption.verifier_ast import (
    load_and_corrupt_re_arc_verifier,
    uncorrupted_re_arc_verifier,
)
from framework.grids import (
    Grid,
    GridPair,
    is_equal_grid,
    normalized_cell_edit_between_outputs,
    pretty_grid,
    connected_component_color_flips_on_pair,
    flip_count_probabilities,
    sample_flip_count_favor_one,
)
from framework.tasks.arc_dataset import load_task
from framework.tasks.base import ArcTask

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

VerifierChoice = Literal["re_arc", "golf-google", "golf-keymoon", "golf-neurips", "both"]


def _grid_to_html(grid: Grid, title: str, cell_px: int = 18) -> str:
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
        f'<div style="margin:8px;"><div style="font-weight:600;margin-bottom:4px;">{escape(title)}</div>'
        + "".join(rows)
        + "</div>"
    )


def _parse_golf_source(verifier: VerifierChoice) -> GolfSource:
    if verifier == "golf-google":
        return "google"
    if verifier == "golf-keymoon":
        return "keymoon"
    if verifier == "golf-neurips":
        return "neurips"
    raise ValueError(f"Not a golf verifier: {verifier!r}")


def sample_demonstration_pair(task: ArcTask, rng: random.Random) -> tuple[GridPair, str]:
    """Pick one random (input, output) demonstration: train, arc-gen stable, or arc-gen dynamic.

    Excludes canonical test-only rows (no paired output in this dataset).
    """
    options: list[tuple[str, GridPair]] = []
    for i, p in enumerate(task.train_pairs):
        options.append((f"train[{i}]", p))
    if task.arc_gen_synthetic_pairs:
        for i, p in enumerate(task.arc_gen_synthetic_pairs):
            options.append((f"arc_gen_stable[{i}]", p))
    if task.arc_gen_generator is not None:
        try:
            dyn = task.arc_gen_generator(50)
            for i, p in enumerate(dyn):
                options.append((f"arc_gen_dynamic[{i}]", p))
        except Exception:
            pass
    if not options:
        raise ValueError("No demonstration pairs (train / arc-gen stable / dynamic) for this task.")
    label, pair = rng.choice(options)
    return copy.deepcopy(pair), label


def sample_arc_gen_pair_for_color_flips(task: ArcTask, rng: random.Random) -> tuple[GridPair, str]:
    """Stable or dynamic ARC-GEN pair only (for the flip augmentation block)."""
    opts: list[tuple[str, GridPair]] = []
    if task.arc_gen_synthetic_pairs:
        for i, p in enumerate(task.arc_gen_synthetic_pairs):
            opts.append((f"arc_gen_stable[{i}]", p))
    if task.arc_gen_generator is not None:
        try:
            dyn = task.arc_gen_generator(50)
            for i, p in enumerate(dyn):
                opts.append((f"arc_gen_dynamic[{i}]", p))
        except Exception:
            pass
    if not opts:
        raise ValueError("No ARC-GEN stable or dynamic pairs for color-flip demo.")
    label, pair = rng.choice(opts)
    return copy.deepcopy(pair), label


def _run_re_arc(
    task_id: str,
    rng: random.Random,
    inp: Grid,
    *,
    max_norm_edit: float | None,
    max_corruption_attempts: int,
) -> tuple[Grid, Grid, str, int, float]:
    corrupt_fn, corrupt_src, drop_idx = load_and_corrupt_re_arc_verifier(
        task_id,
        rng=rng,
        sample_input=inp,
        max_attempts=max_corruption_attempts,
        max_normalized_cell_edit_distance=max_norm_edit,
    )
    gold_out = uncorrupted_re_arc_verifier(task_id)(copy.deepcopy(inp))
    bad_out = corrupt_fn(copy.deepcopy(inp))
    assert not is_equal_grid(
        gold_out, bad_out
    ), "re_arc: corrupted output must differ from gold on the sample input"
    norm = normalized_cell_edit_between_outputs(gold_out, bad_out)
    if max_norm_edit is not None:
        assert norm <= max_norm_edit, "re_arc: normalized edit above threshold"
    return gold_out, bad_out, corrupt_src, drop_idx, norm


def _run_golf(
    task_id: str,
    source: GolfSource,
    rng: random.Random,
    inp: Grid,
    *,
    max_norm_edit: float | None,
    max_corruption_attempts: int,
) -> tuple[Grid, Grid, str, int, float]:
    corrupt_fn, corrupt_src, binop_idx = load_and_corrupt_golf_verifier(
        task_id,
        source,
        rng=rng,
        sample_input=inp,
        max_attempts=max_corruption_attempts,
        max_normalized_cell_edit_distance=max_norm_edit,
    )
    gold_out = uncorrupted_golf_verifier(task_id, source)(copy.deepcopy(inp))
    bad_out = corrupt_fn(copy.deepcopy(inp))
    assert not is_equal_grid(
        gold_out, bad_out
    ), "golf: corrupted output must differ from gold on the sample input"
    norm = normalized_cell_edit_between_outputs(gold_out, bad_out)
    if max_norm_edit is not None:
        assert norm <= max_norm_edit, "golf: normalized edit above threshold"
    return gold_out, bad_out, corrupt_src, binop_idx, norm


def _neighbor_block_html(
    query_label: str,
    query_pair: GridPair,
    neighbors: list[NeighborInstance],
) -> str:
    parts = [
        "<section style='margin-bottom:32px;border:1px solid #444;padding:16px;border-radius:8px;'>"
        "<h2>Nearest alternative-task instances (avg. input / output edit distance)</h2>"
        "<p>Query is a <b>random demonstration pair</b> from this task (not a test-only input). "
        "Each candidate is from another task. Distance = padded Hamming; score = "
        "<code>(norm_in + norm_out) / 2</code> with sentinel <code>-1</code> padding.</p>",
        f"<h3>Query — {escape(query_label)}</h3>",
        "<div style='display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start;'>",
        _grid_to_html(query_pair.input, "Query input"),
        _grid_to_html(query_pair.output, "Query output"),
        "</div>",
    ]
    for i, nb in enumerate(neighbors, start=1):
        parts.append(
            f"<h3>#{i} — {escape(nb.source)} / task <code>{escape(nb.ref_task_id)}</code> / "
            f"{escape(nb.detail)} — "
            f"norm_in={nb.normalized_distance_in:.4f}, norm_out={nb.normalized_distance_out:.4f}, "
            f"avg={nb.normalized_distance_avg:.4f} "
            f"(raw padded cells: in={nb.cell_edit_distance_in}, out={nb.cell_edit_distance_out})</h3>"
        )
        parts.append(
            "<div style='display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start;'>"
            + _grid_to_html(nb.pair.input, "Their input")
            + _grid_to_html(nb.pair.output, "Their output")
            + "</div>"
        )
    parts.append("</section>")
    return "".join(parts)


def _flip_block_html(
    source_label: str,
    original: GridPair,
    flipped: GridPair,
    n_ops: int,
    *,
    flip_max: int,
) -> str:
    m = flip_max
    probs = flip_count_probabilities(m)
    prob_bits = ", ".join(f"P({k})={probs[k]:.3f}" for k in sorted(probs))
    dist_note = (
        f"Number of operations <i>K</i> ∈ {{1,…,{m}}} with weights favoring 1–2 flips and "
        f"downweighting 4–5 (see <code>framework.grids._flip_count_weights</code>). "
        f"For <i>M</i>={m}: {prob_bits}. This run: <b>K={n_ops}</b>."
    )
    return (
        "<section style='margin-bottom:32px;border:1px solid #444;padding:16px;border-radius:8px;'>"
        "<h2>Color-flip augmentation (ARC-GEN stable or dynamic)</h2>"
        "<p>Sampled pair from <b>ARC-GEN stable</b> or <b>dynamic</b> pool. "
        f"{dist_note} Each operation <b>first</b> picks a random cell on input or output "
        "(black cells are less likely). Then the 4-connected same-color neighborhood of that cell "
        "is found: if it has ≥2 cells, the <b>entire</b> neighborhood is recolored to one new color; "
        "otherwise only that cell. Replacement color is biased away from black. Source: "
        f"{escape(source_label)}.</p>"
        "<h3>Original</h3>"
        "<div style='display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start;'>"
        + _grid_to_html(original.input, "Input")
        + _grid_to_html(original.output, "Output")
        + "</div>"
        "<h3>After flips</h3>"
        "<div style='display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start;'>"
        + _grid_to_html(flipped.input, "Input")
        + _grid_to_html(flipped.output, "Output")
        + "</div>"
        "</section>"
    )


def _cross_dynamic_block_html(
    m: CrossDynamicMismatch | None,
    *,
    skipped: bool,
) -> str:
    if skipped:
        return (
            "<section style='margin-bottom:32px;border:1px solid #444;padding:16px;border-radius:8px;'>"
            "<h2>Same-task ARC-GEN dynamic instance mismatch</h2>"
            "<p><i>Skipped (--skip-cross-dynamic).</i></p></section>"
        )
    if m is None:
        return (
            "<section style='margin-bottom:32px;border:1px solid #444;padding:16px;border-radius:8px;'>"
            "<h2>Same-task ARC-GEN dynamic instance mismatch</h2>"
            "<p><i>No valid candidate (no other dynamic instance with different input and non-identical "
            "output in the pool, or generator unavailable).</i></p></section>"
        )
    hybrid = synthetic_input_borrowed_output_pair(m)
    tid = m.task_id
    return (
        "<section style='margin-bottom:32px;border:1px solid #444;padding:16px;border-radius:8px;'>"
        "<h2>Same-task ARC-GEN dynamic instance mismatch</h2>"
        "<p>Task <code>"
        + escape(tid)
        + "</code> only. Anchor is one dynamic sample; we draw more dynamic pairs for the "
        "<b>same</b> task and select an <b>output</b> from a <b>different instance</b> (different "
        "input grid) whose padded edit distance to the anchor <b>output</b> is minimal, excluding "
        "outputs grid-equal to the anchor. The hybrid pairs the anchor <b>input</b> with that "
        "borrowed output.</p>"
        "<p>Normalized output distance <b>"
        f"{m.normalized_output_distance:.4f}</b>, raw padded cells <b>{m.raw_cell_output_distance}</b>.</p>"
        "<h3>Anchor instance (dynamic)</h3>"
        "<div style='display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start;'>"
        + _grid_to_html(m.anchor_pair.input, "Input")
        + _grid_to_html(m.anchor_pair.output, "Output (goes with anchor input)")
        + "</div>"
        "<h3>Hybrid — anchor input, output from another instance</h3>"
        "<div style='display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start;'>"
        + _grid_to_html(hybrid.input, "Input (same as anchor)")
        + _grid_to_html(hybrid.output, "Output (borrowed from other instance)")
        + "</div>"
        "<h3>Other instance (reference — where the borrowed output came from)</h3>"
        "<div style='display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start;'>"
        + _grid_to_html(m.other_instance_pair.input, "Other instance input")
        + _grid_to_html(m.other_instance_pair.output, "Other instance output (= borrowed)")
        + "</div>"
        "</section>"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", default="a85d4709")
    p.add_argument(
        "--verifier",
        default="both",
        choices=[
            "re_arc",
            "golf-google",
            "golf-keymoon",
            "golf-neurips",
            "both",
        ],
        help="re_arc verifiers.py, a golf repo, or both in one HTML page",
    )
    p.add_argument("--golf-source", default="keymoon", choices=["google", "keymoon", "neurips"])
    p.add_argument("--max-input-tries", type=int, default=12)
    p.add_argument(
        "--max-norm-edit",
        type=float,
        default=0.70,
        help="Resample corruption if gold vs corrupt normalized padded edit exceeds this. "
        "Negative disables. Handles different grid sizes via padding.",
    )
    p.add_argument(
        "--max-corruption-attempts",
        type=int,
        default=100,
        help="Per-input attempts for drop/binop indices before giving up.",
    )
    p.add_argument("--neighbor-dynamic", type=int, default=50, help="ARC-GEN dynamic samples in neighbor pool.")
    p.add_argument("--neighbor-seed", type=int, default=0, help="RNG seed for neighbor pool / query sampling.")
    p.add_argument("--demo-seed", type=int, default=1, help="RNG seed for demo pair and flip sampling.")
    p.add_argument(
        "--flip-max",
        type=int,
        default=5,
        help="Largest possible number of recolor ops; count is random (weights favor "
        "1–2 flips, downweight 4–5; see flip_count_probabilities in grids.py).",
    )
    p.add_argument(
        "--skip-neighbors",
        action="store_true",
        help="Skip scanning other tasks (can take several minutes on the full pool).",
    )
    p.add_argument(
        "--skip-cross-dynamic",
        action="store_true",
        help="Skip same-task cross-instance dynamic block (extra generator samples).",
    )
    p.add_argument(
        "--cross-dynamic-pool",
        type=int,
        default=50,
        help="Number of extra ARC-GEN dynamic pairs to draw for the same task when searching.",
    )
    p.add_argument("--out-dir", default="Demonstrations/corruption_demos")
    args = p.parse_args()
    if args.flip_max < 1:
        raise SystemExit("--flip-max must be at least 1")
    if args.cross_dynamic_pool < 1:
        raise SystemExit("--cross-dynamic-pool must be at least 1")

    max_norm: float | None = args.max_norm_edit if args.max_norm_edit >= 0 else None

    task = load_task(args.task_id)
    if task.arc_gen_generator is None:
        raise SystemExit(f"Task {args.task_id} has no ARC-GEN dynamic generator.")

    rng = random.Random(0)
    rng_demo = random.Random(args.demo_seed)
    rng_neighbors = random.Random(args.neighbor_seed)

    verifier: VerifierChoice = args.verifier  # type: ignore[assignment]
    golf_src: GolfSource = args.golf_source  # type: ignore[assignment]

    inp: Grid | None = None
    last_err: str | None = None

    re_block = ""
    golf_block = ""
    neighbor_block = ""
    flip_block = ""
    cross_block = ""
    cross_result: CrossDynamicMismatch | None = None
    gold_r = bad_r = gold_g = bad_g = None
    src_r = ""
    src_g = ""
    drop_idx = -1
    binop_idx = -1
    norm_r = norm_g = 0.0
    neighbors: list[NeighborInstance] = []
    query_demo_pair: GridPair | None = None
    query_demo_label = ""
    flip_source_label = ""
    flip_original: GridPair | None = None
    flip_augmented: GridPair | None = None
    n_color_flips = 0

    for attempt in range(args.max_input_tries):
        pair = task.arc_gen_generator(1)[0]
        inp_try = copy.deepcopy(pair.input)
        dynamic_anchor_pair = copy.deepcopy(pair)
        try:
            query_demo_pair, query_demo_label = sample_demonstration_pair(task, rng_demo)
            flip_original, flip_source_label = sample_arc_gen_pair_for_color_flips(task, rng_demo)
            n_flip_ops = sample_flip_count_favor_one(rng_demo, max_flips=args.flip_max)
            flip_augmented, n_color_flips = connected_component_color_flips_on_pair(
                flip_original,
                rng_demo,
                num_ops=n_flip_ops,
            )
            if verifier in ("re_arc", "both"):
                gold_r, bad_r, src_r, drop_idx, norm_r = _run_re_arc(
                    args.task_id,
                    rng,
                    inp_try,
                    max_norm_edit=max_norm,
                    max_corruption_attempts=args.max_corruption_attempts,
                )
            if verifier in ("golf-google", "golf-keymoon", "golf-neurips", "both"):
                gs = _parse_golf_source(verifier) if verifier != "both" else golf_src
                gold_g, bad_g, src_g, binop_idx, norm_g = _run_golf(
                    args.task_id,
                    gs,
                    rng,
                    inp_try,
                    max_norm_edit=max_norm,
                    max_corruption_attempts=args.max_corruption_attempts,
                )
            inp = inp_try
            if args.skip_neighbors:
                neighbors = []
            else:
                neighbors = find_nearest_alternative_instances(
                    query_demo_pair,
                    args.task_id,
                    rng_neighbors,
                    k=3,
                    num_dynamic=args.neighbor_dynamic,
                )
            if args.skip_cross_dynamic:
                cross_result = None
            else:
                cross_result = find_cross_dynamic_mismatch(
                    args.task_id,
                    dynamic_anchor_pair,
                    pool_size=args.cross_dynamic_pool,
                )
            break
        except Exception as e:
            last_err = repr(e)
            continue
    else:
        raise RuntimeError(
            f"Could not produce valid corruption / neighbors after {args.max_input_tries} input tries. "
            f"Last error: {last_err}"
        )

    assert inp is not None
    assert query_demo_pair is not None
    assert flip_original is not None and flip_augmented is not None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.task_id}_verifier_corruption_demo.html"

    thr_note = "disabled" if max_norm is None else f"{max_norm:.2f}"
    if verifier in ("re_arc", "both") and gold_r is not None and bad_r is not None:
        re_block = (
            "<section style='margin-bottom:32px;border:1px solid #444;padding:16px;border-radius:8px;'>"
            "<h2>re_arc (<code>external/re_arc/verifiers.py</code>)</h2>"
            "<p>Gold: compiled <code>verify_&lt;task&gt;</code>. Corrupt: one middle assignment "
            f"removed; loads rewired to the previous intermediate (<b>drop_index={drop_idx}</b>). "
            f"Normalized <b>padded</b> cell edit distance (gold vs corrupt outputs): "
            f"<b>{norm_r:.4f}</b> (threshold ≤ {escape(thr_note)}).</p>"
            "<div style='display:flex;flex-wrap:wrap;gap:24px;align-items:flex-start;'>"
            + _grid_to_html(inp, "Input (ARC-GEN dynamic)")
            + _grid_to_html(gold_r, "Output — gold (re_arc verifier)")
            + _grid_to_html(bad_r, "Output — corrupted re_arc verifier")
            + "</div>"
            "<h3>Corrupted function (excerpt)</h3>"
            f"<pre style='background:#111;padding:12px;overflow:auto;max-height:280px;'>{escape(src_r[:8000])}</pre>"
            "</section>"
        )

    if verifier in ("golf-google", "golf-keymoon", "golf-neurips", "both"):
        assert gold_g is not None and bad_g is not None
        gs_label = (
            _parse_golf_source(verifier) if verifier != "both" else golf_src
        )
        golf_block = (
            "<section style='margin-bottom:32px;border:1px solid #444;padding:16px;border-radius:8px;'>"
            f"<h2>Golf ({escape(gs_label)})</h2>"
            "<p>Gold: compiled golf solution file. Corrupt: one <code>BinOp</code> replaced by its "
            f"left subtree (<b>binop_index={binop_idx}</b> over binary ops in "
            f"<code>ast.walk</code> order in the lambda body). "
            f"Normalized padded cell edit distance: <b>{norm_g:.4f}</b> (threshold ≤ {escape(thr_note)}).</p>"
            "<div style='display:flex;flex-wrap:wrap;gap:24px;align-items:flex-start;'>"
            + _grid_to_html(inp, "Input (same sample)" if verifier == "both" else "Input (ARC-GEN dynamic)")
            + _grid_to_html(gold_g, "Output — gold (golf)")
            + _grid_to_html(bad_g, "Output — corrupted golf")
            + "</div>"
            "<h3>Corrupted module (excerpt)</h3>"
            f"<pre style='background:#111;padding:12px;overflow:auto;max-height:280px;'>{escape(src_g[:8000])}</pre>"
            "</section>"
        )

    if neighbors:
        neighbor_block = _neighbor_block_html(query_demo_label, query_demo_pair, neighbors)
    elif args.skip_neighbors:
        neighbor_block = (
            "<section style='margin-bottom:32px;border:1px solid #444;padding:16px;border-radius:8px;'>"
            "<h2>Nearest alternative-task instances</h2>"
            "<p><i>Skipped (--skip-neighbors).</i></p></section>"
        )
    else:
        neighbor_block = (
            "<section style='margin-bottom:32px;border:1px solid #444;padding:16px;border-radius:8px;'>"
            "<h2>Nearest alternative-task instances</h2>"
            "<p><i>No candidates found.</i></p></section>"
        )

    flip_block = _flip_block_html(
        flip_source_label,
        flip_original,
        flip_augmented,
        n_color_flips,
        flip_max=args.flip_max,
    )

    cross_block = _cross_dynamic_block_html(
        cross_result,
        skipped=args.skip_cross_dynamic,
    )

    title_bits = []
    if verifier == "both":
        title_bits.append("re_arc + golf")
    elif verifier == "re_arc":
        title_bits.append("re_arc")
    else:
        title_bits.append(verifier.replace("golf-", "golf "))

    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'/>"
        f"<title>Corruption demo {escape(args.task_id)}</title></head>"
        "<body style='font-family:Segoe UI,sans-serif;background:#1a1a1a;color:#eee;'>"
        f"<h1>Verifier corruption — {escape(args.task_id)} ({escape(' / '.join(title_bits))})</h1>"
        "<p>Gold and corrupt outputs differ; normalized <b>padded</b> edit distance caps how far the "
        "corrupt output drifts from gold (works across different grid sizes).</p>"
        + re_block
        + golf_block
        + neighbor_block
        + flip_block
        + cross_block
        + "</body></html>"
    )
    path.write_text(html, encoding="utf-8")
    print(path)
    print(f"query demonstration: {query_demo_label}; max_norm_edit={max_norm}; flips={n_color_flips}")
    if gold_r is not None:
        print("re_arc norm_edit (padded):", norm_r)
        print("re_arc gold grid (text):\n", pretty_grid(gold_r))
        print("re_arc corrupt grid (text):\n", pretty_grid(bad_r))
    if gold_g is not None:
        print("golf norm_edit (padded):", norm_g)
        print("golf gold grid (text):\n", pretty_grid(gold_g))
        print("golf corrupt grid (text):\n", pretty_grid(bad_g))
    for nb in neighbors:
        print(
            nb.source,
            nb.ref_task_id,
            f"avg={nb.normalized_distance_avg:.4f}",
            f"in={nb.normalized_distance_in:.4f}",
            f"out={nb.normalized_distance_out:.4f}",
        )
    if cross_result is not None:
        print(
            "cross-dynamic (same task):",
            cross_result.task_id,
            f"norm_out={cross_result.normalized_output_distance:.4f}",
        )


if __name__ == "__main__":
    main()
