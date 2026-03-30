from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import streamlit as st

# Ensure project root is on sys.path so we can import `framework` when this
# file is executed via `streamlit run Viewer/app.py`.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from framework.grids import Grid, GridPair, is_equal_grid
from framework.tasks.arc_dataset import ARC_ORIGINAL_DIR, load_task
from Viewer.render import grid_html, grid_height_px


def _list_original_task_ids() -> List[str]:
    if not ARC_ORIGINAL_DIR.exists():
        return []
    ids = [p.stem for p in ARC_ORIGINAL_DIR.glob("*.json")]
    ids.sort()
    return ids


def _render_pair(pair: GridPair, *, label: str, max_px: int = 26) -> None:
    height = max(
        grid_height_px(pair.input, max_px=max_px),
        grid_height_px(pair.output, max_px=max_px),
    )
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.components.v1.html(
            grid_html(pair.input, title=f"{label} – input", max_px=max_px), height=height
        )
    with c2:
        st.components.v1.html(
            grid_html(pair.output, title=f"{label} – output", max_px=max_px), height=height
        )


def _render_grid(
    grid: Grid,
    *,
    label: str,
    max_px: int = 26,
    border_color: str | None = None,
) -> None:
    height = grid_height_px(grid, max_px=max_px)
    st.components.v1.html(
        grid_html(grid, title=label, max_px=max_px, border_color=border_color, border_width_px=3),
        height=height,
    )


def _stable_or_train_pairs(task) -> List[GridPair]:
    if task.arc_gen_synthetic_pairs:
        return list(task.arc_gen_synthetic_pairs)
    return list(task.train_pairs)


def _pick_pairs(pool: Sequence[GridPair], count: int, *, seed: int) -> List[GridPair]:
    if not pool:
        return []
    idxs = list(range(len(pool)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    return [pool[i] for i in idxs[: min(count, len(pool))]]


def _pick_other_task_pairs(current_task_id: str, count: int) -> List[GridPair]:
    out: List[GridPair] = []
    for tid in _list_original_task_ids():
        if tid == current_task_id:
            continue
        try:
            t = load_task(tid)
        except Exception:
            continue
        pairs = _stable_or_train_pairs(t)
        if pairs:
            out.append(pairs[0])
        if len(out) >= count:
            break
    return out


def _corrupt_grid(grid: Grid, *, seed: int) -> Grid:
    rng = random.Random(seed)
    corrupted = [list(row) for row in grid]
    h = len(corrupted)
    w = len(corrupted[0]) if h else 0
    if h == 0 or w == 0:
        return corrupted
    k = max(1, (h * w) // 12)
    for _ in range(k):
        r = rng.randrange(h)
        c = rng.randrange(w)
        cur = corrupted[r][c]
        choices = [x for x in range(10) if x != cur]
        corrupted[r][c] = rng.choice(choices)
    return corrupted


def _render_io_with_arrow(pair: GridPair, *, title: str, max_px: int = 16) -> None:
    st.markdown(f"**{title}**")
    c1, c2, c3 = st.columns([1, 0.25, 1], gap="small")
    with c1:
        _render_grid(pair.input, label="Input", max_px=max_px)
    with c2:
        st.markdown("<div style='text-align:center; font-size:1.6rem; margin-top:2.3rem;'>→</div>", unsafe_allow_html=True)
    with c3:
        _render_grid(pair.output, label="Output", max_px=max_px)


def _render_dimension_map(task, task_id: str) -> None:
    stable_pool = _stable_or_train_pairs(task)
    if not stable_pool:
        st.warning("This task has no train/stable examples to render.")
        return

    picked = _pick_pairs(stable_pool, 10, seed=abs(hash(task_id)) % (2**31))
    center_pairs = picked[: min(3, len(picked))]

    # Top panels
    top_left, top_mid, top_right = st.columns(3, gap="large")
    with top_left:
        st.markdown("### ↖ Definition")
        st.markdown(
            "<div style='font-family:monospace; border:1px solid #888; border-radius:8px; "
            "padding:0.5rem; background:#111; color:#ddd;'>"
            "def solve(grid):<br/>"
            "&nbsp;&nbsp;# inferred transformation<br/>"
            "&nbsp;&nbsp;return transform(grid)"
            "</div>",
            unsafe_allow_html=True,
        )
        def_pairs = picked[3:6] if len(picked) >= 6 else center_pairs[:]
        for i, pair in enumerate(def_pairs[:3]):
            _render_io_with_arrow(pair, title=f"Small demo {i+1}", max_px=14)

    with top_mid:
        st.markdown("### ↑ Classification")
        same_pair = picked[6] if len(picked) >= 7 else center_pairs[0]
        other_pairs = _pick_other_task_pairs(task_id, 2)
        cls_pairs = [same_pair] + other_pairs
        labels = ["Same task (correct)", "Different task (incorrect)", "Different task (incorrect)"]
        colors = ["#2ECC40", "#FF4136", "#FF4136"]
        for i, pair in enumerate(cls_pairs):
            if i >= 3:
                break
            st.markdown(f"**{labels[i]}**")
            c1, c2, c3 = st.columns([1, 0.25, 1], gap="small")
            with c1:
                _render_grid(pair.input, label="Input", max_px=14, border_color=colors[i])
            with c2:
                st.markdown("<div style='text-align:center; font-size:1.6rem; margin-top:2.3rem;'>→</div>", unsafe_allow_html=True)
            with c3:
                _render_grid(pair.output, label="Output", max_px=14, border_color=colors[i])

    with top_right:
        st.markdown("### ↗ Constrained Generation")
        gen_pair: Optional[GridPair] = None
        if task.arc_gen_generator is not None:
            try:
                gen_pair = task.arc_gen_generator(1)[0]
            except Exception:
                gen_pair = None
        if gen_pair is None:
            gen_pair = picked[7] if len(picked) >= 8 else center_pairs[0]
        _render_io_with_arrow(gen_pair, title="New generated pair", max_px=14)

    st.markdown("---")

    # Center demonstrations
    st.markdown("## Center Demonstrations")
    st.markdown("<div style='text-align:center; font-size:1.4rem;'>↖ &nbsp; ↑ &nbsp; ↗</div>", unsafe_allow_html=True)
    for i, pair in enumerate(center_pairs):
        _render_pair(pair, label=f"Demonstration {i+1}", max_px=24)
    st.markdown("<div style='text-align:center; font-size:1.4rem;'>↙ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ↘</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Bottom panels with center spacer.
    bot_left, _bot_mid, bot_right = st.columns([1, 0.35, 1], gap="large")
    with bot_left:
        st.markdown("### ↙ Editing")
        edit_pair = picked[8] if len(picked) >= 9 else center_pairs[0]
        corrupted = _corrupt_grid(edit_pair.output, seed=(abs(hash(task_id)) + 17) % (2**31))
        c1, c2 = st.columns(2, gap="small")
        with c1:
            _render_grid(edit_pair.input, label="Input", max_px=14)
        with c2:
            _render_grid(corrupted, label="Corrupted output", max_px=14, border_color="#FF4136")
        st.markdown("<div style='text-align:center; font-size:1.8rem;'>↓</div>", unsafe_allow_html=True)
        _render_grid(edit_pair.output, label="Correct output", max_px=14, border_color="#2ECC40")

    with bot_right:
        st.markdown("### ↘ Inversion")
        inv_pair = picked[9] if len(picked) >= 10 else center_pairs[-1]
        c1, c2, c3 = st.columns([1, 0.25, 1], gap="small")
        with c1:
            _render_grid(inv_pair.output, label="Output (given)", max_px=14)
        with c2:
            st.markdown("<div style='text-align:center; font-size:1.6rem; margin-top:2.3rem;'>→</div>", unsafe_allow_html=True)
        with c3:
            _render_grid(inv_pair.input, label="Input (to infer)", max_px=14)


def _render_diagnostics(task) -> None:
    st.subheader("Task demonstrations & test input")
    if not task.train_pairs:
        st.info("No training pairs found.")
    else:
        for i, pair in enumerate(task.train_pairs):
            st.markdown(f"**Train pair {i+1}**")
            _render_pair(pair, label=f"Train {i+1}", max_px=24)
            st.markdown("---")

    if task.test_inputs:
        st.markdown("**Original test inputs**")
        for i, grid in enumerate(task.test_inputs):
            with st.expander(f"Test input {i+1}", expanded=(i == 0)):
                _render_grid(grid, label=f"Test input {i+1}", max_px=24)

    st.divider()
    st.subheader("Stable synthetic examples")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**re_arc stable synthetic pairs**")
        if task.re_arc_synthetic_pairs:
            st.caption(f"{len(task.re_arc_synthetic_pairs)} pairs available")
            _render_pair(task.re_arc_synthetic_pairs[0], label="re_arc stable (first)", max_px=20)
        else:
            st.info("None available.")
    with c2:
        st.markdown("**ARC-GEN stable synthetic pairs**")
        if task.arc_gen_synthetic_pairs:
            st.caption(f"{len(task.arc_gen_synthetic_pairs)} pairs available")
            _render_pair(task.arc_gen_synthetic_pairs[0], label="ARC-GEN stable (first)", max_px=20)
        else:
            st.info("None available.")

    st.divider()
    st.subheader("Dynamic synthetic examples")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**re_arc dynamic**")
        if task.re_arc_generator is not None:
            try:
                _render_pair(task.re_arc_generator(1)[0], label="re_arc dynamic (1 sample)", max_px=20)
            except Exception as e:
                st.error(f"Generator failed: {e}")
        else:
            st.info("No generator found for this task.")
    with c2:
        st.markdown("**ARC-GEN dynamic**")
        if task.arc_gen_generator is not None:
            try:
                _render_pair(task.arc_gen_generator(1)[0], label="ARC-GEN dynamic (1 sample)", max_px=20)
            except Exception as e:
                st.error(f"Generator failed: {e}")
        else:
            st.info("No generator found for this task.")

    st.divider()
    st.subheader("Verifier quick check")
    if task.verifier is None:
        st.warning("Primary verifier not available.")
        return
    st.success("Primary verifier available")
    sample = None
    if task.arc_gen_synthetic_pairs:
        sample = task.arc_gen_synthetic_pairs[0]
    elif task.re_arc_synthetic_pairs:
        sample = task.re_arc_synthetic_pairs[0]
    elif task.train_pairs:
        sample = task.train_pairs[0]
    if sample is None:
        st.info("No sample pair available for verifier check.")
        return
    try:
        out = task.verifier(sample.input)
        ok = is_equal_grid(out, sample.output)
        st.markdown("✅ Matches sample pair" if ok else "❌ Does not match sample pair")
    except Exception as e:
        st.error(f"Verifier error: {e}")


def main() -> None:
    st.set_page_config(page_title="PotARCin Viewer", layout="wide")

    st.markdown(
        "<h1 style='text-align:center; margin-bottom: 0.5rem;'>PotARCin Viewer</h1>",
        unsafe_allow_html=True,
    )

    task_ids = _list_original_task_ids()
    if not task_ids:
        st.error(f"Could not find ARC tasks in `{ARC_ORIGINAL_DIR}`.")
        st.stop()

    default_idx = task_ids.index("a85d4709") if "a85d4709" in task_ids else 0
    top_c1, top_c2 = st.columns([2, 1], gap="large")
    with top_c1:
        task_id = st.selectbox("Select task", task_ids, index=default_idx)
    with top_c2:
        view_mode = st.selectbox("View mode", ["Dimension map", "Diagnostics"], index=0)

    with st.spinner("Loading task..."):
        task = load_task(task_id)

    st.caption(
        "Dimension map shows definition/classification/constrained generation/editing/inversion. "
        "Diagnostics shows the detailed synthetic/verifier inspection view."
    )
    st.divider()

    if view_mode == "Dimension map":
        _render_dimension_map(task, task_id)
    else:
        _render_diagnostics(task)


if __name__ == "__main__":
    main()

