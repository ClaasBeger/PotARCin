from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

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


def _render_pair(pair: GridPair, *, label: str) -> None:
    height = max(grid_height_px(pair.input), grid_height_px(pair.output))
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.components.v1.html(
            grid_html(pair.input, title=f"{label} – input"), height=height
        )
    with c2:
        st.components.v1.html(
            grid_html(pair.output, title=f"{label} – output"), height=height
        )


def _render_grid(grid: Grid, *, label: str) -> None:
    height = grid_height_px(grid)
    st.components.v1.html(grid_html(grid, title=label), height=height)


def main() -> None:
    st.set_page_config(page_title="PotARCin Viewer", layout="wide")

    st.markdown(
        "<h1 style='text-align:center; margin-bottom: 0.5rem;'>PotARCin Viewer</h1>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Inspect original ARC demonstrations, test inputs, stable synthetic pairs "
        "(re_arc + ARC-GEN), dynamic samples, and verifier availability."
    )

    task_ids = _list_original_task_ids()
    if not task_ids:
        st.error(f"Could not find ARC tasks in `{ARC_ORIGINAL_DIR}`.")
        st.stop()

    default_idx = task_ids.index("a85d4709") if "a85d4709" in task_ids else 0
    task_id = st.selectbox("Select task", task_ids, index=default_idx)

    with st.spinner("Loading task..."):
        task = load_task(task_id)

    st.divider()

    st.subheader("Task demonstrations & test input")
    with st.container():
        if not task.train_pairs:
            st.info("No training pairs found.")
        else:
            for i, pair in enumerate(task.train_pairs):
                st.markdown(f"**Train pair {i+1}**")
                _render_pair(pair, label=f"Train {i+1}")

                # Equivalence check against primary verifier, if available.
                status = "Unknown (no primary verifier)"
                symbol = "⚪"
                color = "gray"
                alt_output: Optional[Grid] = None
                if task.verifier is not None:
                    try:
                        out = task.verifier(pair.input)
                        if is_equal_grid(out, pair.output):
                            status = "Matches primary verifier"
                            symbol = "✅"
                            color = "green"
                        else:
                            status = "Does NOT match primary verifier"
                            symbol = "❌"
                            color = "red"
                            alt_output = out
                    except Exception as e:
                        status = f"Verifier error: {e}"
                        symbol = "⚠️"
                        color = "orange"

                st.markdown(
                    f"<span style='color:{color}; font-size:0.9rem;'>{symbol} {status}</span>",
                    unsafe_allow_html=True,
                )

                # If the verifier produced a valid but different output, show it.
                if alt_output is not None:
                    with st.expander("Verifier alternative output", expanded=False):
                        _render_grid(alt_output, label="Verifier output")

                st.markdown("---")

        if task.test_inputs:
            st.markdown("**Original test inputs**")
            for i, grid in enumerate(task.test_inputs):
                with st.expander(f"Test input {i+1}", expanded=(i == 0)):
                    _render_grid(grid, label=f"Test input {i+1}")

    st.divider()

    st.subheader("Stable synthetic examples")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**re_arc stable synthetic pairs**")
        if task.re_arc_synthetic_pairs:
            st.caption(f"{len(task.re_arc_synthetic_pairs)} pairs available")
            pair = task.re_arc_synthetic_pairs[0]
            _render_pair(pair, label="re_arc stable (first)")
        else:
            st.info("None available.")
    with c2:
        st.markdown("**ARC-GEN stable synthetic pairs**")
        if task.arc_gen_synthetic_pairs:
            st.caption(f"{len(task.arc_gen_synthetic_pairs)} pairs available")
            pair = task.arc_gen_synthetic_pairs[0]
            _render_pair(pair, label="ARC-GEN stable (first)")
        else:
            st.info("None available.")

    st.divider()

    st.subheader("Dynamic synthetic examples")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**re_arc dynamic**")
        st.caption(f"Generator available: `{task.re_arc_generator is not None}`")
        if task.re_arc_generator is not None:
            try:
                gen_pair = task.re_arc_generator(1)[0]
                _render_pair(gen_pair, label="re_arc dynamic (1 sample)")
            except Exception as e:
                st.error(f"Generator failed: {e}")
        else:
            st.info("No generator found for this task.")
    with c2:
        st.markdown("**ARC-GEN dynamic**")
        st.caption(f"Generator available: `{task.arc_gen_generator is not None}`")
        if task.arc_gen_generator is not None:
            try:
                gen_pair = task.arc_gen_generator(1)[0]
                _render_pair(gen_pair, label="ARC-GEN dynamic (1 sample)")
            except Exception as e:
                st.error(f"Generator failed: {e}")
        else:
            st.info("No generator found for this task.")

    st.divider()

    st.subheader("Verifiers")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Primary verifier (re_arc)**")
        if task.verifier is not None:
            st.success("Available")
            # Run on first stable re_arc synthetic pair, if present.
            if task.re_arc_synthetic_pairs:
                pair = task.re_arc_synthetic_pairs[0]
                try:
                    out = task.verifier(pair.input)
                    ok = is_equal_grid(out, pair.output)
                    symbol = "✅" if ok else "❌"
                    color = "green" if ok else "red"
                    msg = (
                        "Matches re_arc synthetic pair"
                        if ok
                        else "Does NOT match re_arc synthetic pair"
                    )
                    st.markdown(
                        f"<span style='color:{color}; font-size:0.9rem;'>{symbol} {msg}</span>",
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.markdown(
                        f"<span style='color:orange; font-size:0.9rem;'>⚠️ Verifier error on re_arc synthetic pair: {e}</span>",
                        unsafe_allow_html=True,
                    )
            # Also, if ARC-GEN stable pairs exist, run on the first of those.
            if task.arc_gen_synthetic_pairs:
                pair = task.arc_gen_synthetic_pairs[0]
                try:
                    out = task.verifier(pair.input)
                    ok = is_equal_grid(out, pair.output)
                    symbol = "✅" if ok else "❌"
                    color = "green" if ok else "red"
                    msg = (
                        "Matches ARC-GEN synthetic pair"
                        if ok
                        else "Does NOT match ARC-GEN synthetic pair"
                    )
                    st.markdown(
                        f"<span style='color:{color}; font-size:0.9rem;'>{symbol} {msg}</span>",
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.markdown(
                        f"<span style='color:orange; font-size:0.9rem;'>⚠️ Verifier error on ARC-GEN synthetic pair: {e}</span>",
                        unsafe_allow_html=True,
                    )
        else:
            st.warning("Not available")
    with c2:
        st.markdown("**Alternative verifiers**")

        def _show_alt(name: str, verifier):
            st.markdown(f"**{name}**")
            if verifier is None:
                st.warning("Not available")
                return
            st.success("Available")
            if task.re_arc_synthetic_pairs:
                pair = task.re_arc_synthetic_pairs[0]
                try:
                    out = verifier(pair.input)
                    ok = is_equal_grid(out, pair.output)
                    symbol = "✅" if ok else "❌"
                    color = "green" if ok else "red"
                    msg = "Matches re_arc synthetic pair" if ok else "Does NOT match re_arc synthetic pair"
                    st.markdown(
                        f"<span style='color:{color}; font-size:0.9rem;'>{symbol} {msg}</span>",
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.markdown(
                        f"<span style='color:orange; font-size:0.9rem;'>⚠️ Verifier error on re_arc synthetic pair: {e}</span>",
                        unsafe_allow_html=True,
                    )
            if task.arc_gen_synthetic_pairs:
                pair = task.arc_gen_synthetic_pairs[0]
                try:
                    out = verifier(pair.input)
                    ok = is_equal_grid(out, pair.output)
                    symbol = "✅" if ok else "❌"
                    color = "green" if ok else "red"
                    msg = "Matches ARC-GEN synthetic pair" if ok else "Does NOT match ARC-GEN synthetic pair"
                    st.markdown(
                        f"<span style='color:{color}; font-size:0.9rem;'>{symbol} {msg}</span>",
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.markdown(
                        f"<span style='color:orange; font-size:0.9rem;'>⚠️ Verifier error on ARC-GEN synthetic pair: {e}</span>",
                        unsafe_allow_html=True,
                    )

        _show_alt("v2 (google-code-golf-2025)", task.secondary_verifier)
        _show_alt("v3 (golf)", getattr(task, "tertiary_verifier", None))
        _show_alt("v4 (NeurIPS-Code-Golf-2025)", getattr(task, "quaternary_verifier", None))


if __name__ == "__main__":
    main()

