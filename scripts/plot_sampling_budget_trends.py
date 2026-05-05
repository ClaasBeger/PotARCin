#!/usr/bin/env python3
"""
Plot sampling-budget trend lines for subset-sampling stress runs.

Input:
  - A ``mode: subset_sampling_stress`` JSON (see ``evaluation/multi_dimension_eval_runs/**``).
    It contains:
      - ``seeds``: the sampled RNG seeds (in the order you want to treat as sample budget 1..N)
      - ``runs``: a list of per-(task_id, seed) dimension evaluation payloads

Computed curves (all in %):
  1. Average full-task accuracy:
       mean over tasks and seeds of strict full-task pass (all 5 strict dimensions).
  2. Hard full-task accuracy:
       for each task, the task counts as correct only if it passes strict full-task
       for *every* seed in the first k seeds; then average over tasks.
  3. Aggregated dimension average:
       for each dimension, compute the fraction of tasks that pass that dimension
       for *every* seed in the first k seeds; then average those fractions across dimensions.

Output:
  - PDF + PNG to ``evaluation/figures/`` by default.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_plot_script_module():
    """
    Reuse strict-pass helpers from the existing plot script so semantics match.
    """
    plot_script = Path(__file__).resolve().parent / "plot_output_grid_vs_potarcin_paper.py"
    spec = importlib.util.spec_from_file_location("potarcin_plot_helpers", plot_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helpers from {plot_script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def compute_trends(
    payload: dict,
    helpers_mod,
) -> dict[str, list[float]]:
    seeds: list[int] = payload.get("seeds") or []
    if not seeds:
        raise ValueError("Missing or empty 'seeds' in subset_sampling_stress payload")

    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Missing or invalid 'runs' in subset_sampling_stress payload")

    dim_order: tuple[str, ...] = tuple(getattr(helpers_mod, "POTARCIN_DIMENSION_ORDER"))
    dim_labels: tuple[str, ...] = tuple(getattr(helpers_mod, "DIMENSION_COLUMN_LABELS"))
    strict_prefix = len(dim_order)

    # Index runs by (task_id, seed).
    by_task_seed: dict[tuple[str, int], dict] = {}
    task_ids: set[str] = set()
    for r in runs:
        if not isinstance(r, dict):
            continue
        tid = r.get("task_id")
        sd = r.get("seed")
        if not isinstance(tid, str) or not isinstance(sd, int):
            continue
        by_task_seed[(tid, sd)] = r
        task_ids.add(tid)

    task_ids_l = sorted(task_ids)
    if not task_ids_l:
        raise ValueError("Could not infer any task_ids from 'runs'")

    # Precompute per-(task, seed) strict pass booleans.
    full_pass: dict[tuple[str, int], bool] = {}
    dim_pass: dict[tuple[str, int, str], bool] = {}

    for tid in task_ids_l:
        for sd in seeds:
            run = by_task_seed.get((tid, sd))
            if not isinstance(run, dict):
                # If missing, treat as failure for that (task, seed).
                full_pass[(tid, sd)] = False
                for d in dim_order:
                    dim_pass[(tid, sd, d)] = False
                continue

            # Full-task strict pass = all 5 strict dimensions.
            full_pass[(tid, sd)] = bool(
                helpers_mod._task_passes_dimension_prefix_strict(run, strict_prefix)
            )
            for d in dim_order:
                dim_pass[(tid, sd, d)] = bool(helpers_mod._task_strict_passes_potarcin_dimension(run, d))

    T = len(task_ids_l)
    N = len(seeds)

    xs = [0] + list(range(1, N + 1))
    # k=0 baseline: vacuous success for "all samples correct", and by request all curves start at 100%.
    avg_full: list[float] = [100.0]
    hard_full: list[float] = [100.0]
    per_dim_pct: dict[str, list[float]] = {d: [100.0] for d in dim_order}

    for k in range(1, N + 1):
        seeds_k = seeds[:k]

        # Average full-task accuracy.
        correct = 0
        for tid in task_ids_l:
            for sd in seeds_k:
                if full_pass[(tid, sd)]:
                    correct += 1
        avg_full.append(100.0 * correct / (T * k))

        # Hard full-task accuracy: all samples correct.
        hard_correct = 0
        for tid in task_ids_l:
            if all(full_pass[(tid, sd)] for sd in seeds_k):
                hard_correct += 1
        hard_full.append(100.0 * hard_correct / T)

        # Per-dimension hard pass: for each dimension, require all samples correct,
        # then track the fraction of tasks that satisfy this per dimension.
        for d in dim_order:
            d_correct = 0
            for tid in task_ids_l:
                if all(dim_pass[(tid, sd, d)] for sd in seeds_k):
                    d_correct += 1
            per_dim_pct[d].append(100.0 * d_correct / T)

    return {
        "x_samples": [float(x) for x in xs],
        "avg_full_task_pct": avg_full,
        "hard_full_task_pct": hard_full,
        "per_dim_pct": per_dim_pct,
        "dim_order": list(dim_order),
        "dim_labels": list(dim_labels),
    }


def plot_trends(trends: dict, run_tag: str, out_dir: Path, basename: str) -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "legend.fontsize": 7.6,
            "figure.constrained_layout.use": True,
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )

    x = np.asarray(trends["x_samples"], dtype=float)

    # Visual palette (chosen to be professional + colorblind-friendly-ish).
    col_avg = "#395D9F"  # blue
    col_hard = "#0B7285"  # teal-ish
    # Per-dimension styling (distinct color + marker per line for readability).
    dim_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
    dim_markers = ["o", "s", "^", "D", "P"]

    fig, ax = plt.subplots(1, 1, figsize=(4.2, 2.8))

    ln_avg, = ax.plot(
        x,
        trends["avg_full_task_pct"],
        color=col_avg,
        marker="X",
        markersize=3.7,
        linewidth=1.3,
        label="Average full-task accuracy",
        zorder=3,
    )
    ln_hard, = ax.plot(
        x,
        trends["hard_full_task_pct"],
        color=col_hard,
        marker="s",
        markersize=3.7,
        linewidth=1.3,
        label="Hard full-task accuracy",
        zorder=3,
    )
    # One line per dimension (hard per-dimension pass over all samples).
    dim_order = trends["dim_order"]
    dim_labels = trends["dim_labels"]
    per_dim_pct = trends["per_dim_pct"]
    for idx, dim in enumerate(dim_order):
        lbl = dim_labels[idx] if idx < len(dim_labels) else dim
        color = dim_colors[idx % len(dim_colors)]
        marker = dim_markers[idx % len(dim_markers)]
        ln_dim, = ax.plot(
            x,
            per_dim_pct[dim],
            color=color,
            marker=marker,
            markersize=3.4,
            linewidth=1.1,
            label=lbl,
            zorder=2,
        )
        _ = ln_dim

    ax.set_xlabel("Number of samples (k)")
    ax.set_ylabel("Accuracy (%)")

    ax.grid(axis="y", linestyle="-", linewidth=0.55, alpha=0.34, color="#d1d5db", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlim(-0.1, float(x.max()) + 0.2)
    ax.set_ylim(0.0, 103.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{basename}.png"
    pdf = out_dir / f"{basename}.pdf"
    fig.savefig(png, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05)
    handles, labels = ax.get_legend_handles_labels()
    plt.close(fig)
    _save_legend_only(handles, labels, out_dir=out_dir, basename=basename)


def _save_legend_only(handles, labels, out_dir: Path, basename: str) -> None:
    fig_leg, ax_leg = plt.subplots(1, 1, figsize=(6.8, 1.0))
    ax_leg.axis("off")
    leg = ax_leg.legend(
        handles,
        labels,
        loc="center",
        ncol=2,
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        edgecolor="#d1d5db",
        fontsize=8,
        handlelength=1.8,
        columnspacing=1.4,
    )
    png_leg = out_dir / f"{basename}_legend.png"
    pdf_leg = out_dir / f"{basename}_legend.pdf"
    fig_leg.savefig(png_leg, bbox_inches="tight", pad_inches=0.04)
    fig_leg.savefig(pdf_leg, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig_leg)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the subset_sampling_stress JSON (e.g. sampling_stress_*.json).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for the plot (default: evaluation/figures).",
    )
    p.add_argument(
        "--basename",
        type=str,
        default="sampling_budget_trends",
        help="Base filename stem for the outputs.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload_path: Path = args.input.resolve()
    payload = _load_json(payload_path)

    run_tag = str(payload.get("run_tag") or payload_path.parent.name)
    # Find repo root (the workspace root folder).
    repo_root = payload_path
    while repo_root.name != "PotARCin" and repo_root.parent != repo_root:
        repo_root = repo_root.parent
    out_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else repo_root / "evaluation" / "figures"
    )

    helpers_mod = _load_plot_script_module()
    trends = compute_trends(payload, helpers_mod)
    plot_trends(trends, run_tag=run_tag, out_dir=out_dir, basename=args.basename)

    print("Wrote:")
    print("  PNG:", out_dir / f"{args.basename}.png")
    print("  PDF:", out_dir / f"{args.basename}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

