#!/usr/bin/env python3
"""Main PotARCin performance figure: output-grid accuracy, the five dimensions, full-task.

Five models x seven measures x two datasets, pooled over both independent runs.
Bars are the pooled rate over n = 800 (ARC-AGI-1) or n = 100 (P-ARC) task evaluations;
whiskers are pooled Wilson 95% intervals.

Rebuilt from the per-task result files rather than the consolidated run summaries: the
Claude run-2 summary computes Definition with the re_arc_stable split included, which is
not the Definition criterion (see the README in that run directory). This script
reproduces every cell of the paper's main results table.

Two metric-key renames are handled explicitly; both silently zero a split if ignored:
  * the generator-sample split is ``dynamic50`` before 2026-05-03 and
    ``generator_dynamic50`` after;
  * Test2 tasks carry their stable split as ``p_arc_stable`` (``stable`` is empty).

Usage:  python scripts/plot_potarcin_performance_v2.py [--output-dir DIR]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "evaluation"
TS = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.json$")

# ---------------------------------------------------------------- palette
# Categorical slots 1-5, fixed order, never cycled. Validated for the light
# surface: lightness band, chroma floor, adjacent-pair CVD separation
# (worst dE 9.1 protan), normal-vision floor (19.6). The contrast warning on
# slots 3-5 is answered by the paper's appendix table, which carries every value.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d9d8d4", "#ffffff"

MODELS = ["GPT-5.4", "Gemini 3.1 Pro", "Claude 4.6 Opus", "Kimi K2.5", "MiniMax M2.5"]
SHORT = {"GPT-5.4": "GPT54", "Gemini 3.1 Pro": "Gemini31Pro", "Claude 4.6 Opus": "ClaudeO46",
         "Kimi K2.5": "KimiK25", "MiniMax M2.5": "MiniMaxM25"}

DIMS = [("Definition", "definition_eval_results", "_definition"),
        ("Classification", "classification_eval_results", "_distribution"),
        ("Generation", "constrained_generation_eval_results", "_constrained"),
        ("Editing", "editing_eval_results", "_editing"),
        ("Inversion", "inversion_eval_results", "_inversion")]
MEASURES = ["Output grid"] + [d[0] for d in DIMS] + ["Full-task"]

MULTI_ARC = {
    "GPT-5.4": ["GPT54/Run1/Evaluation_Run_2026-04-14_gpt54_low",
                "GPT54/Run2/Evaluation_Run_2026-04-14_gpt54_low_run2"],
    "Gemini 3.1 Pro": ["Gemini31Pro/Run1/Evaluation_Run_2026-04-14_gemini31pro_low",
                       "Gemini31Pro/Run2/Evaluation_Run_2026-04-27_gemini31pro_low_run2"],
    "Claude 4.6 Opus": ["ClaudeO46/Run1/Evaluation_Run_2026-04-26_claude_opus46_low_thinking120k",
                        "ClaudeO46/Run2/Evaluation_Run_2026-07-07_claude_opus46_low_thinking120k_run2"],
    "Kimi K2.5": ["KimiK25/Run1/Evaluation_Run_kimi_timeout420_test",
                  "KimiK25/Run2/Evaluation_Run_kimi_test_run2"],
    "MiniMax M2.5": ["MiniMaxM25/Run1/Evaluation_Run_minimax_m25_test",
                     "MiniMaxM25/Run2/Evaluation_Run_minimax_m25_run2"],
}
GRID_ARC = {
    "GPT-5.4": ["GPT54/Run1/Evaluation_Run_gpt54_arc_train_output_grid_run1",
                "GPT54/Run2/Evaluation_Run_gpt54_arc_train_output_grid_run2"],
    "Gemini 3.1 Pro": ["Gemini31Pro/Run1/Evaluation_Run_2026-04-25_gemini31pro_output_grid_run1",
                       "Gemini31Pro/Run2/Evaluation_Run_2026-04-30_gemini31pro_output_grid_run2"],
    "Claude 4.6 Opus": ["ClaudeO46/Run1/Evaluation_Run_2026-05-01_claude_opus46_output_grid_run1",
                        "ClaudeO46/Run2/Evaluation_Run_claude_opus46_output_grid_run2"],
    "Kimi K2.5": ["KimiK25/Run1/Evaluation_Run_kimi_arc_train_output_grid_run1",
                  "KimiK25/Run2/Evaluation_Run_kimi_arc_train_output_grid_run2"],
    "MiniMax M2.5": ["MiniMaxM25/Run1/Evaluation_Run_minimax_output_grid_train",
                     "MiniMaxM25/Run2/Evaluation_Run_minimax_output_grid_train_run2"],
}


def latest(directory: Path, split: str) -> dict[str, Path]:
    """One record per task id: the most recent, so retries do not double-count."""
    best: dict[str, tuple[str, Path]] = {}
    for f in glob.glob(str(directory / "*.json")):
        name = os.path.basename(f)
        if "overview" in name or name.startswith("run_"):
            continue
        m = TS.search(f)
        tid = name.split(split)[0]
        stamp = m.group(1) if m else ""
        if tid not in best or stamp > best[tid][0]:
            best[tid] = (stamp, Path(f))
    return {t: v[1] for t, v in best.items()}


def definition_splits(metrics: dict) -> list[str]:
    """The four splits Definition is scored on, under whichever key names this run used."""
    out = ["train", "test"]
    for key in ("stable", "p_arc_stable"):
        if (metrics.get(key) or {}).get("total", 0) > 0:
            out.append(key)
            break
    for key in ("generator_dynamic50", "dynamic50"):
        if (metrics.get(key) or {}).get("total", 0) > 0:
            out.append(key)
            break
    return out


def dimension_passed(dim: str, rec: dict) -> bool:
    metrics = rec.get("metrics") or {}
    if dim == "Definition":
        if not rec.get("script_loaded"):
            return False
        return all((metrics.get(k) or {}).get("success_rate") == 1.0
                   for k in definition_splits(metrics)
                   if (metrics.get(k) or {}).get("total", 0) > 0)
    if dim == "Classification":
        return metrics.get("accuracy") == 1.0
    if dim == "Editing":
        return bool(metrics.get("success"))
    return metrics.get("success_rate") == 1.0 or rec.get("success_rate") == 1.0


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z / den * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * max(centre - half, 0.0), 100 * min(centre + half, 1.0)


def parc_runs(root: str, short: str) -> list[str]:
    return [os.path.relpath(p, str(EVAL / root))
            for p in sorted(glob.glob(str(EVAL / root / short / "Test2" / "Run*" / "*")))]


def collect() -> dict:
    """-> {(suite, model, measure): (successes, trials)} pooled over both runs."""
    out: dict[tuple[str, str, str], list[int]] = collections.defaultdict(lambda: [0, 0])
    for suite in ("ARC-AGI-1", "P-ARC"):
        for model in MODELS:
            short = SHORT[model]
            # ---- output grid
            grid_dirs = (GRID_ARC[model] if suite == "ARC-AGI-1"
                         else parc_runs("output_grid_eval_results", short))
            canon: set[str] | None = None
            for rel in grid_dirs:
                recs = latest(EVAL / "output_grid_eval_results" / rel, "_output_grid")
                if canon is None:
                    canon = canonical_ids(suite)
                for tid, f in recs.items():
                    if tid not in canon:
                        continue          # extra ids (eval-split spillover) are not scored
                    out[(suite, model, "Output grid")][1] += 1
                    out[(suite, model, "Output grid")][0] += bool(json.load(open(f)).get("exact_match"))
            # ---- five dimensions + full task
            multi_dirs = (MULTI_ARC[model] if suite == "ARC-AGI-1"
                          else parc_runs("classification_eval_results", short))
            for rel in multi_dirs:
                per: dict[str, dict[str, bool]] = collections.defaultdict(dict)
                for dim, root, split in DIMS:
                    for tid, f in latest(EVAL / root / rel, split).items():
                        per[tid][dim] = dimension_passed(dim, json.load(open(f)))
                for tid, res in per.items():
                    for dim, _, _ in DIMS:
                        out[(suite, model, dim)][1] += 1
                        out[(suite, model, dim)][0] += bool(res.get(dim))
                    out[(suite, model, "Full-task")][1] += 1
                    out[(suite, model, "Full-task")][0] += all(res.get(d) for d, _, _ in DIMS)
    return out


_CANON: dict[str, set[str]] = {}


def canonical_ids(suite: str) -> set[str]:
    """Task ids that count, taken from a reference multi-dimension run."""
    if suite not in _CANON:
        if suite == "ARC-AGI-1":
            d = EVAL / "definition_eval_results/GPT54/Run1/Evaluation_Run_2026-04-14_gpt54_low"
            files = glob.glob(str(d / "*.json"))
        else:
            files = glob.glob(str(EVAL / "definition_eval_results/GPT54/Test2/Run*/*/*.json"))
        _CANON[suite] = {os.path.basename(f).split("_definition")[0]
                         for f in files if "overview" not in f}
    return _CANON[suite]


def draw(counts: dict, out_dir: Path) -> None:
    # Authored at final print size (single NeurIPS column) so text renders at its
    # true point size rather than being scaled down by \includegraphics.
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.5, "axes.edgecolor": GRID,
        "xtick.major.width": 0.0, "ytick.major.width": 0.5, "ytick.major.size": 2.0,
        "xtick.color": INK, "ytick.color": INK_MUTED,
        "text.color": INK, "axes.labelcolor": INK,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    })

    n_m = len(MODELS)
    slot, gap = 1.0, 0.20            # gap = the surface separation between bars
    bar_w = (slot - gap) / n_m
    xpos, cursor = [], 0.0
    for i, _ in enumerate(MEASURES):
        if i in (1, 6):
            cursor += 0.46           # air around the two headline measures
        xpos.append(cursor)
        cursor += slot + 0.10

    fig, axes = plt.subplots(2, 1, figsize=(5.5, 2.78), sharex=True,
                             gridspec_kw={"hspace": 0.20})

    for ax, suite in zip(axes, ("ARC-AGI-1", "P-ARC")):
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, linewidth=0.45, linestyle="-")
        ax.xaxis.grid(False)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(GRID)

        for mi, model in enumerate(MODELS):
            offs = (mi - (n_m - 1) / 2) * bar_w
            vals, los, his = [], [], []
            for measure in MEASURES:
                k, n = counts[(suite, model, measure)]
                v = 100 * k / n
                lo, hi = wilson(k, n)
                vals.append(v); los.append(v - lo); his.append(hi - v)
            ax.bar([x + offs for x in xpos], vals, width=bar_w * 0.88,
                   color=SERIES[mi], edgecolor="none", zorder=3,
                   label=model if suite == "ARC-AGI-1" else None)
            ax.errorbar([x + offs for x in xpos], vals, yerr=[los, his],
                        fmt="none", ecolor=INK_MUTED, elinewidth=0.45,
                        capsize=0.0, alpha=0.7, zorder=4)

        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 25))
        ax.set_yticklabels([str(v) for v in range(0, 101, 25)], fontsize=5.6)
        ax.tick_params(axis="y", pad=1.5)
        ax.set_xlim(xpos[0] - 0.80, xpos[-1] + 0.80)
        # top-right: the tall bars sit on the left of both panels
        ax.text(0.997, 0.98, suite, transform=ax.transAxes, fontsize=6.8,
                fontweight="bold", va="top", ha="right", color=INK)

    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels(MEASURES, fontsize=6.2)
    axes[1].tick_params(axis="x", pad=1.5)
    for lbl, measure in zip(axes[1].get_xticklabels(), MEASURES):
        if measure in ("Output grid", "Full-task"):
            lbl.set_fontweight("bold")

    half = slot / 2 + 0.06
    for x0, x1, txt in ((xpos[0] - half, xpos[0] + half, "standard ARC"),
                        (xpos[1] - half, xpos[5] + half, "PotARCin dimensions"),
                        (xpos[6] - half, xpos[6] + half, "all five")):
        axes[0].plot([x0, x1], [1.035, 1.035], transform=axes[0].get_xaxis_transform(),
                     color=GRID, linewidth=0.6, clip_on=False, solid_capstyle="butt")
        axes[0].annotate(txt, xy=((x0 + x1) / 2, 1.075),
                         xycoords=axes[0].get_xaxis_transform(),
                         ha="center", va="bottom", fontsize=5.6, color=INK_MUTED,
                         annotation_clip=False)

    fig.supylabel("Tasks passed (%)", fontsize=6.2, color=INK_MUTED, x=0.016)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               fontsize=6.2, handlelength=0.8, handleheight=0.8,
               columnspacing=0.9, handletextpad=0.35, bbox_to_anchor=(0.53, -0.021))

    fig.subplots_adjust(left=0.072, right=0.997, top=0.905, bottom=0.185)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"potarcin_performance_v2.{ext}", dpi=400,
                    facecolor=SURFACE)
    print(f"wrote {out_dir/'potarcin_performance_v2.pdf'} (+ .png)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=str(EVAL / "figures"))
    args = ap.parse_args()
    counts = collect()

    print(f"{'suite':10s} {'model':17s} " + " ".join(f"{m[:9]:>10s}" for m in MEASURES))
    rows = []
    for suite in ("ARC-AGI-1", "P-ARC"):
        for model in MODELS:
            cells = []
            for measure in MEASURES:
                k, n = counts[(suite, model, measure)]
                cells.append(f"{100*k/n:10.2f}")
                rows.append((suite, model, measure, k, n, 100 * k / n, *wilson(k, n)))
            print(f"{suite:10s} {model:17s} " + " ".join(cells))
    with open(Path(args.output_dir) / "potarcin_performance_v2_data.csv", "w") as fh:
        fh.write("suite,model,measure,successes,trials,percent,wilson_lo,wilson_hi\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")
    draw(counts, Path(args.output_dir))


if __name__ == "__main__":
    main()
