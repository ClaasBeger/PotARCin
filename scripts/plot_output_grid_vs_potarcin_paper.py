#!/usr/bin/env python3
"""
Publication-style figure: output-grid evaluation vs PotARCin (full-task and per-dimension strict).

Uncertainty shown on bars:
  * **Claude on ARC-AGI 1** and **MiniMax on ARC-AGI 1**: Wilson 95% intervals from pooled binomial counts
    (output grid, PotARCin dimensions, and PotARCin full task).
  * **Claude on P-ARC (Test2)**: no whiskers on output grid, PotARCin dimensions, or PotARCin full-task bars.
  * **Models with two runs**: mean run-wise proportion ± sample SD across those runs (pct points).
  * **Other single-run models** (MiniMax on P-ARC only here): point estimate only (no whiskers).

Figure layout: for each model, **seven** grouped bars — **output-grid evaluation**, then the five
PotARCin dimension strict rates in **definition→…→inversion** order, then **PotARCin Full Task strict**
(rightmost). Bar centers use **slightly** wider spacing before/after the dimension block than
between consecutive dimension bars (small gaps vs output grid / PotARCin full task). Whiskers: output grid
uses the run-wise SD / Wilson rules below except **Claude on P-ARC** (points only); dimensions use run-wise SD (two reruns) or Wilson
intervals (single rerun), except **Claude on P-ARC** (points only); PotARCin full task follows the same
Wilson/two-run rule with that same Claude-on-P-ARC exception.

Figure text is limited to axis labels and the legend; add captions in your manuscript.

Outputs PDF + PNG to ``evaluation/figures/`` unless ``--output-dir`` is set.
The companion CSV lists pooled PotARCin **strict** per-dimension pass counts/rates: ARC from merged
unified JSON / batch strict summaries; Test2 from ``run_canonical.json`` using the same strict rules as
``scripts/consolidate_multi_dimension_unified_run.py`` (not ``run_canonical_overview.json``, which counts
loose ``dimensions[*].ok`` flags). Test2 ``mean_primary_metric`` columns pool per-task primary metrics
from canonical runs (same keys as the overview reporter).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from math import sqrt
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def _load(json_path: Path) -> dict:
    return json.loads(json_path.read_text(encoding="utf-8"))


def binomial_se(successes: int, n: int) -> float:
    """Standard error of Binomial(successes | n); returns SE on [0,1] scale."""
    if n <= 0:
        return 0.0
    p = successes / n
    return sqrt(p * (1.0 - p) / n)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for Binomial(successes | n); bounds in [0,1]."""
    if n == 0:
        return (0.0, 1.0)
    p_hat = successes / n
    denom = 1.0 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    inner = (p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    inner = max(0.0, inner)
    margin = z * sqrt(inner) / denom
    lo = max(0.0, centre - margin)
    hi = min(1.0, centre + margin)
    return (lo, hi)


def ci_errbars(successes: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    """Symmetric-ish error offsets for plotting: distance from point est to wilson bounds."""
    lo, hi = wilson_ci(successes, n, z=z)
    p_hat = successes / n
    return (p_hat - lo, hi - p_hat)


def arc_og_kn(workspace: Path, rel: str) -> tuple[int, int]:
    data = _load(workspace / rel / "output_grid_run_overview.json")
    total = int(data.get("exact_match_total", data.get("total_tasks", 400)))
    successes = int(data["exact_match_true"])
    return successes, total


def t2_og_kn(workspace: Path, rel: str) -> tuple[int, int]:
    data = _load(workspace / rel / "joint_summary_test2.json")
    return int(data["exact_match_count"]), 50


def arc_md_kn(workspace: Path, rel: str) -> tuple[int, int]:
    data = _load(workspace / rel / "run_all_400_unified_strict_accuracy.json")
    n = int(data.get("canonical_task_count", data.get("task_count", 400)))
    if "strict_full_task_pass_count" in data:
        successes = int(data["strict_full_task_pass_count"])
    else:
        successes = len(data["full_pass_task_ids"])
        if successes == 0:
            fps = round(float(data.get("strict_full_task_accuracy", data["full_task_accuracy"])) * n)
            successes = int(fps)
    return successes, n


def t2_md_kn(workspace: Path, rel: str) -> tuple[int, int]:
    data = _load(workspace / rel / "run_canonical_strict_full_task.json")
    n_tasks = int(
        data.get("canonical_strict_task_count", data.get("task_count", 50)),
    )
    successes = int(data["strict_full_task_pass_count"])
    return successes, n_tasks


def pool_runs(workspace: Path, paths: list[str], kn_fn) -> tuple[int, int]:
    ks, ns = [], []
    for rel in paths:
        k, n = kn_fn(workspace, rel)
        ks.append(k)
        ns.append(n)
    return sum(ks), sum(ns)


POTARCIN_DIMENSION_ORDER: tuple[str, ...] = (
    "definition",
    "classification_distribution",
    "constrained_generation",
    "editing",
    "inversion",
)

DIMENSION_COLUMN_LABELS: tuple[str, ...] = (
    "Definition",
    "Classification",
    "Generation",
    "Editing",
    "Inversion",
)

DIM_BAR_COLORS: tuple[str, ...] = (
    "#5e357c",
    "#237f52",
    "#c65108",
    "#786088",
    "#156575",
)

ARC_UNIFIED_DIM_FIELDS: tuple[tuple[str, str], ...] = (
    ("definition", "definition_strict_pass"),
    ("classification_distribution", "classification_exact_pass"),
    ("constrained_generation", "constrained_generation_pass"),
    ("editing", "editing_pass"),
    ("inversion", "inversion_pass"),
)


def _extract_primary_metric(metrics: object) -> float | None:
    """Same selection order as ``run_all_dimensions_eval._extract_primary_metric``."""
    if not isinstance(metrics, dict):
        return None
    for key in ("success_rate", "accuracy", "exact_match_rate"):
        val = metrics.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _strict_definition_pass(run: dict) -> bool:
    """Strict definition pass rule aligned with published ``*_unified_strict_accuracy`` summaries.

    Auxiliary ``re_arc_*`` metric blocks (re-simulation buckets) are ignored so per-task evaluation
    matches the dimension pass counts aggregated in ``run_all_400_unified_strict_accuracy.json``.
    """
    d = (((run.get("dimensions") or {}).get("definition")) or {})
    m = d.get("metrics") or {}
    if not isinstance(m, dict) or not m:
        return False
    for key, mm in m.items():
        if isinstance(key, str) and key.startswith("re_arc_"):
            continue
        if not isinstance(mm, dict):
            return False
        if "ok" not in mm or "total" not in mm:
            continue
        tot = mm.get("total")
        if not isinstance(tot, int) or tot <= 0:
            continue
        if mm.get("ok") != tot:
            return False
    return True


def _strict_classification_pass(run: dict) -> bool:
    d = (((run.get("dimensions") or {}).get("classification_distribution")) or {})
    m = d.get("metrics") or {}
    return m.get("exact_match") == 1


def _strict_cg_pass(run: dict) -> bool:
    d = (((run.get("dimensions") or {}).get("constrained_generation")) or {})
    m = d.get("metrics") or {}
    return m.get("success_rate") == 1.0


def _strict_edit_pass(run: dict) -> bool:
    d = (((run.get("dimensions") or {}).get("editing")) or {})
    m = d.get("metrics") or {}
    return m.get("success") is True


def _strict_inv_pass(run: dict) -> bool:
    d = (((run.get("dimensions") or {}).get("inversion")) or {})
    m = d.get("metrics") or {}
    return m.get("success_rate") == 1.0


def _potarcin_canonical_run_list(workspace: Path, rel: str, suite_key: str) -> list[dict]:
    """Per-task payloads for pooled progressive strict curves (merged unified or stitched batches)."""
    base = workspace / rel
    if suite_key == "arc_ag1":
        merged = base / "run_all_400_unified.json"
        if merged.is_file():
            data = _load(merged)
            runs = data.get("runs")
            if isinstance(runs, list):
                out = [r for r in runs if isinstance(r, dict)]
                if out:
                    return out
        stitched: list[dict] = []
        for bp in sorted(base.glob("run_batch*_unified_50.json")):
            try:
                bd = _load(bp)
            except Exception:
                continue
            runs_b = bd.get("runs")
            if not isinstance(runs_b, list):
                continue
            stitched.extend(r for r in runs_b if isinstance(r, dict))
        return stitched

    path = base / "run_canonical.json"
    if not path.is_file():
        return []
    data = _load(path)
    runs = data.get("runs")
    if not isinstance(runs, list):
        return []
    return [r for r in runs if isinstance(r, dict)]


def _task_strict_passes_potarcin_dimension(run: dict, dim: str) -> bool:
    if dim == "definition":
        return _strict_definition_pass(run)
    if dim == "classification_distribution":
        return _strict_classification_pass(run)
    if dim == "constrained_generation":
        return _strict_cg_pass(run)
    if dim == "editing":
        return _strict_edit_pass(run)
    if dim == "inversion":
        return _strict_inv_pass(run)
    return False


def _task_passes_dimension_prefix_strict(run: dict, n_first_dims: int) -> bool:
    """Strict pass on ``POTARCIN_DIMENSION_ORDER[0:n_first_dims]`` (``n_first_dims`` in 1–5)."""
    for dim in POTARCIN_DIMENSION_ORDER[:n_first_dims]:
        if not _task_strict_passes_potarcin_dimension(run, dim):
            return False
    return True


def pooled_progressive_strict_rates_pct(
    workspace: Path,
    md_paths: list[str],
    suite_key: str,
) -> tuple[list[float], int]:
    """Pooled % of tasks passing strict on cumulative dimension prefixes length 1..5; pooled ``n``.

    Mirrors path-wise pooling used for aggregated dimension counts (several reruns concatenate as more
    tasks). Returns zeros and ``n=0`` if no canonical merged runs are available.
    """
    prefix_totals = [0] * len(POTARCIN_DIMENSION_ORDER)
    n_tot = 0

    for rel in md_paths:
        runs = _potarcin_canonical_run_list(workspace, rel, suite_key)
        n_run = len(runs)
        if n_run <= 0:
            continue
        n_tot += n_run
        for kk in range(1, len(POTARCIN_DIMENSION_ORDER) + 1):
            prefix_totals[kk - 1] += sum(
                1 for r in runs if _task_passes_dimension_prefix_strict(r, kk)
            )

    if n_tot <= 0:
        return ([0.0] * len(POTARCIN_DIMENSION_ORDER), 0)
    ys = [100.0 * prefix_totals[j] / n_tot for j in range(len(prefix_totals))]
    return ys, n_tot


def _strict_dim_counts_test2(runs: list[dict]) -> dict[str, int]:
    """Mirrors ``consolidate_multi_dimension_unified_run._strict_dim_counts``."""
    return {
        "definition": sum(1 for r in runs if _strict_definition_pass(r)),
        "classification_distribution": sum(1 for r in runs if _strict_classification_pass(r)),
        "constrained_generation": sum(1 for r in runs if _strict_cg_pass(r)),
        "editing": sum(1 for r in runs if _strict_edit_pass(r)),
        "inversion": sum(1 for r in runs if _strict_inv_pass(r)),
    }


def arc_potarcin_dimension_counts(workspace: Path, rel: str) -> tuple[dict[str, int], int]:
    """Strict per-dimension pass counts for one ARC PotARCin run folder."""
    base = workspace / rel
    unified_path = base / "run_all_400_unified_strict_accuracy.json"
    data = _load(unified_path)
    n = int(data.get("canonical_task_count", data.get("task_count", 400)))

    if all(key in data for _, key in ARC_UNIFIED_DIM_FIELDS):
        counts = {dim: int(data[key]) for dim, key in ARC_UNIFIED_DIM_FIELDS}
        return counts, n

    merged: defaultdict[str, int] = defaultdict(int)
    total_n = 0
    for bp in sorted(base.glob("run_batch*_unified_50_strict_accuracy.json")):
        bd = _load(bp)
        total_n += int(bd.get("task_count", 50))
        for dk, val in (bd.get("strict_by_dimension_pass_count") or {}).items():
            merged[str(dk)] += int(val)

    if merged and total_n > 0:
        return dict(merged), total_n

    return {}, n


def pool_potarcin_dimensions(workspace: Path, paths: list[str], suite_key: str) -> dict:
    """Pooled PotARCin strict dimension passes / rates / Test2 mean primary metrics."""
    counts_tot: defaultdict[str, int] = defaultdict(int)
    mp_num: defaultdict[str, float] = defaultdict(float)
    mp_den: defaultdict[str, float] = defaultdict(float)
    n_tot = 0

    for rel in paths:
        if suite_key == "arc_ag1":
            c_run, n_run = arc_potarcin_dimension_counts(workspace, rel)
            n_tot += n_run
            for dk, val in c_run.items():
                counts_tot[str(dk)] += val
        else:
            data = _load(workspace / rel / "run_canonical.json")
            runs = data.get("runs")
            if not isinstance(runs, list):
                runs = []
            runs_d = [r for r in runs if isinstance(r, dict)]
            n_run = len(runs_d)
            n_tot += n_run
            strict_c = _strict_dim_counts_test2(runs_d)
            for dim in POTARCIN_DIMENSION_ORDER:
                counts_tot[dim] += strict_c.get(dim, 0)
            for run in runs_d:
                dims = run.get("dimensions") or {}
                if not isinstance(dims, dict):
                    continue
                for dim in POTARCIN_DIMENSION_ORDER:
                    entry = dims.get(dim, {})
                    if not isinstance(entry, dict):
                        continue
                    val = _extract_primary_metric(entry.get("metrics"))
                    if val is not None:
                        mp_num[dim] += val
                        mp_den[dim] += 1.0

    counts_out = {dim: int(counts_tot[dim]) for dim in POTARCIN_DIMENSION_ORDER}
    rates_out: dict[str, float | None] = {}
    for dim in POTARCIN_DIMENSION_ORDER:
        rates_out[dim] = (counts_out[dim] / n_tot) if n_tot > 0 else None

    mean_out: dict[str, float | None] = {}
    for dim in POTARCIN_DIMENSION_ORDER:
        if suite_key == "parc_test2" and mp_den[dim] > 0:
            mean_out[dim] = mp_num[dim] / mp_den[dim]
        else:
            mean_out[dim] = None

    return {
        "n_tasks": n_tot,
        "counts": counts_out,
        "rates": rates_out,
        "mean_primary_metric": mean_out,
    }


def potarcin_dimension_bar_height_and_errors(
    workspace: Path,
    md_paths: list[str],
    suite_key: str,
    model: str,
    dim: str,
) -> tuple[float, float, float]:
    """Return ``(height_pct, err_low_pct, err_high_pct)`` for one dimension bar.

    * **Claude on P-ARC (Test2)**: pooled strict rate; **no** uncertainty (zeros).
    * **Claude on ARC**: pooled strict rate; asymmetric Wilson 95% offsets from pooled pass counts.
    * **Two reruns**: height = mean of run-wise strict rates; symmetric SD = sample SD of those rates.
    * **Single rerun (otherwise)**: pooled rate; asymmetric Wilson 95% offsets (percentage points).
    """
    if model == "Claude" and suite_key == "parc_test2":
        b = pool_potarcin_dimensions(workspace, md_paths, suite_key)
        rate = b["rates"][dim]
        h = 100.0 * float(rate if rate is not None else 0.0)
        return h, 0.0, 0.0

    if model == "Claude":
        b = pool_potarcin_dimensions(workspace, md_paths, suite_key)
        k = b["counts"][dim]
        n = b["n_tasks"]
        if n <= 0:
            return 0.0, 0.0, 0.0
        h = 100.0 * k / n
        lo_p, hi_p = ci_errbars(k, n)
        return h, 100.0 * lo_p, 100.0 * hi_p

    if len(md_paths) >= 2:
        rs: list[float] = []
        for rel in md_paths:
            b1 = pool_potarcin_dimensions(workspace, [rel], suite_key)
            r = b1["rates"][dim]
            rs.append(100.0 * float(r if r is not None else 0.0))
        h = statistics.mean(rs)
        sd_p = statistics.stdev(rs)
        return h, sd_p, sd_p

    b = pool_potarcin_dimensions(workspace, md_paths, suite_key)
    k = b["counts"][dim]
    n = b["n_tasks"]
    if n <= 0:
        return 0.0, 0.0, 0.0
    h = 100.0 * k / n
    lo_p, hi_p = ci_errbars(k, n)
    return h, 100.0 * lo_p, 100.0 * hi_p


def _use_wilson_pooled_whiskers(model: str, suite_key: str) -> bool:
    if model == "Claude":
        return True
    return model == "MiniMax" and suite_key == "arc_ag1"


def rate_plot_summary(
    workspace: Path,
    paths: list[str],
    kn_fn,
    model: str,
    *,
    suite_key: str,
) -> dict:
    """Bar height and asymmetric/symmetric error extents (all in percentage points)."""
    pairs = [kn_fn(workspace, rel) for rel in paths]
    k_tot = sum(k for k, _ in pairs)
    n_tot = sum(n for _, n in pairs)

    if _use_wilson_pooled_whiskers(model, suite_key):
        p_hat = k_tot / n_tot
        lo_m, hi_m = ci_errbars(k_tot, n_tot)
        return {
            "display_pct": 100.0 * p_hat,
            "yerr_low_pp": 100.0 * lo_m,
            "yerr_high_pp": 100.0 * hi_m,
            "kind": "wilson_pooled",
            "pooled_kn": (k_tot, n_tot),
            "run_rates_pp": [100.0 * k / n for k, n in pairs],
        }

    if len(pairs) >= 2:
        rates = [k / n for k, n in pairs]
        mean_p = statistics.mean(rates)
        sd_p = statistics.stdev(rates)
        return {
            "display_pct": 100.0 * mean_p,
            "yerr_low_pp": 100.0 * sd_p,
            "yerr_high_pp": 100.0 * sd_p,
            "kind": "two_run_mean_sd",
            "pooled_kn": (k_tot, n_tot),
            "run_rates_pp": [100.0 * r for r in rates],
        }

    k, n = pairs[0]
    return {
        "display_pct": 100.0 * k / n,
        "yerr_low_pp": 0.0,
        "yerr_high_pp": 0.0,
        "kind": "single_run_no_whisker",
        "pooled_kn": (k, n),
        "run_rates_pp": [100.0 * k / n],
    }


MODEL_ORDER = ["GPT", "Gemini", "Claude", "MiniMax"]

def default_layout() -> dict[str, tuple]:
    """Per suite: ``(models, og_paths_by_model, potarcin_paths_by_model)``."""
    return {
        "arc_ag1": (
            MODEL_ORDER,
            {
                "Claude": [
                    "evaluation/output_grid_eval_results/ClaudeO46/Run1/Evaluation_Run_2026-05-01_claude_opus46_output_grid_run1",
                ],
                "Gemini": [
                    "evaluation/output_grid_eval_results/Gemini31Pro/Run1/Evaluation_Run_2026-04-25_gemini31pro_output_grid_run1",
                    "evaluation/output_grid_eval_results/Gemini31Pro/Run2/Evaluation_Run_2026-04-30_gemini31pro_output_grid_run2",
                ],
                "GPT": [
                    "evaluation/output_grid_eval_results/GPT54/Run1/Evaluation_Run_gpt54_arc_train_output_grid_run1",
                    "evaluation/output_grid_eval_results/GPT54/Run2/Evaluation_Run_gpt54_arc_train_output_grid_run2",
                ],
                "MiniMax": [
                    "evaluation/output_grid_eval_results/MiniMaxM25/Run1/Evaluation_Run_minimax_output_grid_train",
                ],
            },
            {
                "Claude": [
                    "evaluation/multi_dimension_eval_runs/ClaudeO46/Run1/Evaluation_Run_2026-04-26_claude_opus46_low_thinking120k",
                ],
                "Gemini": [
                    "evaluation/multi_dimension_eval_runs/Gemini31Pro/Run1/Evaluation_Run_2026-04-14_gemini31pro_low",
                    "evaluation/multi_dimension_eval_runs/Gemini31Pro/Run2/Evaluation_Run_2026-04-27_gemini31pro_low_run2",
                ],
                "GPT": [
                    "evaluation/multi_dimension_eval_runs/GPT54/Run1/Evaluation_Run_2026-04-14_gpt54_low",
                    "evaluation/multi_dimension_eval_runs/GPT54/Run2/Evaluation_Run_2026-04-14_gpt54_low_run2",
                ],
                "MiniMax": [
                    "evaluation/multi_dimension_eval_runs/MiniMaxM25/Run1/Evaluation_Run_minimax_m25_test",
                ],
            },
        ),
        "parc_test2": (
            MODEL_ORDER,
            {
                "Claude": [
                    "evaluation/output_grid_eval_results/ClaudeO46/Test2/Run1/Evaluation_Run_2026-05-02_claude_opus46_test2_output_grid_run1",
                ],
                "Gemini": [
                    "evaluation/output_grid_eval_results/Gemini31Pro/Test2/Run1/Evaluation_Run_2026-04-30_test2_output_grid_run1",
                    "evaluation/output_grid_eval_results/Gemini31Pro/Test2/Run2/Evaluation_Run_2026-04-30_test2_output_grid_run2",
                ],
                "GPT": [
                    "evaluation/output_grid_eval_results/GPT54/Test2/Run1/Evaluation_Run_2026-04-30_test2_output_grid_run1",
                    "evaluation/output_grid_eval_results/GPT54/Test2/Run2/Evaluation_Run_2026-04-30_test2_output_grid_run2",
                ],
                "MiniMax": [
                    "evaluation/output_grid_eval_results/MiniMaxM25/Test2/Run1/Evaluation_Run_minimax_m25_test2_output_grid",
                    "evaluation/output_grid_eval_results/MiniMaxM25/Test2/Run2/Evaluation_Run_minimax_Test2_output_grid_run2",
                ],
            },
            {
                "Claude": [
                    "evaluation/multi_dimension_eval_runs/ClaudeO46/Test2/Run1/Evaluation_Run_claude_Test2_seed77_low",
                ],
                "Gemini": [
                    "evaluation/multi_dimension_eval_runs/Gemini31Pro/Test2/Run1/Evaluation_Run_gemini_Test2_seed77_low",
                    "evaluation/multi_dimension_eval_runs/Gemini31Pro/Test2/Run2/Evaluation_Run_gemini_Test2_seed77_low_run2",
                ],
                "GPT": [
                    "evaluation/multi_dimension_eval_runs/GPT54/Test2/Run1/Evaluation_Run_gpt54_Test2_seed77_low",
                    "evaluation/multi_dimension_eval_runs/GPT54/Test2/Run2/Evaluation_Run_gpt54_Test2_seed77_low_run2",
                ],
                "MiniMax": [
                    "evaluation/multi_dimension_eval_runs/MiniMaxM25/Test2/Run1/Evaluation_Run_minimax_Test2_seed77_low",
                    "evaluation/multi_dimension_eval_runs/MiniMaxM25/Test2/Run2/Evaluation_Run_minimax_Test2_multi_dim_run2",
                ],
            },
        ),
    }


def build_table_rows(workspace: Path) -> list[dict]:
    layout = default_layout()
    rows_out: list[dict] = []
    for suite_key in ("arc_ag1", "parc_test2"):
        model_order, og_map, md_map = layout[suite_key]
        og_fn = arc_og_kn if suite_key == "arc_ag1" else t2_og_kn
        md_fn = arc_md_kn if suite_key == "arc_ag1" else t2_md_kn
        for model in model_order:
            so = rate_plot_summary(workspace, og_map[model], og_fn, model, suite_key=suite_key)
            sp = rate_plot_summary(workspace, md_map[model], md_fn, model, suite_key=suite_key)
            rows_out.append({
                "suite": suite_key,
                "model": model,
                "output_grid": so,
                "potarcin": sp,
                "potarcin_dimensions": pool_potarcin_dimensions(
                    workspace,
                    md_map[model],
                    suite_key,
                ),
            })
    return rows_out


def plot_figure(workspace: Path, out_dir: Path, *, basename: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "serif",
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
    })

    palette_og = "#395D9F"
    palette_md = "#B24A6A"

    layout_spec = default_layout()
    fig, axes = plt.subplots(1, 2, figsize=(11.85, 3.65), sharey=True)

    n_slots = 7
    # Mild asymmetric spacing: tiny extra gap OG↔dims and dims↔Pot full vs dim↔dim (middle ground).
    dim_slots = np.linspace(-0.212, 0.212, 5)
    og_slot = -0.342
    pot_slot = 0.342
    slot_offsets = np.concatenate(([og_slot], dim_slots, [pot_slot]))
    _adj = np.sort(slot_offsets)
    bar_width = float(min(0.086, 0.83 * np.min(np.diff(_adj))))
    # Horizontal distance between model groups (Og + dims + Pot).
    model_center_spacing = 1.15

    ymax_track = 0.0

    for ax, suite_key in zip(axes, ("arc_ag1", "parc_test2")):
        mo, og_map, md_map = layout_spec[suite_key]
        og_fn = arc_og_kn if suite_key == "arc_ag1" else t2_og_kn
        md_fn = arc_md_kn if suite_key == "arc_ag1" else t2_md_kn

        ogs: list[float] = []
        pot: list[float] = []
        eog_lo: list[float] = []
        eog_hi: list[float] = []
        ep_lo: list[float] = []
        ep_hi: list[float] = []

        for m in mo:
            so = rate_plot_summary(workspace, og_map[m], og_fn, m, suite_key=suite_key)
            sp = rate_plot_summary(workspace, md_map[m], md_fn, m, suite_key=suite_key)

            ogs.append(float(so["display_pct"]))
            pot.append(float(sp["display_pct"]))
            if m == "Claude" and suite_key == "parc_test2":
                eog_lo.append(0.0)
                eog_hi.append(0.0)
            else:
                eog_lo.append(float(so["yerr_low_pp"]))
                eog_hi.append(float(so["yerr_high_pp"]))
            ep_lo.append(float(sp["yerr_low_pp"]))
            ep_hi.append(float(sp["yerr_high_pp"]))

        x_idx = np.arange(len(mo), dtype=float) * model_center_spacing

        for i, m in enumerate(mo):
            heights: list[float] = []
            err_lo: list[float] = []
            err_hi: list[float] = []
            colors: list[str] = []

            heights.append(ogs[i])
            err_lo.append(eog_lo[i])
            err_hi.append(eog_hi[i])
            colors.append(palette_og)

            for dim in POTARCIN_DIMENSION_ORDER:
                h_d, el_d, eh_d = potarcin_dimension_bar_height_and_errors(
                    workspace,
                    md_map[m],
                    suite_key,
                    m,
                    dim,
                )
                heights.append(h_d)
                err_lo.append(el_d)
                err_hi.append(eh_d)
                di = POTARCIN_DIMENSION_ORDER.index(dim)
                colors.append(DIM_BAR_COLORS[di])

            heights.append(pot[i])
            if m == "Claude" and suite_key == "parc_test2":
                err_lo.append(0.0)
                err_hi.append(0.0)
            else:
                err_lo.append(ep_lo[i])
                err_hi.append(ep_hi[i])
            colors.append(palette_md)

            xs = x_idx[i] + slot_offsets
            ax.bar(
                xs,
                heights,
                width=bar_width,
                color=colors,
                edgecolor="#111827",
                linewidth=0.42,
                zorder=2,
            )

            cap_top = max(heights)
            for j in range(n_slots):
                el, eh = err_lo[j], err_hi[j]
                if el <= 1e-12 and eh <= 1e-12:
                    continue
                ax.errorbar(
                    xs[j],
                    heights[j],
                    yerr=[[el], [eh]],
                    fmt="none",
                    ecolor="#111827",
                    elinewidth=0.92,
                    capsize=2.65,
                    zorder=3,
                )
                cap_top = max(cap_top, heights[j] + eh)

            ymax_track = max(ymax_track, cap_top)

        ax.set_xticks(x_idx)
        ax.set_xticklabels(mo)
        half_span = float(np.max(np.abs(slot_offsets))) + 0.5 * bar_width + 0.06
        ax.set_xlim(float(x_idx[0]) - half_span, float(x_idx[-1]) + half_span)

        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="-", linewidth=0.62, alpha=0.36, color="#d1d5db", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ymax = max(103.5, ymax_track + 5.5)
    for axis in axes:
        axis.set_ylim(0.0, ymax)

    axes[0].set_ylabel("Accuracy (%)")

    legend_handles = [
        mpatches.Patch(
            facecolor=palette_og,
            edgecolor="#111827",
            linewidth=0.55,
            label="Output grid",
        ),
    ]
    for lab, col in zip(DIMENSION_COLUMN_LABELS, DIM_BAR_COLORS):
        legend_handles.append(
            mpatches.Patch(
                facecolor=col,
                edgecolor="#111827",
                linewidth=0.45,
                label=lab,
            ),
        )
    legend_handles.append(
        mpatches.Patch(
            facecolor=palette_md,
            edgecolor="#111827",
            linewidth=0.55,
            label="PotARCin (Full Task strict)",
        ),
    )
    axes[1].legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="#d1d5db",
        framealpha=0.9,
        borderpad=0.35,
        labelspacing=0.3,
        handlelength=1.25,
        handletextpad=0.45,
    )

    png = out_dir / f"{basename}.png"
    pdf = out_dir / f"{basename}.pdf"

    fig.savefig(png, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return pdf


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--workspace",
        type=Path,
        default=Path(".").resolve(),
        help="Repository root.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PDF/PNG (default workspace/evaluation/figures).",
    )
    p.add_argument(
        "--basename",
        type=str,
        default="figure_output_grid_vs_potarcin",
        help="Base filename stem for outputs.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = Path(args.workspace).resolve()
    out_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else workspace / "evaluation" / "figures"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf = plot_figure(workspace, out_dir=out_dir, basename=args.basename)

    csv_path = out_dir / "figure_output_grid_vs_potarcin_data.csv"
    rows = build_table_rows(workspace)

    dim_cols: list[str] = ["pot_dimension_tasks_total"]
    for dim in POTARCIN_DIMENSION_ORDER:
        dim_cols.extend([
            f"pot_dimension_{dim}_pass_count",
            f"pot_dimension_{dim}_pass_rate",
            f"pot_dimension_{dim}_mean_primary_metric",
        ])
    dim_cols.append("pot_dimensions_json")

    hdr = [
        "suite",
        "model",
        "og_display_pct",
        "og_yerr_low_pp",
        "og_yerr_high_pp",
        "og_uncertainty_kind",
        "og_pooled_successes",
        "og_pooled_trials",
        "og_run_rates_pp_json",
        "pot_display_pct",
        "pot_yerr_low_pp",
        "pot_yerr_high_pp",
        "pot_uncertainty_kind",
        "pot_pooled_successes",
        "pot_pooled_trials",
        "pot_run_rates_pp_json",
        "og_binomial_se_on_pooled",
        "pot_binomial_se_on_pooled",
        *dim_cols,
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        for r in rows:
            so = r["output_grid"]
            sp = r["potarcin"]
            dims = r["potarcin_dimensions"]
            ko, no = so["pooled_kn"]
            km, nm = sp["pooled_kn"]
            og_yerr_lo = float(so["yerr_low_pp"])
            og_yerr_hi = float(so["yerr_high_pp"])
            og_kind = str(so["kind"])
            pot_yerr_lo = float(sp["yerr_low_pp"])
            pot_yerr_hi = float(sp["yerr_high_pp"])
            pot_kind = str(sp["kind"])
            if r["model"] == "Claude" and r["suite"] == "parc_test2":
                og_yerr_lo = 0.0
                og_yerr_hi = 0.0
                og_kind = f"{og_kind}_og_whiskers_suppressed"
                pot_yerr_lo = 0.0
                pot_yerr_hi = 0.0
                pot_kind = f"{pot_kind}_pot_whiskers_suppressed"

            dim_row: list = [dims["n_tasks"]]
            dim_blob = {
                "n_tasks": dims["n_tasks"],
                "strict_pass_counts": dims["counts"],
                "strict_pass_rates": dims["rates"],
                "mean_primary_metric": dims["mean_primary_metric"],
            }
            for dim in POTARCIN_DIMENSION_ORDER:
                cnt = dims["counts"][dim]
                rate = dims["rates"][dim]
                mp = dims["mean_primary_metric"][dim]
                dim_row.extend([
                    cnt,
                    f"{rate:.8f}" if rate is not None else "",
                    f"{mp:.10f}" if mp is not None else "",
                ])
            dim_row.append(json.dumps(dim_blob))

            w.writerow(
                [
                    r["suite"],
                    r["model"],
                    f"{so['display_pct']:.6f}",
                    f"{og_yerr_lo:.6f}",
                    f"{og_yerr_hi:.6f}",
                    og_kind,
                    ko,
                    no,
                    json.dumps(so["run_rates_pp"]),
                    f"{sp['display_pct']:.6f}",
                    f"{pot_yerr_lo:.6f}",
                    f"{pot_yerr_hi:.6f}",
                    pot_kind,
                    km,
                    nm,
                    json.dumps(sp["run_rates_pp"]),
                    f"{100 * binomial_se(ko, no):.6f}",
                    f"{100 * binomial_se(km, nm):.6f}",
                    *dim_row,
                ]
            )

    print("Wrote:", csv_path)
    print("Wrote PDF:", pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
