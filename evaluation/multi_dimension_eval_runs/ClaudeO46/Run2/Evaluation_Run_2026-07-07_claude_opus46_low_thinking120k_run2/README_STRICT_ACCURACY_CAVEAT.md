# Caveat: contaminated Definition / full-task counts in this run directory

The consolidated strict-accuracy artifacts in this directory were produced with a
Definition pass criterion that **includes the `re_arc_stable` split**. That split checks
the generated program against the broad RE-ARC distribution, which the paper reports
separately (~3% accuracy) and deliberately excludes from the Definition dimension.

Because full-task pass requires all five dimensions, the contaminated Definition count
also drags down every full-task number in this directory.

## Affected values

| Metric | In these files | Correct (train + test + stable + generator_dynamic50) |
|---|---|---|
| `definition_strict_pass` | 42 / 400 (10.5%) | 265 / 400 (66.25%) |
| `full_task_pass`         | 30 / 400 (7.5%)  | 141 / 400 (35.25%) |

`classification_exact_pass`, `constrained_generation_pass`, `editing_pass` and
`inversion_pass` are **correct** and match the published numbers.

## Files

Renamed with a `_rearc-def` suffix so they cannot be picked up by accident:

- `run_all_400_unified_strict_accuracy_rearc-def.json`
- `run_all_400_unified_strict_accuracy_by_batch_rearc-def.json`

Still carrying their original names, but containing the same contaminated
`strict_full_task_pass_count` (24 files):

- `run_batch{1..8}_unified_50_strict_accuracy.json`
- `run_batch{1..8}_unified_50_strict_full_task_accuracy.json`
- `run_batch{1..8}_unified_50_full_task_accuracy_all5.json`

## Note for consumers

`scripts/plot_output_grid_vs_potarcin_paper.py` loads
`run_all_400_unified_strict_accuracy.json` by name (lines 93 and 316). After this rename
it will raise rather than silently read bad data if this run is ever wired into the
figure. Regenerate with the correct criterion before using it there.

The per-task result files under `evaluation/definition_eval_results/ClaudeO46/Run2/` are
unaffected and remain the source of truth.
