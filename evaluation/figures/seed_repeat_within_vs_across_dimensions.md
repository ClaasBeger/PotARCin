# Same-seed variability: within-dimension repeats vs across dimensions

**Experiment:** GPT-5.4, `reasoning_effort=low`, fixed per-task seeds from `Evaluation_Run_2026-04-14_gpt54_low`  
**Run tag:** `Evaluation_Run_gpt54_subset20_repeat10_same_request`  
**Design:** 20 ARC tasks × 10 identical-setting repeats (200 API-complete full 5-dimension evaluations after quota fill)  
**Artifact summaries:** `seed_repeat_stress_2026-05-05_14-29-11.json`, `seed_repeat_fill_2026-07-23_22-51-43.json`

## Question

With a **matched evaluation budget**, is a metric failure more likely when you:

1. **Same dimension ×5** — take five same-seed repeats of one dimension and ask whether ≥1 fails, or  
2. **Different dimensions ×1** — on a single task-repeat, ask whether ≥1 of the five dimensions fails?

This is *not* comparing one single-dim attempt (~7%) to any-of-five (~25%). Those scopes are not matched.

## Failure definition

On API-complete runs only (`ok=True` for all five dimensions):

| Dimension | Success criterion |
|---|---|
| Definition | test `success_rate` = 1 |
| Classification | `accuracy` = 1 |
| Constrained generation | `success_rate` = 1 |
| Editing | `metrics.success` = true |
| Inversion | `success_rate` = 1 |

Quota / API failures are excluded (repaired to reach 200/200 complete repeats).

## Estimators for same-dim ×5

All three are empirical P(≥1 fail in 5 same-dim repeats), averaged over the five dimensions (and over tasks). They differ in how 5-subsets are taken from each task’s 10 repeats:

| Estimator | How 5-repeat units are formed | # units / task / dim | Same-dim ×5 (avg) |
|---|---|---|---|
| Non-overlapping blocks | Contiguous `0–4` and `5–9` | 2 | **20.0%** |
| Sliding windows | Six contiguous overlapping windows of length 5 | 6 | **18.8%** |
| \(\binom{10}{5}\) | All unordered 5-subsets of the 10 repeats (not necessarily contiguous) | 252 | **20.2%** |

Sliding ≠ \(\binom{10}{5}\). Blocks and \(\binom{10}{5}\) land near **20%**; sliding is a bit lower because overlapping windows overweight long success streaks.

Cross-dimension rates always use all 10 repeats per task (200 trials). Significance is **task-paired** (n = 20).

## Main result

Different dims once — P(any dim fails): **25.5%**.

| Same-dim ×5 estimator | Rate | Diff vs cross | Paired *t* p | Wilcoxon p | Significant (α=0.05)? |
|---|---|---|---|---|---|
| Blocks | 20.0% | +5.5 pp | 0.037 | 0.028 | **Yes** |
| Sliding | 18.8% | +6.7 pp | 0.008 | 0.003 | **Yes** |
| \(\binom{10}{5}\) | 20.2% | +5.3 pp | 0.030 | 0.028 | **Yes** |

Bootstrap 95% CI for (cross − same), blocks: **[+1.0, +10.5] pp** (similar for the other two; all exclude 0).

**Interpretation:** With five evaluations, checking the **five different dimensions once** is significantly more likely to surface at least one metric failure than repeating a **typical single dimension five times**. This holds for all three same-dim aggregators.

### Per-dimension same×5 (blocks) vs cross 25.5%

| Dimension | Same×5 | vs cross |
|---|---|---|
| Definition | 12.5% | cross higher |
| Classification | 12.5% | cross higher |
| Editing | 17.5% | cross higher (n.s.) |
| Inversion | 20.0% | cross higher (n.s.) |
| Constrained generation | 37.5% | same×5 higher |

Four of five dimensions go in the main-result direction. Constrained generation is the exception (noisier under repetition); if it is dropped, the matched 4-vs-4 gap shrinks to ~15.0% vs ~14.4% and is no longer significant. The headline result still includes all five dimensions as run in the experiment.

## What not to emphasize

- **Binomial ~29% vs cross 25.5%:** i.i.d. assumption at the mean single-dim rate; overstates same-dim risk because failures are task-clustered. Prefer the empirical ~19–20%.
- **6.8% vs 25.5%:** unmatched one-shot comparison; not the target here.

## Takeaways

1. On this GPT-5.4 same-seed stress set, **cross-dimension coverage (5 dims ×1) surfaces failures more often than same-dimension repetition (1 dim ×5)**.
2. That gap is **significant at α=0.05** for all three same-dim estimators: blocks **20.0%**, sliding **18.8%**, and \(\binom{10}{5}\) **20.2%**, each vs cross **25.5%**.
3. Most cross-dimension failures are still **single-dimension** events (only weakly synchronized across dims).

## Paths

- Multi-run dir: `evaluation/multi_dimension_eval_runs/GPT54/Run1/Evaluation_Run_gpt54_subset20_repeat10_same_request/`
- Subset: `Test2/ARC1_subsets/gpt54_nontrivial_success_subset_20_manual.json`
- Runner: `scripts/run_seed_repeat_stress_subset.py` (original); fill: `scripts/fill_seed_repeat_quota_deficits.py`
