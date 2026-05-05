# Test2 multi-dimension — `Evaluation_Run_minimax_Test2_multi_dim_run2`

- **Model:** `minimax/minimax-m2.5:nitro`
- **Tasks:** 50 · **Strict full-task pass:** 1/50 (2.00%)

## Per-dimension (API ok)

| Dimension | OK | Fail | OK rate | Mean primary metric |
|---|---:|---:|---:|---:|
| `definition` | 50 | 0 | 100.00% | n/a |
| `classification_distribution` | 50 | 0 | 100.00% | 0.6200 |
| `constrained_generation` | 50 | 0 | 100.00% | 0.1800 |
| `editing` | 50 | 0 | 100.00% | n/a |
| `inversion` | 50 | 0 | 100.00% | 0.3200 |

## Strict full-task (all five dimensions pass strict checks)

- **definition:** 2/50
- **classification_distribution:** 8/50
- **constrained_generation:** 9/50
- **editing:** 9/50
- **inversion:** 16/50

- **Combined strict pass:** 1/50

## Artifacts

- `run_canonical.json` — merged run
- `run_canonical_overview.json` — dimension OK / mean metrics
- `run_canonical_strict_full_task.json` — strict counts and task lists

