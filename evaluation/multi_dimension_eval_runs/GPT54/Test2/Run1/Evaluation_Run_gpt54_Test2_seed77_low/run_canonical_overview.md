# Test2 multi-dimension — `Evaluation_Run_gpt54_Test2_seed77_low`

- **Model:** `gpt-5.4`
- **Tasks:** 50 · **Strict full-task pass:** 3/50 (6.00%)

## Per-dimension (API ok)

| Dimension | OK | Fail | OK rate | Mean primary metric |
|---|---:|---:|---:|---:|
| `definition` | 50 | 0 | 100.00% | n/a |
| `classification_distribution` | 50 | 0 | 100.00% | 0.8000 |
| `constrained_generation` | 50 | 0 | 100.00% | 0.3200 |
| `editing` | 50 | 0 | 100.00% | n/a |
| `inversion` | 50 | 0 | 100.00% | 0.4600 |

## Strict full-task (all five dimensions pass strict checks)

- **definition:** 7/50
- **classification_distribution:** 17/50
- **constrained_generation:** 16/50
- **editing:** 15/50
- **inversion:** 23/50

- **Combined strict pass:** 3/50

## Artifacts

- `run_canonical.json` — merged run
- `run_canonical_overview.json` — dimension OK / mean metrics
- `run_canonical_strict_full_task.json` — strict counts and task lists

