# Test2 multi-dimension — `Evaluation_Run_claude_Test2_seed77_low`

- **Model:** `claude-opus-4-6`
- **Tasks:** 50 · **Strict full-task pass:** 4/50 (8.00%)

## Per-dimension (API ok)

| Dimension | OK | Fail | OK rate | Mean primary metric |
|---|---:|---:|---:|---:|
| `definition` | 50 | 0 | 100.00% | n/a |
| `classification_distribution` | 50 | 0 | 100.00% | 0.7480 |
| `constrained_generation` | 50 | 0 | 100.00% | 0.5400 |
| `editing` | 50 | 0 | 100.00% | n/a |
| `inversion` | 50 | 0 | 100.00% | 0.6400 |

## Strict full-task (all five dimensions pass strict checks)

- **definition:** 10/50
- **classification_distribution:** 14/50
- **constrained_generation:** 27/50
- **editing:** 28/50
- **inversion:** 32/50

- **Combined strict pass:** 4/50

## Artifacts

- `run_canonical.json` — merged run
- `run_canonical_overview.json` — dimension OK / mean metrics
- `run_canonical_strict_full_task.json` — strict counts and task lists

