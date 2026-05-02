# Output grid run — performance overview

- **Run tag:** `Evaluation_Run_2026-05-01_claude_opus46_output_grid_run1`
- **Model:** Claude Opus 4.6 (`claude-opus-4-6`), provider Anthropic, reasoning effort low
- **Tasks:** 400 (canonical ARC train order)
- **Generated:** 2026-05-02T01:42:01Z (UTC)

## Parse / logging health

- **Per-task JSON on disk:** 400/400
- **Missing artifacts:** 0
- **All `evaluation_ok`:** **yes**
- Every task produced a parsable output grid (`evaluation_ok: true`).

## Accuracy (test index 0)

- **Exact match:** 354/400 (**88.5%**)
- **Not exact match:** 46/400

### Task ids with `exact_match: false`

```
05269061 0ca9ddb6 1190e5a7 137eaa0f 150deff5 1b60fb0c 228f6490 29623171 2bcee788 2dc579da 2dd70a9a 32597951 3906de3d 3de23699 3e980e27 4290ef0e 42a50994 4522001f 46f33fce 5168d44c 662c240a 67385a82 6cf79266 72ca375d 776ffc46 7b6016b9 7ddcd7ec 913fb3ed 93b581b8 9565186b 95990924 9aec4887 9af7a82c 9edfc990 a61ba2ce a61f2674 a64e4611 b60334d2 bbc9ae5d c444b776 cbded52d d06dbe63 d22278a0 d6ad076f e73095fd ff805c23
```

Per-task prompts, raw responses, and parsed grids live in `*_output_grid_eval_*.json` in this directory.

Row-level batch summary: `/Users/claas/Documents/SFI/PotARCin/PotARCin/evaluation/output_grid_eval_results/ClaudeO46/Run1/Evaluation_Run_2026-05-01_claude_opus46_output_grid_run1/batch_2026-05-01_12-40-05.json`
