# Output grid run — performance overview

- **Run tag:** `Evaluation_Run_2026-04-25_gemini31pro_output_grid_run1`
- **Model:** `gemini-3.1-pro-preview` (google)
- **Tasks:** 400 (canonical ARC train order)
- **Generated:** 2026-05-05T03:16:04Z (UTC)

## Parse / logging health

- **Per-task JSON on disk:** 400/400
- **Missing artifacts:** 0
- **All `evaluation_ok`:** **yes**

## Accuracy (test index 0)

- **Exact match:** 355/400 (**88.75%**)
- **Not exact match:** 45/400

### Batch summaries (50 tasks each)

- `batch_arc_train_batch01.json`
- `batch_arc_train_batch02.json`
- `batch_arc_train_batch03.json`
- `batch_arc_train_batch04.json`
- `batch_arc_train_batch05.json`
- `batch_arc_train_batch06.json`
- `batch_arc_train_batch07.json`
- `batch_arc_train_batch08.json`

### Joint batch JSON

- `batch_arc_train_full.json`

### Task ids with `exact_match: false`

```
045e512c 150deff5 1b60fb0c 234bbc79 264363fd 2bee17df 2dd70a9a 36fdfd69 3de23699 3e980e27 4290ef0e 469497ad 4c5c2cf0 543a7ed5 57aa92db 5c2c9af4 5daaa586 6455b5f5 662c240a 6a1e5592 6aa20dc0 6d58a25d 73251a56 83302e8f 846bdb03 890034e9 93b581b8 97a05b5b 9aec4887 9edfc990 a64e4611 a68b268e b527c5c6 b775ac94 b782dc8a caa06a1f ce602527 d07ae81c d13f3404 d22278a0 d406998b e40b9e2f e6721834 e73095fd e8593010
```
