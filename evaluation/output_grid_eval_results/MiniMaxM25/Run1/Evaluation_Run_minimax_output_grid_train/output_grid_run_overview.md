# Output grid run — performance overview

- **Run tag:** `Evaluation_Run_minimax_output_grid_train`
- **Model:** `minimax/minimax-m2.5:nitro` (openrouter)
- **Tasks:** 400 (canonical ARC train order)
- **Generated:** 2026-05-05T03:17:03Z (UTC)

## Parse / logging health

- **Per-task JSON on disk:** 400/400
- **Missing artifacts:** 0
- **All `evaluation_ok`:** **yes**

## Accuracy (test index 0)

- **Exact match:** 291/400 (**72.75%**)
- **Not exact match:** 109/400

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
00d62c1b 017c7c7b 045e512c 09629e4f 0b148d64 0e206a2e 10fcaaa3 150deff5 1b60fb0c 1caeab9d 1e32b0e9 1f0c79e5 1fad071e 2204b7a8 22233c11 234bbc79 239be575 264363fd 28e73c20 29623171 2bcee788 2dd70a9a 3345333e 3631a71a 36d67576 36fdfd69 3906de3d 39a8645d 39e1d7f9 3befdf3e 4290ef0e 447fd412 4522001f 4612dd53 469497ad 4938f0c2 50846271 57aa92db 5c2c9af4 5daaa586 63613498 6455b5f5 6855a6e4 6a1e5592 6aa20dc0 6d0160f0 72322fa7 72ca375d 73251a56 7468f01a 75b8110e 760b3cac 776ffc46 7b6016b9 7df24a62 810b9b61 83302e8f 834ec97d 868de0fa 890034e9 8a004b2b 8d5021e8 8e1813be 8efcae92 90f3ed37 941d9a10 97a05b5b 98cf29f8 9aec4887 9edfc990 a64e4611 a78176bb a8c38be5 aba27056 ae4f1146 b0c4d837 b190f7f5 b230c067 b527c5c6 b775ac94 b8825c91 bc1d5164 c8cbb738 caa06a1f ce9e57f2 d07ae81c d0f5fe59 d10ecb37 d13f3404 d22278a0 d5d6de2d d90796e8 db93a21d dc0a314f e40b9e2f e5062a87 e50d258f e6721834 e73095fd e8dc4411 ec883f72 f15e1fac f35d900a f76d97a5 f8a8fe49 f8c80d96 f9012d9b fcc82909 ff805c23
```
