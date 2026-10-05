---
license: mit
language:
  - en
pretty_name: P-ARC
tags:
  - arc
  - arc-agi
  - abstract-reasoning
  - potarcin
  - program-synthesis
  - evaluation
size_categories:
  - n<1K
task_categories:
  - other
configs:
  - config_name: default
    data_files:
      - split: test
        path: p_arc_dataset.parquet
---

# P-ARC

P-ARC is a held-out set of 50 hand-crafted ARC-style tasks, introduced with **PotARCin: Multi-Dimensional Evaluation of Skill Acquisition in Abstract Reasoning Tasks** (NeurIPS 2026, Evaluations and Datasets Track). Every task comes with a generator and a verifier program, 50 fixed generated examples, three human-made corruptions (erroneous attempts from the feasibility check or errors designed by the task's creator), and a natural-language statement of its rule.

- **Venue:** NeurIPS 2026, Evaluations and Datasets Track
- **Paper:** [arXiv:2609.27288](https://arxiv.org/abs/2609.27288)
- **Authors:** Claas Beger, Ryan Yi, Melanie Mitchell
- **Contact:** claasbeger@santafe.edu

## Summary

Standard ARC evaluation only checks whether a model produces the correct output grid. PotARCin instead tests whether a model has acquired a task's underlying rule, across five dimensions: Definition, Classification, Constrained Generation, Editing and Inversion. P-ARC provides tasks outside the public ARC-AGI-1 distribution for this purpose. In the paper, models reach 1–8% accuracy when a task counts as solved only if all five dimensions are correct. We qualitatively estimate P-ARC to lie between ARC-AGI-1 and ARC-AGI-2 in difficulty.

Grouped by their main operation, the tasks cover physics and dynamics (9 tasks), rigid geometric transforms (8), connection and pathfinding (7), symmetry, completion and repair (8), object composition (5), constraint satisfaction and logic (6), and colour or attribute mapping (5); two tasks have no clean category.

## Files

| File | Contents |
|---|---|
| `p_arc_dataset.parquet` / `p_arc_dataset.csv` | One row per task (`t1`–`t50`) with all columns below. The viewer and `load_dataset` use the Parquet file. |
| `tasks/<task_id>.json` | Each task's train and test pairs in the standard ARC JSON format. |
| `SCHEMA.json` | Column descriptions in machine-readable form. |
| `P-ARC.croissant.json` | Croissant metadata including the responsible-AI fields. |

## Columns

| Column | Description |
|---|---|
| `task_id` | `t1` … `t50` |
| `task_name` | Short descriptive name, e.g. `corner_to_center` |
| `rule` | Natural-language statement of the task's intended rule, reviewed by the authors |
| `rule_source` | `human_reviewed` (redrafted from the task's transform and reviewed, 39 tasks), `human_annotated` (written from scratch, 10) or `draft` (the original hand-written rule, kept after review, 1) |
| `train_demonstrations_json` | JSON list of training pairs `{input, output}` |
| `test_input_json` / `test_output_json` | JSON grids of the held-out test pair |
| `stable_instances_50_json` | JSON object `{"train": [...]}` with 50 fixed generator-produced pairs |
| `human_corruptions_json` | JSON list of 3 `{input, output}` pairs with incorrect outputs, made by people: erroneous attempts from the feasibility check or errors designed by the task's creator |
| `generator_py` | Python source of the task's generator |
| `verifier_py` | Python source of the task's verifier, which maps an input grid to its correct output |

Grids are lists of rows of integers 0–9, following the ARC colour convention.

## Loading

```python
import json
from datasets import load_dataset

tasks = load_dataset("ClaasBeger/P-ARC", split="test")
task = tasks[0]
train = json.loads(task["train_demonstrations_json"])
corruptions = json.loads(task["human_corruptions_json"])
print(task["task_id"], task["rule"])
```

To read the CSV with Python's `csv` module, raise its field size limit first, since some cells are large: `csv.field_size_limit(sys.maxsize)`.

## How the data was made

- **Tasks.** The authors designed each task. A candidate was shown to team members other than its author (at least one, most often two or three) and kept only if at least one of them produced the correct output. This is a feasibility check during design, not a formal human study; solve rates and times were not recorded.
- **Generators and verifiers.** For every task, a team member other than its author inspected at least 50 generator-produced examples and confirmed that they follow the intended rule. All train and test pairs and all 50 stable examples agree with the task's verifier.
- **Corruptions.** Each task has three pairs with incorrect outputs. Some are erroneous attempts collected during the feasibility check; others were designed by the task's creator.
- **Rules.** All 50 rules were reviewed by the authors in September 2026.

## Changes

- **October 2026:** moved from `PotARCin/P-ARC`; corrected the `t1` stable examples (21 of 50 previously disagreed with the verifier) and the `t12` verifier (no longer looks up its own examples); added `task_name`, `rule`, `rule_source` and `human_corruptions_json`, the per-task ARC JSON files, and Croissant metadata; licensed under MIT.

## Citation

```bibtex
@inproceedings{beger2026potarcin,
  title     = {PotARCin: Multi-Dimensional Evaluation of Skill Acquisition in Abstract Reasoning Tasks},
  author    = {Beger, Claas and Yi, Ryan and Mitchell, Melanie},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  eprint    = {2609.27288},
  archivePrefix = {arXiv}
}
```
