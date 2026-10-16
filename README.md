# PotARCin

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-4b44ce.svg)](https://neurips.cc/)
[![Project Page](https://img.shields.io/badge/Project-Page-2ea44f.svg)](https://claasbeger.github.io/potarcin)
[![arXiv](https://img.shields.io/badge/arXiv-2609.27288-b31b1b.svg)](https://arxiv.org/abs/2609.27288)
[![Dataset on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-P--ARC-ffd21e.svg)](https://huggingface.co/datasets/ClaasBeger/P-ARC)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

Code and results for **PotARCin: Multi-Dimensional Evaluation of Skill Acquisition in Abstract Reasoning Tasks** (Beger, Yi and Mitchell, NeurIPS 2026, [arXiv:2609.27288](https://arxiv.org/abs/2609.27288)).

PotARCin tests whether a model has actually acquired the rule behind an ARC task, rather than only producing the right output grid. Each task is evaluated along five dimensions, alongside the standard output-grid setting:

- **Definition** – write a program that implements the task's rule.
- **Classification** – decide whether a new input–output pair follows the same rule.
- **Constrained generation** – produce a new input–output pair that follows the rule.
- **Editing** – repair a corrupted pair so that it follows the rule again.
- **Inversion** – given an output, produce an input that the rule maps to it.

Tasks come from the ARC-AGI-1 training set and from **P-ARC** (`Test2/`), a set of 50 new tasks, also available on Hugging Face as [ClaasBeger/P-ARC](https://huggingface.co/datasets/ClaasBeger/P-ARC). Generated pairs come from ARC-GEN and RE-ARC generators, and answers are checked by per-task verifiers.

## Layout

- `framework/` – the evaluation library.
  - `grids.py` – `Grid` / `GridPair` types and helpers.
  - `tasks/` – task loading: `arc_dataset.py` (ARC-AGI-1 training, with generators and verifiers attached), `arc_eval_dataset.py` (ARC-AGI-1 evaluation split), `test2_dataset.py` (P-ARC), `base.py` (`ArcTask`).
  - `dimensions/` – one evaluator per dimension, plus `output_grid_eval.py` for the standard setting.
  - `verifier_selection.py` – picks the verifier a task is judged by (see [Verifiers](#verifiers)).
  - `custom_verifiers/` – hand-written verifiers that replace buggy third-party ones (`fixes/<task_id>.py`).
  - `integrations/` – adapters around `external/re_arc` and `external/ARC-GEN`.
  - `corruption/` – corrupted pairs for the editing dimension.
  - `llm_api.py` – model calls (OpenAI, Anthropic, Google, Together, OpenRouter).
  - `eval_paths.py` – where results are written.
- `scripts/` – command-line runners, consolidation, plotting and dataset export.
- `Test2/` – the P-ARC tasks ([Hugging Face](https://huggingface.co/datasets/ClaasBeger/P-ARC)), each with a generator, verifier, sample pool and corruptions.
- `evaluation/` – results per dimension and model (`<dimension>_eval_results/<Model>/<Run>/<run_tag>/`) and `figures/`.
- `Demonstrations/` – HTML reports of model answers (see [HTML reports](#html-reports)).
- `datasets/p_arc_hub_bundle/` – the P-ARC dataset bundle, published as [ClaasBeger/P-ARC](https://huggingface.co/datasets/ClaasBeger/P-ARC) on Hugging Face.
- `external/` – vendored generators and verifiers (ARC-GEN, RE-ARC, code-golf solutions); two of these are git submodules.
- `task_valid_verifiers.csv` – audited verifier slots per ARC-AGI-1 task.
- `tests/` – unit and smoke tests.

## Setup

Python 3.10 or later (tested with 3.12). After cloning, fetch the code-golf verifier submodules and install the dependencies:

```bash
git submodule update --init
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Model calls go through the standard library, except that the Anthropic, Google and Together providers need their SDKs (`anthropic`, `google-genai`, `together`), which are imported only when that provider is used. The plotting scripts need `matplotlib` and `numpy`.

Run the tests:

```bash
pytest
```

## Using the framework

Load tasks, build evaluators, and pass a prediction function per dimension. This example scores the inversion dimension on two tasks with a predictor that answers perfectly, so it needs no model:

```python
from framework.dimensions.inversion import InversionEvaluator
from framework.evaluation.runner import run_all_dimensions
from framework.tasks.arc_dataset import load_task
from framework.tasks.base import DimensionName

tasks = [load_task("007bbfb7"), load_task("e76a88a6")]

def predict_inversion(instances):
    # Replace with model calls; here we return the reference input.
    return [inst.target for inst in instances]

summary = run_all_dimensions(
    {DimensionName.INVERSION: InversionEvaluator()},
    tasks,
    {DimensionName.INVERSION: predict_inversion},
)
print(summary.results[DimensionName.INVERSION].metrics)
```

## Running model evaluations

The runners call the provider given by `--provider` and need its key (`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `TOGETHER_API_KEY`). A provider's base URL can be overridden with `<PROVIDER>_BASE_URL`. Results go to `evaluation/`, and re-running a command with the same `--run-tag` resumes it.

**ARC-AGI-1 training, all dimensions (Kimi K2.5 via OpenRouter `:nitro`):**

```bash
python scripts/run_all_dimensions_eval.py \
  --model moonshotai/Kimi-K2.5:nitro \
  --provider openrouter \
  --reasoning-effort low \
  --seed 77 \
  --run-tag Evaluation_Run_kimi_timeout420_test \
  --batch-size 100
```

**P-ARC (`Test2/`), all dimensions:**

```bash
python scripts/run_all_dimensions_eval.py \
  --task-source test2 \
  --model moonshotai/Kimi-K2.5:nitro \
  --provider openrouter \
  --reasoning-effort low \
  --seed 77 \
  --run-tag Evaluation_Run_kimi_Test2_seed77_low \
  --batch-size 50
```

`:nitro` writes to the same `KimiK25/` folders as the base model id, so these commands resume the existing run tags.

### ARC-AGI-1 evaluation split, output grids

The standard ARC setting (demonstrations plus one test input; the model returns the output grid) on the 400 evaluation tasks, read from `external/re_arc/arc_original.zip` (`arc_original/evaluation/`), or from an extracted `external/arc_original_eval/` when present. Results go to `evaluation/output_grid_eval_results/<ModelAlias>/ARC1Eval/<Run1|Run2>/<run_tag>/`, next to the training-split runs.

**Kimi K2.5:**

```bash
python scripts/run_output_grid_eval_arc1_eval_batch.py \
  --all-eval \
  --model moonshotai/Kimi-K2.5:nitro \
  --provider openrouter \
  --reasoning-effort low \
  --run-tag Evaluation_Run_kimi_arc1_eval_output_grid_run1 \
  --batch-size 50
```

**MiniMax M2.5:**

```bash
python scripts/run_output_grid_eval_arc1_eval_batch.py \
  --all-eval \
  --model minimax/minimax-m2.5:nitro \
  --provider openrouter \
  --reasoning-effort low \
  --run-tag Evaluation_Run_minimax_arc1_eval_output_grid_run1 \
  --batch-size 50
```

Re-run the same command to continue: `--skip-completed` (the default) skips tasks that already have a non-error result under the run tag, so the 400 tasks can be walked in `--batch-size` chunks. Other options:

- `--all-test-indices` – evaluate every test example (19 of the 400 tasks have 2); results for index > 0 are named `<task_id>_test<idx>_output_grid_eval_*.json`.
- `--tasks <id> …` / `--tasks-file <path>` – run a subset instead of `--all-eval`.
- `--retries` / `--retry-backoff-s` / `--no-retry-eval-failures` – retry budget for transient errors and parse failures. Failures that exhaust it are logged as `<task_id>_output_grid_eval_error_*.json` and are retried on resume.

## Verifiers

Each ARC-AGI-1 task can have up to five verifiers: `re_arc`, three code-golf solution sets (`google`, `keymoon`, `neurips`) and a hand-written `custom` one. `task_valid_verifiers.csv` lists which of them agree with the task's examples and generator, and `selected_verifier_slot` names the canonical one the task is judged by. `framework/verifier_selection.py` tries the canonical slot first, then the other valid slots, and skips any verifier that disagrees with the specific generated pair being judged.

The CSV was built by checking every slot against the task's train, test and ARC-GEN stable examples and 500 generator draws (seed 0); a verifier call over 10 s counts as a mismatch. To rebuild it (a rebuild only removes listed slots that now fail, and keeps canonical choices that are still valid):

```bash
python scripts/caller_export_valid_verifiers.py --out task_valid_verifiers.csv --num-tasks 400 \
  --dynamic-samples 500 --seed 0 --call-timeout 10 --workers 16
```

To audit every verifier of the first 400 training tasks against their examples and generators, writing a log to `caller_run_<time>.log` and HTML reports of disagreements to `caller_mismatches/`:

```bash
python scripts/caller.py
```

## HTML reports

Results are browsed as static HTML pages: `Demonstrations/` holds reports of model answers, and the `scripts/build_*_html.py` scripts generate new ones. For example, a ground-truth-vs-prediction page for an output-grid run (written to `<results-dir>/compare_gt_vs_pred.html` unless `--out` is given):

```bash
python scripts/build_output_grid_results_dir_html.py \
  --results-dir evaluation/output_grid_eval_results/KimiK25/ARC1Eval/Run1/Evaluation_Run_kimi_arc1_eval_output_grid_run1
```

## Citation

```bibtex
@inproceedings{beger2026potarcin,
  title         = {PotARCin: Multi-Dimensional Evaluation of Skill Acquisition in Abstract Reasoning Tasks},
  author        = {Beger, Claas and Yi, Ryan and Mitchell, Melanie},
  booktitle     = {Advances in Neural Information Processing Systems},
  year          = {2026},
  eprint        = {2609.27288},
  archivePrefix = {arXiv}
}
```

## License

The PotARCin code and data, including P-ARC (`Test2/`, `datasets/p_arc_hub_bundle/`), are released under the MIT License (`LICENSE`).

The third-party code and data under `external/` keep their own licenses:

| Path | Source | License |
|---|---|---|
| `external/ARC-GEN/` | [google/ARC-GEN](https://github.com/google/ARC-GEN) | Apache 2.0. Generators for 110 tasks were modified, and one was added, so that each reproduces its task's training demonstrations and test pairs; see `external/ARC-GEN/POTARCIN_CHANGES.md`. |
| `external/re_arc/` | [michaelhodel/re-arc](https://github.com/michaelhodel/re-arc) | MIT. In `arc_original.zip`, the first training example of 469497ad is corrected. |
| `external/golf/` | [key-moon/golf](https://github.com/key-moon/golf), snapshot from March 2026 | Apache 2.0 |
| `external/google-code-golf-2025/` (submodule) | [xsot/google-code-golf-2025](https://github.com/xsot/google-code-golf-2025) | Apache 2.0 |
| `external/NeurIPS-Code-Golf-2025/` (submodule) | [Seek64/NeurIPS-Code-Golf-2025](https://github.com/Seek64/NeurIPS-Code-Golf-2025) | Apache 2.0 |
| `external/arc_gen_stable.zip` | [The ARC-GEN-100K Dataset](https://www.kaggle.com/datasets/arcgen100k/the-arc-gen-100k-dataset) (August 2025), unchanged | Apache 2.0 |
| `external/arc_original_train/` | ARC-AGI-1 training tasks from [fchollet/ARC-AGI](https://github.com/fchollet/ARC-AGI), unchanged | Apache 2.0 |
| `external/H-ARC/incorrect_submissions.csv` | Incorrect human attempts, filtered, from H-ARC ([LeGris et al., 2025](https://doi.org/10.1038/s41597-025-05687-1); [data](https://osf.io/bh8yq/)) | CC0 1.0 |
