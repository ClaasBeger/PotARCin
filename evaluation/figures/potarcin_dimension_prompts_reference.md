# PotARCin Prompt Templates (Exact)

This file records the exact prompt text templates used by the dimension evaluators in this repository, with placeholders indicating where task-specific grids are inserted.

Source functions:
- `framework/dimensions/definition.py` -> `build_written_prompt()`
- `framework/dimensions/classification.py` -> `build_distribution_classification_prompt()`
- `framework/dimensions/constrained_generation.py` -> `build_constrained_generation_prompt()`
- `framework/dimensions/editing.py` -> `build_editing_prompt()`
- `framework/dimensions/inversion.py` -> `build_inversion_prompt()`
- `framework/dimensions/output_grid_eval.py` -> `build_output_grid_prompt()`

## Definition (`build_written_prompt`)

```text
Find the common rule that maps an input grid to an output grid, given the examples below. Generate a python script that takes a grid as input and transforms it according to the this rule. Anticipate that this program will be applied to several grids, not only the shown test grid.

Your script must define exactly one callable entrypoint:
def solve(grid):
where `grid` is passed as a list of lists of integers (list[list[int]]) and the function returns the transformed grid in the same format.

Example 1:

Input:
<TRAIN_INPUT_1_GRID>
Output:
<TRAIN_OUTPUT_1_GRID>

Example 2:

Input:
<TRAIN_INPUT_2_GRID>
Output:
<TRAIN_OUTPUT_2_GRID>

... (one block per training example)

Below is a test input grid.

Input:
<TEST_INPUT_GRID>

Your final answer should just be the python script itself, no other text or markdown.
```

## Classification Distribution (`build_distribution_classification_prompt`)

```text
Find the common rule that maps an input grid to an output grid given the examples below. You will see five additional input/output pairs (test items 0–4). For each test item, decide whether that pair could have been produced by the same underlying rule and data distribution as the training examples (1 = same distribution / consistent, 0 = not).

Training example 1:
Input:
<TRAIN_INPUT_1_GRID>
Output:
<TRAIN_OUTPUT_1_GRID>

Training example 2:
Input:
<TRAIN_INPUT_2_GRID>
Output:
<TRAIN_OUTPUT_2_GRID>

... (one block per training example)

[OPTIONAL IF PRESENT]
Here is some work on this task you have performed on adjacent requests.
<PRIOR_WORK_CONTEXT>

Test pairs (evaluate each in order, index 0 … 4):

Test 0:
Input:
<TEST0_INPUT_GRID>
Output:
<TEST0_OUTPUT_GRID>

Test 1:
Input:
<TEST1_INPUT_GRID>
Output:
<TEST1_OUTPUT_GRID>

... through Test 4

Your final answer must be only a single Python list of exactly five integers, each 0 or 1, in order for test 0 through test 4, for example [1, 0, 1, 1, 0]. Output nothing else: no other text, no markdown code fences, and no explanation.
```

## Constrained Generation (`build_constrained_generation_prompt`)

```text
Find the common rule that maps an input grid to an output grid given the examples below.

Example 1:
Input:
<TRAIN_INPUT_1_GRID>
Output:
<TRAIN_OUTPUT_1_GRID>

Example 2:
Input:
<TRAIN_INPUT_2_GRID>
Output:
<TRAIN_OUTPUT_2_GRID>

... (one block per training example)

[OPTIONAL IF PRESENT]
Here is some work on this task you have performed on adjacent requests.
<PRIOR_WORK_CONTEXT>

Generate one NEW input/output pair that follows the same transformation rule.
- Do NOT copy any demonstration pair exactly.
- The generated input and output must be different (non-trivial transformation).
- Your pair must be consistent with the same underlying rule.

Your final answer must be ONLY a single Python dict with exactly these keys:
{"input": [[...], ...], "output": [[...], ...]}
Use integer grid cells 0-9. Output nothing else.
```

## Editing (`build_editing_prompt`)

```text
Find the common rule that maps an input grid to an output grid given the examples below.

Example 1:
Input:
<TRAIN_INPUT_1_GRID>
Output:
<TRAIN_OUTPUT_1_GRID>

Example 2:
Input:
<TRAIN_INPUT_2_GRID>
Output:
<TRAIN_OUTPUT_2_GRID>

... (one block per training example)

[OPTIONAL IF PRESENT]
Here is some work on this task you have performed on adjacent requests.
<PRIOR_WORK_CONTEXT>

[IF include_true = false]
The following candidate input/output pair has been corrupted in some way. Your task is to return the corrected input/output pair that is consistent with the rule inferred from the training examples.

[IF include_true = true]
The following candidate input/output pair may already be correct or may have been corrupted in some way. Your task is to return the corrected version: if the pair is already correct, return it unchanged; if it is corrupted, fix whatever is necessary so that it becomes consistent with the rule inferred from the training examples.

Candidate pair:
Input:
<CANDIDATE_INPUT_GRID>
Output:
<CANDIDATE_OUTPUT_GRID>

Your final answer must be ONLY a single Python dict with exactly these keys:
{"input": [[...], ...], "output": [[...], ...]}
Use integer grid cells 0-9. Output nothing else.
```

## Inversion (`build_inversion_prompt`)

```text
Find the common rule that maps an input grid to an output grid given the examples below.

Example 1:
Input:
<TRAIN_INPUT_1_GRID>
Output:
<TRAIN_OUTPUT_1_GRID>

Example 2:
Input:
<TRAIN_INPUT_2_GRID>
Output:
<TRAIN_OUTPUT_2_GRID>

... (one block per training example)

[OPTIONAL IF PRESENT]
Here is some work on this task you have performed on adjacent requests.
<PRIOR_WORK_CONTEXT>

Now infer an input grid that would produce the following output by applying the same rule.
Desired output:
<DESIRED_OUTPUT_GRID>

Your final answer must be ONLY a single Python list-of-lists grid:
[[...], ...]
Use integer grid cells 0-9. Output nothing else.
```

## Output Grid Correctness (`build_output_grid_prompt`)

```text
Find the common rule that maps an input grid to an output grid, given the examples below.

Example 1
Input:
<TRAIN_INPUT_1_GRID>
Output:
<TRAIN_OUTPUT_1_GRID>

Example 2
Input:
<TRAIN_INPUT_2_GRID>
Output:
<TRAIN_OUTPUT_2_GRID>

... (one block per training example)

Below is a test input grid. Predict the corresponding output grid by applying the rule you found.
Test Input:
<TEST_INPUT_GRID>

Your final answer must be ONLY a single Python list-of-lists grid for that output:
[[...], ...]
Use integer grid cells 0-9. Output nothing else.
```

