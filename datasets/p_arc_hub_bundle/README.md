---
language:
  - en
license: other
pretty_name: P-ARC (PotARCin Test Set) tabular export
configs:
  - config_name: default
    data_files:
      - split: train
        path: p_arc_dataset.csv
size_categories:
  - n<1K
tags:
  - arc
  - arc-agi
  - potarcin
  - program-synthesis
task_categories:
  - other
---

# P-ARC CSV export (PotARCin Test2)

One CSV file in UTF-8. Each row is one of the fifty P-ARC tasks from PotARCin (`t1.json` through `t50.json`). Besides the usual train/test grids, each row includes the fifty-sample bundle from `t<n>_samples_50.json` as compact JSON (same structure as the file, without the extra whitespace from pretty-printing), plus the `generator.py` and `verifier.py` sources from the matching task folder.

## Files

| File | Description |
|------|-------------|
| `p_arc_dataset.csv` | One row per task (`t1`–`t50`); column details below and in `SCHEMA.json` |
| `README.md` | Dataset card (what you are reading now) |
| `SCHEMA.json` | Same column layout in JSON for scripts |

## Columns

| Column | Description |
|--------|-------------|
| `task_id` | `t1` … `t50` |
| `train_demonstrations_json` | JSON array of training pairs `{input, output}`; grids are nested lists of integers |
| `test_input_json` | JSON grid for `test[0].input` |
| `test_output_json` | JSON grid for `test[0].output` |
| `stable_instances_50_json` | Compact JSON for the object in `t<n>_samples_50.json` |
| `generator_py` | Full `generator.py` source |
| `verifier_py` | Full `verifier.py` source |

Grids follow the usual ARC convention: each row is a list of cell integers.

## Size and reading the CSV

The file is fairly large because big JSON blobs and full Python files sit inside cells. In Python, the standard library `csv` module caps field length by default; bump it before you read:

```python
csv.field_size_limit(sys.maxsize)
```

### Stdlib example

```python
import csv, json, sys

csv.field_size_limit(sys.maxsize)

with open("p_arc_dataset.csv", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        train = json.loads(row["train_demonstrations_json"])
        test_in = json.loads(row["test_input_json"])
        test_out = json.loads(row["test_output_json"])
        samples = json.loads(row["stable_instances_50_json"])
        # row["generator_py"], row["verifier_py"]
```

### pandas

```python
import sys, pandas as pd
import csv as _csv

_csv.field_size_limit(sys.maxsize)
df = pd.read_csv("p_arc_dataset.csv", encoding="utf-8")
```

## Croissant metadata

Hugging Face generates Croissant metadata automatically for datasets that the viewer can read and convert. With the `configs` block above, the viewer targets `p_arc_dataset.csv`.

After pushing this README, you can check the generated Croissant JSON at:

`https://huggingface.co/api/datasets/<your-dataset-id>/croissant`
