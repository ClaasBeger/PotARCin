#!/usr/bin/env python3
"""Export P-ARC (Test2 t1..t50) into one CSV row per task.

Columns include the task's name and natural-language rule, canonical
demonstrations + held-out test pair, the bundled 50-sample stable JSON, the three
human-error corruptions, and generator/verifier source from each task directory.
Rules are read from ``Test2/p_arc_rules.json``.

Default output: ``evaluation/figures/p_arc_dataset.csv`` (UTF-8, RFC4180-style).

Cells can exceed Python's default ``csv`` field size limit (~128 KiB). When loading
with :mod:`csv`, call ``csv.field_size_limit(sys.maxsize)`` first.

Example::

    cd /path/to/PotARCin && python scripts/export_p_arc_csv_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _discover_task_dirs(test2_dir: Path) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    if not test2_dir.is_dir():
        return mapping
    for p in sorted(test2_dir.iterdir()):
        if not p.is_dir():
            continue
        m = re.match(r"t(\d+)_", p.name)
        if not m:
            continue
        n = int(m.group(1))
        if 1 <= n <= 50 and n not in mapping:
            mapping[n] = p
    return mapping


def _compact_json(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def main() -> int:
    root = _repo_root()

    ap = argparse.ArgumentParser(description="Export Test2 P-ARC rows to CSV.")
    ap.add_argument(
        "--test2-dir",
        type=Path,
        default=root / "Test2",
        help="Directory containing t<n>.json and t<n>_*/ task folders.",
    )
    ap.add_argument(
        "--output",
        "-o",
        type=Path,
        default=root / "evaluation" / "figures" / "p_arc_dataset.csv",
        help="Destination CSV path.",
    )
    args = ap.parse_args()

    test2_dir: Path = args.test2_dir.resolve()
    out_path: Path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    task_dirs = _discover_task_dirs(test2_dir)
    rules = json.loads((test2_dir / "p_arc_rules.json").read_text(encoding="utf-8"))
    fieldnames = (
        "task_id",
        "task_name",
        "rule",
        "rule_source",
        "train_demonstrations_json",
        "test_input_json",
        "test_output_json",
        "stable_instances_50_json",
        "human_corruptions_json",
        "generator_py",
        "verifier_py",
    )

    rows_written = 0
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_NONNUMERIC,
        )
        w.writeheader()

        for n in range(1, 51):
            tid = f"t{n}"
            pair_path = test2_dir / f"t{n}.json"
            if not pair_path.is_file():
                raise FileNotFoundError(f"Missing task JSON: {pair_path}")

            raw = json.loads(pair_path.read_text(encoding="utf-8"))
            train = raw.get("train")
            test_list = raw.get("test")
            if not isinstance(train, list):
                raise ValueError(f"{pair_path}: expected 'train' list")
            if not isinstance(test_list, list) or not test_list:
                raise ValueError(f"{pair_path}: expected non-empty 'test' list")
            te0 = test_list[0]
            if not isinstance(te0, dict) or "input" not in te0 or "output" not in te0:
                raise ValueError(f"{pair_path}: test[0] must have input/output")

            subdir = task_dirs.get(n)
            if subdir is None:
                raise FileNotFoundError(f"Missing Test2 subdirectory for {tid} under {test2_dir}")

            samples_path = subdir / f"t{n}_samples_50.json"
            if not samples_path.is_file():
                raise FileNotFoundError(f"Missing stable samples JSON: {samples_path}")

            gen_path = subdir / "generator.py"
            ver_path = subdir / "verifier.py"
            if not gen_path.is_file():
                raise FileNotFoundError(f"Missing generator: {gen_path}")
            if not ver_path.is_file():
                raise FileNotFoundError(f"Missing verifier: {ver_path}")

            stable_obj = json.loads(samples_path.read_text(encoding="utf-8"))
            stable_json = _compact_json(stable_obj)

            # Three erroneous human attempts; older tasks name them t<n>_corrupt<k>.json,
            # newer ones t<n>_corrupted<k>.json.
            corruption_paths = sorted(subdir.glob(f"t{n}_corrupt*.json"))
            if len(corruption_paths) != 3:
                raise FileNotFoundError(f"Expected 3 corruption files for {tid}, found {len(corruption_paths)}")
            corruptions = [
                json.loads(p.read_text(encoding="utf-8"))["test"][0] for p in corruption_paths
            ]
            rule = rules.get(tid)
            if not rule:
                raise KeyError(f"Missing rule for {tid} in p_arc_rules.json")

            w.writerow(
                {
                    "task_id": tid,
                    "task_name": re.sub(r"^t\d+_", "", subdir.name),
                    "rule": rule["rule"],
                    "rule_source": rule["source"],
                    "train_demonstrations_json": _compact_json(train),
                    "test_input_json": _compact_json(te0["input"]),
                    "test_output_json": _compact_json(te0["output"]),
                    "stable_instances_50_json": stable_json,
                    "human_corruptions_json": _compact_json(
                        [{"input": c["input"], "output": c["output"]} for c in corruptions]
                    ),
                    "generator_py": gen_path.read_text(encoding="utf-8"),
                    "verifier_py": ver_path.read_text(encoding="utf-8"),
                }
            )
            rows_written += 1

    print(f"Wrote {rows_written} rows to {out_path}")
    print(
        "Note: large cells — Python readers should set csv.field_size_limit(sys.maxsize)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
