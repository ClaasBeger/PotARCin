"""Adapter for H-ARC incorrect human submissions.

CSV columns expected:
- ``task_name`` (e.g. ``a85d4709.json``)
- ``task_type`` (training/evaluation; evaluation rows can be ignored)
- ``test_output_grid`` (rows encoded as ``|012|340|...|``)
- ``count`` (frequency)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

from framework.grids import Grid
from framework.tasks.arc_dataset import ROOT_DIR


H_ARC_INCORRECT_CSV = ROOT_DIR / "external" / "H-ARC" / "incorrect_submissions.csv"


@dataclass(frozen=True)
class HArcIncorrectSubmission:
    task_id: str
    task_type: str
    output_grid: Grid
    count: int


def parse_h_arc_pipe_grid(encoded: str) -> Grid:
    """Parse ``|``-delimited rows (e.g. ``|012|340|``) into a grid."""
    s = (encoded or "").strip()
    if not s:
        raise ValueError("Empty H-ARC pipe grid.")
    parts = [p for p in s.split("|") if p != ""]
    if not parts:
        raise ValueError(f"Invalid H-ARC pipe grid: {encoded!r}")
    grid: Grid = []
    width: int | None = None
    for row in parts:
        vals = [int(ch) for ch in row]
        if width is None:
            width = len(vals)
            if width == 0:
                raise ValueError(f"Invalid empty row in H-ARC pipe grid: {encoded!r}")
        elif len(vals) != width:
            raise ValueError(f"Non-rectangular H-ARC pipe grid: {encoded!r}")
        grid.append(vals)
    return grid


def iter_h_arc_incorrect_submissions(
    csv_path: Path = H_ARC_INCORRECT_CSV,
) -> Iterator[HArcIncorrectSubmission]:
    if not csv_path.exists():
        return
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = (row.get("task_name") or "").strip()
            if not task_name:
                continue
            task_id = task_name[:-5] if task_name.endswith(".json") else task_name
            task_type = (row.get("task_type") or "").strip()
            encoded = (row.get("test_output_grid") or "").strip()
            if not encoded:
                continue
            try:
                out = parse_h_arc_pipe_grid(encoded)
                count = int((row.get("count") or "1").strip())
            except Exception:
                continue
            yield HArcIncorrectSubmission(
                task_id=task_id,
                task_type=task_type,
                output_grid=out,
                count=count,
            )


def load_h_arc_incorrect_outputs_for_task(
    task_id: str,
    *,
    csv_path: Path = H_ARC_INCORRECT_CSV,
    exclude_evaluation: bool = True,
) -> List[HArcIncorrectSubmission]:
    out: List[HArcIncorrectSubmission] = []
    for item in iter_h_arc_incorrect_submissions(csv_path):
        if item.task_id != task_id:
            continue
        if exclude_evaluation and item.task_type.lower() == "evaluation":
            continue
        out.append(item)
    return out


def sample_h_arc_incorrect_output_for_task(
    task_id: str,
    *,
    rng_seed: int = 0,
    csv_path: Path = H_ARC_INCORRECT_CSV,
) -> Optional[HArcIncorrectSubmission]:
    import random

    rows = load_h_arc_incorrect_outputs_for_task(
        task_id,
        csv_path=csv_path,
        exclude_evaluation=True,
    )
    if not rows:
        return None
    rng = random.Random(rng_seed)
    weights = [max(1, r.count) for r in rows]
    (picked,) = rng.choices(rows, weights=weights, k=1)
    return picked
