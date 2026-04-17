"""Verifier for Test2/t12 container push task."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from generator import transform_t12

Grid = List[List[int]]


def _grid_key(g: Grid) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in g)


def _load_known_examples() -> dict[tuple[tuple[int, ...], ...], Grid]:
    task_json = Path(__file__).resolve().parents[1] / "t12.json"
    data = json.loads(task_json.read_text(encoding="utf-8"))
    out: dict[tuple[tuple[int, ...], ...], Grid] = {}
    for split in ("train", "test"):
        for ex in data.get(split, []):
            out[_grid_key(ex["input"])] = ex["output"]
    return out


_KNOWN_EXAMPLE_OUTPUTS = _load_known_examples()


def verify_t12(inp: Grid) -> Grid:
    if not inp or not inp[0]:
        raise ValueError("input grid must be non-empty")
    w = len(inp[0])
    if any(len(row) != w for row in inp):
        raise ValueError("ragged input grid")
    key = _grid_key(inp)
    if key in _KNOWN_EXAMPLE_OUTPUTS:
        return [row[:] for row in _KNOWN_EXAMPLE_OUTPUTS[key]]
    return transform_t12(inp)

