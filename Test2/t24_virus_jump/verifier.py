"""Verifier for Test2/t24 virus jump spread rule."""

from __future__ import annotations

from typing import List

from generator import transform_t24

Grid = List[List[int]]


def verify_t24(inp: Grid) -> Grid:
    if not inp or not inp[0]:
        raise ValueError("input grid must be non-empty")
    w = len(inp[0])
    if any(len(row) != w for row in inp):
        raise ValueError("ragged input grid")
    return transform_t24(inp)
