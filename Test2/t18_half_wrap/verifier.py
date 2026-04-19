"""Verifier for Test2/t18 horizontal wrap-and-continue rule."""

from __future__ import annotations

from typing import List

from generator import transform_t18

Grid = List[List[int]]


def verify_t18(inp: Grid) -> Grid:
    if not inp or not inp[0]:
        raise ValueError("input grid must be non-empty")
    w = len(inp[0])
    if any(len(row) != w for row in inp):
        raise ValueError("ragged input grid")
    return transform_t18(inp)
