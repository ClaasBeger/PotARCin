"""Verifier for t31 mirror-block transfer task."""

from __future__ import annotations

from typing import List

from generator import transform_t31

Grid = List[List[int]]


def verify_t31(inp: Grid) -> Grid:
    if not inp or not inp[0]:
        raise ValueError('input grid must be non-empty')
    w = len(inp[0])
    if any(len(row) != w for row in inp):
        raise ValueError('input grid must be rectangular')
    return transform_t31(inp)
