"""Verifier for t29: color-directed diagonal motion of objects."""

from __future__ import annotations

from typing import List

from generator import transform_t29

Grid = List[List[int]]


def verify_t29(inp: Grid) -> Grid:
    if not inp or not inp[0]:
        raise ValueError('Input grid must be non-empty')
    w = len(inp[0])
    if any(len(row) != w for row in inp):
        raise ValueError('Input grid must be rectangular')
    return transform_t29(inp)
