"""Verifier for Test2/t46 4-color sudoku quadrants rule."""

from __future__ import annotations

from typing import List

from generator import transform_t46

Grid = List[List[int]]


def verify_t46(inp: Grid) -> Grid:
    if not inp or not inp[0]:
        raise ValueError("input grid must be non-empty")
    if len(inp) != 5 or any(len(row) != 5 for row in inp):
        raise ValueError("input grid must be 5x5")
    return transform_t46(inp)
