"""Generator for Test2/t11: diagonal rays with wall bounces.

Rule implemented:
- Input has exactly one non-zero start cell (source color).
- From start, shoot along all 4 diagonal directions.
- For each direction:
  1) March to the first wall (boundary cell), coloring visited cells with source color.
  2) Bounce off that wall and move the same number of steps again, with reflective
     boundary behavior (can bounce multiple times while consuming remaining steps).
  3) Mark the final cell blue (1).
- Finally, color each final cell's 4 diagonal neighbors green (3), potentially
  overwriting existing colors.
"""

from __future__ import annotations

import random
from typing import List

Grid = List[List[int]]

BLUE = 1
GREEN = 3
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
SOURCE_PALETTE = [2, 4, 5, 6, 7, 8, 9]


def _empty(h: int, w: int) -> Grid:
    return [[0 for _ in range(w)] for _ in range(h)]


def _find_single_start(inp: Grid) -> tuple[int, int, int]:
    cells = [(r, c, inp[r][c]) for r in range(len(inp)) for c in range(len(inp[0])) if inp[r][c] != 0]
    if len(cells) != 1:
        raise ValueError("t11 expects exactly one non-zero start cell in input")
    return cells[0]


def _reflect_on_wall(dr: int, dc: int, r: int, c: int, h: int, w: int) -> tuple[int, int]:
    if r == 0 or r == h - 1:
        dr = -dr
    if c == 0 or c == w - 1:
        dc = -dc
    return dr, dc


def _step_with_reflect(r: int, c: int, dr: int, dc: int, h: int, w: int) -> tuple[int, int, int, int]:
    nr, nc = r + dr, c + dc
    if nr < 0 or nr >= h:
        dr = -dr
        nr = r + dr
    if nc < 0 or nc >= w:
        dc = -dc
        nc = c + dc
    return nr, nc, dr, dc


def transform_t11(inp: Grid) -> Grid:
    h = len(inp)
    w = len(inp[0]) if h else 0
    if h == 0 or w == 0 or any(len(row) != w for row in inp):
        raise ValueError("input must be a non-empty rectangular grid")

    sr, sc, source_color = _find_single_start(inp)
    out = [row[:] for row in inp]
    endpoints: list[tuple[int, int]] = []

    for dr0, dc0 in DIAGONALS:
        r, c = sr, sc
        dr, dc = dr0, dc0
        steps_to_wall = 0

        # Phase 1: move to first wall.
        while True:
            r += dr
            c += dc
            steps_to_wall += 1
            out[r][c] = source_color
            if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                break

        # Phase 2: bounce and move same number of steps, allowing additional bounces.
        dr, dc = _reflect_on_wall(dr, dc, r, c, h, w)
        remaining = steps_to_wall
        while remaining > 0:
            r, c, dr, dc = _step_with_reflect(r, c, dr, dc, h, w)
            out[r][c] = source_color
            remaining -= 1
        endpoints.append((r, c))

    # Mark final cells blue.
    for r, c in endpoints:
        out[r][c] = BLUE

    # Mark diagonal neighbors of final cells green (overwriting allowed).
    for r, c in endpoints:
        for dr, dc in DIAGONALS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out[rr][cc] = GREEN

    return out


def generate(
    width: int | None = None,
    height: int | None = None,
    rng: random.Random | None = None,
) -> dict[str, Grid]:
    rr = rng if rng is not None else random
    w = width if width is not None else rr.randint(9, 14)
    h = height if height is not None else rr.randint(9, 14)
    if h < 5 or w < 5:
        raise ValueError("height and width must be >= 5")

    inp = _empty(h, w)
    color = rr.choice(SOURCE_PALETTE)
    sr = rr.randint(1, h - 2)
    sc = rr.randint(1, w - 2)
    inp[sr][sc] = color
    out = transform_t11(inp)
    return {"input": inp, "output": out}
