"""Generator for Test2/t41: refraction through a dotted vertical surface.

Rule:
- The input contains a vertical dotted gray line (color 5) and a single colored
  cell to its left.
- The colored cell is imagined to move horizontally right, passing through the
  surface and ending at the far-right edge on the same row.
- From the intersection point with the vertical line, a second copy moves
  diagonally up-right until it hits either the top edge or the right edge.
- The output contains the same dotted line and exactly those two resulting
  colored cells on the right; the original left-side colored cell disappears.

Generation constraints used to keep examples well-formed and unambiguous:
- exactly one non-gray colored cell appears in the input;
- all gray cells lie in one column and form an every-other-row dotted line;
- the colored cell is strictly left of the line;
- the chosen row is not the top row, so the diagonal copy moves at least one
  step to the right and yields a distinct second result cell.
"""

from __future__ import annotations

import json
import random
from typing import List, Tuple

Grid = List[List[int]]
Cell = Tuple[int, int]

BG = 0
GRAY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _empty(h: int, w: int, color: int = BG) -> Grid:
    return [[color for _ in range(w)] for _ in range(h)]


def _rectangular(grid: Grid) -> bool:
    return bool(grid) and bool(grid[0]) and all(len(row) == len(grid[0]) for row in grid)


def _find_surface_col(grid: Grid) -> int:
    cols = {c for r, row in enumerate(grid) for c, v in enumerate(row) if v == GRAY}
    if len(cols) != 1:
        raise ValueError("expected all gray cells to lie in a single column")
    col = next(iter(cols))
    expected_rows = list(range(0, len(grid), 2))
    actual_rows = [r for r, row in enumerate(grid) if row[col] == GRAY]
    if actual_rows != expected_rows:
        raise ValueError("gray cells must form a vertical every-other-row dotted line starting at row 0")
    return col


def _find_colored_cell(grid: Grid) -> tuple[int, Cell]:
    cells = [(r, c, v) for r, row in enumerate(grid) for c, v in enumerate(row) if v not in (BG, GRAY)]
    if len(cells) != 1:
        raise ValueError("expected exactly one non-gray colored cell")
    r, c, color = cells[0]
    return color, (r, c)


def _diag_endpoint(start: Cell, w: int) -> Cell:
    r, c = start
    while r - 1 >= 0 and c + 1 < w:
        r -= 1
        c += 1
    return (r, c)


def transform_t41(grid: Grid) -> Grid:
    if not _rectangular(grid):
        raise ValueError("input grid must be non-empty and rectangular")
    h, w = len(grid), len(grid[0])
    surf_c = _find_surface_col(grid)
    color, (r, c) = _find_colored_cell(grid)
    if c >= surf_c:
        raise ValueError("colored cell must be strictly left of the surface")

    out = _empty(h, w)
    for rr in range(0, h, 2):
        out[rr][surf_c] = GRAY

    out[r][w - 1] = color
    dr, dc = _diag_endpoint((r, surf_c), w)
    out[dr][dc] = color
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def generate(
    rng: random.Random | None = None,
    h: int | None = None,
    w: int | None = None,
    surface_col: int | None = None,
    color: int | None = None,
) -> dict[str, Grid]:
    rng = rng or random.Random()

    for _attempt in range(1000):
        hh = h if h is not None else rng.randint(6, 20)
        ww = w if w is not None else rng.randint(8, 30)
        if not (2 <= hh <= 30 and 4 <= ww <= 30):
            continue

        surf = surface_col if surface_col is not None else rng.randint(1, ww - 2)
        if not (1 <= surf <= ww - 2):
            continue

        # Need a left-side source cell and a non-top row so the diagonal branch moves.
        src_r = rng.randint(1, hh - 1)
        src_c = rng.randint(0, surf - 1)
        col = color if color is not None else rng.choice(COLORS)
        if col in (BG, GRAY):
            continue

        inp = _empty(hh, ww)
        for rr in range(0, hh, 2):
            inp[rr][surf] = GRAY
        inp[src_r][src_c] = col

        out = transform_t41(inp)

        # Ensure the two outputs are distinct and both are to the right of the line.
        positions = [(r, c) for r, row in enumerate(out) for c, v in enumerate(row) if v == col]
        if len(positions) != 2:
            continue
        if not all(c > surf for r, c in positions):
            continue
        return {"input": inp, "output": out}

    raise RuntimeError("failed to generate a valid t41 sample")


if __name__ == "__main__":
    print(json.dumps(generate(rng=random.Random(0)), indent=2))
