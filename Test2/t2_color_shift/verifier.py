"""Verifier for Test2/t2 color-shift-by-leftmost-object rule."""

from __future__ import annotations

from typing import List, Sequence, Tuple

Grid = List[List[int]]
Cell = Tuple[int, int]


def _neighbors8(r: int, c: int) -> list[Cell]:
    out: list[Cell] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _leftmost_key(cells: Sequence[Cell]) -> tuple[int, int]:
    min_col = min(c for _, c in cells)
    min_row = min(r for r, c in cells if c == min_col)
    return (min_col, min_row)


def verify_t2(grid: Grid) -> Grid:
    h = len(grid)
    w = len(grid[0]) if h else 0
    if h == 0 or w == 0:
        return []

    seen: set[Cell] = set()
    objs: list[list[Cell]] = []
    colors: list[int] = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0 or (r, c) in seen:
                continue
            color = grid[r][c]
            stack = [(r, c)]
            comp: list[Cell] = []
            while stack:
                rr, cc = stack.pop()
                if (rr, cc) in seen or grid[rr][cc] != color:
                    continue
                seen.add((rr, cc))
                comp.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and (nr, nc) not in seen
                        and grid[nr][nc] == color
                    ):
                        stack.append((nr, nc))
            objs.append(comp)
            colors.append(color)

    if len(objs) <= 1:
        return [list(row) for row in grid]

    order = sorted(range(len(objs)), key=lambda i: _leftmost_key(objs[i]))
    out: Grid = [[0 for _ in range(w)] for _ in range(h)]
    for pos, obj_idx in enumerate(order):
        src_idx = order[pos - 1]  # shift right with wrap
        col = colors[src_idx]
        for r, c in objs[obj_idx]:
            out[r][c] = col
    return out
