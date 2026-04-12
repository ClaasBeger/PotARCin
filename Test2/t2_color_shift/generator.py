"""Generator for Test2/t2: cyclic color shift by leftmost-object order.

Rule:
- detect 8-connected non-zero objects
- sort objects by the position of their leftmost cell (col, then row)
- shift object colors to the right with wrap-around
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

Grid = List[List[int]]
Cell = Tuple[int, int]


def _empty(h: int, w: int) -> Grid:
    return [[0 for _ in range(w)] for _ in range(h)]


def _neighbors8(r: int, c: int) -> list[Cell]:
    out: list[Cell] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _grow_component(
    rng: random.Random,
    start: Cell,
    target_size: int,
    h: int,
    w: int,
    blocked: set[Cell],
) -> list[Cell]:
    comp = {start}
    while len(comp) < target_size:
        frontier: set[Cell] = set()
        for r, c in comp:
            for rr, cc in _neighbors8(r, c):
                if rr < 0 or cc < 0 or rr >= h or cc >= w:
                    continue
                if (rr, cc) in comp or (rr, cc) in blocked:
                    continue
                frontier.add((rr, cc))
        if not frontier:
            break
        comp.add(rng.choice(list(frontier)))
    return sorted(comp)


def _leftmost_key(cells: Sequence[Cell]) -> tuple[int, int]:
    min_col = min(c for _, c in cells)
    min_row = min(r for r, c in cells if c == min_col)
    return (min_col, min_row)


def _leftmost_col(cells: Sequence[Cell]) -> int:
    return min(c for _, c in cells)


def _shift_colors_right(grid: Grid) -> Grid:
    h = len(grid)
    w = len(grid[0]) if h else 0
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
    out = _empty(h, w)
    for pos, obj_idx in enumerate(order):
        src_idx = order[pos - 1]  # right-shift with wrap-around
        new_color = colors[src_idx]
        for r, c in objs[obj_idx]:
            out[r][c] = new_color
    return out


def generate(
    width: int | None = None,
    height: int | None = None,
    object_count: int | None = None,
    objects: Sequence[Sequence[Cell]] | None = None,
    colors: Sequence[int] | None = None,
    rng: random.Random | None = None,
) -> dict[str, Grid]:
    """Return one input/output pair for the t2 rule."""
    rr = rng if rng is not None else random
    if width is None:
        width = rr.randint(8, 14)
    if height is None:
        height = rr.randint(8, 14)
    if width < 6 or height < 6:
        raise ValueError("width/height must be >= 6")

    if objects is None:
        if object_count is None:
            object_count = rr.randint(2, 5)
        if object_count < 2:
            raise ValueError("need at least 2 objects for color shifting")
        max_layout_attempts = 120
        objects = None
        for _ in range(max_layout_attempts):
            blocked: set[Cell] = set()
            built: list[list[Cell]] = []
            attempts = 0
            max_attempts = 4000
            while len(built) < object_count and attempts < max_attempts:
                attempts += 1
                r0 = rr.randint(0, height - 1)
                c0 = rr.randint(0, width - 1)
                if (r0, c0) in blocked:
                    continue
                size = rr.randint(2, 8)
                comp = _grow_component(rr, (r0, c0), size, height, width, blocked)
                if len(comp) < 2:
                    continue
                # Reserve 8-neighborhood halo so objects stay distinct.
                halo = set(comp)
                for r, c in comp:
                    for nr, nc in _neighbors8(r, c):
                        if 0 <= nr < height and 0 <= nc < width:
                            halo.add((nr, nc))
                blocked.update(halo)
                built.append(comp)
            if len(built) < object_count:
                continue
            # No ties on leftmost position criterion shown in provided examples:
            # require strictly unique leftmost columns across objects.
            left_cols = [_leftmost_col(comp) for comp in built]
            if len(set(left_cols)) != len(left_cols):
                continue
            objects = built
            break
        if objects is None:
            raise RuntimeError("could not sample non-tied objects by leftmost-column order")
    else:
        if len(objects) < 2:
            raise ValueError("objects must contain at least 2 objects")
        left_cols = [_leftmost_col(comp) for comp in objects]
        if len(set(left_cols)) != len(left_cols):
            raise ValueError("objects must have unique leftmost columns (no ordering ties)")

    if colors is None:
        colors = rr.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], len(objects))
    if len(colors) != len(objects):
        raise ValueError("colors length must equal objects length")

    inp = _empty(height, width)
    for comp, col in zip(objects, colors):
        for r, c in comp:
            if r < 0 or c < 0 or r >= height or c >= width:
                raise ValueError("object cell outside grid")
            inp[r][c] = int(col)
    out = _shift_colors_right(inp)
    return {"input": inp, "output": out}
