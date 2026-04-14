"""Generator for Test2/t7 diagonal-connect-and-wrap task."""

from __future__ import annotations

import copy
import random
from typing import List, Optional

Grid = List[List[int]]

CYAN = 8
PALETTE = [1, 2, 3, 4, 5, 6, 7, 9]
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _empty(h: int, w: int) -> Grid:
    return [[0 for _ in range(w)] for _ in range(h)]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _components8_same_color(grid: Grid) -> list[tuple[int, set[tuple[int, int]]]]:
    h = len(grid)
    w = len(grid[0]) if h else 0
    seen: set[tuple[int, int]] = set()
    comps: list[tuple[int, set[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            col = grid[r][c]
            if col in (0, CYAN) or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            comp: set[tuple[int, int]] = set()
            while stack:
                cr, cc = stack.pop()
                comp.add((cr, cc))
                for nr, nc in _neighbors8(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and grid[nr][nc] == col:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            comps.append((col, comp))
    return comps


def _find_cyan(grid: Grid) -> tuple[int, int]:
    cells = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == CYAN]
    if len(cells) != 1:
        raise ValueError("expected exactly one cyan cell")
    return cells[0]


def _first_diagonal_hit(
    grid: Grid,
    cr: int,
    cc: int,
) -> Optional[tuple[int, tuple[int, int], tuple[int, int], int]]:
    """Return (distance, direction, hit_cell, hit_color) for first diagonal contact."""
    h = len(grid)
    w = len(grid[0]) if h else 0
    best: Optional[tuple[int, tuple[int, int], tuple[int, int], int]] = None
    for dr, dc in DIAGONALS:
        step = 1
        while True:
            nr = cr + step * dr
            nc = cc + step * dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                break
            val = grid[nr][nc]
            if val not in (0, CYAN):
                cand = (step, (dr, dc), (nr, nc), val)
                if best is None or step < best[0]:
                    best = cand
                break
            step += 1
    return best


def _component_containing(
    comps: list[tuple[int, set[tuple[int, int]]]],
    cell: tuple[int, int],
) -> set[tuple[int, int]]:
    for _col, comp in comps:
        if cell in comp:
            return comp
    raise ValueError("hit cell not in any component")


def transform_t7(inp: Grid) -> Grid:
    h = len(inp)
    w = len(inp[0]) if h else 0
    if h == 0 or w == 0 or any(len(row) != w for row in inp):
        raise ValueError("input must be a non-empty rectangular grid")

    cr, cc = _find_cyan(inp)
    comps = _components8_same_color(inp)
    if not comps:
        raise ValueError("expected at least one non-cyan object")
    hit = _first_diagonal_hit(inp, cr, cc)
    if hit is None:
        raise ValueError("cyan does not diagonally hit any object")
    _dist, (dr, dc), (hr, hc), _hit_color = hit
    target = _component_containing(comps, (hr, hc))

    out = copy.deepcopy(inp)

    # Connect cyan diagonally to the first-hit object.
    rr, cc2 = cr + dr, cc + dc
    while (rr, cc2) != (hr, hc):
        if out[rr][cc2] == 0:
            out[rr][cc2] = CYAN
        rr += dr
        cc2 += dc

    # Wrap target object with cyan so no target cell touches black orthogonally.
    add: set[tuple[int, int]] = set()
    for r, c in target:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and out[nr][nc] == 0:
                add.add((nr, nc))
    for r, c in add:
        out[r][c] = CYAN
    return out


def _random_component(
    grid: Grid,
    color: int,
    rng: random.Random,
    *,
    min_cells: int = 3,
    max_cells: int = 8,
) -> bool:
    h = len(grid)
    w = len(grid[0]) if h else 0
    def can_place_cell(r: int, c: int) -> bool:
        if not (0 < r < h - 1 and 0 < c < w - 1):
            return False
        if grid[r][c] != 0:
            return False
        # Keep different objects at most diagonally adjacent (no orthogonal contact).
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] not in (0, color):
                return False
        return True

    empties = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1) if can_place_cell(r, c)]
    if not empties:
        return False
    seed = rng.choice(empties)
    cells = {seed}
    target = rng.randint(min_cells, max_cells)
    frontier = [seed]
    while frontier and len(cells) < target:
        r, c = frontier.pop(rng.randrange(len(frontier)))
        nbrs = [
            (nr, nc)
            for nr, nc in _neighbors8(r, c)
            if (nr, nc) not in cells and can_place_cell(nr, nc)
        ]
        rng.shuffle(nbrs)
        for nr, nc in nbrs[:2]:
            cells.add((nr, nc))
            frontier.append((nr, nc))
            if len(cells) >= target:
                break
    if len(cells) < min_cells:
        return False
    for r, c in cells:
        grid[r][c] = color
    return True


def _unique_min_hit_component(grid: Grid, cr: int, cc: int) -> bool:
    comps = _components8_same_color(grid)
    if not comps:
        return False
    comp_by_cell: dict[tuple[int, int], int] = {}
    for i, (_col, comp) in enumerate(comps):
        for rc in comp:
            comp_by_cell[rc] = i

    hits: list[tuple[int, int]] = []  # (distance, comp_id)
    h = len(grid)
    w = len(grid[0]) if h else 0
    for dr, dc in DIAGONALS:
        step = 1
        while True:
            nr = cr + step * dr
            nc = cc + step * dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                break
            v = grid[nr][nc]
            if v not in (0, CYAN):
                hits.append((step, comp_by_cell[(nr, nc)]))
                break
            step += 1
    if not hits:
        return False
    best_d = min(d for d, _ in hits)
    winners = {cid for d, cid in hits if d == best_d}
    return len(winners) == 1


def generate(
    width: int | None = None,
    height: int | None = None,
    n_shapes: int | None = None,
    rng: random.Random | None = None,
) -> dict[str, Grid]:
    rr = rng if rng is not None else random
    if width is None:
        width = rr.randint(9, 13)
    if height is None:
        height = rr.randint(9, 13)
    if width < 7 or height < 7:
        raise ValueError("width and height must be >= 7")
    if n_shapes is None:
        n_shapes = rr.randint(2, 4)
    if n_shapes < 2:
        raise ValueError("n_shapes must be >= 2")

    for _ in range(600):
        inp = _empty(height, width)
        colors = rr.sample(PALETTE, k=min(n_shapes, len(PALETTE)))
        placed = 0
        for col in colors:
            if _random_component(inp, col, rr):
                placed += 1
        if placed < 2:
            continue

        empties = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1) if inp[r][c] == 0]
        if not empties:
            continue
        cr, cc = rr.choice(empties)
        inp[cr][cc] = CYAN

        if not _unique_min_hit_component(inp, cr, cc):
            continue
        try:
            out = transform_t7(inp)
        except Exception:
            continue
        return {"input": inp, "output": out}

    raise RuntimeError("failed to sample valid t7 instance")
