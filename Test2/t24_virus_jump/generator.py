"""Generator for Test2/t24: virus jump-spread recoloring.

Rule:
- Exactly one red virus cell (color 2) is present; no other red cells exist.
- Other cells form colored shapes (4-connected components of same color).
- Virus spreads from infected cells to shapes by either:
  * direct orthogonal adjacency,
  * orthogonal jump over exactly one black cell, or
  * direct diagonal adjacency (distance (1,1)).
- When a shape is reached, the whole shape becomes infected and can further
  spread from any of its cells.
- Output recolors all infected shapes to red and removes the singular initial
  virus cell.
"""

from __future__ import annotations

import random
from collections import deque
from typing import List, Set, Tuple

Grid = List[List[int]]
Pos = Tuple[int, int]
Component = Set[Pos]

BG = 0
RED = 2
NEI4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _empty(h: int, w: int) -> Grid:
    return [[BG for _ in range(w)] for _ in range(h)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _components_nonred(inp: Grid) -> list[Component]:
    h = len(inp)
    w = len(inp[0]) if h else 0
    vis: set[Pos] = set()
    comps: list[Component] = []
    for r in range(h):
        for c in range(w):
            col = inp[r][c]
            if col in (BG, RED) or (r, c) in vis:
                continue
            q = deque([(r, c)])
            vis.add((r, c))
            comp: Component = {(r, c)}
            while q:
                rr, cc = q.popleft()
                for dr, dc in NEI4:
                    nr, nc = rr + dr, cc + dc
                    if (
                        _in_bounds(h, w, nr, nc)
                        and inp[nr][nc] == col
                        and (nr, nc) not in vis
                    ):
                        vis.add((nr, nc))
                        comp.add((nr, nc))
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _jump_reachable(a: Pos, b: Pos, inp: Grid) -> bool:
    ar, ac = a
    br, bc = b
    dr = br - ar
    dc = bc - ac
    # Direct orthogonal neighbor
    if abs(dr) + abs(dc) == 1:
        return True
    # Direct diagonal neighbor
    if abs(dr) == 1 and abs(dc) == 1:
        return True
    # Orthogonal jump over exactly one black cell
    if dr == 0 and abs(dc) == 2:
        mid = (ar, ac + dc // 2)
        return inp[mid[0]][mid[1]] == BG
    if dc == 0 and abs(dr) == 2:
        mid = (ar + dr // 2, ac)
        return inp[mid[0]][mid[1]] == BG
    return False


def transform_t24(inp: Grid) -> Grid:
    if not inp or not inp[0]:
        raise ValueError("input grid must be non-empty")
    h = len(inp)
    w = len(inp[0])
    if any(len(row) != w for row in inp):
        raise ValueError("input grid must be rectangular")

    red_cells = [(r, c) for r in range(h) for c in range(w) if inp[r][c] == RED]
    if len(red_cells) != 1:
        raise ValueError("expected exactly one red virus cell")
    virus = red_cells[0]

    comps = _components_nonred(inp)
    infected = [False] * len(comps)

    frontier: set[Pos] = {virus}
    changed = True
    while changed:
        changed = False
        for i, comp in enumerate(comps):
            if infected[i]:
                continue
            reached = False
            for p in frontier:
                if reached:
                    break
                for q in comp:
                    if _jump_reachable(p, q, inp):
                        reached = True
                        break
            if reached:
                infected[i] = True
                frontier |= comp
                changed = True

    out = [row[:] for row in inp]
    # Remove initial singular virus cell.
    out[virus[0]][virus[1]] = BG
    # Recolor infected shapes.
    for i, comp in enumerate(comps):
        if not infected[i]:
            continue
        for r, c in comp:
            out[r][c] = RED
    return out


def _grow_component(
    rr: random.Random, h: int, w: int, seed: Pos, size_max: int, occupied: set[Pos]
) -> list[Pos]:
    if seed in occupied:
        return []
    cells = {seed}
    target = rr.randint(2, size_max)
    while len(cells) < target:
        boundary: list[Pos] = []
        for r, c in cells:
            for dr, dc in NEI4:
                nr, nc = r + dr, c + dc
                if _in_bounds(h, w, nr, nc) and (nr, nc) not in cells and (nr, nc) not in occupied:
                    boundary.append((nr, nc))
        if not boundary:
            break
        cells.add(rr.choice(boundary))
    return sorted(cells)


def _place_component(grid: Grid, cells: list[Pos], color: int, occupied: set[Pos]) -> None:
    for r, c in cells:
        grid[r][c] = color
        occupied.add((r, c))


def generate(
    rng: random.Random | None = None,
    *,
    min_h: int = 10,
    max_h: int = 14,
    min_w: int = 10,
    max_w: int = 14,
) -> dict[str, Grid]:
    rr = rng or random.Random()
    palette = [1, 3, 4, 5, 6, 7, 8, 9]  # no red

    for _ in range(3000):
        h = rr.randint(min_h, max_h)
        w = rr.randint(min_w, max_w)
        inp = _empty(h, w)
        occupied: set[Pos] = set()

        n_shapes = rr.randint(5, 9)
        colors = [rr.choice(palette) for _ in range(n_shapes)]

        ok = True
        for col in colors:
            placed = False
            for _ in range(80):
                seed = (rr.randint(0, h - 1), rr.randint(0, w - 1))
                comp = _grow_component(rr, h, w, seed, size_max=6, occupied=occupied)
                if len(comp) < 2:
                    continue
                _place_component(inp, comp, col, occupied)
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        # Place the singular red virus cell on an empty location.
        empties = [(r, c) for r in range(h) for c in range(w) if inp[r][c] == BG]
        if not empties:
            continue
        vr, vc = rr.choice(empties)
        inp[vr][vc] = RED

        try:
            out = transform_t24(inp)
        except ValueError:
            continue
        if out == inp:
            continue

        # Keep useful variety: at least one infected and one untouched shape.
        comps = _components_nonred(inp)
        infected_count = 0
        untouched_count = 0
        for comp in comps:
            in_col = inp[next(iter(comp))[0]][next(iter(comp))[1]]
            out_cols = {out[r][c] for r, c in comp}
            if out_cols == {RED} and in_col != RED:
                infected_count += 1
            elif out_cols == {in_col}:
                untouched_count += 1
        if infected_count < 2 or untouched_count < 1:
            continue

        return {"input": inp, "output": out}

    raise RuntimeError("failed to sample valid t24 instance")
