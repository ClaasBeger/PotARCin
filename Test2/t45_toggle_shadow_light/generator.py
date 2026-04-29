"""Generator for Test2/t45: toggle shadow and light from a yellow marker.

Rule:
- A yellow (4) marker is placed at a corner or the midpoint of one edge.
- The grid has a single non-black/non-yellow background color.
- Non-background, non-yellow cells cast black shadows away from the marker:
    * from an edge midpoint, shadows are horizontal/vertical;
    * from a corner, shadows are diagonal.
- Yellow cells do not cast black shadows.
- If a yellow cell lies in another object's shadow, it shines light in the same
  direction instead: consecutive background cells after it are restored to the
  background color until the ray hits another non-background object.

Generation constraints used to keep examples well-formed and unambiguous:
- exactly one yellow marker appears at an allowed boundary location;
- the background is a single dominant non-black/non-yellow color;
- there are at least two non-yellow colored objects;
- examples include visible shadows;
- many generated examples include shadowed yellow light-cells that clear part
  of an otherwise black shadow ray.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from typing import List, Sequence, Tuple

Grid = List[List[int]]
Cell = Tuple[int, int]

BG_OUT = 0
YELLOW = 4
COLORS = list(range(1, 10))


def _empty(h: int, w: int, color: int) -> Grid:
    return [[color for _ in range(w)] for _ in range(h)]


def _rectangular(grid: Grid) -> bool:
    return bool(grid) and bool(grid[0]) and all(len(row) == len(grid[0]) for row in grid)


def _allowed_marker_positions(h: int, w: int) -> list[Cell]:
    positions = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    if w % 2 == 1:
        positions.extend([(0, w // 2), (h - 1, w // 2)])
    if h % 2 == 1:
        positions.extend([(h // 2, 0), (h // 2, w - 1)])
    # Deduplicate for very small grids.
    return sorted(set(positions))


def _find_marker(grid: Grid) -> Cell:
    h, w = len(grid), len(grid[0])
    candidates = [p for p in _allowed_marker_positions(h, w) if grid[p[0]][p[1]] == YELLOW]
    if len(candidates) != 1:
        raise ValueError("expected exactly one yellow marker at a corner or edge midpoint")
    return candidates[0]


def _direction_from_marker(marker: Cell, h: int, w: int) -> tuple[int, int]:
    r, c = marker
    if r == 0 and c == 0:
        return (1, 1)
    if r == 0 and c == w - 1:
        return (1, -1)
    if r == h - 1 and c == 0:
        return (-1, 1)
    if r == h - 1 and c == w - 1:
        return (-1, -1)
    if r == 0:
        return (1, 0)
    if r == h - 1:
        return (-1, 0)
    if c == 0:
        return (0, 1)
    if c == w - 1:
        return (0, -1)
    raise ValueError("marker is not at an allowed location")


def _infer_background(grid: Grid) -> int:
    counts = Counter(v for row in grid for v in row if v not in (BG_OUT, YELLOW))
    if not counts:
        raise ValueError("expected a non-yellow background color")
    return counts.most_common(1)[0][0]


def _ray_cells(start: Cell, dr: int, dc: int, h: int, w: int) -> list[Cell]:
    r, c = start
    out: list[Cell] = []
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w:
        out.append((r, c))
        r += dr
        c += dc
    return out


def transform_t45(grid: Grid) -> Grid:
    if not _rectangular(grid):
        raise ValueError("input grid must be non-empty and rectangular")
    h, w = len(grid), len(grid[0])
    marker = _find_marker(grid)
    dr, dc = _direction_from_marker(marker, h, w)
    bg = _infer_background(grid)

    shadowed: set[Cell] = set()
    for r in range(h):
        for c in range(w):
            val = grid[r][c]
            if val != bg and val != YELLOW:
                shadowed.update(_ray_cells((r, c), dr, dc, h, w))

    out = [row[:] for row in grid]
    for r, c in shadowed:
        if grid[r][c] == bg:
            out[r][c] = BG_OUT

    # Shadowed yellow cells act as light sources: restore background along the
    # ray, but only through original background cells. Light does not pass
    # through colored objects.
    for r in range(h):
        for c in range(w):
            if grid[r][c] == YELLOW and (r, c) in shadowed:
                for rr, cc in _ray_cells((r, c), dr, dc, h, w):
                    if grid[rr][cc] != bg:
                        break
                    out[rr][cc] = bg

    return out


def _place_cells(grid: Grid, cells: Sequence[Cell], color: int) -> None:
    h, w = len(grid), len(grid[0])
    for r, c in cells:
        if 0 <= r < h and 0 <= c < w:
            grid[r][c] = color


def _random_object(rng: random.Random, h: int, w: int, max_size: int = 9) -> list[Cell]:
    start = (rng.randrange(h), rng.randrange(w))
    target = rng.randint(1, max_size)
    cells = {start}
    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(list(frontier)))
    return sorted(cells)


def _choose_marker(rng: random.Random, h: int, w: int) -> Cell:
    positions = _allowed_marker_positions(h, w)
    return rng.choice(positions)


def _place_light_cell_in_shadow(
    rng: random.Random,
    grid: Grid,
    shadow_source: Cell,
    dr: int,
    dc: int,
    bg: int,
) -> bool:
    h, w = len(grid), len(grid[0])
    ray = _ray_cells(shadow_source, dr, dc, h, w)
    candidates = [(r, c) for r, c in ray[:-1] if grid[r][c] == bg]
    rng.shuffle(candidates)
    for r, c in candidates:
        # Need at least one background cell after it before another object.
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and grid[rr][cc] == bg:
            grid[r][c] = YELLOW
            return True
    return False


def generate(
    rng: random.Random | None = None,
    h: int | None = None,
    w: int | None = None,
    background: int | None = None,
    marker: Cell | None = None,
) -> dict[str, Grid]:
    rng = rng or random.Random()

    for _attempt in range(2000):
        hh = h if h is not None else rng.randint(8, 23)
        ww = w if w is not None else rng.randint(8, 23)
        if hh < 5 or ww < 5:
            continue

        bg_choices = [c for c in COLORS if c != YELLOW]
        bg = background if background is not None else rng.choice(bg_choices)
        if bg in (BG_OUT, YELLOW):
            continue

        grid = _empty(hh, ww, bg)
        mark = marker if marker is not None else _choose_marker(rng, hh, ww)
        if mark not in _allowed_marker_positions(hh, ww):
            continue
        grid[mark[0]][mark[1]] = YELLOW
        dr, dc = _direction_from_marker(mark, hh, ww)

        obj_colors = [c for c in COLORS if c not in (bg, YELLOW)]
        rng.shuffle(obj_colors)
        placed_sources: list[Cell] = []
        num_objects = rng.randint(3, 7)
        for i in range(num_objects):
            color = obj_colors[i % len(obj_colors)]
            cells = _random_object(rng, hh, ww, max_size=rng.randint(2, 10))
            cells = [p for p in cells if p != mark]
            if not cells:
                continue
            _place_cells(grid, cells, color)
            placed_sources.extend(cells)

        # Often include a yellow cell in an existing shadow, to demonstrate light.
        if placed_sources and rng.random() < 0.75:
            rng.shuffle(placed_sources)
            for src in placed_sources:
                if _place_light_cell_in_shadow(rng, grid, src, dr, dc, bg):
                    break

        # Ensure the boundary marker is still the only allowed-position yellow.
        for p in _allowed_marker_positions(hh, ww):
            if p != mark and grid[p[0]][p[1]] == YELLOW:
                grid[p[0]][p[1]] = bg

        try:
            out = transform_t45(grid)
        except Exception:
            continue

        changed = sum(
            1
            for r in range(hh)
            for c in range(ww)
            if out[r][c] != grid[r][c]
        )
        if changed < max(4, (hh * ww) // 25):
            continue
        non_bg_colors = {v for row in grid for v in row if v not in (bg, YELLOW)}
        if len(non_bg_colors) < 2:
            continue
        return {"input": grid, "output": out}

    raise RuntimeError("failed to generate a valid t45 sample")


if __name__ == "__main__":
    print(json.dumps(generate(rng=random.Random(0)), indent=2))
