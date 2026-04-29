"""Generator for t29: color-directed diagonal motion of V/L-shaped objects.

Rule:
- Each connected monochrome object moves rigidly in a direction determined only by
  its color:
    6 (magenta): up-right    (-1, +1)
    3 (green):   up-left     (-1, -1)
    4 (yellow):  down-left   (+1, -1)
    8 (cyan):    down-right  (+1, +1)
- Each object moves as far as possible until moving farther would place some cell
  outside the grid. Equivalently, it is translated by the maximal nonnegative k
  such that all translated cells remain inside the grid.
- The objects are not rotated or deformed.

Generation goals:
- objects are usually 3-cell V/L-shapes in a 2x2 box, reflecting the demos;
- every generated example contains at least one object of each movement color
  (magenta, green, yellow, cyan), so all four directions are demonstrated;
- most objects are oriented to "point" in their movement direction, but every
  generated example contains at least one object whose orientation does *not*
  match its movement direction, encouraging the color-based rule;
- outputs are generated so translated objects do not overlap.
"""

from __future__ import annotations

import json
import random
from collections import deque
from typing import Dict, Iterable, List, Sequence, Set, Tuple

Grid = List[List[int]]
Cell = Tuple[int, int]
COLOR_TO_DIR: Dict[int, Cell] = {
    6: (-1, 1),
    3: (-1, -1),
    4: (1, -1),
    8: (1, 1),
}
ALLOWED_COLORS = [3, 4, 6, 8]

# 2x2 box with one corner missing. We interpret the missing corner as the direction
# the V/L-shape "points" toward.
POINTING_SHAPES: Dict[Cell, Set[Cell]] = {
    (-1, -1): {(0, 1), (1, 0), (1, 1)},  # missing top-left
    (-1, 1): {(0, 0), (1, 0), (1, 1)},   # missing top-right
    (1, -1): {(0, 0), (0, 1), (1, 1)},   # missing bottom-left
    (1, 1): {(0, 0), (0, 1), (1, 0)},    # missing bottom-right
}


def _empty(h: int, w: int, fill: int = 0) -> Grid:
    return [[fill for _ in range(w)] for _ in range(h)]


def _rectangular(grid: Grid) -> bool:
    return bool(grid) and bool(grid[0]) and all(len(row) == len(grid[0]) for row in grid)


def _components(grid: Grid) -> list[tuple[int, Set[Cell]]]:
    h, w = len(grid), len(grid[0])
    seen: Set[Cell] = set()
    out: list[tuple[int, Set[Cell]]] = []
    for r in range(h):
        for c in range(w):
            col = grid[r][c]
            if col == 0 or (r, c) in seen:
                continue
            q = deque([(r, c)])
            seen.add((r, c))
            cells: Set[Cell] = set()
            while q:
                x, y = q.popleft()
                cells.add((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and (nx, ny) not in seen and grid[nx][ny] == col:
                        seen.add((nx, ny))
                        q.append((nx, ny))
            out.append((col, cells))
    return out


def _max_translation(cells: Set[Cell], direction: Cell, h: int, w: int) -> int:
    dr, dc = direction
    if dr == -1:
        lim_r = min(r for r, _ in cells)
    else:
        lim_r = min(h - 1 - r for r, _ in cells)
    if dc == -1:
        lim_c = min(c for _, c in cells)
    else:
        lim_c = min(w - 1 - c for _, c in cells)
    return min(lim_r, lim_c)


def _translate(cells: Iterable[Cell], direction: Cell, k: int) -> Set[Cell]:
    dr, dc = direction
    return {(r + k * dr, c + k * dc) for r, c in cells}


def transform_t29(grid: Grid) -> Grid:
    if not _rectangular(grid):
        raise ValueError('Input grid must be non-empty and rectangular')
    h, w = len(grid), len(grid[0])
    out = _empty(h, w)
    comps = _components(grid)
    translated: list[tuple[int, Set[Cell]]] = []
    for color, cells in comps:
        if color not in COLOR_TO_DIR:
            raise ValueError(f'Unexpected color {color}; allowed colors are {sorted(COLOR_TO_DIR)}')
        k = _max_translation(cells, COLOR_TO_DIR[color], h, w)
        translated.append((color, _translate(cells, COLOR_TO_DIR[color], k)))
    # The intended task distribution forbids all translated collisions.
    occ: Dict[Cell, int] = {}
    for color, cells in translated:
        for cell in cells:
            if cell in occ:
                raise ValueError('Translated objects overlap; malformed example')
            occ[cell] = color
    for (r, c), color in occ.items():
        out[r][c] = color
    return out


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def _bbox(cells: Iterable[Cell]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)



def _can_place(cells: Set[Cell], occupied: Set[Cell], h: int, w: int, margin: int = 1) -> bool:
    for r, c in cells:
        if not (margin <= r < h - margin and margin <= c < w - margin):
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied:
                    return False
    return True


def generate(rng: random.Random | None = None) -> Dict[str, Grid]:
    rng = rng or random.Random()
    for _ in range(5000):
        h = rng.randint(10, 18)
        w = rng.randint(10, 18)
        grid = _empty(h, w)
        input_occ: Set[Cell] = set()
        output_occ: Set[Cell] = set()
        objects: list[tuple[int, Set[Cell], bool]] = []  # color, input cells, is_violation

        # Ensure every direction/color appears at least once in every example.
        n_objects = rng.randint(4, 8)
        colors = ALLOWED_COLORS[:] + [rng.choice(ALLOWED_COLORS) for _ in range(n_objects - len(ALLOWED_COLORS))]
        rng.shuffle(colors)
        violate_idx = rng.randrange(n_objects)

        success = True
        for i, color in enumerate(colors):
            move_dir = COLOR_TO_DIR[color]
            if i == violate_idx:
                pointing_choices = [d for d in POINTING_SHAPES if d != move_dir]
                pointing = rng.choice(pointing_choices)
                is_violation = True
            else:
                # Usually point in motion direction, but not always.
                if rng.random() < 0.8:
                    pointing = move_dir
                    is_violation = False
                else:
                    pointing_choices = [d for d in POINTING_SHAPES if d != move_dir]
                    pointing = rng.choice(pointing_choices)
                    is_violation = True
            shape = set(POINTING_SHAPES[pointing])

            # Pick a top-left placement such that both input and translated output fit,
            # and translated outputs do not overlap.
            max_r = max(r for r, _ in shape)
            max_c = max(c for _, c in shape)
            positions = [(r0, c0) for r0 in range(1, h - 1 - max_r) for c0 in range(1, w - 1 - max_c)]
            rng.shuffle(positions)
            placed = False
            for r0, c0 in positions:
                cells = {(r0 + r, c0 + c) for r, c in shape}
                if not _can_place(cells, input_occ, h, w, margin=1):
                    continue
                k = _max_translation(cells, move_dir, h, w)
                if k <= 0:
                    continue
                moved = _translate(cells, move_dir, k)
                # Keep output objects separated and away from touching each other.
                clash = False
                for r, c in moved:
                    if not (0 <= r < h and 0 <= c < w):
                        clash = True
                        break
                    for rr in range(r - 1, r + 2):
                        for cc in range(c - 1, c + 2):
                            if (rr, cc) in output_occ:
                                clash = True
                                break
                        if clash:
                            break
                    if clash:
                        break
                if clash:
                    continue
                # Also avoid an object's output directly overlapping any existing input
                # object only for cleanliness of the generated examples.
                if moved & input_occ:
                    continue
                for rr, cc in cells:
                    grid[rr][cc] = color
                input_occ |= cells
                output_occ |= moved
                objects.append((color, cells, is_violation))
                placed = True
                break
            if not placed:
                success = False
                break
        if not success or not objects or not any(v for _, _, v in objects):
            continue
        try:
            out = transform_t29(grid)
        except Exception:
            continue
        if out == grid:
            continue
        return {'input': grid, 'output': out}
    raise RuntimeError('Failed to generate valid t29 example')


if __name__ == '__main__':
    print(json.dumps(generate(rng=random.Random(0)), indent=2))
