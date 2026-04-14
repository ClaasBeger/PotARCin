"""Generator for Test2/t10: 180° partner fill (center-symmetric sparse overlay).

Rule:
- For each cell (r, c): output[r, c] = input[r, c] if nonzero, else
  input[H - 1 - r][W - 1 - c] (180° partner).

Training-aligned input constraint (not required by the map algebra, but matches t10.json):
- On every 180° orbit, at most one cell is nonzero in the input. Equivalently, no nonzero
  cell sits on top of an orbit that already has a nonzero on its partner — so partner
  cells stay empty and each orbit has a single “source” color that fills both endpoints in
  the output (output is constant on each orbit).

Generation places a few small 4-connected monochrome shapes; each new cell must lie on an
orbit that is still completely empty, so shapes never “claim” the same orbit twice.
"""

from __future__ import annotations

import random
from collections import deque
from typing import List

Grid = List[List[int]]

PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def _empty(h: int, w: int) -> Grid:
    return [[0 for _ in range(w)] for _ in range(h)]


def _partner(h: int, w: int, r: int, c: int) -> tuple[int, int]:
    return h - 1 - r, w - 1 - c


def _orbit_clear_for_placement(inp: Grid, h: int, w: int, r: int, c: int) -> bool:
    """True if we may place a new nonzero at (r,c) under the one-nonzero-per-orbit rule."""
    if inp[r][c] != 0:
        return False
    pr, pc = _partner(h, w, r, c)
    if (pr, pc) == (r, c):
        return True
    return inp[pr][pc] == 0


def transform_t10(inp: Grid) -> Grid:
    h = len(inp)
    w = len(inp[0]) if h else 0
    if h == 0 or w == 0 or any(len(row) != w for row in inp):
        raise ValueError("input must be a non-empty rectangular grid")
    out = _empty(h, w)
    for r in range(h):
        for c in range(w):
            v = inp[r][c]
            if v != 0:
                out[r][c] = v
            else:
                pr, pc = _partner(h, w, r, c)
                out[r][c] = inp[pr][pc]
    return out


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _grow_shape(
    inp: Grid,
    h: int,
    w: int,
    rng: random.Random,
    color: int,
    target_size: int,
) -> int:
    """Grow one 4-connected component of `color`; return cells placed (may be 0)."""
    seeds = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if _orbit_clear_for_placement(inp, h, w, r, c)
    ]
    if not seeds:
        return 0
    rng.shuffle(seeds)
    sr, sc = seeds[0]
    q: deque[tuple[int, int]] = deque()
    q.append((sr, sc))
    placed = 0
    while q and placed < target_size:
        r, c = q.popleft()
        if not _orbit_clear_for_placement(inp, h, w, r, c):
            continue
        inp[r][c] = color
        placed += 1
        nbrs = [(nr, nc) for nr, nc in _neighbors4(r, c) if 0 <= nr < h and 0 <= nc < w]
        rng.shuffle(nbrs)
        for nr, nc in nbrs:
            if placed >= target_size:
                break
            if _orbit_clear_for_placement(inp, h, w, nr, nc):
                q.append((nr, nc))
    return placed


def _output_nonzero_fraction(out: Grid) -> float:
    h, w = len(out), len(out[0])
    nz = sum(1 for r in range(h) for c in range(w) if out[r][c] != 0)
    return nz / float(h * w)


def _input_orbit_constraint_ok(inp: Grid) -> bool:
    h, w = len(inp), len(inp[0])
    seen: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            pr, pc = _partner(h, w, r, c)
            seen.add((r, c))
            if (pr, pc) != (r, c):
                seen.add((pr, pc))
            a, b = inp[r][c], inp[pr][pc]
            if a != 0 and b != 0:
                return False
    return True


def generate(
    width: int | None = None,
    height: int | None = None,
    rng: random.Random | None = None,
) -> dict[str, Grid]:
    rr = rng if rng is not None else random
    h = height if height is not None else rr.randint(6, 12)
    w = width if width is not None else rr.randint(6, 12)
    if h < 3 or w < 3:
        raise ValueError("height and width must be >= 3")

    max_out_density = 0.48
    min_in_cells = 4

    for _ in range(600):
        inp = _empty(h, w)
        n_shapes = rr.randint(2, 5)
        total_placed = 0
        for _ in range(n_shapes):
            color = rr.choice(PALETTE)
            cap = max(2, min(9, (h * w) // 14))
            target = rr.randint(2, cap)
            total_placed += _grow_shape(inp, h, w, rr, color, target)

        if total_placed < min_in_cells:
            continue
        if not _input_orbit_constraint_ok(inp):
            continue

        out = transform_t10(inp)
        if _output_nonzero_fraction(out) > max_out_density:
            continue

        return {"input": inp, "output": out}

    raise RuntimeError("failed to sample valid t10 instance")
