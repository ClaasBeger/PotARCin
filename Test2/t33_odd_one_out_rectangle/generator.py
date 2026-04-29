"""Generator for t33: odd one out rectangle."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from verifier import _extract_objects, _normalize, _odd_candidates, verify_t33

Grid = List[List[int]]
Cell = Tuple[int, int]
Shape = Tuple[Cell, ...]

MAX_DIM = 22  # intentionally smaller than the global 30x30 cap


def _empty(h: int, w: int) -> Grid:
    return [[0 for _ in range(w)] for _ in range(h)]


def _dims(shape: Shape) -> Tuple[int, int]:
    if not shape:
        return (0, 0)
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _paint(grid: Grid, shape: Shape, r0: int, c0: int, color: int) -> None:
    for r, c in shape:
        grid[r0 + r][c0 + c] = color


def _shape_to_grid(shape: Shape, color: int) -> Grid:
    h, w = _dims(shape)
    g = _empty(h, w)
    _paint(g, shape, 0, 0, color)
    return g


def _shape_signature(shape: Shape) -> Shape:
    return _normalize(shape)


def _random_connected_partition(rr: random.Random, H: int, W: int, m: int) -> Tuple[Shape, ...] | None:
    cells = [(r, c) for r in range(H) for c in range(W)]
    cell_set = set(cells)
    min_piece_area = 2
    if H * W < m * min_piece_area:
        return None
    for _ in range(500):
        seeds = rr.sample(cells, m)
        pieces = [{s} for s in seeds]
        unassigned = cell_set - set(seeds)
        while unassigned:
            frontier = []
            sizes = [len(p) for p in pieces]
            min_size = min(sizes)
            for i, piece in enumerate(pieces):
                nbrs = set()
                for r, c in piece:
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        q = (r + dr, c + dc)
                        if q in unassigned:
                            nbrs.add(q)
                # bias toward growing smaller pieces
                weight = 3 if len(piece) == min_size else 1
                frontier.extend([(i, q)] * weight for q in nbrs)
            flat = [item for sub in frontier for item in sub]
            if not flat:
                break
            i, q = rr.choice(flat)
            pieces[i].add(q)
            unassigned.remove(q)
        if unassigned:
            continue
        sizes = [len(p) for p in pieces]
        if min(sizes) < min_piece_area:
            continue
        # keep pieces reasonably balanced
        if max(sizes) > H * W - (m - 1) * min_piece_area:
            continue
        shapes = tuple(_normalize(piece) for piece in pieces)
        if len({_shape_signature(s) for s in shapes}) != m:
            continue
        return shapes
    return None


def _random_connected_shape(rr: random.Random, area: int, max_h: int = 6, max_w: int = 6) -> Shape | None:
    for _ in range(300):
        cells = {(0, 0)}
        frontier = {(0, 1), (1, 0), (-1, 0), (0, -1)}
        while len(cells) < area and frontier:
            cell = rr.choice(list(frontier))
            frontier.remove(cell)
            cells.add(cell)
            r, c = cell
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (r + dr, c + dc)
                if q not in cells:
                    frontier.add(q)
        shape = _normalize(cells)
        h, w = _dims(shape)
        if len(shape) == area and h <= max_h and w <= max_w:
            return shape
    return None


def _has_clearance(occupied: Set[Cell], shape: Shape, r0: int, c0: int, h: int, w: int) -> bool:
    for r, c in shape:
        rr, cc = r0 + r, c0 + c
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (rr + dr, cc + dc) in occupied:
                    return False
    return True


def _place_shapes(rr: random.Random, shapes: Sequence[Shape], colors: Sequence[int], h: int, w: int) -> Grid | None:
    grid = _empty(h, w)
    occupied: Set[Cell] = set()
    slots = [
        (1, 1),
        (1, w - 8),
        (h - 8, 1),
        (h - 8, w - 8),
        (1, w // 2 - 3),
        (h // 2 - 3, 1),
        (h // 2 - 3, w - 8),
        (h - 8, w // 2 - 3),
    ]
    rr.shuffle(slots)
    order = list(range(len(shapes)))
    rr.shuffle(order)
    positions: Dict[int, Tuple[int, int]] = {}
    for idx, base in zip(order, slots):
        shape = shapes[idx]
        sh, sw = _dims(shape)
        placed = False
        for _ in range(150):
            r0 = max(0, min(h - sh, base[0] + rr.randint(-1, 1)))
            c0 = max(0, min(w - sw, base[1] + rr.randint(-1, 1)))
            if _has_clearance(occupied, shape, r0, c0, h, w):
                positions[idx] = (r0, c0)
                for r, c in shape:
                    occupied.add((r0 + r, c0 + c))
                placed = True
                break
        if not placed:
            return None
    for idx, shape in enumerate(shapes):
        r0, c0 = positions[idx]
        _paint(grid, shape, r0, c0, colors[idx])
    return grid


def _dims_ok(grid: Grid) -> bool:
    return len(grid) <= MAX_DIM and len(grid[0]) <= MAX_DIM


def _make_case(rr: random.Random) -> dict[str, Grid] | None:
    # Strongly favor 4 total pieces: odd one out + 3 rectangle pieces.
    # Keep some 5-piece cases and only rare 6-piece cases for variety.
    total_pieces = rr.choices([4, 5, 6], weights=[72, 25, 3])[0]
    rect_piece_count = total_pieces - 1

    rect_dims = [(3, 4), (3, 5), (4, 4), (4, 5), (5, 5), (4, 6)]
    rr.shuffle(rect_dims)
    rect_shapes: Tuple[Shape, ...] | None = None
    chosen_dims = None
    for H, W in rect_dims:
        rect_shapes = _random_connected_partition(rr, H, W, rect_piece_count)
        if rect_shapes is not None:
            chosen_dims = (H, W)
            break
    if rect_shapes is None:
        return None

    rect_sigs = {_shape_signature(s) for s in rect_shapes}
    total_rect_area = sum(len(s) for s in rect_shapes)

    odd_shape = None
    for _ in range(500):
        area = rr.randint(3, min(8, total_rect_area - 1))
        cand = _random_connected_shape(rr, area)
        if cand is None:
            continue
        if _shape_signature(cand) in rect_sigs:
            continue
        odd_shape = cand
        break
    if odd_shape is None:
        return None

    shapes = list(rect_shapes) + [odd_shape]
    colors = rr.sample(range(1, 10), len(shapes))

    max_h = max(_dims(s)[0] for s in shapes)
    max_w = max(_dims(s)[1] for s in shapes)
    grid_h = rr.randint(max(10, max_h + 4), min(MAX_DIM, 16 + (1 if total_pieces >= 5 else 0)))
    grid_w = rr.randint(max(10, max_w + 4), min(MAX_DIM, 16 + (1 if total_pieces >= 5 else 0)))
    grid = _place_shapes(rr, shapes, colors, grid_h, grid_w)
    if grid is None:
        return None

    if not _dims_ok(grid):
        return None
    out = verify_t33(grid)
    if not out:
        return None
    objs = _extract_objects(grid)
    if len(objs) != len(shapes):
        return None
    # Explicit single-solution constraint: exactly one removable object must make
    # the rest translatable into a rectangle, and it must be the intended odd one.
    if len(_odd_candidates(objs)) != 1:
        return None
    if out != _shape_to_grid(odd_shape, colors[-1]):
        # verifier chose a different odd object or failed orientation
        return None
    return {"input": grid, "output": out}


def generate(rng: random.Random | None = None) -> dict[str, Grid]:
    rr = rng if rng is not None else random
    for _ in range(1500):
        ex = _make_case(rr)
        if ex is not None and verify_t33(ex["input"]) == ex["output"]:
            return ex
    raise RuntimeError("failed to generate valid t33 sample")
