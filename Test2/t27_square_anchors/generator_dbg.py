"""Generator for t27_square_anchors.

Inputs contain colored anchor shapes.  The shapes split into one or two groups;
each group can be rotated/reassembled into a filled square.  The output completes
one full square around every anchor shape without moving or rotating that anchor.

The generator samples from a library of uniquely reconstructible square tilings,
applies random dihedral transforms, combines one or two square arrangements, and
uses the verifier to reject any ambiguous, cropped, or overlapping result.
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from verifier import _extract_objects, _find_partitions, _shape_signature, verify_t27

Grid = List[List[int]]
Cell = Tuple[int, int]
Shape = Tuple[Cell, ...]
PlacedGroup = Tuple[int, Tuple[Shape, ...]]


def _empty(h: int, w: int) -> Grid:
    return [[0 for _ in range(w)] for _ in range(h)]


def _normalize(cells: Iterable[Cell]) -> Shape:
    pts = list(cells)
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return tuple(sorted((r - min_r, c - min_c) for r, c in pts))


def _dims(shape: Shape) -> Tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _paint(grid: Grid, shape: Shape, r0: int, c0: int, color: int) -> None:
    for r, c in shape:
        grid[r0 + r][c0 + c] = color


def _nonzero_count(grid: Grid) -> int:
    return sum(1 for row in grid for v in row if v != 0)


def _input_preserved(inp: Grid, out: Grid) -> bool:
    return all(v == 0 or out[r][c] == v for r, row in enumerate(inp) for c, v in enumerate(row))


def _dihedral_cell(r: int, c: int, k: int, t: int) -> Cell:
    # Four rotations, then their horizontal reflections.  This gives reflected
    # chiral shapes too, unlike rotation-only augmentation.
    if t == 0:
        return r, c
    if t == 1:
        return c, k - 1 - r
    if t == 2:
        return k - 1 - r, k - 1 - c
    if t == 3:
        return k - 1 - c, r
    if t == 4:
        return r, k - 1 - c
    if t == 5:
        return k - 1 - c, k - 1 - r
    if t == 6:
        return k - 1 - r, c
    return c, r


def _transform_group(group: PlacedGroup, t: int) -> PlacedGroup:
    k, shapes = group
    out: List[Shape] = []
    for shape in shapes:
        out.append(_normalize(_dihedral_cell(r, c, k, t) for r, c in shape))
    return k, tuple(out)


def _base_groups() -> List[PlacedGroup]:
    # Two 3x3 two-piece arrangements from the demonstrations.
    g2a = (
        3,
        (
            ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1)),
            ((1, 1), (1, 2), (2, 2)),
        ),
    )
    g2b = (
        3,
        (
            ((0, 1), (1, 1), (2, 1)),
            ((0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 2)),
        ),
    )

    # A 4x4 four-piece arrangement from the demonstrations.
    p0 = ((0, 0), (1, 0), (2, 0), (2, 1), (3, 0))
    p1 = ((1, 1), (1, 2), (2, 2))
    p2 = ((0, 1), (0, 2), (0, 3), (1, 3))
    p3 = ((2, 3), (3, 1), (3, 2), (3, 3))
    g4 = (4, (p0, p1, p2, p3))

    # Six 3-piece variants formed by merging two pieces of the 4-piece square.
    merged: List[PlacedGroup] = []
    pieces = [p0, p1, p2, p3]
    for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        shapes: List[Shape] = [_normalize(list(pieces[a]) + list(pieces[b]))]
        for i, p in enumerate(pieces):
            if i not in (a, b):
                shapes.append(_normalize(p))
        merged.append((4, tuple(shapes)))

    return [g2a, g2b, g4] + merged


_BASE_GROUPS = _base_groups()


def _choose_groups(rr: random.Random) -> List[PlacedGroup]:
    """Choose already-transformed group templates.

    To keep generation fast and unambiguous, two-square examples combine either
    two 2-piece squares or one 2-piece square with one 3-piece square.  One-square
    examples include both 3-piece and 4-piece squares.  The random dihedral
    transform still gives reflected and rotated shape variants.
    """
    t = rr.randrange(8)
    two_piece = [0, 1]
    three_piece = [4, 5]
    four_piece = [2]

    roll = rr.random()
    if roll < 0.42:
        gi = rr.choice(three_piece)
        return [_transform_group(_BASE_GROUPS[gi], t)]
    if roll < 0.58:
        return [_transform_group(_BASE_GROUPS[2], t)]
    if roll < 0.74:
        a, b = rr.sample(two_piece, 2)
        return [_transform_group(_BASE_GROUPS[a], t), _transform_group(_BASE_GROUPS[b], t)]

    # A 5-anchor case: one small 3x3 two-piece square and one 4x4 three-piece square.
    a = rr.choice(two_piece)
    b = rr.choice(three_piece)
    if rr.random() < 0.5:
        return [_transform_group(_BASE_GROUPS[a], t), _transform_group(_BASE_GROUPS[b], t)]
    return [_transform_group(_BASE_GROUPS[b], t), _transform_group(_BASE_GROUPS[a], t)]

def _has_clearance(occupied: Set[Cell], shape: Shape, r0: int, c0: int) -> bool:
    for r, c in shape:
        rr, cc = r0 + r, c0 + c
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (rr + dr, cc + dc) in occupied:
                    return False
    return True


def _place_shapes(rr: random.Random, shapes: Sequence[Shape], colors: Sequence[int], h: int, w: int) -> Grid | None:
    """Place anchors in widely separated regions.

    The task's outputs contain one completed square per anchor, so random dense
    placement wastes many attempts on overlapping completions.  These slots keep
    examples varied through jitter while leaving enough room for every 3x3/4x4
    completion used by this generator.
    """
    grid = _empty(h, w)
    slots = [
        (2, 2),
        (2, max(10, w - 9)),
        (max(10, h - 9), 2),
        (max(10, h - 9), max(10, w - 9)),
        (h // 2 - 2, 2),
        (h // 2 - 2, max(10, w - 9)),
    ]
    order = list(range(len(shapes)))
    rr.shuffle(order)
    slot_order = slots[:]
    rr.shuffle(slot_order)
    placed: Dict[int, Tuple[int, int]] = {}
    occupied: Set[Cell] = set()
    for idx, base in zip(order, slot_order):
        shape = shapes[idx]
        ph, pw = _dims(shape)
        for _ in range(40):
            r0 = base[0] + rr.randint(-1, 1)
            c0 = base[1] + rr.randint(-1, 1)
            if r0 < 1 or c0 < 1 or r0 + ph >= h - 1 or c0 + pw >= w - 1:
                continue
            if _has_clearance(occupied, shape, r0, c0):
                placed[idx] = (r0, c0)
                for r, c in shape:
                    occupied.add((r0 + r, c0 + c))
                break
        else:
            return None
    for idx, shape in enumerate(shapes):
        r0, c0 = placed[idx]
        _paint(grid, shape, r0, c0, colors[idx])
    return grid

def generate(rng: random.Random | None = None) -> dict[str, Grid]:
    rr = rng if rng is not None else random

    for _ in range(1200):
        raw_groups = _choose_groups(rr)
        groups = [_transform_group(g, rr.randrange(8)) for g in raw_groups]
        shapes = [s for _, gs in groups for s in gs]

        # No repeated shapes up to rotation anywhere in the grid.
        sigs = [_shape_signature(s) for s in shapes]
        if len(set(sigs)) != len(sigs):
            continue

        colors = rr.sample(range(1, 10), len(shapes))
        h = rr.randint(24, 30)
        w = rr.randint(24, 30)
        grid = _place_shapes(rr, shapes, colors, h, w)
        if grid is None:
            continue

        objs = _extract_objects(grid)
        if len(objs) != len(shapes):
            continue
        partitions = _find_partitions(objs)
        if len(partitions) != 1:
            continue
        if sorted(len(g) for g, _ in partitions[0]) != sorted(len(gs) for _, gs in groups):
            continue

        out = verify_t27(grid)
        expected = sum(len(gs) * k * k for k, gs in groups)
        if _nonzero_count(out) != expected:
            continue
        if not _input_preserved(grid, out):
            continue
        return {"input": grid, "output": out}

    raise RuntimeError("failed to generate a valid t27 sample")
