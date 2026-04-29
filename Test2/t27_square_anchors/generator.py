"""Diverse generator for t27_square_anchors.

Generated inputs contain several colored anchor shapes. The shapes can be
partitioned into one or two square arrangements. Within each arrangement, the
member shapes tile a filled 3x3, 4x4, or 5x5 square when rotations are allowed.
For each member shape, the output completes that square around the anchor while
keeping that anchor fixed in place and orientation.
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from verifier import _extract_objects, _find_partitions, _shape_signature, verify_t27

Grid = List[List[int]]
Cell = Tuple[int, int]
Shape = Tuple[Cell, ...]
PlacedGroup = Tuple[int, Tuple[Shape, ...]]


# ------------------------------ basic helpers ------------------------------

def _empty(h: int, w: int) -> Grid:
    return [[0 for _ in range(w)] for _ in range(h)]


def _normalize(cells: Iterable[Cell]) -> Shape:
    pts = list(cells)
    if not pts:
        return tuple()
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


def _rotate(shape: Shape, times: int) -> Shape:
    pts = list(shape)
    for _ in range(times % 4):
        pts = [(c, -r) for r, c in pts]
        pts = list(_normalize(pts))
    return tuple(sorted(pts))


def _unique_rotations(shape: Shape) -> List[Shape]:
    rots: List[Shape] = []
    seen: Set[Shape] = set()
    for t in range(4):
        s = _rotate(shape, t)
        if s not in seen:
            seen.add(s)
            rots.append(s)
    return rots


def _placements_for(shape: Shape, k: int) -> List[Shape]:
    ph, pw = _dims(shape)
    out: List[Shape] = []
    for r0 in range(k - ph + 1):
        for c0 in range(k - pw + 1):
            out.append(tuple(sorted((r + r0, c + c0) for r, c in shape)))
    return out


# --------------------- square-partition / orientation logic ---------------------

def _random_connected_partition(rr: random.Random, k: int, m: int) -> Tuple[Shape, ...] | None:
    """Randomly partition a k x k square into m connected pieces.

    This intentionally explores a broad variety of two-, three-, and four-piece
    square decompositions.  All pieces are 4-connected.
    """
    cells = [(r, c) for r in range(k) for c in range(k)]
    cell_set = set(cells)

    for _ in range(200):
        seeds = rr.sample(cells, m)
        pieces = [{seed} for seed in seeds]
        unassigned = cell_set - set(seeds)

        while unassigned:
            frontier: List[Tuple[int, Cell]] = []
            for i, piece in enumerate(pieces):
                nbrs: Set[Cell] = set()
                for r, c in piece:
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        q = (r + dr, c + dc)
                        if q in unassigned:
                            nbrs.add(q)
                frontier.extend((i, q) for q in nbrs)
            if not frontier:
                break
            i, q = rr.choice(frontier)
            pieces[i].add(q)
            unassigned.remove(q)
        if unassigned:
            continue

        # Mild size balancing so the pieces are usually all visually present.
        sizes = sorted(len(p) for p in pieces)
        if sizes[0] == 1 and k >= 4:
            continue
        if sizes[-1] > k * k - (m - 1) * 2:
            continue

        shapes = tuple(_normalize(piece) for piece in pieces)
        sigs = [_shape_signature(s) for s in shapes]
        if len(set(sigs)) != m:
            continue
        return shapes
    return None


def _random_group(rr: random.Random, k: int, m: int) -> PlacedGroup | None:
    for _ in range(200):
        shapes = _random_connected_partition(rr, k, m)
        if shapes is None:
            continue
        # Prefer groups containing at least one asymmetric piece, which makes
        # rotation-based reasoning more common and more visible.
        if not any(len(_unique_rotations(s)) > 1 for s in shapes):
            continue
        return (k, shapes)
    return None


def _tilings_fixed_orientation(shapes: Sequence[Shape], k: int, limit: int = 1) -> int:
    """Number of exact tilings using current orientations only (translations allowed)."""
    target = {(r, c) for r in range(k) for c in range(k)}
    options: Dict[int, List[Set[Cell]]] = {}
    for i, shape in enumerate(shapes):
        opts = [set(p) for p in _placements_for(shape, k)]
        if not opts:
            return 0
        options[i] = opts

    solutions = 0

    def backtrack(used: Set[Cell], remaining: Set[int]) -> None:
        nonlocal solutions
        if solutions >= limit:
            return
        if not remaining:
            if used == target:
                solutions += 1
            return

        uncovered = target - used
        coverable: Set[Cell] = set()
        for idx in remaining:
            for p in options[idx]:
                if not (p & used):
                    coverable |= p
        if uncovered - coverable:
            return

        best_idx = None
        best_valid: List[Set[Cell]] | None = None
        for idx in remaining:
            valid = [p for p in options[idx] if not (p & used)]
            if best_valid is None or len(valid) < len(best_valid):
                best_idx = idx
                best_valid = valid
            if best_valid == []:
                return
        assert best_idx is not None and best_valid is not None

        rem2 = set(remaining)
        rem2.remove(best_idx)
        for p in best_valid:
            backtrack(used | p, rem2)
            if solutions >= limit:
                return

    backtrack(set(), set(range(len(shapes))))
    return solutions


def _orient_group(rr: random.Random, group: PlacedGroup, force_rotation_need: bool) -> PlacedGroup | None:
    """Randomly rotate each member shape independently.

    If ``force_rotation_need`` is True, reject orientations that still tile the
    square with no piece rotation at all.
    """
    k, shapes = group
    for _ in range(200):
        oriented = tuple(_rotate(shape, rr.randrange(4)) for shape in shapes)
        # Avoid the trivial "everything shares the same rotation" cases most of the time.
        rotable = [i for i, s in enumerate(shapes) if len(_unique_rotations(s)) > 1]
        if len(rotable) >= 2:
            rots = [oriented[i] for i in rotable]
            if len({_shape_signature(s) for s in rots}) == 1 and rr.random() < 0.85:
                continue
        if force_rotation_need and _tilings_fixed_orientation(oriented, k, limit=1) > 0:
            continue
        return (k, oriented)
    return None


def _choose_group_specs(rr: random.Random) -> List[Tuple[int, int]]:
    """Return one or two (k, piece_count) specs within the allowed constraints."""
    layouts = [
        [(3, 3)],
        [(4, 3)],
        [(4, 4)],
        [(5, 3)],
        [(5, 4)],
        [(3, 2), (3, 2)],
        [(3, 2), (4, 3)],
        [(3, 2), (4, 4)],
        [(3, 2), (5, 3)],
        [(4, 2), (4, 2)],
        [(4, 2), (4, 3)],
        [(4, 2), (5, 3)],
        [(5, 2), (3, 2)],
        [(5, 2), (4, 2)],
        [(5, 2), (4, 3)],
        [(3, 3), (3, 2)],
        [(4, 3), (3, 2)],
        [(4, 3), (4, 2)],
        [(5, 3), (3, 2)],
    ]
    return list(rr.choice(layouts))


def _build_groups(rr: random.Random) -> List[PlacedGroup] | None:
    specs = _choose_group_specs(rr)
    base_groups: List[PlacedGroup] = []
    used_sigs: Set[Shape] = set()

    for k, m in specs:
        for _ in range(250):
            group = _random_group(rr, k, m)
            if group is None:
                continue
            sigs = [_shape_signature(s) for s in group[1]]
            if any(sig in used_sigs for sig in sigs):
                continue
            # Strongly bias toward examples that require rotation for at least one
            # arrangement; single-group tasks always require it.
            force = len(specs) == 1 or rr.random() < 0.8
            oriented = _orient_group(rr, group, force_rotation_need=force)
            if oriented is None:
                continue
            osigs = [_shape_signature(s) for s in oriented[1]]
            if any(sig in used_sigs for sig in osigs):
                continue
            base_groups.append(oriented)
            used_sigs.update(osigs)
            break
        else:
            return None

    if len(base_groups) == 2:
        # Ensure at least one group in two-square tasks really needs rotation.
        needs = [_tilings_fixed_orientation(gs, k, limit=1) == 0 for k, gs in base_groups]
        if not any(needs):
            return None
    return base_groups


# ------------------------------ input placement ------------------------------

def _has_clearance(occupied: Set[Cell], shape: Shape, r0: int, c0: int) -> bool:
    for r, c in shape:
        rr, cc = r0 + r, c0 + c
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (rr + dr, cc + dc) in occupied:
                    return False
    return True


def _place_shapes(rr: random.Random, shapes: Sequence[Shape], colors: Sequence[int], h: int, w: int) -> Grid | None:
    grid = _empty(h, w)
    # Widely separated anchor slots so the completed output squares also remain separated.
    slots = [
        (2, 2),
        (2, w - 9),
        (h - 9, 2),
        (h - 9, w - 9),
        (2, w // 2 - 4),
        (h // 2 - 4, 2),
        (h // 2 - 4, w - 9),
        (h - 9, w // 2 - 4),
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
        for _ in range(120):
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


# ------------------------------ public generator ------------------------------

def generate(rng: random.Random | None = None) -> dict[str, Grid]:
    rr = rng if rng is not None else random

    for _ in range(1000):
        groups = _build_groups(rr)
        if groups is None:
            continue
        shapes = [s for _, gs in groups for s in gs]
        sigs = [_shape_signature(s) for s in shapes]
        if len(set(sigs)) != len(sigs):
            continue

        total_shapes = len(shapes)
        h = rr.randint(22, 30)
        w = rr.randint(22, 30)
        if total_shapes >= 5:
            h = max(h, 26)
            w = max(w, 26)
        colors = rr.sample(range(1, 10), total_shapes)
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
