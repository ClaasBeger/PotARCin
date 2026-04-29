"""Verifier for t27: square completion around each anchor shape.

Valid inputs contain several monochrome objects, identified by color.  The
objects can be partitioned into one or two groups.  Within each group, the member
shapes can be rotated and packed to form one filled k x k square.  For every
member shape, the output places a full copy of that square so that the member
itself remains fixed in its original input position and orientation.
"""

from __future__ import annotations

from itertools import combinations
from math import isqrt
from typing import Dict, Iterable, List, Sequence, Set, Tuple

Grid = List[List[int]]
Cell = Tuple[int, int]
Shape = Tuple[Cell, ...]


def _empty(h: int, w: int) -> Grid:
    return [[0 for _ in range(w)] for _ in range(h)]


def _normalize(cells: Iterable[Cell]) -> Shape:
    pts = list(cells)
    if not pts:
        return tuple()
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return tuple(sorted((r - min_r, c - min_c) for r, c in pts))


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


def _shape_signature(shape: Shape) -> Shape:
    return min(_unique_rotations(shape))


def _dims(shape: Shape) -> Tuple[int, int]:
    if not shape:
        return (0, 0)
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _extract_objects(grid: Grid) -> List[dict]:
    h = len(grid)
    w = len(grid[0]) if h else 0
    by_color: Dict[int, List[Cell]] = {}
    for r in range(h):
        for c in range(w):
            color = grid[r][c]
            if color:
                by_color.setdefault(color, []).append((r, c))

    objs: List[dict] = []
    for color, cells in by_color.items():
        norm = _normalize(cells)
        min_r = min(rr for rr, _ in cells)
        min_c = min(cc for _, cc in cells)
        objs.append({"cells": tuple(sorted(cells)), "shape": norm, "origin": (min_r, min_c), "color": color})
    objs.sort(key=lambda o: (o["origin"], o["color"], len(o["cells"])))
    return objs


def _placements_for(shape: Shape, k: int) -> List[Shape]:
    ph, pw = _dims(shape)
    out: List[Shape] = []
    for r0 in range(k - ph + 1):
        for c0 in range(k - pw + 1):
            out.append(tuple(sorted((r + r0, c + c0) for r, c in shape)))
    return out


def _tilings_for_group(objs: Sequence[dict], group: Sequence[int], k: int, anchor: int | None = None, limit: int = 2) -> List[Dict[int, Shape]]:
    """Find up to ``limit`` exact square tilings for a group.

    If anchor is supplied, that object must be used in its current orientation
    (rotation 0), which is the condition required for output reconstruction
    around that fixed anchor.
    """
    target = {(r, c) for r in range(k) for c in range(k)}
    options: Dict[int, List[Shape]] = {}
    option_sets: Dict[int, List[Set[Cell]]] = {}

    for idx in group:
        rotations = [objs[idx]["shape"]] if idx == anchor else _unique_rotations(objs[idx]["shape"])
        opts: List[Shape] = []
        seen: Set[Shape] = set()
        for rot in rotations:
            for placed in _placements_for(rot, k):
                if placed not in seen:
                    seen.add(placed)
                    opts.append(placed)
        if not opts:
            return []
        options[idx] = opts
        option_sets[idx] = [set(o) for o in opts]

    solutions: List[Dict[int, Shape]] = []
    calls = 0
    max_calls = 10000

    def backtrack(used: Set[Cell], chosen: Dict[int, Shape], remaining: Set[int]) -> None:
        nonlocal calls
        calls += 1
        if calls > max_calls or len(solutions) >= limit:
            return
        if not remaining:
            if used == target:
                solutions.append(dict(chosen))
            return

        uncovered = target - used
        coverable: Set[Cell] = set()
        for idx in remaining:
            for pset in option_sets[idx]:
                if not (pset & used):
                    coverable |= pset
        if uncovered - coverable:
            return

        best_idx = None
        best_opts: List[int] | None = None
        for idx in remaining:
            valid = [j for j, pset in enumerate(option_sets[idx]) if not (pset & used)]
            if best_opts is None or len(valid) < len(best_opts):
                best_idx = idx
                best_opts = valid
            if best_opts == []:
                return
        assert best_idx is not None and best_opts is not None

        next_remaining = set(remaining)
        next_remaining.remove(best_idx)
        for opt_i in best_opts:
            placed = options[best_idx][opt_i]
            pset = option_sets[best_idx][opt_i]
            chosen[best_idx] = placed
            backtrack(used | pset, chosen, next_remaining)
            chosen.pop(best_idx, None)
            if len(solutions) >= limit:
                return

    backtrack(set(), {}, set(group))
    return solutions


def _candidate_groups(objs: Sequence[dict]) -> List[Tuple[Tuple[int, ...], int]]:
    n = len(objs)
    cands: List[Tuple[Tuple[int, ...], int]] = []
    for m in range(2, min(4, n) + 1):
        for group in combinations(range(n), m):
            area = sum(len(objs[i]["shape"]) for i in group)
            k = isqrt(area)
            if k * k != area or k < 2:
                continue
            ok = True
            for idx in group:
                if len(_tilings_for_group(objs, group, k, anchor=idx, limit=2)) != 1:
                    ok = False
                    break
            if ok:
                cands.append((tuple(group), k))
    return cands


def _find_partitions(objs: Sequence[dict]) -> List[List[Tuple[Tuple[int, ...], int]]]:
    n = len(objs)
    all_idxs = set(range(n))
    cands = _candidate_groups(objs)
    parts: List[List[Tuple[Tuple[int, ...], int]]] = []

    for g, k in cands:
        s = set(g)
        if s == all_idxs and len(g) >= 3:
            parts.append([(g, k)])
        rem = all_idxs - s
        if rem:
            for g2, k2 in cands:
                s2 = set(g2)
                if s2 == rem and min(g) < min(g2):
                    parts.append([(g, k), (g2, k2)])

    dedup: List[List[Tuple[Tuple[int, ...], int]]] = []
    seen = set()
    for part in parts:
        key = tuple(sorted((tuple(sorted(g)), k) for g, k in part))
        if key not in seen:
            seen.add(key)
            dedup.append(sorted(part, key=lambda x: x[0]))
    return dedup


def verify_t27(grid: Grid) -> Grid:
    h = len(grid)
    w = len(grid[0]) if h else 0
    if h == 0 or w == 0 or any(len(row) != w for row in grid):
        return []

    objs = _extract_objects(grid)
    if not objs:
        return _empty(h, w)

    sigs = [_shape_signature(o["shape"]) for o in objs]
    if len(set(sigs)) != len(sigs):
        return _empty(h, w)

    partitions = _find_partitions(objs)
    if not partitions:
        return _empty(h, w)
    part = partitions[0]

    out = _empty(h, w)
    for group, k in part:
        for anchor in group:
            tiling = _tilings_for_group(objs, group, k, anchor=anchor, limit=1)[0]
            ar0, ac0 = objs[anchor]["origin"]
            anchor_tile = tiling[anchor]
            tile_min_r = min(r for r, _ in anchor_tile)
            tile_min_c = min(c for _, c in anchor_tile)
            dr = ar0 - tile_min_r
            dc = ac0 - tile_min_c
            for idx, placed in tiling.items():
                color = objs[idx]["color"]
                for r, c in placed:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        out[rr][cc] = color
    return out
