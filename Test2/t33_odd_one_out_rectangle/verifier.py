"""Verifier for t33: odd one out rectangle.

The input contains several colored objects. Exactly one object is the odd one
out. If that object is removed, the remaining objects can be translated (without
rotation) to tile one filled rectangle. The output is the odd object alone,
cropped to its bounding box and preserving its original orientation and color.
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


def _dims(shape: Shape) -> Tuple[int, int]:
    if not shape:
        return (0, 0)
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _paint(grid: Grid, shape: Shape, r0: int, c0: int, color: int) -> None:
    for r, c in shape:
        grid[r0 + r][c0 + c] = color


def _extract_objects(grid: Grid) -> List[dict]:
    h = len(grid)
    w = len(grid[0]) if h else 0
    seen: Set[Cell] = set()
    objs: List[dict] = []
    for r in range(h):
        for c in range(w):
            color = grid[r][c]
            if color == 0 or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            cells: List[Cell] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    q = (rr + dr, cc + dc)
                    qr, qc = q
                    if 0 <= qr < h and 0 <= qc < w and q not in seen and grid[qr][qc] == color:
                        seen.add(q)
                        stack.append(q)
            min_r = min(rr for rr, _ in cells)
            min_c = min(cc for _, cc in cells)
            shape = _normalize(cells)
            objs.append({
                "color": color,
                "cells": tuple(sorted(cells)),
                "origin": (min_r, min_c),
                "shape": shape,
                "area": len(cells),
            })
    objs.sort(key=lambda o: (o["origin"], o["color"], o["area"]))
    return objs


def _factor_pairs(area: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for h in range(2, isqrt(area) + 1):
        if area % h == 0:
            w = area // h
            out.append((h, w))
            if h != w:
                out.append((w, h))
    return out


def _placements_for(shape: Shape, H: int, W: int) -> List[Shape]:
    ph, pw = _dims(shape)
    out: List[Shape] = []
    if ph > H or pw > W:
        return out
    for r0 in range(H - ph + 1):
        for c0 in range(W - pw + 1):
            out.append(tuple(sorted((r0 + r, c0 + c) for r, c in shape)))
    return out


def _can_tile_rectangle(shapes: Sequence[Shape]) -> bool:
    area = sum(len(s) for s in shapes)
    if len(shapes) < 2:
        return False
    for H, W in _factor_pairs(area):
        target = {(r, c) for r in range(H) for c in range(W)}
        options: Dict[int, List[Set[Cell]]] = {}
        valid = True
        for i, shape in enumerate(shapes):
            opts = [set(p) for p in _placements_for(shape, H, W)]
            if not opts:
                valid = False
                break
            options[i] = opts
        if not valid:
            continue

        def backtrack(used: Set[Cell], remaining: Set[int]) -> bool:
            if not remaining:
                return used == target
            uncovered = target - used
            coverable: Set[Cell] = set()
            for idx in remaining:
                for p in options[idx]:
                    if not (p & used):
                        coverable |= p
            if uncovered - coverable:
                return False

            best_idx = None
            best_valid: List[Set[Cell]] | None = None
            for idx in remaining:
                val = [p for p in options[idx] if not (p & used)]
                if best_valid is None or len(val) < len(best_valid):
                    best_idx = idx
                    best_valid = val
                if best_valid == []:
                    return False
            assert best_idx is not None and best_valid is not None
            rem2 = set(remaining)
            rem2.remove(best_idx)
            for p in best_valid:
                if backtrack(used | p, rem2):
                    return True
            return False

        if backtrack(set(), set(range(len(shapes)))):
            return True
    return False


def _odd_candidates(objs: Sequence[dict]) -> List[int]:
    out: List[int] = []
    n = len(objs)
    for odd in range(n):
        shapes = [objs[i]["shape"] for i in range(n) if i != odd]
        if _can_tile_rectangle(shapes):
            out.append(odd)
    return out


def _shape_to_grid(shape: Shape, color: int) -> Grid:
    h, w = _dims(shape)
    g = _empty(h, w)
    _paint(g, shape, 0, 0, color)
    return g


def verify_t33(grid: Grid) -> Grid:
    objs = _extract_objects(grid)
    if len(objs) < 3:
        return []
    cands = _odd_candidates(objs)
    if len(cands) != 1:
        return []
    odd = objs[cands[0]]
    return _shape_to_grid(odd["shape"], odd["color"])
