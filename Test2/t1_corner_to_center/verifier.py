"""Verifier for the Test2/t1 corner-to-center rule."""

from __future__ import annotations

from typing import Dict, List, Tuple

Grid = List[List[int]]


def _empty(size: int) -> Grid:
    return [[0 for _ in range(size)] for _ in range(size)]


def _corner_span(size: int) -> int:
    return size // 2 - 2


def _origins(size: int) -> Dict[str, Tuple[int, int]]:
    span = _corner_span(size)
    return {
        "tl": (0, 0),
        "tr": (0, size - span),
        "bl": (size - span, 0),
        "br": (size - span, size - span),
    }


def verify_t1(grid: Grid) -> Grid:
    """Moves each corner component to its corresponding center window."""
    h = len(grid)
    w = len(grid[0]) if h else 0
    if h == 0 or w == 0 or h != w:
        return []
    size = h
    if size < 8 or size % 2 != 0:
        return _empty(size)

    span = _corner_span(size)
    out = _empty(size)
    src_o = _origins(size)
    half = size // 2

    by_corner: dict[str, list[tuple[int, int, int]]] = {c: [] for c in ("tl", "tr", "bl", "br")}
    for corner in ("tl", "tr", "bl", "br"):
        sr, sc = src_o[corner]
        for rr in range(span):
            for cc in range(span):
                val = grid[sr + rr][sc + cc]
                if val:
                    by_corner[corner].append((sr + rr, sc + cc, val))

    for corner, cells in by_corner.items():
        if not cells:
            continue
        rows = [r for r, _, _ in cells]
        cols = [c for _, c, _ in cells]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        if corner == "tl":
            dr, dc = (half - 1) - max_r, (half - 1) - max_c
        elif corner == "tr":
            dr, dc = (half - 1) - max_r, half - min_c
        elif corner == "bl":
            dr, dc = half - min_r, (half - 1) - max_c
        else:  # br
            dr, dc = half - min_r, half - min_c
        for r, c, val in cells:
            out[r + dr][c + dc] = val
    return out
