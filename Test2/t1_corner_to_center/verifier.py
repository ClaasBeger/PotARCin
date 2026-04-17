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

    top_cells = by_corner["tl"] + by_corner["tr"]
    bot_cells = by_corner["bl"] + by_corner["br"]
    left_cells = by_corner["tl"] + by_corner["bl"]
    right_cells = by_corner["tr"] + by_corner["br"]

    top_max_r = max((r for r, _, _ in top_cells), default=half - 1)
    bot_min_r = min((r for r, _, _ in bot_cells), default=half)
    left_max_c = max((c for _, c, _ in left_cells), default=half - 1)
    right_min_c = min((c for _, c, _ in right_cells), default=half)

    dr_top = (half - 1) - top_max_r
    dr_bot = half - bot_min_r
    dc_left = (half - 1) - left_max_c
    dc_right = half - right_min_c

    for corner, cells in by_corner.items():
        if not cells:
            continue
        dr = dr_top if corner in ("tl", "tr") else dr_bot
        dc = dc_left if corner in ("tl", "bl") else dc_right
        for r, c, val in cells:
            out[r + dr][c + dc] = val
    return out
