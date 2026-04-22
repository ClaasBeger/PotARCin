"""Generator for Test2/t22: center anchor connects to same-color cells.

Rule:
- For each non-background color, choose its most central cell (nearest to grid
  center by squared Euclidean distance) as anchor.
- Process colors in order of anchor row, top to bottom.
- For each target cell of that color, draw an L path from anchor:
  vertical segment first (same column as anchor), then horizontal segment.
- Current color overwrites existing cells, except cells that belong to colors
  that will be processed later (those are protected until their turn).
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

Grid = List[List[int]]
Pos = Tuple[int, int]

BG = 0


def _empty(h: int, w: int) -> Grid:
    return [[BG for _ in range(w)] for _ in range(h)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _center_distance2(r: int, c: int, h: int, w: int) -> float:
    cr = (h - 1) / 2.0
    cc = (w - 1) / 2.0
    dr = r - cr
    dc = c - cc
    return dr * dr + dc * dc


def _color_cells(inp: Grid) -> Dict[int, List[Pos]]:
    h = len(inp)
    w = len(inp[0]) if h else 0
    out: Dict[int, List[Pos]] = {}
    for r in range(h):
        for c in range(w):
            v = inp[r][c]
            if v == BG:
                continue
            out.setdefault(v, []).append((r, c))
    return out


def _unique_anchor_for_color(cells: List[Pos], h: int, w: int) -> Pos:
    ranked = sorted(cells, key=lambda p: (_center_distance2(p[0], p[1], h, w), p[0], p[1]))
    if len(ranked) >= 2:
        d0 = _center_distance2(ranked[0][0], ranked[0][1], h, w)
        d1 = _center_distance2(ranked[1][0], ranked[1][1], h, w)
        if abs(d0 - d1) < 1e-9:
            raise ValueError("ambiguous central cell for color")
    return ranked[0]


def _draw_path(
    out: Grid,
    color: int,
    anchor: Pos,
    target: Pos,
    protected: set[Pos],
) -> None:
    ar, ac = anchor
    tr, tc = target

    # Vertical first along anchor column.
    step = 1 if tr >= ar else -1
    for r in range(ar, tr + step, step):
        if (r, ac) in protected:
            continue
        out[r][ac] = color

    # Then horizontal along target row.
    step = 1 if tc >= ac else -1
    for c in range(ac, tc + step, step):
        if (tr, c) in protected:
            continue
        out[tr][c] = color


def transform_t22(inp: Grid) -> Grid:
    if not inp or not inp[0]:
        raise ValueError("input grid must be non-empty")
    h = len(inp)
    w = len(inp[0])
    if any(len(row) != w for row in inp):
        raise ValueError("input grid must be rectangular")

    by_color = _color_cells(inp)
    if len(by_color) < 2:
        raise ValueError("need at least two colors")

    anchor_of: Dict[int, Pos] = {}
    for col, cells in by_color.items():
        anchor_of[col] = _unique_anchor_for_color(cells, h, w)

    order = sorted(by_color.keys(), key=lambda col: (anchor_of[col][0], anchor_of[col][1], col))
    # Avoid ambiguous same-row processing intent.
    rows = [anchor_of[c][0] for c in order]
    if len(set(rows)) != len(rows):
        raise ValueError("ambiguous color order (anchor-row tie)")

    out = [row[:] for row in inp]
    for i, col in enumerate(order):
        later_colors = order[i + 1 :]
        protected = {p for lc in later_colors for p in by_color[lc]}
        anchor = anchor_of[col]
        # Stable order of targets; anchor cell included naturally.
        for tgt in sorted(by_color[col]):
            _draw_path(out, col, anchor, tgt, protected)
    return out


def _choose_anchor(rr: random.Random, h: int, w: int, used_rows: set[int], occupied: set[Pos]) -> Pos:
    candidates = []
    for r in range(1, h - 1):
        if r in used_rows:
            continue
        for c in range(1, w - 1):
            if (r, c) in occupied:
                continue
            candidates.append((r, c))
    if not candidates:
        raise ValueError("no anchor candidates")
    # Prefer spread over rows.
    rr.shuffle(candidates)
    return candidates[0]


def _place_points_for_color(
    rr: random.Random,
    h: int,
    w: int,
    anchor: Pos,
    n_points: int,
    occupied: set[Pos],
) -> list[Pos]:
    ar, ac = anchor
    ad = _center_distance2(ar, ac, h, w)
    points = [anchor]
    occupied.add(anchor)

    candidates: list[Pos] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in occupied:
                continue
            # Keep anchor uniquely central by only adding farther cells.
            if _center_distance2(r, c, h, w) <= ad + 0.2:
                continue
            candidates.append((r, c))
    rr.shuffle(candidates)
    for p in candidates:
        if len(points) >= n_points:
            break
        points.append(p)
        occupied.add(p)
    if len(points) < n_points:
        raise ValueError("could not place enough points")
    return points


def generate(
    rng: random.Random | None = None,
    *,
    min_h: int = 10,
    max_h: int = 14,
    min_w: int = 10,
    max_w: int = 14,
) -> dict[str, Grid]:
    rr = rng or random.Random()
    palette = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    for _ in range(2500):
        h = rr.randint(min_h, max_h)
        w = rr.randint(min_w, max_w)
        inp = _empty(h, w)

        n_colors = rr.randint(3, 5)
        cols = rr.sample(palette, n_colors)
        occupied: set[Pos] = set()
        used_rows: set[int] = set()

        color_points: Dict[int, list[Pos]] = {}
        ok = True
        # Build anchors first so row-order is unambiguous.
        anchors: Dict[int, Pos] = {}
        for col in cols:
            try:
                a = _choose_anchor(rr, h, w, used_rows, occupied)
            except ValueError:
                ok = False
                break
            anchors[col] = a
            used_rows.add(a[0])
        if not ok:
            continue

        # Add extra points per color.
        for col in cols:
            n_points = rr.randint(4, 7)
            try:
                pts = _place_points_for_color(rr, h, w, anchors[col], n_points, occupied)
            except ValueError:
                ok = False
                break
            color_points[col] = pts
        if not ok:
            continue

        for col, pts in color_points.items():
            for r, c in pts:
                inp[r][c] = col

        try:
            out = transform_t22(inp)
        except ValueError:
            continue

        if out == inp:
            continue

        # Ensure substantial path growth.
        in_count = sum(1 for r in range(h) for c in range(w) if inp[r][c] != BG)
        out_count = sum(1 for r in range(h) for c in range(w) if out[r][c] != BG)
        if out_count - in_count < 10:
            continue

        return {"input": inp, "output": out}

    raise RuntimeError("failed to sample valid t22 instance")
