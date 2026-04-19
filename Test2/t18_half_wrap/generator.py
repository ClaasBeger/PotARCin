"""Generator for Test2/t18: horizontal periodic pattern wrapping between halves.

Rule: a horizontally periodic motif occupies one half of the grid, with its
axis row reaching one of the side edges (left or right) and NOT touching the
opposite edge. The motif then "wraps" into the other half on the SAME side as
the wrap edge, just shifted vertically into the other half. From there it
continues across the other half all the way to the opposite grid edge. The
periodic structure is preserved across the wrap.

Equivalently, parameterize the wave by an integer position x that increases
along its path from the input-half interior, out to the wrap edge, then jumps
across to the same edge in the other half and proceeds to the opposite edge:
    - Input half, wrap=right: col c -> x = c.            (col 0 interior, col W-1 wrap edge)
    - Other half, wrap=right: col c -> x = 2*W - 1 - c.  (col W-1 wrap edge, col 0 opposite)
    - Input half, wrap=left : col c -> x = W - 1 - c.    (col W-1 interior, col 0 wrap edge)
    - Other half, wrap=left : col c -> x = W + c.        (col 0 wrap edge, col W-1 opposite)

Cells are drawn at (axis_row + dr, c) iff (dr, x mod P) belongs to the motif M.
The same period P and motif M apply to both halves; the axis row in the other
half is at the same RELATIVE position within its half.
"""

from __future__ import annotations

import random
from typing import Iterable, List, Tuple

Grid = List[List[int]]
Pos = Tuple[int, int]
MotifCell = Tuple[int, int]  # (dr, dx)

BG = 0


def _empty(h: int, w: int) -> Grid:
    return [[BG for _ in range(w)] for _ in range(h)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _x_for_col(c: int, w: int, wrap_edge: str, half: str, *, input_half: str) -> int:
    """Snake parameter x for a given column.

    The wave is parameterised so that x increases along its physical path:
    starting at the interior corner of the input half (x = 0), reaching the
    wrap edge of the input half at x = W-1, then jumping vertically to the
    SAME side in the other half (x = W at that wrap-edge column) and
    continuing to the opposite edge of the other half (x = 2W-1).
    """
    if half == input_half:
        # Input half: x grows toward the wrap edge.
        return c if wrap_edge == "right" else (w - 1 - c)
    # Other half: enter at the wrap edge (x=W) and march toward the opposite edge.
    if wrap_edge == "right":
        return w + (w - 1 - c)
    return w + c


def _draw_motif_in_half(
    grid: Grid,
    color: int,
    motif: Iterable[MotifCell],
    period: int,
    axis_row: int,
    wrap_edge: str,
    half: str,
    *,
    input_half: str,
    col_extent: Tuple[int, int] | None = None,
) -> None:
    """Paint motif cells inside one half of the grid.

    `col_extent` lets the input half be drawn with a partial extent (so the
    wave only covers part of the half), matching the look of the manual
    examples; the other half is always drawn for every column.
    """
    h = len(grid)
    w = len(grid[0]) if h else 0
    motif_set = set(motif)
    drs = sorted({dr for dr, _ in motif_set})
    cols = range(w) if col_extent is None else range(col_extent[0], col_extent[1] + 1)
    for c in cols:
        x = _x_for_col(c, w, wrap_edge, half, input_half=input_half)
        dx = x % period
        for dr in drs:
            if (dr, dx) in motif_set:
                rr = axis_row + dr
                if _in_bounds(h, w, rr, c):
                    grid[rr][c] = color


def _row_counts(grid: Grid, color: int) -> dict[int, int]:
    counts: dict[int, int] = {}
    for r, row in enumerate(grid):
        n = sum(1 for v in row if v == color)
        if n:
            counts[r] = n
    return counts


def _detect_motif(
    grid: Grid,
    color: int,
    axis_row: int,
    wrap_edge: str,
    input_half: str,
) -> tuple[int, set[MotifCell]]:
    """Find the smallest period P and motif consistent with the input pattern.

    Consistency means: within the pattern's column bbox, every cell predicted
    by the motif (for each (dr, x mod P)) is actually painted, and no painted
    cell goes unexplained.
    """
    h = len(grid)
    w = len(grid[0]) if h else 0
    cells = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == color]
    if not cells:
        raise ValueError("no foreground cells")

    cell_set = set(cells)
    drs_present = sorted({r - axis_row for r, _ in cells})
    cols_present = [c for _, c in cells]
    col_lo = min(cols_present)
    col_hi = max(cols_present)

    pairs = [
        (r - axis_row, _x_for_col(c, w, wrap_edge, input_half, input_half=input_half))
        for r, c in cells
    ]

    for period in range(1, w + 1):
        motif = {(dr, x % period) for dr, x in pairs}

        # Check both directions of consistency.
        ok = True
        for dr in drs_present:
            allowed = {dx for d, dx in motif if d == dr}
            for c in range(col_lo, col_hi + 1):
                x = _x_for_col(c, w, wrap_edge, input_half, input_half=input_half)
                rr = axis_row + dr
                predicted = (x % period) in allowed
                actual = (rr, c) in cell_set
                if predicted != actual:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return period, motif
    raise ValueError("could not infer periodic motif")


def transform_t18(inp: Grid) -> Grid:
    if not inp or not inp[0]:
        raise ValueError("input grid must be non-empty")
    h = len(inp)
    w = len(inp[0])
    if any(len(row) != w for row in inp):
        raise ValueError("input grid must be rectangular")
    if h % 2 != 0:
        raise ValueError("input grid height must be even")

    colors = sorted({v for row in inp for v in row if v != BG})
    if len(colors) != 1:
        raise ValueError("expected exactly one foreground color")
    color = colors[0]

    counts = _row_counts(inp, color)
    if not counts:
        raise ValueError("no foreground cells")
    half_h = h // 2

    # Axis row = row with the most cells; tie-broken by closeness to a half center.
    axis_row = max(
        counts.keys(),
        key=lambda r: (counts[r], -min(abs(r - half_h // 2), abs(r - (h - half_h // 2)))),
    )

    axis_cols = sorted(c for r in counts for c in range(w) if inp[r][c] == color and r == axis_row)
    if axis_cols and axis_cols[-1] == w - 1:
        wrap_edge = "right"
    elif axis_cols and axis_cols[0] == 0:
        wrap_edge = "left"
    else:
        # Fallback: pick the side closer to the rightmost or leftmost cell.
        wrap_edge = "right" if axis_cols and (w - 1 - axis_cols[-1]) <= axis_cols[0] else "left"

    input_half = "upper" if axis_row < half_h else "lower"
    other_axis_row = axis_row + half_h if input_half == "upper" else axis_row - half_h

    period, motif = _detect_motif(inp, color, axis_row, wrap_edge, input_half)

    dr_min = min(dr for dr, _ in motif)
    dr_max = max(dr for dr, _ in motif)

    # Re-anchor the axis row so the entire pattern fits inside the input half.
    if input_half == "upper":
        lo = -dr_min
        hi = half_h - 1 - dr_max
    else:
        lo = half_h + (-dr_min)
        hi = h - 1 - dr_max
    adjusted_axis = max(lo, min(hi, axis_row))
    other_axis_row = (
        adjusted_axis + half_h if input_half == "upper" else adjusted_axis - half_h
    )

    cells_present = [(r, c) for r in range(h) for c in range(w) if inp[r][c] == color]
    cols_present = [c for _, c in cells_present]
    col_lo = min(cols_present)
    col_hi = max(cols_present)

    out = _empty(h, w)
    _draw_motif_in_half(
        out,
        color,
        motif,
        period,
        adjusted_axis,
        wrap_edge,
        input_half,
        input_half=input_half,
        col_extent=(col_lo, col_hi),
    )
    other_half = "lower" if input_half == "upper" else "upper"
    _draw_motif_in_half(
        out,
        color,
        motif,
        period,
        other_axis_row,
        wrap_edge,
        other_half,
        input_half=input_half,
    )
    return out


# ---------------------------------------------------------------------------
# Generator-side helpers: motif library and sampling.
# ---------------------------------------------------------------------------


def _wave_motif(rng: random.Random) -> tuple[int, list[MotifCell]]:
    """3-row wave, axis on alternating cols with above/below offshoots."""
    period = rng.choice([4, 4, 6])
    motif: list[MotifCell] = [(0, dx) for dx in range(period) if dx % 2 == 0]
    # Offshoots above/below at odd dx, alternating.
    odds = [dx for dx in range(period) if dx % 2 == 1]
    rng.shuffle(odds)
    if len(odds) == 1:
        motif.append((rng.choice([-1, 1]), odds[0]))
    else:
        for i, dx in enumerate(odds):
            motif.append(((-1) ** i, dx))
    return period, motif


def _diamond_motif(rng: random.Random) -> tuple[int, list[MotifCell]]:
    """5-row diamond/star with axis at every other dx."""
    period = 4
    motif: list[MotifCell] = [(0, 0), (0, 2)]
    # dx=1 gets a narrow vertical, dx=3 a wide vertical (or swapped).
    narrow = [(-1, 1), (0, 1), (1, 1)]
    wide = [(-2, 3), (0, 3), (2, 3)]
    if rng.random() < 0.5:
        narrow = [(dr, 3) for dr, _ in narrow]
        wide = [(dr, 1) for dr, _ in wide]
    motif.extend(narrow)
    motif.extend(wide)
    return period, motif


def _double_diamond_motif(rng: random.Random) -> tuple[int, list[MotifCell]]:
    """Star-like 5-row motif at period 4 (no axis cell at full-vertical dx)."""
    period = 4
    motif: list[MotifCell] = [
        (0, 0),
        (0, 2),
        (-1, 1), (1, 1),
        (-2, 3), (-1, 3), (1, 3), (2, 3),
    ]
    if rng.random() < 0.5:
        motif = [(dr, (period - dx) % period) for dr, dx in motif]
    return period, motif


def _long_arm_motif(rng: random.Random) -> tuple[int, list[MotifCell]]:
    """Period-6 motif with a 'long arm' axis spanning every column."""
    period = 6
    motif: list[MotifCell] = [(0, dx) for dx in range(period)]
    above_dx = rng.choice([1, 2])
    below_dx = (above_dx + 3) % period
    above_depth = rng.choice([1, 2])
    below_depth = rng.choice([1, 2])
    for k in range(1, above_depth + 1):
        motif.append((-k, above_dx))
    for k in range(1, below_depth + 1):
        motif.append((k, below_dx))
    return period, motif


_MOTIF_BUILDERS = [
    _wave_motif,
    _diamond_motif,
    _double_diamond_motif,
    _long_arm_motif,
]


def _pattern_height(motif: Iterable[MotifCell]) -> tuple[int, int]:
    drs = [dr for dr, _ in motif]
    return min(drs), max(drs)


def _phase_align_motif(
    motif: list[MotifCell], period: int, target_dx: int
) -> list[MotifCell]:
    """Rotate motif (shift all dx by a constant) so that an axis cell lands on
    `target_dx`. This guarantees the axis row reaches the wrap-edge column of
    the input half. If there is no axis cell, the motif is returned unchanged.
    """
    axis_dxs = [dx for dr, dx in motif if dr == 0]
    if not axis_dxs:
        return motif
    if target_dx in axis_dxs:
        return motif
    shift = (target_dx - axis_dxs[0]) % period
    return [(dr, (dx + shift) % period) for dr, dx in motif]


def generate(
    rng: random.Random | None = None,
    *,
    min_h: int = 10,
    max_h: int = 14,
    min_w: int = 10,
    max_w: int = 14,
) -> dict[str, Grid]:
    rr = rng or random.Random()

    for _ in range(2000):
        h = rr.randrange(min_h, max_h + 1, 2)  # ensure even height
        w = rr.randint(min_w, max_w)
        color = rr.choice([1, 2, 3, 4, 6, 7, 8, 9])
        wrap_edge = rr.choice(["left", "right"])
        input_half = rr.choice(["upper", "lower"])

        builder = rr.choice(_MOTIF_BUILDERS)
        period, motif = builder(rr)
        # Rotate motif so that the wrap-edge column carries an axis cell.
        # Under the snake parameterisation, the wrap-edge column always has
        # x = W - 1 regardless of which side it is on, so the target dx is the
        # same for both wrap directions.
        target_dx = (w - 1) % period
        motif = _phase_align_motif(motif, period, target_dx)

        dr_min, dr_max = _pattern_height(motif)
        pattern_h = dr_max - dr_min + 1
        half_h = h // 2
        if pattern_h > half_h:
            continue
        # Choose axis position so the pattern fits inside the input half.
        rel_axis_min = -dr_min
        rel_axis_max = half_h - 1 - dr_max
        if rel_axis_min > rel_axis_max:
            continue
        rel_axis = rr.randint(rel_axis_min, rel_axis_max)
        axis_row = rel_axis if input_half == "upper" else rel_axis + half_h

        # Decide a partial column extent for the input half, anchored at wrap edge.
        # The pattern must touch the wrap edge but NOT the opposite edge, so we
        # always leave at least one empty column on the far side.
        max_extent = max(period + 2, w - 1)
        max_extent = min(max_extent, w - 1)
        min_extent = max(period + 2, w - period * 2)
        min_extent = min(min_extent, max_extent)
        extent = rr.randint(min_extent, max_extent)
        if wrap_edge == "right":
            col_lo = w - extent
            col_hi = w - 1
        else:
            col_lo = 0
            col_hi = extent - 1

        inp = _empty(h, w)
        _draw_motif_in_half(
            inp,
            color,
            motif,
            period,
            axis_row,
            wrap_edge,
            input_half,
            input_half=input_half,
            col_extent=(col_lo, col_hi),
        )
        # Sanity 1: axis row must touch the wrap edge inside the input half.
        axis_in_row = [c for c in range(w) if inp[axis_row][c] == color]
        if not axis_in_row:
            continue
        if wrap_edge == "right" and axis_in_row[-1] != w - 1:
            continue
        if wrap_edge == "left" and axis_in_row[0] != 0:
            continue
        # Sanity 2: NO row of the input pattern may touch the opposite edge.
        any_cells = [(r, c) for r in range(h) for c in range(w) if inp[r][c] == color]
        if not any_cells:
            continue
        all_cols = {c for _, c in any_cells}
        if wrap_edge == "right" and 0 in all_cols:
            continue
        if wrap_edge == "left" and (w - 1) in all_cols:
            continue

        try:
            out = transform_t18(inp)
        except ValueError:
            continue
        if out == inp:
            continue
        # Verify the snake produced extra cells in the other half.
        new_cells = sum(
            1
            for r in range(h)
            for c in range(w)
            if inp[r][c] == BG and out[r][c] == color
        )
        if new_cells < pattern_h:
            continue
        return {"input": inp, "output": out}

    raise RuntimeError("failed to sample valid t18 instance")
