"""Generator for Test2/t5 cyan-flow task."""

from __future__ import annotations

import random
from typing import List

Grid = List[List[int]]

CYAN = 8
ORANGE = 7


def _empty(h: int, w: int) -> Grid:
    return [[0 for _ in range(w)] for _ in range(h)]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _non_orange_components(grid: Grid) -> list[set[tuple[int, int]]]:
    h = len(grid)
    w = len(grid[0]) if h else 0
    seen: set[tuple[int, int]] = set()
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == ORANGE or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            comp: set[tuple[int, int]] = set()
            while stack:
                cr, cc = stack.pop()
                comp.add((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen:
                        if grid[nr][nc] != ORANGE:
                            seen.add((nr, nc))
                            stack.append((nr, nc))
            comps.append(comp)
    return comps


def _center_order(cols: list[int]) -> list[int]:
    center = (cols[0] + cols[-1]) / 2.0
    return sorted(cols, key=lambda c: (abs(c - center), c))


def _pack_component(comp: set[tuple[int, int]], count: int) -> set[tuple[int, int]]:
    if count <= 0:
        return set()
    rows = sorted({r for r, _ in comp}, reverse=True)
    row_cells = {r: sorted([c for rr, c in comp if rr == r]) for r in rows}
    bottom = rows[0]

    filled: set[tuple[int, int]] = set()
    remaining = count

    # Bottom-up fill; each row fills from the middle, but is anchored to filled row below.
    for r in rows:
        if remaining <= 0:
            break
        cols = row_cells[r]
        if not cols:
            continue
        i = 0
        while i < len(cols):
            j = i
            while j + 1 < len(cols) and cols[j + 1] == cols[j] + 1:
                j += 1
            seg = cols[i : j + 1]
            i = j + 1
            if remaining <= 0:
                break

            if r == bottom:
                order = _center_order(seg)
            else:
                anchors = [c for c in seg if (r + 1, c) in filled]
                if not anchors:
                    continue
                center = (seg[0] + seg[-1]) / 2.0
                order = sorted(
                    seg,
                    key=lambda c: (min(abs(c - a) for a in anchors), abs(c - center), c),
                )

            take = min(remaining, len(order))
            for c in order[:take]:
                filled.add((r, c))
            remaining -= take

    # Rare fallback for irregular cavities: continue from frontier.
    while remaining > 0:
        candidates: list[tuple[int, int, int]] = []
        for r, c in comp:
            if (r, c) in filled:
                continue
            support = (r == bottom) or ((r + 1, c) in filled) or ((r, c - 1) in filled) or ((r, c + 1) in filled)
            if not support:
                continue
            center = 0.0
            row = row_cells[r]
            if row:
                center = (row[0] + row[-1]) / 2.0
            candidates.append((r, int(abs(c - center) * 1000), c))
        if not candidates:
            break
        # Highest row index (lower in grid) first.
        candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
        r, _, c = candidates[0]
        filled.add((r, c))
        remaining -= 1

    return filled


def transform_t5(inp: Grid) -> Grid:
    h = len(inp)
    w = len(inp[0]) if h else 0
    out = _empty(h, w)
    for r in range(h):
        for c in range(w):
            if inp[r][c] == ORANGE:
                out[r][c] = ORANGE

    for comp in _non_orange_components(inp):
        n_cyan = sum(1 for r, c in comp if inp[r][c] == CYAN)
        for r, c in _pack_component(comp, n_cyan):
            out[r][c] = CYAN
    return out


def _add_random_boundaries(
    grid: Grid, rng: random.Random
) -> tuple[int, set[int], set[tuple[int, int]]]:
    h = len(grid)
    w = len(grid[0]) if h else 0
    barrier_r = rng.randint(max(1, h // 4), min(h - 4, h // 2))
    for c in range(w):
        grid[barrier_r][c] = ORANGE
    gap_cols: set[int] = set()
    # Keep openings center-symmetric to avoid ambiguous left-vs-right cases.
    if w % 2 == 0:
        gap_cols = {w // 2 - 1, w // 2}
    else:
        gap_cols = {w // 2}
    for c in gap_cols:
        grid[barrier_r][c] = 0

    reserved_no_cyan: set[tuple[int, int]] = {(barrier_r, c) for c in gap_cols}
    for c in gap_cols:
        for rr, cc in (
            (barrier_r - 1, c - 1),
            (barrier_r - 1, c + 1),
            (barrier_r + 1, c - 1),
            (barrier_r + 1, c + 1),
        ):
            if 0 <= rr < h and 0 <= cc < w:
                reserved_no_cyan.add((rr, cc))

    # Add a second horizontal barrier sometimes (still center-symmetric opening).
    if rng.random() < 0.65 and barrier_r + 2 < h - 1:
        r2 = rng.randint(barrier_r + 2, h - 2)
        for c in range(w):
            grid[r2][c] = ORANGE
        if w % 2 == 0:
            gaps2 = {w // 2 - 1, w // 2}
        else:
            gaps2 = {w // 2}
        for c in gaps2:
            grid[r2][c] = 0
            reserved_no_cyan.add((r2, c))
            for rr, cc in (
                (r2 - 1, c - 1),
                (r2 - 1, c + 1),
                (r2 + 1, c - 1),
                (r2 + 1, c + 1),
            ):
                if 0 <= rr < h and 0 <= cc < w:
                    reserved_no_cyan.add((rr, cc))

    # Add one or two vertical walls for richer basin geometry.
    wall_count = rng.randint(0, 2)
    for _ in range(wall_count):
        candidates = [c for c in range(1, w - 1) if all(abs(c - gc) >= 2 for gc in gap_cols)]
        if not candidates:
            break
        c = rng.choice(candidates)
        start_r = rng.randint(barrier_r + 1, max(barrier_r + 1, h - 2))
        for r in range(start_r, h):
            grid[r][c] = ORANGE
        # Optional small opening near bottom to keep compartments varied.
        if h - start_r >= 4 and rng.random() < 0.45:
            hole_r = rng.randint(start_r + 1, h - 2)
            grid[hole_r][c] = 0
            reserved_no_cyan.add((hole_r, c))
            for rr, cc in (
                (hole_r - 1, c - 1),
                (hole_r - 1, c + 1),
                (hole_r + 1, c - 1),
                (hole_r + 1, c + 1),
            ):
                if 0 <= rr < h and 0 <= cc < w:
                    reserved_no_cyan.add((rr, cc))
    return barrier_r, gap_cols, reserved_no_cyan


def _row_segments(cells: set[tuple[int, int]]) -> dict[int, list[list[int]]]:
    by_row: dict[int, list[int]] = {}
    for r, c in cells:
        by_row.setdefault(r, []).append(c)
    out: dict[int, list[list[int]]] = {}
    for r, cols in by_row.items():
        cols = sorted(cols)
        segs: list[list[int]] = []
        start = [cols[0]]
        for c in cols[1:]:
            if c == start[-1] + 1:
                start.append(c)
            else:
                segs.append(start)
                start = [c]
        segs.append(start)
        out[r] = segs
    return out


def _is_ambiguous_fill(inp: Grid, out: Grid) -> bool:
    """True if fill needs unresolved side tie-breaks or violates center-on-water behavior."""
    h = len(inp)
    w = len(inp[0]) if h else 0

    # Reject patterns where cyan appears wedged by orange support that would be ambiguous.
    for r in range(h):
        for c in range(w):
            if inp[r][c] != CYAN:
                continue
            below_orange = (r + 1 >= h) or (inp[r + 1][c] == ORANGE)
            left_orange = (c - 1 < 0) or (inp[r][c - 1] == ORANGE)
            right_orange = (c + 1 >= w) or (inp[r][c + 1] == ORANGE)
            if below_orange and (left_orange or right_orange):
                # Keep such cells fixed; if they disappear, dynamics are unclear in examples.
                if out[r][c] != CYAN:
                    return True

    for comp in _non_orange_components(inp):
        segs_by_row = _row_segments(comp)
        bottom = max(r for r, _ in comp)
        for r, segs in segs_by_row.items():
            for seg in segs:
                chosen = [c for c in seg if out[r][c] == CYAN]
                if not chosen or len(chosen) == len(seg):
                    continue
                if r == bottom:
                    center = (seg[0] + seg[-1]) / 2.0
                else:
                    supports = [c for c in seg if r + 1 < h and out[r + 1][c] == CYAN]
                    center = (
                        (sum(supports) / len(supports))
                        if supports
                        else (seg[0] + seg[-1]) / 2.0
                    )
                chosen = sorted(chosen)
                # Partial rows must be one contiguous interval.
                if any(chosen[i + 1] != chosen[i] + 1 for i in range(len(chosen) - 1)):
                    return True
                left = chosen[0]
                right = chosen[-1]
                # Strict symmetric centering around the selected center.
                if abs((center - left) - (right - center)) > 1e-9:
                    return True
    return False


def generate(
    width: int | None = None,
    height: int | None = None,
    rng: random.Random | None = None,
) -> dict[str, Grid]:
    rr = rng if rng is not None else random
    if width is None:
        width = rr.randint(8, 12)
    if height is None:
        height = rr.randint(8, 12)
    if width < 8 or height < 8:
        raise ValueError("width and height must be >= 8")

    for _ in range(1000):
        inp = _empty(height, width)
        barrier_r, gap_cols, reserved_no_cyan = _add_random_boundaries(inp, rr)

        comps = _non_orange_components(inp)
        for comp in comps:
            if not comp:
                continue
            frac = rr.uniform(0.2, 0.6)
            cells = [
                (r, c)
                for (r, c) in comp
                if (r, c) not in reserved_no_cyan and not (r == barrier_r and c in gap_cols)
            ]
            if not cells:
                continue
            n = max(1, min(len(cells), int(round(len(cells) * frac))))
            rr.shuffle(cells)
            for r, c in cells[:n]:
                inp[r][c] = CYAN

        out = transform_t5(inp)
        if not _is_ambiguous_fill(inp, out):
            return {"input": inp, "output": out}

    raise RuntimeError("failed to sample unambiguous t5 instance")
