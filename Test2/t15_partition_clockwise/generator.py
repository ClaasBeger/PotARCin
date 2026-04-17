"""Generator for Test2/t15: move independent cell one partition clockwise."""

from __future__ import annotations

import random
from collections import Counter
from typing import List, Sequence, Tuple

Grid = List[List[int]]
Pos = Tuple[int, int]


def _empty(h: int, w: int) -> Grid:
    return [[0 for _ in range(w)] for _ in range(h)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _rot_cw(r: int, c: int, cr: int, cc: int) -> Pos:
    dr, dc = r - cr, c - cc
    return cr + dc, cc - dr


def _best_center(inp: Grid) -> Pos:
    h = len(inp)
    w = len(inp[0]) if h else 0
    nz = [(r, c) for r in range(h) for c in range(w) if inp[r][c] != 0]
    if not nz:
        raise ValueError("input has no non-zero cells")

    best: tuple[int, int, int, int, int] | None = None
    for cr in range(h):
        for cc in range(w):
            good = 0
            miss = 0
            out = 0
            for r, c in nz:
                rr, rc = _rot_cw(r, c, cr, cc)
                if not _in_bounds(h, w, rr, rc):
                    out += 1
                elif inp[rr][rc] != 0:
                    good += 1
                else:
                    miss += 1
            cand = (good, -miss, -out, -abs(cr - h // 2), -abs(cc - w // 2))
            if best is None or cand > best:
                best = cand
                best_center = (cr, cc)
    return best_center


def _pick_mover(inp: Grid, center: Pos) -> tuple[Pos, Pos]:
    h = len(inp)
    w = len(inp[0]) if h else 0
    cr, cc = center
    freq = Counter(v for row in inp for v in row if v != 0)
    candidates: list[tuple[int, int, int, int]] = []
    for r in range(h):
        for c in range(w):
            v = inp[r][c]
            if v == 0:
                continue
            rr, rc = _rot_cw(r, c, cr, cc)
            if not _in_bounds(h, w, rr, rc):
                continue
            if inp[rr][rc] != 0:
                continue
            # Prefer likely "independent" colors first (rarer color count).
            candidates.append((freq[v], abs(r - cr) + abs(c - cc), r, c))
    if not candidates:
        raise ValueError("could not identify independent movable cell")
    candidates.sort()
    _f, _d, r, c = candidates[0]
    rr, rc = _rot_cw(r, c, cr, cc)
    return (r, c), (rr, rc)


def transform_t15(inp: Grid) -> Grid:
    if not inp or not inp[0]:
        raise ValueError("input must be non-empty")
    w = len(inp[0])
    if any(len(row) != w for row in inp):
        raise ValueError("input must be rectangular")

    center = _best_center(inp)
    src, dst = _pick_mover(inp, center)
    out = [row[:] for row in inp]
    sr, sc = src
    dr, dc = dst
    out[dr][dc] = out[sr][sc]
    out[sr][sc] = 0
    return out


def _orbit4(cr: int, cc: int, r: int, c: int) -> list[Pos]:
    pts = []
    rr, rc = r, c
    for _ in range(4):
        pts.append((rr, rc))
        rr, rc = _rot_cw(rr, rc, cr, cc)
    # dedupe while preserving order
    seen: set[Pos] = set()
    out: list[Pos] = []
    for p in pts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _orbit_fully_in_bounds(h: int, w: int, cr: int, cc: int, r: int, c: int) -> bool:
    for rr, rc in _orbit4(cr, cc, r, c):
        if not _in_bounds(h, w, rr, rc):
            return False
    return True


def _is_occupancy_cw_closed(shape: set[Pos], cr: int, cc: int) -> bool:
    for r, c in shape:
        if _rot_cw(r, c, cr, cc) not in shape:
            return False
    return True


def _is_connected_8(cells: set[Pos]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    stack = [start]
    seen: set[Pos] = {start}
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in cells and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
    return len(seen) == len(cells)


def _add_plus_skeleton(cells: set[Pos], h: int, w: int, cr: int, cc: int, arm: int) -> None:
    for k in range(arm + 1):
        for p in ((cr, cc + k), (cr, cc - k), (cr + k, cc), (cr - k, cc)):
            cells.add(p)


def _add_x_skeleton(cells: set[Pos], h: int, w: int, cr: int, cc: int, arm: int) -> None:
    for k in range(arm + 1):
        for p in ((cr + k, cc + k), (cr + k, cc - k), (cr - k, cc + k), (cr - k, cc - k)):
            cells.add(p)


def _adjacent_to_shape(r: int, c: int, shape: set[Pos]) -> bool:
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if (r + dr, c + dc) in shape:
            return True
    return False


def _mover_candidates(h: int, w: int, shape: set[Pos], cr: int, cc: int) -> list[tuple[Pos, Pos]]:
    out: list[tuple[Pos, Pos]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in shape:
                continue
            rr, rc = _rot_cw(r, c, cr, cc)
            if not _in_bounds(h, w, rr, rc):
                continue
            if (rr, rc) in shape:
                continue
            if (rr, rc) == (r, c):
                continue
            out.append(((r, c), (rr, rc)))
    return out


def generate(
    rng: random.Random | None = None,
    *,
    min_h: int = 9,
    max_h: int = 13,
    min_w: int = 9,
    max_w: int = 13,
) -> dict[str, Grid]:
    rr = rng or random.Random()

    for _ in range(1200):
        h = rr.randint(min_h, max_h)
        w = rr.randint(min_w, max_w)
        inp = _empty(h, w)

        cr = rr.randint(2, h - 3)
        cc = rr.randint(2, w - 3)
        max_arm = min(cr, h - 1 - cr, cc, w - 1 - cc)
        if max_arm < 2:
            continue

        shape: set[Pos] = set()
        arm = rr.randint(2, max_arm)
        if rr.random() < 0.5:
            _add_plus_skeleton(shape, h, w, cr, cc, arm)
        else:
            _add_x_skeleton(shape, h, w, cr, cc, arm)

        # Add extra 4-orbit points to get varied 4-part structures.
        extra = rr.randint(1, 4)
        for _ in range(extra):
            dr = rr.randint(-max_arm, max_arm)
            dc = rr.randint(-max_arm, max_arm)
            if dr == 0 and dc == 0:
                continue
            r0, c0 = cr + dr, cc + dc
            if not _orbit_fully_in_bounds(h, w, cr, cc, r0, c0):
                continue
            orbit = set(_orbit4(cr, cc, r0, c0))
            candidate = shape | orbit
            # Keep the main structure as one 8-connected component.
            if _is_connected_8(candidate):
                shape = candidate

        if len(shape) < 8 or not _is_occupancy_cw_closed(shape, cr, cc) or not _is_connected_8(shape):
            continue

        base_color = rr.randint(1, 9)
        for r, c in shape:
            inp[r][c] = base_color

        # Miscolorings on the shape can break visual symmetry.
        mis_n = rr.randint(1, min(4, len(shape) // 4))
        for r, c in rr.sample(list(shape), mis_n):
            inp[r][c] = rr.randint(1, 9)

        pairs = _mover_candidates(h, w, shape, cr, cc)
        if not pairs:
            continue
        # Prefer adjacency-allowed placements.
        pairs.sort(
            key=lambda p: (
                0 if _adjacent_to_shape(p[0][0], p[0][1], shape) else 1,
                abs(p[0][0] - cr) + abs(p[0][1] - cc),
            )
        )
        (sr, sc), (tr, tc) = rr.choice(pairs[: max(1, min(12, len(pairs)))])
        mover_color = rr.randint(1, 9)
        inp[sr][sc] = mover_color

        out = transform_t15(inp)
        # Ensure output matches intended destination.
        if out[tr][tc] != mover_color or out[sr][sc] != 0:
            continue
        return {"input": inp, "output": out}

    raise RuntimeError("failed to sample valid t15 instance")
