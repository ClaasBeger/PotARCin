"""Generator for Test2/t19: split-by-dots line and side-wise box growth/fill.

Rule:
- Exactly two gray dots (color 5) lie on two different grid edges.
- The (invisible) line through these dots splits the grid into two halves.
- Each half is processed independently:
  * If the half contains multiple shapes (8-connected components), fill the
    bounding box of all shapes in that half with gray.
  * If the half contains exactly one shape, grow it to an x*x square in its
    own color, where x = max(bbox_height, bbox_width) of that shape.
- Nothing painted by the transformation may lie on the divider line.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Iterable, List, Sequence, Tuple

Grid = List[List[int]]
Pos = Tuple[int, int]

BG = 0
GRAY = 5
NEI8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _empty(h: int, w: int) -> Grid:
    return [[BG for _ in range(w)] for _ in range(h)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _is_edge(r: int, c: int, h: int, w: int) -> bool:
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _edge_ids(r: int, c: int, h: int, w: int) -> set[str]:
    ids: set[str] = set()
    if r == 0:
        ids.add("top")
    if r == h - 1:
        ids.add("bottom")
    if c == 0:
        ids.add("left")
    if c == w - 1:
        ids.add("right")
    return ids


def _cross(a: Pos, b: Pos, p: Pos) -> int:
    """2D cross product sign for directed line a->b vs point p."""
    ar, ac = a
    br, bc = b
    pr, pc = p
    return (br - ar) * (pc - ac) - (bc - ac) * (pr - ar)


def _find_anchor_dots(grid: Grid) -> tuple[Pos, Pos]:
    h = len(grid)
    w = len(grid[0]) if h else 0
    dots = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == GRAY]
    if len(dots) != 2:
        raise ValueError("expected exactly two gray anchor dots")
    a, b = dots
    if a == b:
        raise ValueError("anchor dots must differ")
    if not _is_edge(a[0], a[1], h, w) or not _is_edge(b[0], b[1], h, w):
        raise ValueError("gray anchor dots must be on edges")
    if _edge_ids(a[0], a[1], h, w) & _edge_ids(b[0], b[1], h, w):
        raise ValueError("gray anchor dots must be on two different edges")
    return a, b


def _shape_components(cells: set[Pos]) -> list[set[Pos]]:
    rem = set(cells)
    comps: list[set[Pos]] = []
    while rem:
        s = next(iter(rem))
        q = deque([s])
        rem.remove(s)
        comp = {s}
        while q:
            r, c = q.popleft()
            for dr, dc in NEI8:
                rr, cc = r + dr, c + dc
                if (rr, cc) in rem:
                    rem.remove((rr, cc))
                    comp.add((rr, cc))
                    q.append((rr, cc))
        comps.append(comp)
    return comps


def _bbox(cells: Iterable[Pos]) -> tuple[int, int, int, int]:
    xs = list(cells)
    rs = [r for r, _ in xs]
    cs = [c for _, c in xs]
    return min(rs), max(rs), min(cs), max(cs)


def _paint_rect(out: Grid, r0: int, r1: int, c0: int, c1: int, color: int) -> None:
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            out[r][c] = color


def transform_t19(inp: Grid) -> Grid:
    if not inp or not inp[0]:
        raise ValueError("input grid must be non-empty")
    h = len(inp)
    w = len(inp[0])
    if any(len(row) != w for row in inp):
        raise ValueError("input grid must be rectangular")

    a, b = _find_anchor_dots(inp)

    # Partition all non-background, non-gray cells by side of divider line.
    side_cells: dict[int, set[Pos]] = {-1: set(), 1: set()}
    for r in range(h):
        for c in range(w):
            v = inp[r][c]
            if v in (BG, GRAY):
                continue
            s = _cross(a, b, (r, c))
            if s == 0:
                raise ValueError("shape cell on divider line")
            side_cells[-1 if s < 0 else 1].add((r, c))

    out = _empty(h, w)
    out[a[0]][a[1]] = GRAY
    out[b[0]][b[1]] = GRAY

    for side in (-1, 1):
        comps = _shape_components(side_cells[side])
        if not comps:
            continue
        if len(comps) >= 2:
            # Multiple shapes: fill aggregate bbox with gray.
            all_cells = set().union(*comps)
            r0, r1, c0, c1 = _bbox(all_cells)
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    s = _cross(a, b, (rr, cc))
                    if s == 0:
                        raise ValueError("multi-shape fill overlaps divider line")
                    if (-1 if s < 0 else 1) != side:
                        raise ValueError("multi-shape fill crosses divider")
            _paint_rect(out, r0, r1, c0, c1, GRAY)
        else:
            # Single shape: grow to x*x square (x = max bbox side).
            # To avoid ambiguity, select the unique valid square that contains
            # the shape and is as close as possible to the shape's top-left
            # bbox corner (i.e., minimal up/left shift).
            comp = comps[0]
            colors = {inp[r][c] for r, c in comp}
            if len(colors) != 1:
                raise ValueError("single shape must be monochrome")
            color = next(iter(colors))
            minr, maxr, minc, maxc = _bbox(comp)
            x = max(maxr - minr + 1, maxc - minc + 1)

            r0_lo = max(0, maxr - x + 1)
            r0_hi = min(minr, h - x)
            c0_lo = max(0, maxc - x + 1)
            c0_hi = min(minc, w - x)
            if r0_lo > r0_hi or c0_lo > c0_hi:
                raise ValueError("single-shape square has no in-bounds placement")

            candidates: list[tuple[int, int]] = []
            for r0 in range(r0_lo, r0_hi + 1):
                for c0 in range(c0_lo, c0_hi + 1):
                    rr1 = r0 + x - 1
                    cc1 = c0 + x - 1
                    ok = True
                    for rr in range(r0, rr1 + 1):
                        for cc in range(c0, cc1 + 1):
                            s = _cross(a, b, (rr, cc))
                            if s == 0 or (-1 if s < 0 else 1) != side:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        candidates.append((r0, c0))
            if not candidates:
                raise ValueError("single-shape square overlaps/crosses divider")

            # Prefer maximal r0, then maximal c0: closest to (minr, minc) from
            # below/left constraints while keeping containment.
            r0, c0 = max(candidates, key=lambda rc: (rc[0], rc[1]))
            _paint_rect(out, r0, r0 + x - 1, c0, c0 + x - 1, color)

    return out


def _all_edge_positions(h: int, w: int, edge: str) -> list[Pos]:
    if edge == "top":
        return [(0, c) for c in range(w)]
    if edge == "bottom":
        return [(h - 1, c) for c in range(w)]
    if edge == "left":
        return [(r, 0) for r in range(h)]
    if edge == "right":
        return [(r, w - 1) for r in range(h)]
    raise ValueError("unknown edge")


def _is_direct_connect_pair(a: Pos, b: Pos) -> bool:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return dr == 0 or dc == 0 or dr == dc


def _all_valid_anchor_pairs(h: int, w: int) -> list[tuple[Pos, Pos]]:
    edges = ["top", "bottom", "left", "right"]
    points_by_edge = {e: _all_edge_positions(h, w, e) for e in edges}
    pairs: list[tuple[Pos, Pos]] = []
    for e1 in edges:
        for e2 in edges:
            if e2 == e1:
                continue
            for p1 in points_by_edge[e1]:
                for p2 in points_by_edge[e2]:
                    if p1 == p2:
                        continue
                    if _edge_ids(p1[0], p1[1], h, w) & _edge_ids(p2[0], p2[1], h, w):
                        continue
                    # Avoid very short separators.
                    if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) < max(4, min(h, w) // 2):
                        continue
                    pairs.append((p1, p2))
    # De-duplicate reversed pairs.
    uniq: list[tuple[Pos, Pos]] = []
    seen: set[tuple[Pos, Pos]] = set()
    for a, b in pairs:
        k = (a, b) if a <= b else (b, a)
        if k in seen:
            continue
        seen.add(k)
        uniq.append((a, b))
    return uniq


def _pick_anchors(rr: random.Random, h: int, w: int) -> tuple[Pos, Pos]:
    pairs = _all_valid_anchor_pairs(h, w)
    if not pairs:
        raise ValueError("no valid anchor pairs")
    direct = [p for p in pairs if _is_direct_connect_pair(p[0], p[1])]
    ambiguous = [p for p in pairs if not _is_direct_connect_pair(p[0], p[1])]

    # Most samples should be directly connectable (horizontal/vertical/diagonal).
    use_direct = bool(direct) and (not ambiguous or rr.random() < 0.85)
    pool = direct if use_direct else ambiguous
    return rr.choice(pool)


def _raster_line_cells(a: Pos, b: Pos) -> set[Pos]:
    """Integer raster cells of the segment using Bresenham."""
    r0, c0 = a
    r1, c1 = b
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr
    cells: set[Pos] = set()
    while True:
        cells.add((r0, c0))
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c0 += sc
        if e2 < dc:
            err += dc
            r0 += sr
    return cells


def _cells_on_side(h: int, w: int, a: Pos, b: Pos, side: int) -> list[Pos]:
    out: list[Pos] = []
    for r in range(h):
        for c in range(w):
            s = _cross(a, b, (r, c))
            if s == 0:
                continue
            if (-1 if s < 0 else 1) == side:
                out.append((r, c))
    return out


def _rect_fits_side(a: Pos, b: Pos, side: int, r0: int, r1: int, c0: int, c1: int) -> bool:
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            s = _cross(a, b, (r, c))
            if s == 0:
                return False
            if (-1 if s < 0 else 1) != side:
                return False
    return True


def _random_connected_subset(
    rr: random.Random, base_cells: Sequence[Pos], min_size: int, max_size: int
) -> set[Pos]:
    base = set(base_cells)
    if not base:
        return set()
    start = rr.choice(list(base))
    target = rr.randint(min_size, max_size)
    chosen = {start}
    frontier = {start}
    while frontier and len(chosen) < target:
        r, c = rr.choice(list(frontier))
        neigh = []
        for dr, dc in NEI8:
            p = (r + dr, c + dc)
            if p in base and p not in chosen:
                neigh.append(p)
        if neigh:
            p = rr.choice(neigh)
            chosen.add(p)
            frontier.add(p)
        else:
            frontier.remove((r, c))
            if not frontier and len(chosen) < target:
                # Restart growth from any chosen cell with potential neighbors.
                frontier = set(chosen)
    return chosen


def _place_single_side(
    grid: Grid,
    rr: random.Random,
    a: Pos,
    b: Pos,
    side: int,
    color: int,
) -> bool:
    h = len(grid)
    w = len(grid[0]) if h else 0
    # Choose a square target that is entirely on one side and away from divider.
    for _ in range(120):
        x = rr.randint(2, min(5, min(h, w) - 1))
        r0 = rr.randint(0, h - x)
        c0 = rr.randint(0, w - x)
        if not _rect_fits_side(a, b, side, r0, r0 + x - 1, c0, c0 + x - 1):
            continue
        # Build one connected non-full shape inside this box with bbox that has
        # max(height, width)=x, anchored at (r0, c0).
        # Use either horizontal line, vertical line, or L-shape.
        pattern = rr.choice(["hline", "vline", "L"])
        cells: set[Pos]
        if pattern == "hline":
            row = r0 + rr.randint(0, x - 1)
            cells = {(row, c) for c in range(c0, c0 + x)}
        elif pattern == "vline":
            col = c0 + rr.randint(0, x - 1)
            cells = {(r, col) for r in range(r0, r0 + x)}
        else:
            arm1 = rr.randint(2, x)
            arm2 = rr.randint(2, x)
            cells = {(r0, c0 + dc) for dc in range(arm1)} | {(r0 + dr, c0) for dr in range(arm2)}
            # Ensure bbox side length is exactly x.
            if rr.random() < 0.5:
                cells.add((r0, c0 + x - 1))
            else:
                cells.add((r0 + x - 1, c0))
        # Keep exactly one component and avoid touching gray dots/other shapes.
        if any(grid[r][c] != BG for r, c in cells):
            continue
        for r, c in cells:
            grid[r][c] = color
        return True
    return False


def _place_multi_side(
    grid: Grid,
    rr: random.Random,
    a: Pos,
    b: Pos,
    side: int,
    colors: Sequence[int],
) -> bool:
    h = len(grid)
    w = len(grid[0]) if h else 0
    for _ in range(150):
        bh = rr.randint(3, min(6, h - 1))
        bw = rr.randint(3, min(6, w - 1))
        r0 = rr.randint(0, h - bh)
        c0 = rr.randint(0, w - bw)
        r1 = r0 + bh - 1
        c1 = c0 + bw - 1
        if not _rect_fits_side(a, b, side, r0, r1, c0, c1):
            continue
        # Reserve this bbox region for multiple separated components.
        if any(grid[r][c] != BG for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)):
            continue
        base = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        placed: list[set[Pos]] = []
        ok = True
        n_comps = rr.randint(2, 3)
        for i in range(n_comps):
            comp = _random_connected_subset(rr, base, min_size=2, max_size=min(6, len(base)))
            if len(comp) < 2:
                ok = False
                break
            # Keep components separated (including diagonal) to avoid accidental merge.
            for old in placed:
                if any(abs(r1_ - r2_) <= 1 and abs(c1_ - c2_) <= 1 for r1_, c1_ in comp for r2_, c2_ in old):
                    ok = False
                    break
            if not ok:
                break
            placed.append(comp)
        if not ok:
            continue
        # Paint components.
        for i, comp in enumerate(placed):
            color = colors[i % len(colors)]
            for r, c in comp:
                grid[r][c] = color
        return True
    return False


def generate(
    rng: random.Random | None = None,
    *,
    min_h: int = 10,
    max_h: int = 14,
    min_w: int = 10,
    max_w: int = 14,
) -> dict[str, Grid]:
    rr = rng or random.Random()

    for _ in range(2500):
        h = rr.randint(min_h, max_h)
        w = rr.randint(min_w, max_w)
        inp = _empty(h, w)
        a, b = _pick_anchors(rr, h, w)
        inp[a[0]][a[1]] = GRAY
        inp[b[0]][b[1]] = GRAY

        # Ensure both sides are sufficiently populated with available cells.
        neg_cells = _cells_on_side(h, w, a, b, -1)
        pos_cells = _cells_on_side(h, w, a, b, 1)
        if len(neg_cells) < 12 or len(pos_cells) < 12:
            continue

        palette = [1, 2, 3, 4, 6, 7, 8, 9]
        rr.shuffle(palette)

        # Randomly choose each side as single/multi; force at least one multi for variety.
        side_mode = {
            -1: "single" if rr.random() < 0.45 else "multi",
            1: "single" if rr.random() < 0.45 else "multi",
        }
        if side_mode[-1] == "single" and side_mode[1] == "single":
            side_mode[rr.choice([-1, 1])] = "multi"

        ok = True
        color_idx = 0
        for side in (-1, 1):
            if side_mode[side] == "single":
                color = palette[color_idx]
                color_idx += 1
                if not _place_single_side(inp, rr, a, b, side, color):
                    ok = False
                    break
            else:
                colors = [palette[color_idx], palette[color_idx + 1], palette[color_idx + 2]]
                color_idx += 3
                if not _place_multi_side(inp, rr, a, b, side, colors):
                    ok = False
                    break
        if not ok:
            continue

        try:
            out = transform_t19(inp)
        except ValueError:
            continue

        if out == inp:
            continue
        # Sanity 1: output should not place non-bg on analytic divider line except anchors.
        bad = False
        for r in range(h):
            for c in range(w):
                if (r, c) in (a, b):
                    continue
                if _cross(a, b, (r, c)) == 0 and out[r][c] != BG:
                    bad = True
                    break
            if bad:
                break
        if bad:
            continue
        # Sanity 2: avoid perceived ambiguity: do not place transformed cells on
        # Bresenham divider raster cells either (except anchors).
        raster = _raster_line_cells(a, b)
        raster.discard(a)
        raster.discard(b)
        if any(out[r][c] != BG for r, c in raster):
            continue

        return {"input": inp, "output": out}

    raise RuntimeError("failed to sample valid t19 instance")
