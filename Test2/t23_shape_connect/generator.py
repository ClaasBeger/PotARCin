"""Generator for Test2/t23: connect same-color shapes with gray straight lines.

Interpretation used here:
- Shapes are 8-connected components of equal color (diagonal touch connects).
- For each color, every pair of its shapes connects at most once.
- A connection is valid only when the endpoint neighborhood orientation matches
  on both ends of the segment (e.g. horizontal: left-neighbor at start and
  right-neighbor at end).
- Among valid connections, use the shortest orthogonal straight segment.
- Draw connections in gray (5), excluding component endpoint cells.
- Remove all original colored shapes in the output.
"""

from __future__ import annotations

import random
from collections import defaultdict, deque
from typing import Dict, List, Sequence, Set, Tuple

Grid = List[List[int]]
Pos = Tuple[int, int]
Component = Set[Pos]

BG = 0
GRAY = 5
NEI8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
NEI4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _empty(h: int, w: int) -> Grid:
    return [[BG for _ in range(w)] for _ in range(h)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _components_of_color(grid: Grid, color: int) -> list[Component]:
    h = len(grid)
    w = len(grid[0]) if h else 0
    vis: set[Pos] = set()
    out: list[Component] = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] != color or (r, c) in vis:
                continue
            q = deque([(r, c)])
            vis.add((r, c))
            comp: Component = {(r, c)}
            while q:
                rr, cc = q.popleft()
                for dr, dc in NEI8:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(h, w, nr, nc) and grid[nr][nc] == color and (nr, nc) not in vis:
                        vis.add((nr, nc))
                        comp.add((nr, nc))
                        q.append((nr, nc))
            out.append(comp)
    return out


def _best_horizontal_segment(a: Component, b: Component) -> tuple[int, int, int, int] | None:
    cands = _pair_connection_candidates(a, b)
    hs = [(gap, axis, s, e) for gap, typ, axis, s, e in cands if typ == "H"]
    return min(hs) if hs else None


def _best_vertical_segment(a: Component, b: Component) -> tuple[int, int, int, int] | None:
    cands = _pair_connection_candidates(a, b)
    vs = [(gap, axis, s, e) for gap, typ, axis, s, e in cands if typ == "V"]
    return min(vs) if vs else None


def _pair_connection_candidates(a: Component, b: Component) -> list[tuple[int, str, int, int, int]]:
    """All valid orthogonal straight connection candidates for a shape pair.

    Returns tuples of the form:
      (gap, 'H' or 'V', axis_value, start_idx, end_idx)
    where axis_value is row for H and col for V.
    """
    rows_a: Dict[int, List[int]] = defaultdict(list)
    rows_b: Dict[int, List[int]] = defaultdict(list)
    for r, c in a:
        rows_a[r].append(c)
    for r, c in b:
        rows_b[r].append(c)

    cands: list[tuple[int, str, int, int, int]] = []
    for r in set(rows_a.keys()) & set(rows_b.keys()):
        a_min, a_max = min(rows_a[r]), max(rows_a[r])
        b_min, b_max = min(rows_b[r]), max(rows_b[r])

        # A on left, B on right.
        if a_max < b_min - 1:
            gap = b_min - a_max - 1
            cands.append((gap, "H", r, a_max + 1, b_min - 1))
        # B on left, A on right.
        if b_max < a_min - 1:
            gap = a_min - b_max - 1
            cands.append((gap, "H", r, b_max + 1, a_min - 1))

    cols_a: Dict[int, List[int]] = defaultdict(list)
    cols_b: Dict[int, List[int]] = defaultdict(list)
    for r, c in a:
        cols_a[c].append(r)
    for r, c in b:
        cols_b[c].append(r)

    for c in set(cols_a.keys()) & set(cols_b.keys()):
        a_min, a_max = min(cols_a[c]), max(cols_a[c])
        b_min, b_max = min(cols_b[c]), max(cols_b[c])

        # A above, B below.
        if a_max < b_min - 1:
            gap = b_min - a_max - 1
            cands.append((gap, "V", c, a_max + 1, b_min - 1))
        # B above, A below.
        if b_max < a_min - 1:
            gap = a_min - b_max - 1
            cands.append((gap, "V", c, b_max + 1, a_min - 1))
    return cands


def _has_undefined_tie_for_color(grid: Grid, color: int) -> bool:
    """True iff any same-color shape pair has >1 equally-short valid segment."""
    comps = _components_of_color(grid, color)
    if len(comps) < 2:
        return False
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            cands = _pair_connection_candidates(comps[i], comps[j])
            if not cands:
                continue
            best_gap = min(g for g, _, _, _, _ in cands)
            best = {(t, axis, s, e) for g, t, axis, s, e in cands if g == best_gap}
            if len(best) > 1:
                return True
    return False


def transform_t23(inp: Grid) -> Grid:
    if not inp or not inp[0]:
        raise ValueError("input grid must be non-empty")
    h = len(inp)
    w = len(inp[0])
    if any(len(row) != w for row in inp):
        raise ValueError("input grid must be rectangular")

    out = _empty(h, w)
    colors = sorted({v for row in inp for v in row if v != BG})
    for color in colors:
        shapes = _components_of_color(inp, color)
        if len(shapes) < 2:
            continue
        for i in range(len(shapes)):
            for j in range(i + 1, len(shapes)):
                hseg = _best_horizontal_segment(shapes[i], shapes[j])
                vseg = _best_vertical_segment(shapes[i], shapes[j])
                if hseg is None and vseg is None:
                    continue
                if hseg is not None and vseg is not None:
                    # Use the shorter straight connection; tie -> horizontal first.
                    use_h = hseg[0] <= vseg[0]
                else:
                    use_h = hseg is not None
                if use_h:
                    _, r, c1, c2 = hseg  # type: ignore[misc]
                    for c in range(c1, c2 + 1):
                        out[r][c] = GRAY
                else:
                    _, c, r1, r2 = vseg  # type: ignore[misc]
                    for r in range(r1, r2 + 1):
                        out[r][c] = GRAY
    return out


def _paint_component(grid: Grid, cells: Sequence[Pos], color: int) -> None:
    for r, c in cells:
        grid[r][c] = color


def _random_component(
    rr: random.Random,
    h: int,
    w: int,
    seed: Pos,
    max_size: int,
    *,
    forbidden: set[Pos] | None = None,
) -> list[Pos]:
    block = forbidden or set()
    if seed in block:
        return []
    cells = {seed}
    target = rr.randint(2, max_size)
    while len(cells) < target:
        frontier: list[Pos] = []
        for r, c in cells:
            for dr, dc in NEI4:
                nr, nc = r + dr, c + dc
                if _in_bounds(h, w, nr, nc) and (nr, nc) not in cells and (nr, nc) not in block:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(rr.choice(frontier))
    return sorted(cells)


def _place_color_with_two_components(
    grid: Grid,
    rr: random.Random,
    color: int,
) -> bool:
    h = len(grid)
    w = len(grid[0]) if h else 0
    # Build two components guaranteed to share a row or column for line creation.
    mode = rr.choice(["row", "col"])
    for _ in range(120):
        if mode == "row":
            r = rr.randint(1, h - 2)
            c1 = rr.randint(0, max(0, w // 3))
            c2 = rr.randint(max(c1 + 3, (2 * w) // 3), w - 1)
            s1 = (r, c1)
            s2 = (r, c2)
        else:
            c = rr.randint(1, w - 2)
            r1 = rr.randint(0, max(0, h // 3))
            r2 = rr.randint(max(r1 + 3, (2 * h) // 3), h - 1)
            s1 = (r1, c)
            s2 = (r2, c)

        if grid[s1[0]][s1[1]] != BG or grid[s2[0]][s2[1]] != BG:
            continue
        comp1 = _random_component(rr, h, w, s1, max_size=6)
        if not comp1:
            continue
        block = set(comp1)
        # keep one-cell moat around first component to avoid diagonal touching
        for r, c in comp1:
            for dr, dc in NEI8:
                block.add((r + dr, c + dc))
        comp2 = _random_component(rr, h, w, s2, max_size=6, forbidden=block)
        if not comp2:
            continue
        if any(grid[r][c] != BG for r, c in comp1 + comp2):
            continue
        # Under 8-connectivity, these must remain separate shapes.
        tmp = [row[:] for row in grid]
        _paint_component(tmp, comp1, color)
        _paint_component(tmp, comp2, color)
        if len(_components_of_color(tmp, color)) < 2:
            continue
        comps = _components_of_color(tmp, color)
        if len(comps) != 2:
            continue
        if _best_horizontal_segment(comps[0], comps[1]) is None and _best_vertical_segment(comps[0], comps[1]) is None:
            continue
        _paint_component(grid, comp1, color)
        _paint_component(grid, comp2, color)
        return True
    return False


def _place_single_component_noise(grid: Grid, rr: random.Random, color: int) -> bool:
    h = len(grid)
    w = len(grid[0]) if h else 0
    for _ in range(100):
        seed = (rr.randint(0, h - 1), rr.randint(0, w - 1))
        if grid[seed[0]][seed[1]] != BG:
            continue
        comp = _random_component(rr, h, w, seed, max_size=5)
        if any(grid[r][c] != BG for r, c in comp):
            continue
        _paint_component(grid, comp, color)
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
    palette = [1, 2, 3, 4, 6, 7, 8, 9]

    for _ in range(2500):
        h = rr.randint(min_h, max_h)
        w = rr.randint(min_w, max_w)
        inp = _empty(h, w)
        cols = rr.sample(palette, rr.randint(4, 6))

        ok = True
        # Most colors have two components (to create connectable pairs).
        n_link_colors = rr.randint(3, max(3, len(cols) - 1))
        link_cols = cols[:n_link_colors]
        noise_cols = cols[n_link_colors:]

        for col in link_cols:
            if not _place_color_with_two_components(inp, rr, col):
                ok = False
                break
        if not ok:
            continue
        for col in noise_cols:
            if not _place_single_component_noise(inp, rr, col):
                ok = False
                break
        if not ok:
            continue
        # Avoid undefined cases: no same-color pair may have a shortest-connection tie.
        if any(_has_undefined_tie_for_color(inp, c) for c in cols):
            continue

        try:
            out = transform_t23(inp)
        except ValueError:
            continue
        if out == inp:
            continue

        gray_count = sum(1 for r in range(h) for c in range(w) if out[r][c] == GRAY)
        if gray_count < 8:
            continue
        return {"input": inp, "output": out}

    raise RuntimeError("failed to sample valid t23 instance")
