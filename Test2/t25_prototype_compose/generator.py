"""Generator for Test2/t25: keep prototype-composable shapes, delete others.

Interpretation:
- Grid contains one foreground color split into multiple 4-connected shapes.
- Prototype candidates are present shapes that:
  (a) cannot be composed from multiple non-overlapping copies of any other
      present shape (under symmetry), and
  (b) can compose at least one other present shape.
- Among those candidates, the unique largest one is the prototype.
- Any shape that can be exactly composed by one or more non-overlapping copies
  of the prototype (using translation/rotation/reflection) is kept.
- All other shapes are removed.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Dict, Iterable, List, Sequence, Set, Tuple

Grid = List[List[int]]
Pos = Tuple[int, int]
Shape = Set[Pos]

BG = 0
NEI4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _empty(h: int, w: int) -> Grid:
    return [[BG for _ in range(w)] for _ in range(h)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _components_of_color(grid: Grid, color: int) -> list[Shape]:
    h = len(grid)
    w = len(grid[0]) if h else 0
    vis: set[Pos] = set()
    comps: list[Shape] = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] != color or (r, c) in vis:
                continue
            q = deque([(r, c)])
            vis.add((r, c))
            comp: Shape = {(r, c)}
            while q:
                rr, cc = q.popleft()
                for dr, dc in NEI4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(h, w, nr, nc) and grid[nr][nc] == color and (nr, nc) not in vis:
                        vis.add((nr, nc))
                        comp.add((nr, nc))
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize(shape: Iterable[Pos]) -> tuple[Pos, ...]:
    pts = list(shape)
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _symmetries(norm: tuple[Pos, ...]) -> list[tuple[Pos, ...]]:
    pts = list(norm)
    out: set[tuple[Pos, ...]] = set()
    for rx in (1, -1):
        for ry in (1, -1):
            for swap in (False, True):
                t = []
                for r, c in pts:
                    a, b = (c, r) if swap else (r, c)
                    t.append((rx * a, ry * b))
                out.add(_normalize(t))
    return sorted(out)


def _shape_buildable(shape: Shape, proto_variants: list[tuple[Pos, ...]]) -> bool:
    """Exact cover of shape cells by translated prototype variants."""
    n = len(shape)
    psize = len(proto_variants[0]) if proto_variants else 0
    if psize == 0 or n % psize != 0:
        return False
    target = set(shape)

    variants = [set(v) for v in proto_variants]

    # Precompute placements that fit fully inside target.
    placements: list[Shape] = []
    for v in variants:
        vr = [r for r, _ in v]
        vc = [c for _, c in v]
        v_anchor = min(v)  # deterministic anchor cell in variant
        ar, ac = v_anchor
        for tr, tc in target:
            dr, dc = tr - ar, tc - ac
            placed = {(r + dr, c + dc) for r, c in v}
            if placed <= target:
                placements.append(placed)

    # Deduplicate placements.
    uniq = []
    seen: set[frozenset[Pos]] = set()
    for p in placements:
        fp = frozenset(p)
        if fp in seen:
            continue
        seen.add(fp)
        uniq.append(p)
    placements = uniq

    # Inverted index for exact-cover search.
    by_cell: Dict[Pos, list[int]] = {cell: [] for cell in target}
    for i, p in enumerate(placements):
        for cell in p:
            by_cell[cell].append(i)

    def backtrack(remaining: set[Pos]) -> bool:
        if not remaining:
            return True
        # choose constrained cell
        cell = min(remaining, key=lambda x: len(by_cell[x]))
        for idx in by_cell[cell]:
            p = placements[idx]
            if not p <= remaining:
                continue
            if backtrack(remaining - p):
                return True
        return False

    return backtrack(set(target))


def _shape_buildable_min_copies(
    shape: Shape,
    proto_variants: list[tuple[Pos, ...]],
    min_copies: int,
) -> bool:
    psize = len(proto_variants[0]) if proto_variants else 0
    if psize == 0:
        return False
    if len(shape) % psize != 0:
        return False
    if (len(shape) // psize) < min_copies:
        return False
    return _shape_buildable(shape, proto_variants)


def _infer_prototype(comps: list[Shape]) -> list[tuple[Pos, ...]]:
    """Infer prototype by candidate filtering and largest-shape disambiguation."""
    if not comps:
        raise ValueError("no shapes")

    norms = [_normalize(c) for c in comps]
    norms = [n for n in norms if len(n) >= 2]
    if not norms:
        raise ValueError("no valid prototype candidates")

    # Deduplicate present shapes by symmetry class.
    present_by_class: dict[tuple[Pos, ...], tuple[Pos, ...]] = {}
    for n in norms:
        canon = min(_symmetries(n))
        present_by_class.setdefault(canon, n)
    present = list(present_by_class.values())

    candidates: list[tuple[Pos, ...]] = []
    for a in present:
        a_shape = set(a)
        composed_from_other = False
        for b in present:
            if b == a:
                continue
            if _shape_buildable_min_copies(a_shape, _symmetries(b), min_copies=2):
                composed_from_other = True
                break
        if composed_from_other:
            continue

        # Additional criterion: candidate prototype must compose at least
        # one distinct present shape.
        pvars = _symmetries(a)
        builds_other = False
        for b in present:
            if b == a:
                continue
            if _shape_buildable(set(b), pvars):
                builds_other = True
                break
        if builds_other:
            candidates.append(a)

    if not candidates:
        raise ValueError("no prototype candidates after filtering")
    max_size = max(len(c) for c in candidates)
    largest = [c for c in candidates if len(c) == max_size]
    if len(largest) != 1:
        raise ValueError("ambiguous largest prototype candidate")
    return _symmetries(largest[0])


def transform_t25(inp: Grid) -> Grid:
    if not inp or not inp[0]:
        raise ValueError("input grid must be non-empty")
    h = len(inp)
    w = len(inp[0])
    if any(len(row) != w for row in inp):
        raise ValueError("input grid must be rectangular")

    colors = sorted({v for row in inp for v in row if v != BG})
    if len(colors) != 1:
        raise ValueError("expected exactly one foreground color")
    color = colors[0]

    comps = _components_of_color(inp, color)
    pvars = _infer_prototype(comps)

    out = _empty(h, w)
    for comp in comps:
        if _shape_buildable(comp, pvars):
            for r, c in comp:
                out[r][c] = color
    return out


def _random_connected_shape(rr: random.Random, h: int, w: int, size: int) -> list[Pos]:
    seed = (rr.randint(0, h - 1), rr.randint(0, w - 1))
    cells = {seed}
    while len(cells) < size:
        boundary: list[Pos] = []
        for r, c in cells:
            for dr, dc in NEI4:
                nr, nc = r + dr, c + dc
                if _in_bounds(h, w, nr, nc) and (nr, nc) not in cells:
                    boundary.append((nr, nc))
        if not boundary:
            break
        cells.add(rr.choice(boundary))
    return sorted(cells)


def _translate(shape: Sequence[Pos], dr: int, dc: int) -> list[Pos]:
    return [(r + dr, c + dc) for r, c in shape]


def _place_shape(grid: Grid, shape: Sequence[Pos], color: int) -> bool:
    h = len(grid)
    w = len(grid[0]) if h else 0
    for r, c in shape:
        if not _in_bounds(h, w, r, c) or grid[r][c] != BG:
            return False
    shape_set = set(shape)
    # Prevent accidental 4-neighbor merges between nominally separate shapes.
    for r, c in shape:
        for dr, dc in NEI4:
            nr, nc = r + dr, c + dc
            if (nr, nc) in shape_set:
                continue
            if _in_bounds(h, w, nr, nc) and grid[nr][nc] == color:
                return False
    for r, c in shape:
        grid[r][c] = color
    return True


def _compose_from_proto(rr: random.Random, proto: tuple[Pos, ...], copies: int) -> tuple[Pos, ...]:
    variants = _symmetries(proto)
    placed: set[Pos] = set(variants[rr.randrange(len(variants))])
    for _ in range(copies - 1):
        var = set(variants[rr.randrange(len(variants))])
        # try to place touching existing union without overlap
        ok = False
        for _ in range(80):
            pr, pc = rr.choice(list(placed))
            vr, vc = rr.choice(list(var))
            dr, dc = pr - vr, pc - vc
            cand = {(r + dr, c + dc) for r, c in var}
            if cand & placed:
                continue
            # require at least one edge-touch for connectivity
            touch = any((r + dr4, c + dc4) in placed for r, c in cand for dr4, dc4 in NEI4)
            if not touch:
                continue
            placed |= cand
            ok = True
            break
        if not ok:
            # fallback: disjoint placement, still valid component only if touching elsewhere later
            shift = rr.randint(1, 3)
            cand = {(r + shift, c + shift) for r, c in var}
            placed |= cand
    return _normalize(placed)


def generate(
    rng: random.Random | None = None,
    *,
    min_h: int = 10,
    max_h: int = 14,
    min_w: int = 10,
    max_w: int = 14,
) -> dict[str, Grid]:
    rr = rng or random.Random()

    for _ in range(3500):
        h = rr.randint(min_h, max_h)
        w = rr.randint(min_w, max_w)
        color = rr.choice([1, 3, 7, 9])
        inp = _empty(h, w)

        psize = rr.randint(3, 6)
        proto_cells = _random_connected_shape(rr, 4, 4, psize)
        proto = _normalize(proto_cells)

        # Build keepers from 1..3 prototype copies.
        keep_shapes: list[tuple[Pos, ...]] = []
        keep_shapes.append(proto)
        for _ in range(rr.randint(2, 4)):
            copies = rr.choice([1, 2, 2, 3])
            keep_shapes.append(_compose_from_proto(rr, proto, copies))

        # Build rejectors (size not multiple / random non-composable).
        reject_shapes: list[tuple[Pos, ...]] = []
        for _ in range(rr.randint(2, 4)):
            sz = rr.randint(2, 10)
            if sz % len(proto) == 0:
                sz += 1
            rej = _normalize(_random_connected_shape(rr, 5, 5, min(sz, 8)))
            reject_shapes.append(rej)

        all_shapes = [(s, True) for s in keep_shapes] + [(s, False) for s in reject_shapes]
        rr.shuffle(all_shapes)

        ok = True
        for shape, _is_keep in all_shapes:
            placed = False
            for _ in range(120):
                maxr = max(r for r, _ in shape)
                maxc = max(c for _, c in shape)
                if maxr >= h or maxc >= w:
                    break
                r0 = rr.randint(0, h - maxr - 1)
                c0 = rr.randint(0, w - maxc - 1)
                if _place_shape(inp, _translate(shape, r0, c0), color):
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        try:
            out = transform_t25(inp)
        except ValueError:
            continue
        if out == inp:
            continue

        # Ensure intended prototype is actually represented as a standalone shape.
        comps_in = _components_of_color(inp, color)
        pvars_intended = set(_symmetries(proto))
        has_isolated_proto = any(_normalize(c) in pvars_intended for c in comps_in)
        if not has_isolated_proto:
            continue

        # Keep prototype inference unambiguous and aligned with intended class.
        try:
            inferred = _infer_prototype(comps_in)
        except ValueError:
            continue
        if min(inferred) != min(_symmetries(proto)):
            continue

        # Ensure both kept and removed shapes exist.
        in_cells = sum(v != BG for row in inp for v in row)
        out_cells = sum(v != BG for row in out for v in row)
        if out_cells < 8 or out_cells >= in_cells:
            continue
        return {"input": inp, "output": out}

    raise RuntimeError("failed to sample valid t25 instance")
