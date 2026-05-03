"""Generator for Test2/t17 (pattern fall).

There is exactly **one** prototype (no rotations): a fixed set of foreground cells in
the plane, **king-connected** (edge or diagonal adjacency counts *within* that stencil).
The prototype is never a single cell.

Each *placement* is a translation of that stencil. Viewport clipping can hide parts above,
below, or beside the grid; shifting is by **one prototype height** (vertical span).

**Between** different placements of the same prototype, diagonal touching must **not**
merge them into one instance (only orthogonal edge contact merges on the visible grid).

``generate()`` only samples common cases; authored puzzles may use rarer geometry.
"""

from __future__ import annotations

import random
from collections import Counter, deque
from typing import Iterable, List, Tuple

Grid = List[List[int]]
Pos = Tuple[int, int]
ShapeSig = Tuple[Pos, ...]

BG = 0


def _empty(h: int, w: int) -> Grid:
    return [[BG for _ in range(w)] for _ in range(h)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> Iterable[Pos]:
    yield (r + 1, c)
    yield (r - 1, c)
    yield (r, c + 1)
    yield (r, c - 1)


def _neighbors8(r: int, c: int) -> Iterable[Pos]:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                yield (r + dr, c + dc)


def _king_connected(sig: ShapeSig) -> bool:
    """True iff cells of ``sig`` are connected under king (8-neighbor) moves."""
    cells = set(sig)
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _extract_components(grid: Grid, color: int) -> list[set[Pos]]:
    h = len(grid)
    w = len(grid[0]) if h else 0
    seen: set[Pos] = set()
    comps: list[set[Pos]] = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] != color or (r, c) in seen:
                continue
            q = deque([(r, c)])
            seen.add((r, c))
            comp: set[Pos] = set()
            while q:
                rr, cc = q.popleft()
                comp.add((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(h, w, nr, nc) and grid[nr][nc] == color and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: set[Pos]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _normalize(cells: Iterable[Pos]) -> ShapeSig:
    cells_t = tuple(cells)
    if not cells_t:
        return ()
    min_r = min(r for r, _ in cells_t)
    min_c = min(c for _, c in cells_t)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells_t))


def _draw_shape(grid: Grid, color: int, sig: ShapeSig, top: int, left: int) -> None:
    h = len(grid)
    w = len(grid[0]) if h else 0
    for dr, dc in sig:
        rr, cc = top + dr, left + dc
        if _in_bounds(h, w, rr, cc):
            grid[rr][cc] = color


def _visible_from_anchor(sig: ShapeSig, top: int, left: int, h: int, w: int) -> set[Pos]:
    out: set[Pos] = set()
    for dr, dc in sig:
        rr, cc = top + dr, left + dc
        if _in_bounds(h, w, rr, cc):
            out.add((rr, cc))
    return out


def _touches_boundary(comp: set[Pos], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)


def _infer_anchor_for_component(
    comp: set[Pos],
    sig: ShapeSig,
    h: int,
    w: int,
) -> tuple[int, int]:
    if not comp:
        raise ValueError("empty component is not allowed")
    max_r = max(r for r, _ in sig)
    max_c = max(c for _, c in sig)
    candidates: list[tuple[int, int]] = []
    # Allow clipping by any border; this supports top-completion and bottom cut-offs.
    for top in range(-max_r, h):
        for left in range(-max_c, w):
            if _visible_from_anchor(sig, top, left, h, w) == comp:
                candidates.append((top, left))
    if not candidates:
        raise ValueError("could not infer anchor for component")
    # Deterministic tie-break for ambiguous clipped fragments.
    candidates.sort(key=lambda p: (p[0], p[1]))
    return candidates[0]


def _infer_anchors_for_proto(inp: Grid, color: int, sig: ShapeSig) -> list[tuple[int, int]]:
    """Infer placement anchors for a single prototype across the whole grid.

    Placements can touch and merge in the visible grid, so we cannot assume one
    connected component corresponds to one placement. Instead, solve an exact
    cover over all viewport-clipped prototype placements.
    """
    h = len(inp)
    w = len(inp[0]) if h else 0
    fg = {(r, c) for r in range(h) for c in range(w) if inp[r][c] == color}
    if not fg:
        return []

    max_r = max(r for r, _ in sig)
    max_c = max(c for _, c in sig)

    # Deduplicate by visible cell-set; keep deterministic representative anchor.
    vis_to_anchor: dict[frozenset[Pos], tuple[int, int]] = {}
    for top in range(-max_r, h):
        for left in range(-max_c, w):
            vis = frozenset(_visible_from_anchor(sig, top, left, h, w))
            if not vis or not vis.issubset(fg):
                continue
            prev = vis_to_anchor.get(vis)
            if prev is None or (top, left) < prev:
                vis_to_anchor[vis] = (top, left)

    if not vis_to_anchor:
        raise ValueError("no feasible clipped placements for inferred prototype")

    sets = sorted(vis_to_anchor.items(), key=lambda kv: (-len(kv[0]), kv[1]))
    cell_to_set_ids: dict[Pos, list[int]] = {p: [] for p in fg}
    for idx, (vis, _anchor) in enumerate(sets):
        for p in vis:
            cell_to_set_ids[p].append(idx)

    if any(len(cell_to_set_ids[p]) == 0 for p in fg):
        raise ValueError("prototype cannot explain all foreground cells")

    covered: set[Pos] = set()
    chosen_ids: list[int] = []

    def _dfs_cover() -> bool:
        if len(covered) == len(fg):
            return True
        # Branch on the tightest uncovered cell first.
        uncovered = [p for p in fg if p not in covered]
        pivot = min(
            uncovered,
            key=lambda p: sum(1 for sid in cell_to_set_ids[p] if not (sets[sid][0] & covered)),
        )
        cands = [sid for sid in cell_to_set_ids[pivot] if not (sets[sid][0] & covered)]
        # Deterministic: larger placement first, then anchor order from `sets`.
        cands.sort(key=lambda sid: -len(sets[sid][0]))
        for sid in cands:
            vis, _anchor = sets[sid]
            if vis & covered:
                continue
            chosen_ids.append(sid)
            covered.update(vis)
            if _dfs_cover():
                return True
            chosen_ids.pop()
            covered.difference_update(vis)
        return False

    if not _dfs_cover():
        raise ValueError("could not decompose foreground into clipped prototype placements")

    return [sets[sid][1] for sid in chosen_ids]


def transform_t17(inp: Grid) -> Grid:
    if not inp or not inp[0]:
        raise ValueError("input grid must be non-empty")
    w = len(inp[0])
    if any(len(row) != w for row in inp):
        raise ValueError("input grid must be rectangular")

    h = len(inp)
    colors = sorted({v for row in inp for v in row if v != BG})
    if len(colors) != 1:
        raise ValueError("expected exactly one foreground color")
    color = colors[0]

    comps = _extract_components(inp, color)
    if not comps:
        raise ValueError("no foreground components found")

    interior: list[set[Pos]] = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        if r0 > 0 and r1 < h - 1 and c0 > 0 and c1 < w - 1:
            interior.append(comp)
    if not interior:
        raise ValueError("need at least one full interior instance to infer shape")

    # Prefer larger interior clips so tiny dots do not dominate prototype choice.
    sig_counter = Counter(_normalize(c) for c in interior)
    proto_sig = max(sig_counter.items(), key=lambda kv: (len(kv[0]), kv[1]))[0]
    if not proto_sig:
        raise ValueError("invalid empty prototype")
    if len(proto_sig) < 2:
        raise ValueError("prototype is too small")
    if not _king_connected(proto_sig):
        raise ValueError("prototype must be king-connected")
    shape_h = max(r for r, _ in proto_sig) + 1

    anchors = _infer_anchors_for_proto(inp, color, proto_sig)

    anchors = sorted(set(anchors))
    out = _empty(h, w)
    for top, left in anchors:
        _draw_shape(out, color, proto_sig, top + shape_h, left)
    return out


def _random_connected_shape(rr: random.Random, height: int, width: int) -> ShapeSig:
    # Start with one cell per row, connected row-to-row.
    cols: list[int] = [rr.randint(0, width - 1)]
    for _r in range(1, height):
        prev = cols[-1]
        cands = [c for c in (prev - 1, prev, prev + 1) if 0 <= c < width]
        cols.append(rr.choice(cands))

    cells: set[Pos] = {(r, c) for r, c in enumerate(cols)}
    extra = rr.randint(1, max(2, height + width))
    for _ in range(extra):
        r = rr.randint(0, height - 1)
        c = rr.randint(0, width - 1)
        # Keep additions adjacent to existing cells to preserve connectivity.
        if any((nr, nc) in cells for nr, nc in _neighbors4(r, c)):
            cells.add((r, c))
    return _normalize(cells)


def generate(
    rng: random.Random | None = None,
    *,
    min_h: int = 10,
    max_h: int = 16,
    min_w: int = 10,
    max_w: int = 16,
) -> dict[str, Grid]:
    rr = rng or random.Random()

    for _ in range(1400):
        h = rr.randint(min_h, max_h)
        w = rr.randint(min_w, max_w)
        color = rr.choice([1, 2, 3, 4, 6, 7, 8, 9])

        shape_h = rr.randint(2, min(6, h - 4))
        shape_w = rr.randint(2, min(6, w - 4))
        sig = _random_connected_shape(rr, shape_h, shape_w)
        if len(sig) < 3:
            continue

        n_instances = rr.randint(6, 16)
        anchors: list[tuple[int, int]] = []
        occupied: set[Pos] = set()
        # Track component starts for style balancing.
        top_cut_count = 0
        bot_cut_count = 0
        left_cut_count = 0
        right_cut_count = 0
        full_count = 0

        for _j in range(n_instances * 6):
            if len(anchors) >= n_instances:
                break
            top = rr.randint(-shape_h + 1, h - 1)
            left = rr.randint(-shape_w + 1, w - 1)
            visible = _visible_from_anchor(sig, top, left, h, w)
            if not visible:
                continue
            # Keep instances separated so components stay interpretable.
            expanded = set(visible)
            for r, c in list(visible):
                for nr, nc in _neighbors4(r, c):
                    if _in_bounds(h, w, nr, nc):
                        expanded.add((nr, nc))
            if occupied & expanded:
                continue
            anchors.append((top, left))
            occupied |= expanded
            if top < 0:
                top_cut_count += 1
            if top + shape_h - 1 >= h:
                bot_cut_count += 1
            if left < 0:
                left_cut_count += 1
            if left + shape_w - 1 >= w:
                right_cut_count += 1
            if (
                0 <= top
                and top + shape_h - 1 < h
                and 0 <= left
                and left + shape_w - 1 < w
                and len(visible) == len(sig)
            ):
                full_count += 1

        if len(anchors) < 5:
            continue
        if full_count < 2:
            continue
        # Edge clipping is common in your examples, but not mandatory every time.
        if rr.random() < 0.75 and (top_cut_count + bot_cut_count) == 0:
            continue
        if rr.random() < 0.7 and (left_cut_count + right_cut_count) == 0:
            continue

        inp = _empty(h, w)
        for top, left in anchors:
            _draw_shape(inp, color, sig, top, left)

        try:
            out = transform_t17(inp)
        except ValueError:
            continue
        if out == inp:
            continue
        if not any(out[r][c] == color and inp[r][c] == BG for r in range(h) for c in range(w)):
            continue
        return {"input": inp, "output": out}

    raise RuntimeError("failed to sample valid t17 instance")
