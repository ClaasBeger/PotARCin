"""Generator for Test2/t17 (pattern fall).

The prototype is always **fully visible** somewhere on the grid (an unclipped copy),
with at least one placement **strictly inside** the outer border (not on row/column 0
or the last row/column).

**Inference.** Prefer strictly interior 4‑connected foreground blobs of size 2–10
(unique normalised shape, or most frequent among ties). If that does not determine the
prototype (merged blobs, ambiguous fragments), fall back to the **largest** ``n``≤10
connected shape that **occurs as an induced subgraph of the foreground** (orthogonal
connectivity first, then king) and admits an exact disjoint cover of the foreground.

The fallback never scans the full catalogue of fixed ``n``‑ominoes (that was
prohibitively slow); it only considers shapes that literally appear on the grid.

``generate()`` samples orth‑connected prototypes with two fully interior placements.
"""

from __future__ import annotations

import random
from collections import Counter, deque
from typing import Callable, Iterable, List, Tuple

Grid = List[List[int]]
Pos = Tuple[int, int]
ShapeSig = Tuple[Pos, ...]

BG = 0


def _empty(h: int, w: int) -> Grid:
    return [[BG for _ in range(w)] for _ in range(h)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _max_orth_component_size(fg: set[Pos]) -> int:
    """Largest 4‑connected size among cells in ``fg`` (induced on the grid)."""
    if not fg:
        return 0
    seen: set[Pos] = set()
    best = 0
    for start in sorted(fg):
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        comp: set[Pos] = {start}
        while q:
            r, c = q.popleft()
            for nr, nc in _neighbors4(r, c):
                p = (nr, nc)
                if p in fg and p not in seen:
                    seen.add(p)
                    comp.add(p)
                    q.append(p)
        if len(comp) > best:
            best = len(comp)
    return best


def _max_king_component_size(fg: set[Pos]) -> int:
    """Largest 8‑connected size among ``fg``."""
    if not fg:
        return 0
    seen: set[Pos] = set()
    best = 0
    for start in sorted(fg):
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        comp: set[Pos] = {start}
        while q:
            r, c = q.popleft()
            for nr, nc in _neighbors8(r, c):
                p = (nr, nc)
                if p in fg and p not in seen:
                    seen.add(p)
                    comp.add(p)
                    q.append(p)
        if len(comp) > best:
            best = len(comp)
    return best


def _orth_connected_components(inp: Grid, color: int) -> list[set[Pos]]:
    h = len(inp)
    w = len(inp[0]) if h else 0
    seen: set[Pos] = set()
    comps: list[set[Pos]] = []
    for r in range(h):
        for c in range(w):
            if inp[r][c] != color or (r, c) in seen:
                continue
            q = deque([(r, c)])
            seen.add((r, c))
            comp: set[Pos] = set()
            while q:
                rr, cc = q.popleft()
                comp.add((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(h, w, nr, nc) and inp[nr][nc] == color and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _is_strictly_interior(comp: set[Pos], h: int, w: int) -> bool:
    if not comp:
        return False
    return (
        min(r for r, _ in comp) > 0
        and max(r for r, _ in comp) < h - 1
        and min(c for _, c in comp) > 0
        and max(c for _, c in comp) < w - 1
    )


def _try_infer_proto_from_interior_blobs(
    inp: Grid, color: int
) -> tuple[ShapeSig, list[tuple[int, int]]] | None:
    """Use strictly interior 4‑connected blobs of size 2–10 (unique or most frequent)."""
    h = len(inp)
    w = len(inp[0]) if h else 0
    fg0 = {(r, c) for r in range(h) for c in range(w) if inp[r][c] == color}
    sigs: list[ShapeSig] = []
    for comp in _orth_connected_components(inp, color):
        if not _is_strictly_interior(comp, h, w):
            continue
        if 2 <= len(comp) <= 10:
            sigs.append(_normalize(comp))
    if not sigs:
        return None
    if len(sigs) == 1:
        sig = sigs[0]
        anchors = _try_infer_anchors_for_proto(inp, color, sig, fg0)
        return (sig, anchors) if anchors is not None else None
    cnt = Counter(sigs)
    max_c = max(cnt.values())
    for sig in sorted(s for s, c in cnt.items() if c == max_c):
        anchors = _try_infer_anchors_for_proto(inp, color, sig, fg0)
        if anchors is not None:
            return sig, anchors
    return None


def _neighbors4(r: int, c: int) -> Iterable[Pos]:
    yield (r + 1, c)
    yield (r - 1, c)
    yield (r, c + 1)
    yield (r, c - 1)


def _normalize(cells: Iterable[Pos]) -> ShapeSig:
    cells_t = tuple(cells)
    if not cells_t:
        return ()
    min_r = min(r for r, _ in cells_t)
    min_c = min(c for _, c in cells_t)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells_t))


def _neighbors8(r: int, c: int) -> Iterable[Pos]:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                yield (r + dr, c + dc)


def _induced_connected_exact_size(
    fg: set[Pos],
    neighbor_iter: Callable[[int, int], Iterable[Pos]],
    n: int,
    max_calls: int,
) -> set[ShapeSig]:
    """Connected ``n``‑subsets of ``fg`` (``neighbor_iter``), translation‑normalised. Pruned to exact size only."""
    uniq: set[ShapeSig] = set()
    calls = 0

    def rec(active: frozenset[Pos]) -> None:
        nonlocal calls
        calls += 1
        if calls > max_calls:
            return
        na = len(active)
        if na == n:
            uniq.add(_normalize(active))
            return
        if na > n:
            return
        cand: set[Pos] = set()
        for r, c in active:
            for nr, nc in neighbor_iter(r, c):
                if (nr, nc) in fg and (nr, nc) not in active:
                    cand.add((nr, nc))
        for nb in sorted(cand):
            rec(active | {nb})

    for start in sorted(fg):
        rec(frozenset({start}))
    return uniq


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


def _strict_interior_full_placement_count(sig: ShapeSig, fg: set[Pos], h: int, w: int) -> int:
    """How many anchors place a full unclipped copy strictly inside the outer border and inside ``fg``."""
    max_r = max(r for r, _ in sig)
    max_c = max(c for _, c in sig)
    n = 0
    for top in range(-max_r, h):
        for left in range(-max_c, w):
            vis = _visible_from_anchor(sig, top, left, h, w)
            if len(vis) != len(sig) or not vis.issubset(fg):
                continue
            if min(r for r, _ in vis) <= 0:
                continue
            if max(r for r, _ in vis) >= h - 1:
                continue
            if min(c for _, c in vis) <= 0:
                continue
            if max(c for _, c in vis) >= w - 1:
                continue
            n += 1
    return n


def _strict_interior_full_placement_exists(sig: ShapeSig, fg: set[Pos], h: int, w: int) -> bool:
    """True iff some anchor places the entire prototype on-grid, strictly inside the border (unclipped copy)."""
    return _strict_interior_full_placement_count(sig, fg, h, w) > 0


def _try_infer_anchors_for_proto(
    inp: Grid, color: int, sig: ShapeSig, fg: set[Pos] | None = None
) -> list[tuple[int, int]] | None:
    """Return anchors whose disjoint clipped placements cover all ``color`` cells, or ``None``."""
    h = len(inp)
    w = len(inp[0]) if h else 0
    if fg is None:
        fg = {(r, c) for r in range(h) for c in range(w) if inp[r][c] == color}
    if not fg:
        return []

    max_r = max(r for r, _ in sig)
    max_c = max(c for _, c in sig)

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
        return None

    sets = sorted(vis_to_anchor.items(), key=lambda kv: (-len(kv[0]), kv[1]))
    cell_to_set_ids: dict[Pos, list[int]] = {p: [] for p in fg}
    for idx, (vis, _anchor) in enumerate(sets):
        for p in vis:
            cell_to_set_ids[p].append(idx)

    if any(len(cell_to_set_ids[p]) == 0 for p in fg):
        return None

    covered: set[Pos] = set()
    chosen_ids: list[int] = []

    def _dfs_cover() -> bool:
        if len(covered) == len(fg):
            return True
        uncovered = [p for p in fg if p not in covered]
        pivot = min(
            uncovered,
            key=lambda p: sum(1 for sid in cell_to_set_ids[p] if not (sets[sid][0] & covered)),
        )
        cands = [sid for sid in cell_to_set_ids[pivot] if not (sets[sid][0] & covered)]
        cands.sort(key=lambda sid: (-len(sets[sid][0]), sets[sid][1]))
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
        return None

    return [sets[sid][1] for sid in chosen_ids]


def _infer_anchors_for_proto(inp: Grid, color: int, sig: ShapeSig) -> list[tuple[int, int]]:
    out = _try_infer_anchors_for_proto(inp, color, sig)
    if out is None:
        raise ValueError("could not decompose foreground into clipped prototype placements")
    return out


def _resolve_proto_signature(inp: Grid, color: int) -> tuple[ShapeSig, list[tuple[int, int]]]:
    """Interior blobs first; else largest induced foreground shape (orth, then king) with exact cover."""
    hinted = _try_infer_proto_from_interior_blobs(inp, color)
    if hinted is not None:
        return hinted

    h = len(inp)
    w = len(inp[0]) if h else 0
    fg = {(r, c) for r in range(h) for c in range(w) if inp[r][c] == color}

    nf = len(fg)
    max_orth = _max_orth_component_size(fg)
    max_king = _max_king_component_size(fg)
    for neighbors, budget, n_hi in (
        (_neighbors4, 200_000, min(10, nf, max_orth)),
        (_neighbors8, 400_000, min(10, nf, max_king)),
    ):
        if n_hi < 2:
            continue
        for n in range(n_hi, 1, -1):
            raw = _induced_connected_exact_size(fg, neighbors, n, max_calls=budget)
            scored: list[tuple[int, ShapeSig]] = []
            for s in raw:
                pc = _strict_interior_full_placement_count(s, fg, h, w)
                if pc > 0:
                    scored.append((-pc, s))
            scored.sort(key=lambda t: (t[0], t[1]))
            for _neg_pc, sig in scored:
                anchors = _try_infer_anchors_for_proto(inp, color, sig, fg)
                if anchors is not None:
                    return sig, anchors

    raise ValueError("no prototype (2–10 cells, orth or king) tiles the foreground")


def _pattern_fall_output(
    inp: Grid,
    color: int,
    proto_sig: ShapeSig,
    anchors: list[tuple[int, int]] | None = None,
) -> Grid:
    """Stamp each inferred anchor placement downward by the prototype height (authoritative rule)."""
    h = len(inp)
    w = len(inp[0]) if h else 0
    if anchors is None:
        anchors = _infer_anchors_for_proto(inp, color, proto_sig)
    shape_h = max(r for r, _ in proto_sig) + 1
    anchors = sorted(set(anchors))
    out = _empty(h, w)
    for top, left in anchors:
        _draw_shape(out, color, proto_sig, top + shape_h, left)
    return out


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

    fg = {(r, c) for r in range(h) for c in range(w) if inp[r][c] == color}
    if not fg:
        raise ValueError("no foreground components found")

    proto_sig, anchors = _resolve_proto_signature(inp, color)
    return _pattern_fall_output(inp, color, proto_sig, anchors)


def _random_connected_shape(rr: random.Random, height: int, width: int) -> ShapeSig:
    cols: list[int] = [rr.randint(0, width - 1)]
    for _r in range(1, height):
        prev = cols[-1]
        cands = [c for c in (prev - 1, prev, prev + 1) if 0 <= c < width]
        cols.append(rr.choice(cands))

    cells: set[Pos] = {(r, c) for r, c in enumerate(cols)}
    max_cells = 10
    max_extra = max(0, max_cells - len(cells))
    extra_n = rr.randint(0, min(max(2, height + width), max_extra))
    for _ in range(extra_n):
        r = rr.randint(0, height - 1)
        c = rr.randint(0, width - 1)
        if len(cells) >= max_cells:
            break
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

    # Enough placement draws per candidate grid; low values waste outer iterations.
    for _ in range(1000):
        h = rr.randint(min_h, max_h)
        w = rr.randint(min_w, max_w)
        color = rr.choice([1, 2, 3, 4, 6, 7, 8, 9])

        shape_h = rr.randint(2, min(6, h - 4))
        shape_w = rr.randint(2, min(6, w - 4))
        sig = _random_connected_shape(rr, shape_h, shape_w)
        if not (2 <= len(sig) <= 10):
            continue

        n_instances = rr.randint(6, 16)
        anchors: list[tuple[int, int]] = []
        occupied: set[Pos] = set()
        top_cut_count = 0
        bot_cut_count = 0
        left_cut_count = 0
        right_cut_count = 0
        full_count = 0

        for _j in range(n_instances * 14):
            if len(anchors) >= n_instances:
                break
            top = rr.randint(-shape_h + 1, h - 1)
            left = rr.randint(-shape_w + 1, w - 1)
            visible = _visible_from_anchor(sig, top, left, h, w)
            if not visible:
                continue
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
