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


def _pack_component(
    comp: set[tuple[int, int]], count: int, preferred_center: float | None = None
) -> set[tuple[int, int]]:
    if count <= 0:
        return set()
    rows = sorted({r for r, _ in comp}, reverse=True)
    row_cells = {r: sorted([c for rr, c in comp if rr == r]) for r in rows}
    bottom = rows[0]

    filled: set[tuple[int, int]] = set()
    remaining = count

    # Bottom-up fill; each row selects a single flow-front segment and fills from
    # the middle. This avoids non-physical "teleport" fills into side cavities on
    # the same level.
    for r in rows:
        if remaining <= 0:
            break
        cols = row_cells[r]
        if not cols:
            continue
        segments: list[list[int]] = []
        i = 0
        while i < len(cols):
            j = i
            while j + 1 < len(cols) and cols[j + 1] == cols[j] + 1:
                j += 1
            seg = cols[i : j + 1]
            segments.append(seg)
            i = j + 1

        if not segments:
            continue

        if r == bottom:
            global_center = preferred_center if preferred_center is not None else (cols[0] + cols[-1]) / 2.0
            chosen_seg = min(segments, key=lambda seg: (abs(((seg[0] + seg[-1]) / 2.0) - global_center), seg[0]))
            order = _center_order(chosen_seg)
        else:
            anchored: list[tuple[int, list[int], list[int]]] = []
            for seg in segments:
                anchors = [c for c in seg if (r + 1, c) in filled]
                if anchors:
                    anchored.append((len(anchors), seg, anchors))
            if not anchored:
                continue
            # Prefer the segment with the strongest support from below.
            anchored.sort(
                key=lambda t: (
                    -t[0],
                    abs(
                        ((t[1][0] + t[1][-1]) / 2.0)
                        - (preferred_center if preferred_center is not None else ((cols[0] + cols[-1]) / 2.0))
                    ),
                    t[1][0],
                )
            )
            _, chosen_seg, anchors = anchored[0]
            # Center from underlying cyan support row (anchors), not full row span.
            center = sum(anchors) / float(len(anchors))
            order = sorted(
                chosen_seg,
                key=lambda c: (abs(c - center), min(abs(c - a) for a in anchors), c),
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


def _flow_component(
    comp: set[tuple[int, int]], initial_cyan: set[tuple[int, int]], steps_cap: int = 2000
) -> set[tuple[int, int]]:
    """Simulate deterministic gravity/flow inside one non-orange component."""
    if not initial_cyan:
        return set()
    by_row: dict[int, set[int]] = {}
    for r, c in comp:
        by_row.setdefault(r, set()).add(c)

    def row_segment(r: int, c: int) -> tuple[int, int] | None:
        row = by_row.get(r)
        if row is None or c not in row:
            return None
        lo = c
        hi = c
        while (lo - 1) in row:
            lo -= 1
        while (hi + 1) in row:
            hi += 1
        return (lo, hi)

    occ = set(initial_cyan)
    global_center = sum(c for _, c in comp) / float(len(comp))

    steps = 0
    changed = True
    while changed and steps < steps_cap:
        steps += 1
        changed = False
        # Bottom-up update reduces oscillations and keeps settling stable.
        order = sorted(occ, key=lambda rc: (-rc[0], abs(rc[1] - global_center), rc[1]))
        for r, c in order:
            if (r, c) not in occ:
                continue
            # 1) Pure gravity: fall straight down whenever possible.
            below = (r + 1, c)
            if below in comp and below not in occ:
                occ.remove((r, c))
                occ.add(below)
                changed = True
                continue
            # Lateral travel is only on solid support (orange/boundary below),
            # not when simply stacked on another cyan.
            if below in comp:
                continue

            # 2) Blocked below: move laterally toward a reachable drop if any.
            seg = row_segment(r, c)
            if seg is None:
                continue
            lo, hi = seg

            drop_targets: list[int] = []
            for tc in range(lo, hi + 1):
                if (r, tc) in occ and tc != c:
                    continue
                tb = (r + 1, tc)
                if tb in comp and tb not in occ:
                    drop_targets.append(tc)

            moved = False
            if drop_targets:
                best_tc = min(
                    drop_targets,
                    key=lambda tc: (abs(tc - c), abs(tc - global_center), tc),
                )
                step = 0 if best_tc == c else (1 if best_tc > c else -1)
                if step != 0:
                    nxt = (r, c + step)
                    if nxt in comp and nxt not in occ:
                        occ.remove((r, c))
                        occ.add(nxt)
                        changed = True
                        moved = True
            if moved:
                continue

            # 3) No reachable drop: settle on the current solid lane from middle.
            seg_center = (lo + hi) / 2.0
            dirs = sorted(
                [-1, 1],
                key=lambda dc: (
                    abs((c + dc) - seg_center),
                    abs((c + dc) - global_center),
                    c + dc,
                ),
            )
            for dc in dirs:
                nxt = (r, c + dc)
                if nxt in comp and nxt not in occ:
                    occ.remove((r, c))
                    occ.add(nxt)
                    changed = True
                    break
    return occ


def transform_t5(inp: Grid) -> Grid:
    h = len(inp)
    w = len(inp[0]) if h else 0
    out = _empty(h, w)
    for r in range(h):
        for c in range(w):
            if inp[r][c] == ORANGE:
                out[r][c] = ORANGE

    for comp in _non_orange_components(inp):
        cyan_cols = [c for r, c in comp if inp[r][c] == CYAN]
        n_cyan = len(cyan_cols)
        preferred_center = (sum(cyan_cols) / float(n_cyan)) if n_cyan > 0 else None
        for r, c in _pack_component(comp, n_cyan, preferred_center=preferred_center):
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

    # Keep one main horizontal barrier only; stacked barriers can produce
    # non-physical-looking fills under a gravity interpretation.

    # Add one or two vertical walls for richer basin geometry/cavities.
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
    """Reject generated samples with unresolved center-fill side ambiguity."""
    h = len(inp)
    for comp in _non_orange_components(inp):
        segs_by_row = _row_segments(comp)
        for r, segs in segs_by_row.items():
            for seg in segs:
                chosen = [c for c in seg if out[r][c] == CYAN]
                if not chosen or len(chosen) == len(seg):
                    continue
                # Partial fills are validated against deterministic center ranking
                # from the immediate supporting row.
                support = [c for c in seg if r + 1 < h and out[r + 1][c] == CYAN]
                if not support:
                    return True
                center = sum(support) / float(len(support))
                ordered = sorted(seg, key=lambda c: (abs(c - center), c))
                k = len(chosen)
                # If the boundary between chosen and unchosen has equal distance
                # to center, then side choice is ambiguous.
                if 0 < k < len(ordered):
                    d_k = abs(ordered[k - 1] - center)
                    d_next = abs(ordered[k] - center)
                    if abs(d_k - d_next) < 1e-9:
                        return True
                if set(chosen) != set(ordered[:k]):
                    return True
                # If an unchosen column in this segment has a free drop below
                # (inside component and not already occupied), dynamics would be
                # ambiguous between "center now" vs "slide then drop".
                if r + 1 < h:
                    for c in seg:
                        if c in chosen:
                            continue
                        if (r + 1, c) in comp and out[r + 1][c] != CYAN:
                            return True
    return False


def _has_multi_basin_drop_ambiguity(inp: Grid) -> bool:
    """Reject layouts with multiple simultaneously fed bottom basins.

    If a non-orange component has a split row near its bottom (multiple disjoint
    segments) and more than one segment has cyan somewhere above it in-column,
    the intended drop destination can be read ambiguously.
    """
    for comp in _non_orange_components(inp):
        if not comp:
            continue
        by_row = _row_segments(comp)
        rows = sorted(by_row.keys(), reverse=True)  # bottom-up
        # Focus on the lowest part of each cavity where ambiguity is most visible.
        for r in rows[:3]:
            segs = by_row[r]
            if len(segs) <= 1:
                continue
            fed = 0
            for seg in segs:
                cols = set(seg)
                has_source = any(
                    inp[rr][cc] == CYAN
                    for rr, cc in comp
                    if rr < r and cc in cols
                )
                if has_source:
                    fed += 1
            if fed >= 2:
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
            frac = rr.uniform(0.2, 0.5)
            cells = [
                (r, c)
                for (r, c) in comp
                if (r, c) not in reserved_no_cyan and not (r == barrier_r and c in gap_cols)
            ]
            # Avoid seeding directly ambiguous local states:
            # cyan above orange with both lateral routes open.
            safe_cells: list[tuple[int, int]] = []
            for r, c in cells:
                if r + 1 < height and inp[r + 1][c] == ORANGE:
                    continue
                safe_cells.append((r, c))
            if not safe_cells:
                continue
            # Favor sources above/near the first barrier and near its opening(s),
            # so motion reads like mostly vertical falling into the main basin.
            gap_min = min(gap_cols)
            gap_max = max(gap_cols)
            upper_band = [
                rc
                for rc in safe_cells
                if rc[0] <= barrier_r + 1 and (gap_min - 2) <= rc[1] <= (gap_max + 2)
            ]
            # If filtering is too strict for a particular component, relax gently.
            if len(upper_band) < max(1, int(0.5 * len(safe_cells))):
                upper_band = [rc for rc in safe_cells if rc[0] <= barrier_r + 1]

            n = max(1, min(len(safe_cells), int(round(len(safe_cells) * frac))))
            upper_target = n
            chosen: list[tuple[int, int]] = []

            rr.shuffle(upper_band)
            chosen.extend(upper_band[: min(len(upper_band), upper_target)])

            remaining = n - len(chosen)
            if remaining > 0:
                # Top up from just-above-barrier cells before using deeper cells.
                chosen_set = set(chosen)
                near_band = [rc for rc in safe_cells if rc[0] <= barrier_r + 2 and rc not in chosen_set]
                rr.shuffle(near_band)
                chosen.extend(near_band[:remaining])

            if len(chosen) < n:
                chosen_set = set(chosen)
                rest = [rc for rc in safe_cells if rc not in chosen_set and rc[0] <= barrier_r + 3]
                rr.shuffle(rest)
                chosen.extend(rest[: n - len(chosen)])

            for r, c in chosen:
                inp[r][c] = CYAN

        out = transform_t5(inp)
        if not _is_ambiguous_fill(inp, out) and not _has_multi_basin_drop_ambiguity(inp):
            return {"input": inp, "output": out}

    raise RuntimeError("failed to sample unambiguous t5 instance")
