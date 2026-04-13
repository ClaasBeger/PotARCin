"""Generator for Test2/t6 partition-distance ranking task."""

from __future__ import annotations

import random
from typing import List

Grid = List[List[int]]

GREY = 5
GREEN = 3
RANK_COLORS = [3, 4, 2]  # closest, second, third
# Only green is forbidden for partition objects.
OBJECT_PALETTE = [1, 2, 4, 6, 7, 8, 9]


def _empty(h: int, w: int) -> Grid:
    return [[0 for _ in range(w)] for _ in range(h)]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_partitions(grid: Grid, top: int, left: int, bottom: int, right: int) -> list[set[tuple[int, int]]]:
    interior = [
        (r, c)
        for r in range(top + 1, bottom)
        for c in range(left + 1, right)
        if grid[r][c] != GREY
    ]
    inside = set(interior)
    seen: set[tuple[int, int]] = set()
    comps: list[set[tuple[int, int]]] = []
    for rc in interior:
        if rc in seen:
            continue
        stack = [rc]
        seen.add(rc)
        comp: set[tuple[int, int]] = set()
        while stack:
            r, c = stack.pop()
            comp.add((r, c))
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in inside and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _paint_outer_rectangle(grid: Grid, top: int, left: int, bottom: int, right: int) -> None:
    for c in range(left, right + 1):
        grid[top][c] = GREY
        grid[bottom][c] = GREY
    for r in range(top + 1, bottom):
        grid[r][left] = GREY
        grid[r][right] = GREY


def _paint_internal_walls(
    grid: Grid,
    top: int,
    left: int,
    bottom: int,
    right: int,
    partition_count: int,
    rng: random.Random,
) -> None:
    ih = bottom - top - 1
    iw = right - left - 1

    if partition_count == 1:
        return

    if partition_count == 2:
        modes: list[str] = ["diag_pos", "diag_neg"]
        if iw >= 5:
            modes.append("vertical")
        if ih >= 5:
            modes.append("horizontal")
        if iw >= 5 and ih >= 5:
            modes.append("zigzag2")
        mode = rng.choice(modes)
        if mode == "vertical":
            cut = rng.randint(2, iw - 2)
            c = left + cut
            for r in range(top + 1, bottom):
                grid[r][c] = GREY
            return
        if mode == "horizontal":
            cut = rng.randint(2, ih - 2)
            r = top + cut
            for c in range(left + 1, right):
                grid[r][c] = GREY
            return
        if mode == "diag_pos":
            # interior coords satisfy (rr - cc == k)
            k = rng.randint(-(iw - 3), ih - 3)
            for rr in range(1, ih + 1):
                for cc in range(1, iw + 1):
                    if rr - cc == k:
                        grid[top + rr][left + cc] = GREY
            return
        if mode == "zigzag2":
            # Staircase-style separating wall.
            base = rng.randint(2, iw - 2)
            for rr in range(1, ih + 1):
                cc = base + (rr % 2)
                if cc > iw - 1:
                    cc = base - (rr % 2)
                cc = max(1, min(iw, cc))
                grid[top + rr][left + cc] = GREY
            return
        # diag_neg: interior coords satisfy (rr + cc == k)
        k = rng.randint(4, ih + iw - 2)
        for rr in range(1, ih + 1):
            for cc in range(1, iw + 1):
                if rr + cc == k:
                    grid[top + rr][left + cc] = GREY
        return

    # partition_count == 3
    modes3: list[str] = []
    if iw >= 6:
        modes3.append("vertical3")
    if ih >= 6:
        modes3.append("horizontal3")
    # Parallel diagonals need enough spread to leave 3 regions.
    if ih + iw >= 10:
        modes3.extend(["diag_pos3", "diag_neg3"])
    if iw >= 8 and ih >= 6:
        modes3.append("zigzag3")
    if not modes3:
        # Fallback: single wall; caller will reject if partition count mismatches.
        return
    mode = rng.choice(modes3)
    if mode == "vertical3":
        a = rng.randint(2, iw - 4)
        b = rng.randint(a + 2, iw - 2)
        for r in range(top + 1, bottom):
            grid[r][left + a] = GREY
            grid[r][left + b] = GREY
        return
    if mode == "horizontal3":
        a = rng.randint(2, ih - 4)
        b = rng.randint(a + 2, ih - 2)
        for c in range(left + 1, right):
            grid[top + a][c] = GREY
            grid[top + b][c] = GREY
        return
    if mode == "diag_pos3":
        k1 = rng.randint(-(iw - 4), ih - 6)
        k2 = rng.randint(k1 + 2, ih - 3)
        for rr in range(1, ih + 1):
            for cc in range(1, iw + 1):
                if rr - cc == k1 or rr - cc == k2:
                    grid[top + rr][left + cc] = GREY
        return
    if mode == "zigzag3":
        # Two parallel staircase walls, similar to the provided test style.
        base1 = rng.randint(2, max(2, iw // 2 - 1))
        base2 = rng.randint(base1 + 3, iw - 1)
        for rr in range(1, ih + 1):
            off = rr % 2
            c1 = base1 + off
            c2 = base2 + off
            if c2 > iw:
                c1 -= 1
                c2 -= 1
            c1 = max(1, min(iw, c1))
            c2 = max(1, min(iw, c2))
            if c2 - c1 < 2:
                c2 = min(iw, c1 + 2)
            grid[top + rr][left + c1] = GREY
            grid[top + rr][left + c2] = GREY
        return
    k1 = rng.randint(4, ih + iw - 6)
    k2 = rng.randint(k1 + 2, ih + iw - 2)
    for rr in range(1, ih + 1):
        for cc in range(1, iw + 1):
            if rr + cc == k1 or rr + cc == k2:
                grid[top + rr][left + cc] = GREY


def _sample_object_cells(part: set[tuple[int, int]], rng: random.Random) -> set[tuple[int, int]]:
    # Build a small connected object (1-4 cells) inside this partition.
    seed = rng.choice(list(part))
    obj = {seed}
    target = rng.randint(1, min(4, len(part)))
    frontier = [seed]
    while len(obj) < target and frontier:
        r, c = frontier.pop(rng.randrange(len(frontier)))
        candidates = [(nr, nc) for nr, nc in _neighbors4(r, c) if (nr, nc) in part and (nr, nc) not in obj]
        rng.shuffle(candidates)
        for rc in candidates[:2]:
            obj.add(rc)
            frontier.append(rc)
            if len(obj) >= target:
                break
    return obj


def transform_t6(inp: Grid) -> Grid:
    h = len(inp)
    w = len(inp[0]) if h else 0
    if any(len(row) != w for row in inp):
        raise ValueError("ragged input grid")

    grey_cells = [(r, c) for r in range(h) for c in range(w) if inp[r][c] == GREY]
    if not grey_cells:
        raise ValueError("no grey rectangle found")
    top = min(r for r, _ in grey_cells)
    bottom = max(r for r, _ in grey_cells)
    left = min(c for _, c in grey_cells)
    right = max(c for _, c in grey_cells)

    green_cells = [(r, c) for r in range(h) for c in range(w) if inp[r][c] == GREEN]
    if len(green_cells) != 1:
        raise ValueError("expected exactly one green cell")
    gr, gc = green_cells[0]

    parts = _find_partitions(inp, top, left, bottom, right)
    part_info: list[tuple[int, set[tuple[int, int]]]] = []
    for part in parts:
        objs = [(r, c) for r, c in part if inp[r][c] not in (0, GREY, GREEN)]
        if not objs:
            raise ValueError("each partition must contain one object")
        d = min(abs(r - gr) + abs(c - gc) for r, c in objs)
        part_info.append((d, part))
    part_info.sort(key=lambda x: x[0])

    out = _empty(h, w)
    for r, c in grey_cells:
        out[r][c] = GREY
    for i, (_, part) in enumerate(part_info):
        col = RANK_COLORS[min(i, len(RANK_COLORS) - 1)]
        for r, c in part:
            out[r][c] = col
    return out


def generate(
    width: int | None = None,
    height: int | None = None,
    partition_count: int | None = None,
    rng: random.Random | None = None,
) -> dict[str, Grid]:
    rr = rng if rng is not None else random
    if width is None:
        width = rr.randint(10, 14)
    if height is None:
        height = rr.randint(10, 14)
    if width < 9 or height < 9:
        raise ValueError("width and height must be >= 9")

    if partition_count is None:
        # Bias toward richer examples: 3 partitions most likely, then 2, then 1.
        partition_count = rr.choices([1, 2, 3], weights=[1, 2, 9], k=1)[0]
    if partition_count not in (1, 2, 3):
        raise ValueError("partition_count must be in {1,2,3}")

    for _ in range(300):
        inp = _empty(height, width)

        rect_h = rr.randint(7, min(height - 1, 11))
        rect_w = rr.randint(7, min(width - 1, 11))
        top = rr.randint(1, height - rect_h)
        left = rr.randint(1, width - rect_w)
        bottom = top + rect_h - 1
        right = left + rect_w - 1

        _paint_outer_rectangle(inp, top, left, bottom, right)
        _paint_internal_walls(inp, top, left, bottom, right, partition_count, rr)

        parts = _find_partitions(inp, top, left, bottom, right)
        if len(parts) != partition_count:
            continue
        if any(len(p) < 3 for p in parts):
            continue

        # One object per partition.
        used_colors = rr.sample(OBJECT_PALETTE, k=partition_count)
        for part, color in zip(parts, used_colors):
            for r, c in _sample_object_cells(part, rr):
                inp[r][c] = color

        # One green target may be outside or inside the grey shape.
        outside = {
            (r, c)
            for r in range(height)
            for c in range(width)
            if not (top <= r <= bottom and left <= c <= right)
        }
        inside_open = {
            (r, c)
            for r in range(top + 1, bottom)
            for c in range(left + 1, right)
            if inp[r][c] == 0
        }
        outside_list = list(outside)
        inside_list = list(inside_open)
        if not outside_list and not inside_list:
            continue
        # Bias placement outside the grey shape; inside remains possible but less likely.
        if outside_list and inside_list:
            pick_outside = rr.random() < 0.8
            gr, gc = rr.choice(outside_list if pick_outside else inside_list)
        elif outside_list:
            gr, gc = rr.choice(outside_list)
        else:
            gr, gc = rr.choice(inside_list)
        inp[gr][gc] = GREEN

        # Enforce unique distances to avoid tie ambiguity.
        dists = []
        for part in parts:
            objs = [(r, c) for r, c in part if inp[r][c] not in (0, GREY, GREEN)]
            d = min(abs(r - gr) + abs(c - gc) for r, c in objs)
            dists.append(d)
        if len(set(dists)) != len(dists):
            continue

        out = transform_t6(inp)
        return {"input": inp, "output": out}

    raise RuntimeError("failed to sample valid t6 instance")
