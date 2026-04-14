"""Generator for Test2/t8 enclosed-body red gravity task."""

from __future__ import annotations

import copy
import random
from typing import List

Grid = List[List[int]]

BLUE = 1
RED = 2
GREY = 5


def _empty(h: int, w: int) -> Grid:
    return [[0 for _ in range(w)] for _ in range(h)]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _find_grey_component(inp: Grid) -> set[tuple[int, int]]:
    h = len(inp)
    w = len(inp[0]) if h else 0
    greys = [(r, c) for r in range(h) for c in range(w) if inp[r][c] == GREY]
    if not greys:
        raise ValueError("no grey cells")
    seen: set[tuple[int, int]] = set()
    comp: set[tuple[int, int]] = set()
    stack = [greys[0]]
    seen.add(greys[0])
    while stack:
        r, c = stack.pop()
        comp.add((r, c))
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and inp[nr][nc] == GREY:
                seen.add((nr, nc))
                stack.append((nr, nc))
    if len(comp) != len(greys):
        raise ValueError("grey shape must be a single 8-connected component")
    return comp


def _opening_seed_cells(inp: Grid, grey: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h = len(inp)
    top = min(r for r, _ in grey)
    cols = [c for r, c in grey if r == top]
    if not cols:
        return []
    lo, hi = min(cols), max(cols)
    seeds: list[tuple[int, int]] = []
    if top + 1 >= h:
        return seeds
    for c in range(lo, hi + 1):
        if inp[top][c] == GREY:
            continue
        if inp[top + 1][c] in (BLUE, RED):
            seeds.append((top + 1, c))
    return seeds


def _enclosed_component(inp: Grid) -> set[tuple[int, int]]:
    grey = _find_grey_component(inp)
    seeds = _opening_seed_cells(inp, grey)
    if not seeds:
        raise ValueError("could not find enclosed opening seeds")
    h = len(inp)
    w = len(inp[0]) if h else 0
    allowed = {(r, c) for r in range(h) for c in range(w) if inp[r][c] in (BLUE, RED)}
    seen: set[tuple[int, int]] = set()
    stack = list(seeds)
    for s in seeds:
        seen.add(s)
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in allowed and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    if not seen:
        raise ValueError("empty enclosed component")
    return seen


def _is_4_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    return (
        min(r for r, _ in cells),
        max(r for r, _ in cells),
        min(c for _, c in cells),
        max(c for _, c in cells),
    )


def transform_t8(inp: Grid) -> Grid:
    h = len(inp)
    w = len(inp[0]) if h else 0
    if h == 0 or w == 0 or any(len(row) != w for row in inp):
        raise ValueError("input must be a non-empty rectangular grid")
    enclosed = _enclosed_component(inp)

    body_enclosed = {(r, c) for (r, c) in enclosed if inp[r][c] in (BLUE, RED)}
    if not body_enclosed:
        raise ValueError("no enclosed body cells")
    min_r = min(r for r, _ in body_enclosed)
    max_r = max(r for r, _ in body_enclosed)
    min_c = min(c for _, c in body_enclosed)
    max_c = max(c for _, c in body_enclosed)
    oh = max_r - min_r + 1
    ow = max_c - min_c + 1
    out = _empty(oh, ow)

    # Column-wise gravity for enclosed red cells into the extracted dimensions.
    for c in range(min_c, max_c + 1):
        reds = sum(
            1
            for r in range(min_r, max_r + 1)
            if (r, c) in enclosed and inp[r][c] == RED
        )
        for k in range(reds):
            out[oh - 1 - k][c - min_c] = RED
    return out


def _paint_body(inp: Grid, top: int, rng: random.Random) -> None:
    h = len(inp)
    w = len(inp[0]) if h else 0
    for r in range(top, h):
        for c in range(w):
            inp[r][c] = BLUE


def _paint_enclosure(
    inp: Grid,
    *,
    top: int,
    left_tip: int,
    right_tip: int,
    depth: int,
    rng: random.Random,
) -> None:
    """Paint non-rectangular 8-connected grey enclosure.

    The topmost grey tips are on the first blue row (*top*), matching task style.
    """
    h = len(inp)
    w = len(inp[0]) if h else 0
    if not (0 <= top < h and 1 <= left_tip < right_tip <= w - 2):
        raise ValueError("invalid enclosure anchors")

    inp[top][left_tip] = GREY
    inp[top][right_tip] = GREY

    # Build two side arms descending from the top tips.
    # Keep them single-cell thick to avoid doubled/protruding grey artifacts.
    lcol = left_tip
    rcol = right_tip
    left_path: list[tuple[int, int]] = [(top, lcol)]
    right_path: list[tuple[int, int]] = [(top, rcol)]
    for step in range(1, depth + 1):
        r = top + step
        if r >= h:
            break
        # Mild slant, mostly vertical; no side bulges.
        lcol += rng.choice([0, 0, -1, 1])
        rcol += rng.choice([0, 0, -1, 1])
        lcol = max(1, min(w - 3, lcol))
        rcol = max(lcol + 2, min(w - 2, rcol))
        left_path.append((r, lcol))
        right_path.append((r, rcol))
        inp[r][lcol] = GREY
        inp[r][rcol] = GREY

    # Bottom connector: clean horizontal span on the arm depth row.
    br = left_path[-1][0]
    lc = left_path[-1][1]
    rc = right_path[-1][1]
    if br >= h:
        br = h - 1
    for c in range(lc, rc + 1):
        inp[br][c] = GREY


def generate(
    width: int | None = None,
    height: int | None = None,
    rng: random.Random | None = None,
) -> dict[str, Grid]:
    rr = rng if rng is not None else random
    if width is None:
        width = rr.randint(9, 13)
    if height is None:
        height = rr.randint(9, 13)
    if width < 9 or height < 9:
        raise ValueError("width and height must be >= 9")

    for _ in range(700):
        inp = _empty(height, width)
        body_top = rr.randint(2, max(2, height // 2 - 1))
        _paint_body(inp, body_top, rr)

        max_depth = min(8, height - body_top - 2)
        if max_depth < 4:
            continue
        depth = rr.randint(4, max_depth)
        span = rr.randint(3, min(6, width - 3))
        max_left_tip = width - span - 2
        if max_left_tip < 1:
            continue
        left_tip = rr.randint(1, max_left_tip)
        right_tip = left_tip + span
        _paint_enclosure(
            inp,
            top=body_top,
            left_tip=left_tip,
            right_tip=right_tip,
            depth=depth,
            rng=rr,
        )

        # Add red cells strictly within the enclosed blue/red component only.
        try:
            enclosed_now = _enclosed_component(inp)
        except Exception:
            continue
        enclosed_blue = [(r, c) for r, c in enclosed_now if inp[r][c] == BLUE]
        if len(enclosed_blue) < 3:
            continue
        blue_set_before = set(enclosed_blue)
        if not _is_4_connected(blue_set_before):
            continue
        bbox_before = _bbox(blue_set_before)

        # Place red cells while preserving 4-connectivity and blue bbox.
        candidates = list(blue_set_before)
        rr.shuffle(candidates)
        want_red = rr.randint(3, min(10, len(candidates)))
        placed: list[tuple[int, int]] = []
        current_blue = set(blue_set_before)
        for rc in candidates:
            if len(placed) >= want_red:
                break
            trial_blue = set(current_blue)
            trial_blue.remove(rc)
            if not trial_blue:
                continue
            if not _is_4_connected(trial_blue):
                continue
            if _bbox(trial_blue) != bbox_before:
                continue
            placed.append(rc)
            current_blue = trial_blue
        if len(placed) < 3:
            continue
        for r, c in placed:
            inp[r][c] = RED

        # Re-check user's requested structural constraints on final enclosed blue.
        enclosed_after = _enclosed_component(inp)
        blue_after = {(r, c) for r, c in enclosed_after if inp[r][c] == BLUE}
        if len(blue_after) < 3 or not _is_4_connected(blue_after):
            continue

        try:
            out = transform_t8(inp)
        except Exception:
            continue

        # Ensure the enclosed component is meaningful and output not empty.
        reds_out = sum(v == RED for row in out for v in row)
        if reds_out == 0:
            continue
        return {"input": inp, "output": out}

    raise RuntimeError("failed to sample valid t8 instance")
