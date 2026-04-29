"""Generator for Test2/t47: XOR two halves separated by a gray line.

Rule:
- The input consists of two equal-size halves separated by a full gray (5) line.
- The halves share one background color.
- Each half has exactly one non-background foreground color.
- For each corresponding location, output:
    - the left/top foreground color if only the first half has foreground there;
    - the right/bottom foreground color if only the second half has foreground there;
    - 0 if neither or both halves have foreground there.

Generation constraints used to keep examples well-formed and unambiguous:
- exactly one full gray separator row or column is present;
- the separator splits the grid into two equal-size halves;
- both halves use the same background color;
- each half has exactly one distinct foreground color;
- both foreground colors are different from the background and gray;
- generated examples include cells of both foreground colors and some cancellations.
"""

from __future__ import annotations

import json
import random
from typing import List

Grid = List[List[int]]

BG_OUT = 0
GRAY = 5
COLORS = list(range(1, 10))


def _empty(h: int, w: int, color: int = BG_OUT) -> Grid:
    return [[color for _ in range(w)] for _ in range(h)]


def _rectangular(grid: Grid) -> bool:
    return bool(grid) and bool(grid[0]) and all(len(row) == len(grid[0]) for row in grid)


def _find_separator(grid: Grid) -> tuple[str, int]:
    """Return ('vertical', col) or ('horizontal', row) for the unique gray separator."""
    if not _rectangular(grid):
        raise ValueError("input grid must be non-empty and rectangular")
    h, w = len(grid), len(grid[0])

    vertical = [c for c in range(w) if all(grid[r][c] == GRAY for r in range(h))]
    horizontal = [r for r in range(h) if all(grid[r][c] == GRAY for c in range(w))]

    candidates: list[tuple[str, int]] = []
    for c in vertical:
        if c > 0 and w - c - 1 == c:
            candidates.append(("vertical", c))
    for r in horizontal:
        if r > 0 and h - r - 1 == r:
            candidates.append(("horizontal", r))

    if len(candidates) != 1:
        raise ValueError("expected exactly one centered full gray separator")
    return candidates[0]


def _split_halves(grid: Grid) -> tuple[Grid, Grid]:
    kind, idx = _find_separator(grid)
    if kind == "vertical":
        left = [row[:idx] for row in grid]
        right = [row[idx + 1 :] for row in grid]
        return left, right

    top = grid[:idx]
    bottom = grid[idx + 1 :]
    return [row[:] for row in top], [row[:] for row in bottom]


def _infer_colors(a: Grid, b: Grid) -> tuple[int, int, int]:
    colors_a = {v for row in a for v in row}
    colors_b = {v for row in b for v in row}
    shared = (colors_a & colors_b) - {GRAY}
    if len(shared) != 1:
        raise ValueError("expected exactly one shared background color")
    bg = next(iter(shared))

    fg_a = sorted(colors_a - {bg, GRAY})
    fg_b = sorted(colors_b - {bg, GRAY})
    if len(fg_a) != 1 or len(fg_b) != 1:
        raise ValueError("expected exactly one foreground color per half")
    if fg_a[0] == fg_b[0]:
        raise ValueError("foreground colors should be distinct")
    return bg, fg_a[0], fg_b[0]


def transform_t47(grid: Grid) -> Grid:
    a, b = _split_halves(grid)
    h, w = len(a), len(a[0])
    if len(b) != h or len(b[0]) != w:
        raise ValueError("halves must have matching dimensions")

    bg, fg_a, fg_b = _infer_colors(a, b)
    out = _empty(h, w, BG_OUT)

    for r in range(h):
        for c in range(w):
            av = a[r][c]
            bv = b[r][c]
            if av not in (bg, fg_a) or bv not in (bg, fg_b):
                raise ValueError("unexpected color in half")
            has_a = av == fg_a
            has_b = bv == fg_b
            if has_a and not has_b:
                out[r][c] = fg_a
            elif has_b and not has_a:
                out[r][c] = fg_b
            else:
                out[r][c] = BG_OUT
    return out


def _render_vertical(a: Grid, b: Grid) -> Grid:
    return [ra + [GRAY] + rb for ra, rb in zip(a, b)]


def _render_horizontal(a: Grid, b: Grid) -> Grid:
    width = len(a[0])
    return [row[:] for row in a] + [[GRAY for _ in range(width)]] + [row[:] for row in b]


def _choose_colors(rng: random.Random) -> tuple[int, int, int]:
    bg = rng.choice([c for c in COLORS if c != GRAY])
    remaining = [c for c in COLORS if c not in (bg, GRAY)]
    fg_a, fg_b = rng.sample(remaining, 2)
    return bg, fg_a, fg_b


def _random_pattern(
    rng: random.Random,
    h: int,
    w: int,
    bg: int,
    fg_a: int,
    fg_b: int,
) -> tuple[Grid, Grid]:
    a = _empty(h, w, bg)
    b = _empty(h, w, bg)

    forced = ["a", "b", "both", "none"]
    cells = [(r, c) for r in range(h) for c in range(w)]
    rng.shuffle(cells)

    for state, (r, c) in zip(forced, cells):
        if state in ("a", "both"):
            a[r][c] = fg_a
        if state in ("b", "both"):
            b[r][c] = fg_b

    for r, c in cells[len(forced) :]:
        state = rng.choices(
            ["none", "a", "b", "both"],
            weights=[4, 5, 5, 3],
            k=1,
        )[0]
        if state in ("a", "both"):
            a[r][c] = fg_a
        if state in ("b", "both"):
            b[r][c] = fg_b

    return a, b


def generate(
    rng: random.Random | None = None,
    h: int | None = None,
    w: int | None = None,
    orientation: str | None = None,
    background: int | None = None,
    fg_a: int | None = None,
    fg_b: int | None = None,
) -> dict[str, Grid]:
    """Generate one valid ARC-style example."""
    rng = rng or random.Random()

    for _attempt in range(1000):
        hh = h if h is not None else rng.randint(3, 12)
        ww = w if w is not None else rng.randint(3, 12)
        if hh < 1 or ww < 1:
            continue

        if background is None or fg_a is None or fg_b is None:
            bg, ca, cb = _choose_colors(rng)
            if background is not None:
                bg = background
            if fg_a is not None:
                ca = fg_a
            if fg_b is not None:
                cb = fg_b
        else:
            bg, ca, cb = background, fg_a, fg_b

        if len({bg, ca, cb, GRAY}) != 4:
            continue

        a, b = _random_pattern(rng, hh, ww, bg, ca, cb)
        orient = orientation if orientation is not None else rng.choice(["vertical", "horizontal"])
        if orient == "vertical":
            inp = _render_vertical(a, b)
        elif orient == "horizontal":
            inp = _render_horizontal(a, b)
        else:
            raise ValueError("orientation must be 'vertical' or 'horizontal'")

        out = transform_t47(inp)
        out_colors = {v for row in out for v in row}
        both_count = sum(1 for r in range(hh) for c in range(ww) if a[r][c] == ca and b[r][c] == cb)
        neither_count = sum(1 for r in range(hh) for c in range(ww) if a[r][c] == bg and b[r][c] == bg)
        if ca in out_colors and cb in out_colors and BG_OUT in out_colors and both_count > 0 and neither_count > 0:
            return {"input": inp, "output": out}

    raise RuntimeError("failed to generate a valid t47 sample")


if __name__ == "__main__":
    print(json.dumps(generate(rng=random.Random(0)), indent=2))
