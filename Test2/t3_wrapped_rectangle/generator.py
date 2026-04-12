"""Generator for Test2/t3: extract wrapped rectangle interior."""

from __future__ import annotations

import random
from typing import List

Grid = List[List[int]]


def _empty(h: int, w: int) -> Grid:
    return [[0 for _ in range(w)] for _ in range(h)]


def _random_content(rng: random.Random, h: int, w: int, frame_color: int) -> Grid:
    out = _empty(h, w)
    for r in range(h):
        for c in range(w):
            # Interior is allowed to use the same color as the frame.
            if rng.random() < 0.2:
                out[r][c] = frame_color
            else:
                out[r][c] = rng.randint(0, 9)
    return out


def _pick_wrapped_start(
    rng: random.Random,
    span: int,
    limit: int,
    require_interior_on_both_sides: bool = True,
) -> int:
    """Pick start so the interval wraps exactly once on this axis."""
    lo = limit - span + 1
    hi = limit - 1
    if lo > hi:
        raise ValueError("span too large for wrapped placement")
    for _ in range(200):
        start = rng.randint(lo, hi)
        first_len = limit - start
        second_len = span - first_len
        if not require_interior_on_both_sides:
            return start
        # Each side should include at least one interior row/col, not only border.
        if first_len >= 2 and second_len >= 2:
            return start
    raise ValueError("could not find valid wrapped start")


def _paint_blob_noise(
    grid: Grid,
    protected: set[tuple[int, int]],
    rng: random.Random,
    target_cells: int,
) -> None:
    h = len(grid)
    w = len(grid[0]) if h else 0
    if h == 0 or w == 0 or target_cells <= 0:
        return
    free = [(r, c) for r in range(h) for c in range(w) if (r, c) not in protected]
    free_set = set(free)
    if not free:
        return
    placed = 0
    while placed < target_cells and free:
        seed = rng.choice(free)
        color = rng.randint(0, 9)
        blob_size = rng.randint(3, 9)
        frontier = [seed]
        seen = {seed}
        for _ in range(blob_size):
            if not frontier:
                break
            r, c = frontier.pop(rng.randrange(len(frontier)))
            if (r, c) in protected:
                continue
            if (r, c) not in free_set:
                continue
            grid[r][c] = color
            free_set.remove((r, c))
            placed += 1
            if placed >= target_cells:
                break
            # Local growth gives larger same-color neighborhoods.
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    if (nr, nc) not in protected and rng.random() < 0.8:
                        frontier.append((nr, nc))
        free = [rc for rc in free if rc in free_set]


def generate(
    width: int | None = None,
    height: int | None = None,
    content_width: int | None = None,
    content_height: int | None = None,
    top: int | None = None,
    left: int | None = None,
    frame_color: int | None = None,
    noise_cells: int | None = None,
    content: Grid | None = None,
    rng: random.Random | None = None,
) -> dict[str, Grid]:
    """Return one input/output pair.

    Input: noisy grid with one outlined rectangle that wraps over edges.
    Output: only the rectangle's inner content (outline removed).
    """
    rr = rng if rng is not None else random
    if width is None:
        width = rr.randint(8, 14)
    if height is None:
        height = rr.randint(8, 14)
    if width < 7 or height < 7:
        raise ValueError("width and height must be >= 7")

    # Bias toward smaller rectangles for better mimicry.
    max_cw = min(width - 3, 6)
    max_ch = min(height - 3, 6)
    if content_width is None:
        content_width = rr.randint(2, max_cw)
    if content_height is None:
        content_height = rr.randint(2, max_ch)
    if not (2 <= content_width <= width - 2 and 2 <= content_height <= height - 2):
        raise ValueError("content size out of range")

    frame_h = content_height + 2
    frame_w = content_width + 2

    if frame_color is None:
        frame_color = rr.randint(1, 9)
    if content is None:
        content = _random_content(rr, content_height, content_width, frame_color)
    if len(content) != content_height or any(len(row) != content_width for row in content):
        raise ValueError("content shape must match content size")

    # Exactly one-axis wrap (XOR), with content present on both split halves.
    if top is None or left is None:
        wrap_row = rr.choice([True, False])
        wrap_col = not wrap_row
        if top is None:
            if wrap_row:
                top = _pick_wrapped_start(rr, frame_h, height)
            else:
                top = rr.randint(0, height - frame_h)
        if left is None:
            if wrap_col:
                left = _pick_wrapped_start(rr, frame_w, width)
            else:
                left = rr.randint(0, width - frame_w)
    assert top is not None and left is not None

    wraps_row = (top + frame_h - 1) >= height
    wraps_col = (left + frame_w - 1) >= width
    if wraps_row == wraps_col:
        raise ValueError("rectangle must wrap across exactly one axis")
    if wraps_row:
        first_len = height - top
        second_len = frame_h - first_len
        if first_len < 2 or second_len < 2:
            raise ValueError("row split must leave content on both halves")
    if wraps_col:
        first_len = width - left
        second_len = frame_w - first_len
        if first_len < 2 or second_len < 2:
            raise ValueError("column split must leave content on both halves")

    inp = _empty(height, width)
    frame_cells: set[tuple[int, int]] = set()
    all_rect_cells: set[tuple[int, int]] = set()
    for r in range(frame_h):
        for c in range(frame_w):
            gr = (top + r) % height
            gc = (left + c) % width
            all_rect_cells.add((gr, gc))
            if r == 0 or c == 0 or r == frame_h - 1 or c == frame_w - 1:
                inp[gr][gc] = frame_color
                frame_cells.add((gr, gc))
            else:
                inp[gr][gc] = content[r - 1][c - 1]

    # Add post-hoc blob-like noise without touching the rectangle outline.
    if noise_cells is None:
        noise_cells = rr.randint((width * height) // 9, (width * height) // 4)
    protected = set(all_rect_cells)
    _paint_blob_noise(inp, protected, rr, noise_cells)

    return {"input": inp, "output": [list(row) for row in content]}
