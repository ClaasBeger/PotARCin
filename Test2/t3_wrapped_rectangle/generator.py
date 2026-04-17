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


def _map_local_to_global(
    *,
    r: int,
    c: int,
    top: int,
    left: int,
    frame_h: int,
    frame_w: int,
    grid_h: int,
    grid_w: int,
    shift: int,
    wraps_row: bool,
) -> tuple[int, int]:
    """Map local frame coordinates to global grid, allowing wrapped-half shift."""
    if wraps_row:
        base_r = top + r
        if base_r < grid_h:
            return base_r, left + c
        # Wrapped top half is shifted horizontally.
        return base_r - grid_h, left + c + shift

    base_c = left + c
    if base_c < grid_w:
        return top + r, base_c
    # Wrapped left half is shifted vertically.
    return top + r + shift, base_c - grid_w


def generate(
    width: int | None = None,
    height: int | None = None,
    content_width: int | None = None,
    content_height: int | None = None,
    top: int | None = None,
    left: int | None = None,
    frame_color_a: int | None = None,
    frame_color_b: int | None = None,
    half_shift: int | None = None,
    noise_cells: int | None = None,
    content: Grid | None = None,
    rng: random.Random | None = None,
) -> dict[str, Grid]:
    """Return one input/output pair.

    Input: noisy grid with one outlined rectangle that wraps over edges.
    Output: only the rectangle's inner content (outline removed).
    """
    rr = rng if rng is not None else random
    max_attempts = 400
    for _ in range(max_attempts):
        w = width if width is not None else rr.randint(8, 14)
        h = height if height is not None else rr.randint(8, 14)
        if w < 7 or h < 7:
            raise ValueError("width and height must be >= 7")

        max_cw = min(w - 3, 6)
        max_ch = min(h - 3, 6)
        cw = content_width if content_width is not None else rr.randint(2, max_cw)
        ch = content_height if content_height is not None else rr.randint(2, max_ch)
        if not (2 <= cw <= w - 2 and 2 <= ch <= h - 2):
            raise ValueError("content size out of range")
        frame_h = ch + 2
        frame_w = cw + 2

        color_a = frame_color_a if frame_color_a is not None else rr.randint(1, 9)
        color_b = frame_color_b if frame_color_b is not None else rr.randint(1, 9)
        content_grid = (
            [row[:] for row in content] if content is not None else _random_content(rr, ch, cw, color_a)
        )
        if len(content_grid) != ch or any(len(row) != cw for row in content_grid):
            raise ValueError("content shape must match content size")

        if top is None or left is None:
            wrap_row = rr.choice([True, False])
            wrap_col = not wrap_row
            top0 = _pick_wrapped_start(rr, frame_h, h) if wrap_row else rr.randint(0, h - frame_h)
            left0 = _pick_wrapped_start(rr, frame_w, w) if wrap_col else rr.randint(0, w - frame_w)
        else:
            top0, left0 = top, left

        wraps_row = (top0 + frame_h - 1) >= h
        wraps_col = (left0 + frame_w - 1) >= w
        if wraps_row == wraps_col:
            if top is not None and left is not None:
                raise ValueError("rectangle must wrap across exactly one axis")
            continue
        if wraps_row:
            first_len = h - top0
            second_len = frame_h - first_len
            if first_len < 2 or second_len < 2:
                continue
        if wraps_col:
            first_len = w - left0
            second_len = frame_w - first_len
            if first_len < 2 or second_len < 2:
                continue

        def _is_contained_shift(shift_val: int) -> bool:
            if wraps_row:
                return 0 <= left0 + shift_val and left0 + shift_val + frame_w <= w
            return 0 <= top0 + shift_val and top0 + shift_val + frame_h <= h

        def _non_overlapping(shift_val: int) -> bool:
            if not _is_contained_shift(shift_val):
                return False
            coords: set[tuple[int, int]] = set()
            for rr0 in range(frame_h):
                for cc0 in range(frame_w):
                    gr, gc = _map_local_to_global(
                        r=rr0,
                        c=cc0,
                        top=top0,
                        left=left0,
                        frame_h=frame_h,
                        frame_w=frame_w,
                        grid_h=h,
                        grid_w=w,
                        shift=shift_val,
                        wraps_row=wraps_row,
                    )
                    coords.add((gr, gc))
            return len(coords) == frame_h * frame_w

        if half_shift is None:
            if wraps_row:
                min_shift = -left0
                max_shift = w - frame_w - left0
            else:
                min_shift = -top0
                max_shift = h - frame_h - top0
            shifts = list(range(min_shift, max_shift + 1))
            rr.shuffle(shifts)
            shift0 = next((s for s in shifts if _non_overlapping(s)), None)
            if shift0 is None:
                continue
        else:
            shift0 = half_shift
            if not _non_overlapping(shift0):
                if top is not None and left is not None:
                    raise ValueError("provided half_shift causes frame overlap")
                continue

        inp = _empty(h, w)
        all_rect_cells: set[tuple[int, int]] = set()

        def border_color(r: int, c: int) -> int:
            if wraps_row:
                return color_a if (top0 + r) < h else color_b
            return color_a if (left0 + c) < w else color_b

        for rr0 in range(frame_h):
            for cc0 in range(frame_w):
                gr, gc = _map_local_to_global(
                    r=rr0,
                    c=cc0,
                    top=top0,
                    left=left0,
                    frame_h=frame_h,
                    frame_w=frame_w,
                    grid_h=h,
                    grid_w=w,
                    shift=shift0,
                    wraps_row=wraps_row,
                )
                all_rect_cells.add((gr, gc))
                if rr0 == 0 or cc0 == 0 or rr0 == frame_h - 1 or cc0 == frame_w - 1:
                    inp[gr][gc] = border_color(rr0, cc0)
                else:
                    inp[gr][gc] = content_grid[rr0 - 1][cc0 - 1]

        n_noise = noise_cells if noise_cells is not None else rr.randint((w * h) // 9, (w * h) // 4)
        _paint_blob_noise(inp, set(all_rect_cells), rr, n_noise)

        # Ensure generator/verifier coherence for ambiguous scenes.
        from verifier import verify_t3

        out = [row[:] for row in content_grid]
        if verify_t3(inp) == out:
            return {"input": inp, "output": out}
        if top is not None and left is not None and half_shift is not None and content is not None:
            raise RuntimeError("fixed parameters produce verifier-ambiguous sample")

    raise RuntimeError("failed to sample valid shifted wrapped-rectangle instance")
