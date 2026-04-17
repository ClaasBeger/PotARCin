"""Generator for Test2/t14: bottom-runner jump/duck with conditional star."""

from __future__ import annotations

import random
from typing import List

Grid = List[List[int]]

BG = 0
DOT = 2
BLOCK = 3
STAR = 4


def _copy_grid(g: Grid) -> Grid:
    return [row[:] for row in g]


def _grounded_height(g: Grid, c: int) -> int:
    """Height of the contiguous bottom-anchored block stack in column c."""
    h = len(g)
    k = 0
    r = h - 1
    while r >= 0 and g[r][c] == BLOCK:
        k += 1
        r -= 1
    return k


def _add_top_right_star(g: Grid) -> None:
    h = len(g)
    w = len(g[0]) if h else 0
    if h < 3 or w < 3:
        return
    center_r, center_c = 1, w - 2
    for r, c in (
        (center_r, center_c),
        (center_r - 1, center_c),
        (center_r + 1, center_c),
        (center_r, center_c - 1),
        (center_r, center_c + 1),
    ):
        if 0 <= r < h and 0 <= c < w:
            g[r][c] = STAR


def transform_t14(inp: Grid) -> Grid:
    """Move the red dot right with jump/duck rules and add star on success."""
    if not inp or not inp[0]:
        raise ValueError("input grid must be non-empty")
    w = len(inp[0])
    if any(len(row) != w for row in inp):
        raise ValueError("input grid must be rectangular")

    h = len(inp)
    dot_cells = [(r, c) for r in range(h) for c in range(w) if inp[r][c] == DOT]
    if len(dot_cells) != 1:
        raise ValueError("expected exactly one red dot (2)")
    dot_r, dot_c = dot_cells[0]
    if dot_r != h - 1:
        raise ValueError("dot must be on bottom row")

    pos = dot_c
    while pos < w - 1:
        nxt = pos + 1
        if inp[dot_r][nxt] == BG:
            pos = nxt
            continue

        # A grounded block directly ahead can be jumped only if its height is 1 or 2.
        if inp[dot_r][nxt] == BLOCK:
            height = _grounded_height(inp, nxt)
            if height in (1, 2):
                landing = pos + 2
                if landing < w and inp[dot_r][landing] == BG:
                    pos = landing
                    continue
        break

    out = _copy_grid(inp)
    out[dot_r][dot_c] = BG
    out[dot_r][pos] = DOT
    if pos == w - 1:
        _add_top_right_star(out)
    return out


def _make_empty(h: int, w: int) -> Grid:
    return [[BG for _ in range(w)] for _ in range(h)]


def _place_grounded_block(g: Grid, c: int, height: int) -> None:
    h = len(g)
    for r in range(h - 1, h - height - 1, -1):
        g[r][c] = BLOCK


def _place_floating_cell(g: Grid, c: int, r: int) -> None:
    g[r][c] = BLOCK


def _sample_nonadjacent_columns(
    candidates: list[int], count: int, rng: random.Random
) -> list[int]:
    pool = candidates[:]
    rng.shuffle(pool)
    chosen: list[int] = []
    for c in pool:
        if all(abs(c - x) > 1 for x in chosen):
            chosen.append(c)
            if len(chosen) >= count:
                break
    return sorted(chosen)


def _has_invalid_horizontal_adjacency(g: Grid) -> bool:
    """Adjacent 3s are allowed only if both are fully floating (clear shaft below)."""
    h = len(g)
    w = len(g[0]) if h else 0
    for r in range(h):
        for c in range(w - 1):
            if g[r][c] != BLOCK or g[r][c + 1] != BLOCK:
                continue
            left_clear = all(g[rr][c] == BG for rr in range(r + 1, h))
            right_clear = all(g[rr][c + 1] == BG for rr in range(r + 1, h))
            if not (left_clear and right_clear):
                return True
    return False


def _jump_clearance_cells(g: Grid) -> set[tuple[int, int]]:
    """Cells that must stay empty so 1/2-high grounded blocks are visibly jumpable."""
    h = len(g)
    w = len(g[0]) if h else 0
    forbidden: set[tuple[int, int]] = set()
    for c in range(w):
        gh = _grounded_height(g, c)
        if gh not in (1, 2):
            continue
        for k in range(gh):
            r = h - 1 - k
            nr = r - 1
            if nr < 0:
                continue
            if c - 1 >= 0:
                forbidden.add((nr, c - 1))
            if c + 1 < w:
                forbidden.add((nr, c + 1))
    return forbidden


def _has_blocked_jump_diagonal(g: Grid) -> bool:
    for r, c in _jump_clearance_cells(g):
        if g[r][c] == BLOCK:
            return True
    return False


def generate(
    rng: random.Random | None = None,
    *,
    min_w: int = 10,
    max_w: int = 16,
    min_h: int = 6,
    max_h: int = 10,
) -> dict[str, Grid]:
    rr = rng or random.Random()

    for _ in range(1000):
        w = rr.randint(min_w, max_w)
        h = rr.randint(min_h, max_h)
        inp = _make_empty(h, w)
        inp[h - 1][0] = DOT

        should_reach_border = rr.random() < 0.55
        candidates = list(range(1, w))
        rr.shuffle(candidates)
        grounded_cols: set[int] = set()

        if should_reach_border:
            n_grounded = rr.randint(2, max(2, min(4, (w - 2) // 2 + 1)))
            grounded_list = _sample_nonadjacent_columns(list(range(1, w - 1)), n_grounded, rr)
            for c in grounded_list:
                _place_grounded_block(inp, c, rr.choice([1, 2]))
                grounded_cols.add(c)
        else:
            stop_col = rr.randint(3, w - 2)
            before_candidates = [c for c in range(1, stop_col) if abs(c - stop_col) > 1]
            n_before_max = max(0, min(3, len(before_candidates)))
            n_before = rr.randint(0, n_before_max) if n_before_max else 0
            for c in _sample_nonadjacent_columns(before_candidates, n_before, rr):
                _place_grounded_block(inp, c, rr.choice([1, 2]))
                grounded_cols.add(c)
            _place_grounded_block(inp, stop_col, 3)
            grounded_cols.add(stop_col)
            after_candidates = [
                c
                for c in range(stop_col + 1, w - 1)
                if abs(c - stop_col) > 1 and all(abs(c - x) > 1 for x in grounded_cols)
            ]
            n_after_max = max(0, min(2, len(after_candidates)))
            n_after = rr.randint(0, n_after_max) if n_after_max else 0
            for c in _sample_nonadjacent_columns(after_candidates, n_after, rr):
                _place_grounded_block(inp, c, rr.choice([1, 2, 3]))
                grounded_cols.add(c)

        forbidden_jump_cells = _jump_clearance_cells(inp)

        # Add floating cells/runs with a full empty shaft below.
        for c in range(1, w - 1):
            if inp[h - 1][c] != BG:
                continue
            if rr.random() < 0.16:
                r = rr.randint(1, max(1, h - 3))
                if (r, c) in forbidden_jump_cells:
                    continue
                if all(inp[rrr][c] == BG for rrr in range(r, h)):
                    _place_floating_cell(inp, c, r)

        # Occasionally place one horizontal floating pair/triple, which is demo-compatible.
        if rr.random() < 0.35:
            run_len = 2 if rr.random() < 0.8 else 3
            start = rr.randint(1, max(1, w - run_len - 1))
            cols = list(range(start, start + run_len))
            r = rr.randint(1, max(1, h - 3))
            if all(inp[h - 1][c] == BG for c in cols) and all(
                all(inp[rrr][c] == BG for rrr in range(r, h)) for c in cols
            ) and all((r, c) not in forbidden_jump_cells for c in cols):
                for c in cols:
                    _place_floating_cell(inp, c, r)

        out = transform_t14(inp)

        # Ensure the task is non-trivial.
        if out == inp or _has_invalid_horizontal_adjacency(inp) or _has_blocked_jump_diagonal(inp):
            continue
        return {"input": inp, "output": out}

    raise RuntimeError("failed to sample valid t14 instance")
