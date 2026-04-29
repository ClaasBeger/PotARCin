"""Generator for t28: breakout-style diagonal bouncing stop.

Task rule:
- The input contains several obstacle cells/shapes, a single colored moving cell near
  the bottom, and a 2-cell grey diagonal pointer extending from that cell.
- The pointer indicates the moving cell's initial diagonal direction.
- Imagine the cell moving one step at a time in that direction, bouncing elastically
  off the grid walls (45-degree billiard motion).
- Movement stops when the *next* step after any required wall-bounce would land on an
  already colored obstacle cell. The output shows the moving cell at its last empty
  position, with the original moving cell and grey pointer removed.

Generator constraints:
- the moving cell color differs from both obstacle color and grey;
- the initial pointer is always a length-2 diagonal chain from the start cell;
- generated examples are restricted to trajectories requiring at most 3 bounces,
  except for an occasional 4-bounce example;
- an obstacle is always placed on the future path so the motion halts finitely;
- extra obstacles are placed away from the early path prefix so the intended stop is
  unambiguous.
"""

from __future__ import annotations

import json
import random
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Grid = List[List[int]]
Cell = Tuple[int, int]
DIRS = [(-1, -1), (-1, 1)]
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _empty(h: int, w: int, fill: int = 0) -> Grid:
    return [[fill for _ in range(w)] for _ in range(h)]


def _rectangular(grid: Grid) -> bool:
    return bool(grid) and bool(grid[0]) and all(len(row) == len(grid[0]) for row in grid)


def _diag_dirs() -> list[Cell]:
    return [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _parse_start_and_direction(grid: Grid) -> tuple[Cell, Cell]:
    h, w = len(grid), len(grid[0])
    # Look for a non-zero/non-grey cell that begins a color-grey-grey diagonal chain.
    for r in range(h):
        for c in range(w):
            if grid[r][c] in (0, GREY):
                continue
            for dr, dc in _diag_dirs():
                r1, c1 = r + dr, c + dc
                r2, c2 = r + 2 * dr, c + 2 * dc
                if 0 <= r2 < h and 0 <= c2 < w:
                    if grid[r1][c1] == GREY and grid[r2][c2] == GREY:
                        # ensure the chain starts here, not earlier in same diagonal
                        rp, cp = r - dr, c - dc
                        if 0 <= rp < h and 0 <= cp < w and grid[rp][cp] == GREY:
                            continue
                        return (r, c), (dr, dc)
    raise ValueError('Could not find moving cell with 2-cell grey diagonal pointer')


def _remove_markers(grid: Grid, start: Cell, direction: Cell) -> tuple[Grid, int]:
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    sr, sc = start
    color = out[sr][sc]
    out[sr][sc] = 0
    dr, dc = direction
    for k in (1, 2):
        rr, cc = sr + k * dr, sc + k * dc
        if not (0 <= rr < h and 0 <= cc < w and out[rr][cc] == GREY):
            raise ValueError('Malformed marker line')
        out[rr][cc] = 0
    return out, color


def _step_with_bounce(r: int, c: int, dr: int, dc: int, h: int, w: int) -> tuple[int, int, int, int]:
    nr, nc = r + dr, c + dc
    if nr < 0 or nr >= h:
        dr = -dr
        nr = r + dr
    if nc < 0 or nc >= w:
        dc = -dc
        nc = c + dc
    return nr, nc, dr, dc


def _step_with_bounce_event(r: int, c: int, dr: int, dc: int, h: int, w: int) -> tuple[int, int, int, int, int]:
    """Like _step_with_bounce, but also returns whether this step bounced.

    A corner reflection counts as a single bounce event for generation purposes.
    """
    nr, nc = r + dr, c + dc
    bounced = int(nr < 0 or nr >= h or nc < 0 or nc >= w)
    if nr < 0 or nr >= h:
        dr = -dr
        nr = r + dr
    if nc < 0 or nc >= w:
        dc = -dc
        nc = c + dc
    return nr, nc, dr, dc, bounced


def _simulate_stop(grid: Grid, start: Cell, direction: Cell) -> Cell:
    h, w = len(grid), len(grid[0])
    r, c = start
    dr, dc = direction
    seen_states: Set[tuple[int, int, int, int]] = set()
    while True:
        state = (r, c, dr, dc)
        if state in seen_states:
            raise ValueError('Path cycles without ever reaching an obstacle')
        seen_states.add(state)
        nr, nc, dr2, dc2 = _step_with_bounce(r, c, dr, dc, h, w)
        if grid[nr][nc] != 0:
            return (r, c)
        r, c, dr, dc = nr, nc, dr2, dc2


def transform_t28(grid: Grid) -> Grid:
    if not _rectangular(grid):
        raise ValueError('Input grid must be non-empty and rectangular')
    start, direction = _parse_start_and_direction(grid)
    board, ball_color = _remove_markers(grid, start, direction)
    fr, fc = _simulate_stop(board, start, direction)
    out = [row[:] for row in board]
    out[fr][fc] = ball_color
    return out


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

_SMALL_SHAPES: list[Set[Cell]] = [
    {(0, 0)},
    {(0, 0), (0, 1)},
    {(0, 0), (1, 0)},
    {(0, 0), (0, 1), (0, 2)},
    {(0, 0), (1, 0), (1, 1)},
    {(0, 1), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (1, 0), (1, 1)},
    {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)},
]


def _normalize(shape: Iterable[Cell]) -> Set[Cell]:
    cells = set(shape)
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _rotate(shape: Iterable[Cell]) -> Set[Cell]:
    cells = set(shape)
    return _normalize({(c, -r) for r, c in cells})


def _orientations(shape: Set[Cell]) -> list[Set[Cell]]:
    out = []
    cur = _normalize(shape)
    for _ in range(4):
        if cur not in out:
            out.append(cur)
        cur = _rotate(cur)
    return out


def _empty_path_sequence(h: int, w: int, start: Cell, direction: Cell) -> list[Cell]:
    """Visited positions in an empty board until state repeats (excluding start)."""
    r, c = start
    dr, dc = direction
    seen: Set[tuple[int, int, int, int]] = set()
    seq: list[Cell] = []
    while True:
        state = (r, c, dr, dc)
        if state in seen:
            return seq
        seen.add(state)
        nr, nc, dr, dc = _step_with_bounce(r, c, dr, dc, h, w)
        seq.append((nr, nc))
        r, c = nr, nc


def _empty_path_sequence_with_bounces(h: int, w: int, start: Cell, direction: Cell) -> list[tuple[Cell, int]]:
    """Empty-board path annotated with cumulative bounce count.

    Each item corresponds to the cell reached after one step and the number of
    wall-bounce events encountered up to and including that attempted step.
    """
    r, c = start
    dr, dc = direction
    seen: Set[tuple[int, int, int, int]] = set()
    seq: list[tuple[Cell, int]] = []
    bounces = 0
    while True:
        state = (r, c, dr, dc)
        if state in seen:
            return seq
        seen.add(state)
        nr, nc, dr, dc, bounced = _step_with_bounce_event(r, c, dr, dc, h, w)
        bounces += bounced
        seq.append(((nr, nc), bounces))
        r, c = nr, nc


def _place_shape(board: Grid, cells: Set[Cell], color: int) -> None:
    for r, c in cells:
        board[r][c] = color


def _can_place(board: Grid, cells: Set[Cell], forbidden: Set[Cell]) -> bool:
    h, w = len(board), len(board[0])
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if board[r][c] != 0 or (r, c) in forbidden:
            return False
    return True


def _random_blob_through(target: Cell, board: Grid, forbidden: Set[Cell], rng: random.Random) -> Optional[Set[Cell]]:
    shapes = _SMALL_SHAPES[:]
    rng.shuffle(shapes)
    for base in shapes:
        orients = _orientations(base)
        rng.shuffle(orients)
        for shape in orients:
            anchor_cells = list(shape)
            rng.shuffle(anchor_cells)
            for ar, ac in anchor_cells:
                off_r, off_c = target[0] - ar, target[1] - ac
                cells = {(off_r + r, off_c + c) for r, c in shape}
                if _can_place(board, cells, forbidden):
                    return cells
    # fallback: single-cell obstacle if available
    if _can_place(board, {target}, forbidden):
        return {target}
    return None


def _random_blob_anywhere(board: Grid, color: int, forbidden: Set[Cell], rng: random.Random) -> bool:
    h, w = len(board), len(board[0])
    shapes = _SMALL_SHAPES[:]
    rng.shuffle(shapes)
    for base in shapes:
        for shape in _orientations(base):
            max_r = max(r for r, _ in shape)
            max_c = max(c for _, c in shape)
            positions = [(rr, cc) for rr in range(h - max_r) for cc in range(w - max_c)]
            rng.shuffle(positions)
            for off_r, off_c in positions:
                cells = {(off_r + r, off_c + c) for r, c in shape}
                if _can_place(board, cells, forbidden):
                    _place_shape(board, cells, color)
                    return True
    return False


def generate(rng: random.Random | None = None) -> Dict[str, Grid]:
    rng = rng or random.Random()
    for _ in range(4000):
        h = rng.randint(9, 18)
        w = rng.randint(9, 18)
        start_row = h - 1
        direction = rng.choice(DIRS)
        dc = direction[1]
        if dc == 1:
            start_col = rng.randint(0, w - 3)
        else:
            start_col = rng.randint(2, w - 1)
        start = (start_row, start_col)
        grey1 = (start_row - 1, start_col + dc)
        grey2 = (start_row - 2, start_col + 2 * dc)
        if not (0 <= grey2[1] < w):
            continue

        obstacle_color = rng.choice(COLORS)
        ball_color = rng.choice([c for c in COLORS if c != obstacle_color])
        board = _empty(h, w)

        # Early-path prefix to protect: no obstacles may appear there, so the chosen
        # stop location remains the first collision. We also restrict examples to
        # paths with at most 3 bounces, except for occasional 4-bounce cases.
        empty_path_info = _empty_path_sequence_with_bounces(h, w, start, direction)
        empty_path = [cell for cell, _b in empty_path_info]
        if len(empty_path) < 5:
            continue
        bounce_cap = 4 if rng.random() < 0.08 else 3
        candidate_indices = [
            idx for idx in range(3, len(empty_path_info))
            if empty_path_info[idx][1] <= bounce_cap
        ]
        if not candidate_indices:
            continue
        # Keep the chosen collision reasonably early to avoid overly long traces.
        upper = max(6, len(empty_path_info) // 2)
        preferred = [idx for idx in candidate_indices if idx <= upper]
        pool = preferred if preferred else candidate_indices
        stop_index = rng.choice(pool)
        hit_cell = empty_path[stop_index]
        protected_prefix = set(empty_path[:stop_index])
        marker_cells = {start, grey1, grey2}
        forbidden_for_main = protected_prefix | marker_cells

        main_blob = _random_blob_through(hit_cell, board, forbidden_for_main, rng)
        if main_blob is None:
            continue
        _place_shape(board, main_blob, obstacle_color)

        # Additional obstacle blobs for visual richness, kept off the protected prefix.
        n_extra = rng.randint(1, 6)
        forbidden_extra = protected_prefix | marker_cells
        for _k in range(n_extra):
            _random_blob_anywhere(board, obstacle_color, forbidden_extra, rng)

        inp = [row[:] for row in board]
        inp[start[0]][start[1]] = ball_color
        inp[grey1[0]][grey1[1]] = GREY
        inp[grey2[0]][grey2[1]] = GREY

        try:
            out = transform_t28(inp)
        except Exception:
            continue

        # Sanity-check that the output ball stops exactly at the intended prefix end.
        expected_stop = empty_path[stop_index - 1]
        if out[expected_stop[0]][expected_stop[1]] != ball_color:
            continue
        # Keep outputs meaningfully changed.
        if out == inp:
            continue
        return {'input': inp, 'output': out}
    raise RuntimeError('Failed to generate valid t28 example')


if __name__ == '__main__':
    print(json.dumps(generate(rng=random.Random(0)), indent=2))
