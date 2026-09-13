"""484b58aa: fill the holes back in from the pattern's repeat.

The picture is one tile laid down edge to edge with a few rectangles punched out
of it. The repeat is whatever period the visible cells agree on -- found here by
trying each period in turn and keeping the smallest that never contradicts
itself -- and each hole is then filled from any cell that shares its place in
that repeat.

Deriving the period from the whole picture rather than row by row is what makes
this work when a hole swallows most of one row: the rest of the picture still
pins the repeat down.
"""
from typing import List, Optional

Grid = List[List[int]]

EMPTY = 0


def _period(grid: Grid, along_rows: bool) -> Optional[int]:
    h, w = len(grid), len(grid[0])
    span, other = (h, w) if along_rows else (w, h)
    for step in range(1, span):
        ok = True
        for i in range(span - step):
            for j in range(other):
                a = grid[i][j] if along_rows else grid[j][i]
                b = grid[i + step][j] if along_rows else grid[j][i + step]
                if a != EMPTY and b != EMPTY and a != b:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return step
    return None


def solve(grid: Grid) -> Grid:
    h = len(grid)
    w = len(grid[0]) if h else 0
    if h == 0 or w == 0 or any(len(row) != w for row in grid):
        return [list(row) for row in grid]

    down = _period(grid, True)
    across = _period(grid, False)
    if down is None and across is None:
        return [list(row) for row in grid]

    def agreed(cells: List[int]) -> int:
        """The colour these cells share, or nothing if they disagree.

        A tile wider than the picture leaves no real horizontal repeat, and the
        shortest step that happens not to contradict the few visible pairs is an
        accident. Demanding that every cell drawn from a repeat agrees is what
        tells a real period from an accidental one -- one official example is a
        42-wide tile on a 29-wide grid, where only the vertical repeat is real.
        """
        shown = [v for v in cells if v != EMPTY]
        return shown[0] if shown and len(set(shown)) == 1 else EMPTY

    out = [list(row) for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != EMPTY:
                continue
            fill = EMPTY
            if down:      # the same column, a whole number of repeats away
                fill = agreed([grid[rr][c] for rr in range(r % down, h, down)])
            if fill == EMPTY and across:   # otherwise along the row
                fill = agreed([grid[r][cc] for cc in range(c % across, w, across)])
            if fill == EMPTY and down and across:
                fill = agreed([grid[rr][cc] for rr in range(r % down, h, down)
                               for cc in range(c % across, w, across)])
            out[r][c] = fill
    return out
