"""b782dc8a: spread the two seed colours through the maze, alternating.

A seed sits somewhere in a maze: one cell of its own colour with four arms of a
second colour around it. Both colours flow together through every open cell the
seed can reach, swapping on each step, so a cell takes the centre's colour when
it is an even number of steps away and the arms' colour when it is odd. Walls
and any pocket the seed cannot reach are left as they are.

Written to replace a verifier that is correct but roughly quadratic in the cell
count: it takes over a second on the larger grids this task's generator draws,
which is slow enough to time out the harness. A flood fill is linear.
"""
from collections import deque
from typing import Dict, List

Grid = List[List[int]]


def solve(grid: Grid) -> Grid:
    h = len(grid)
    w = len(grid[0]) if h else 0
    if h == 0 or w == 0 or any(len(row) != w for row in grid):
        return [list(row) for row in grid]

    tally: Dict[int, int] = {}
    for row in grid:
        for v in row:
            tally[v] = tally.get(v, 0) + 1
    # The seed is five cells in all, so its two colours are much the rarest;
    # the maze itself and the open ground make up everything else.
    seeds = sorted((v for v in tally if v != 0), key=lambda v: tally[v])
    if len(seeds) < 3:
        return [list(row) for row in grid]
    middle, arms = seeds[0], seeds[1]
    wall = max(tally, key=lambda v: tally[v] if v not in (0, middle, arms) else -1)

    start = next(((r, c) for r in range(h) for c in range(w)
                  if grid[r][c] == middle), None)
    if start is None:
        return [list(row) for row in grid]

    out = [list(row) for row in grid]
    seen = [[False] * w for _ in range(h)]
    seen[start[0]][start[1]] = True
    queue = deque([(start[0], start[1], 0)])
    while queue:
        r, c, step = queue.popleft()
        out[r][c] = middle if step % 2 == 0 else arms
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] != wall:
                seen[nr][nc] = True
                queue.append((nr, nc, step + 1))
    return out
