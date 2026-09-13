"""e3497940: fold the far half of the picture onto the near half.

A solid line of one colour splits the picture into two halves of equal size.
The answer has the shape of one half: the far half is folded over the line, and
wherever it carries ink that ink shows through onto the near half.  Cells that
are empty on both halves stay empty.

The fold is a mirror about the divider, so the far half's column next to the
line lands on the near half's column next to the line.  The divider's colour,
the ink colours and the half width are all read off the grid; only the empty
colour (0) is the task's own, and it has to be, because a half can carry more
ink than empty space and the majority colour then names the wrong thing.
"""

from typing import List, Optional, Tuple

EMPTY = 0


def _halves(grid: List[List[int]]) -> Optional[Tuple[List[List[int]], List[List[int]]]]:
    """Return (near half, far half already folded over the divider), or None."""
    h = len(grid)
    w = len(grid[0])

    if w % 2 == 1:
        j = (w - 1) // 2
        if len({row[j] for row in grid}) == 1:
            near = [row[:j] for row in grid]
            # Folding about column j reverses the far half's columns.
            far = [row[j + 1:][::-1] for row in grid]
            return near, far

    if h % 2 == 1:
        i = (h - 1) // 2
        if len(set(grid[i])) == 1:
            near = [list(row) for row in grid[:i]]
            far = [list(row) for row in grid[i + 1:]][::-1]
            return near, far

    return None


def solve(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return [list(row) for row in grid]

    halves = _halves(grid)
    if halves is None:
        return [list(row) for row in grid]
    near, far = halves
    if not near or not near[0] or len(near) != len(far) or len(near[0]) != len(far[0]):
        return [list(row) for row in grid]

    return [
        [near[r][c] if near[r][c] != EMPTY else far[r][c] for c in range(len(near[0]))]
        for r in range(len(near))
    ]
