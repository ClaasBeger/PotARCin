"""f2829549: mark the cells that are empty on *both* sides of the divider.

The picture is two panels of equal size laid either side of a solid line of a
single colour.  Each panel carries its own ink on an empty field.  The answer
has the shape of one panel and is green wherever the matching cell is empty in
*both* panels; everywhere else it stays empty.

The empty colour (0) and the answer colour (green, 3) belong to the task
itself.  Everything geometric -- which axis the divider runs along, where it
sits, how big a panel is -- is read off the grid, so a picture whose panels are
a different size, or whose ink colours differ from the official ones, is still
handled.  In particular a panel that happens to carry no empty cell at all is
fine: no cell is then empty on both sides and the answer is uniformly empty.
"""

from typing import List, Optional, Tuple

EMPTY = 0
MARK = 3


def _split(grid: List[List[int]]) -> Optional[Tuple[List[List[int]], List[List[int]]]]:
    """Return the two panels either side of the divider, or None.

    The divider is a full line of one colour that leaves the same number of
    rows/columns on each side, so its index follows from the grid's own
    dimensions rather than from any remembered layout.
    """
    h = len(grid)
    w = len(grid[0])

    if w % 2 == 1:
        j = (w - 1) // 2
        column = [row[j] for row in grid]
        if len(set(column)) == 1:
            left = [row[:j] for row in grid]
            right = [row[j + 1:] for row in grid]
            return left, right

    if h % 2 == 1:
        i = (h - 1) // 2
        if len(set(grid[i])) == 1:
            top = [list(row) for row in grid[:i]]
            bottom = [list(row) for row in grid[i + 1:]]
            return top, bottom

    return None


def solve(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return [list(row) for row in grid]

    panels = _split(grid)
    if panels is None:
        return [list(row) for row in grid]
    a, b = panels
    if not a or not a[0] or len(a) != len(b) or len(a[0]) != len(b[0]):
        return [list(row) for row in grid]

    return [
        [MARK if a[r][c] == EMPTY and b[r][c] == EMPTY else EMPTY
         for c in range(len(a[0]))]
        for r in range(len(a))
    ]
