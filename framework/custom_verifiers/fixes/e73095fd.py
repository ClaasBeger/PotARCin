"""e73095fd: paint the inside of every box in the wire frame.

The picture is a set of hollow rectangular boxes joined by one-cell-thick lines,
all drawn in a single wall colour on an empty field.  The answer fills the
inside of each box -- and only the inside of a box -- with yellow.

An inside is an empty region shaped like a solid rectangle with wall all the way
round it.  Wiring leaves gaps that look the same, so two further marks of a real
box are used, both visible in the picture:

* A box is always drawn with its full height, so a gap touching the top or
  bottom border cannot be one; only the left and right borders may cut a box off
  (and then by a single column, the frame's outermost one).
* Nothing ever meets a box at a corner: lines join a box along a side, never at
  the corner cell, and no two boxes come within a cell of each other.  So a gap
  whose frame has wall carrying on past one of its corners is wiring -- that is
  what a gap fenced in by two crossing lines, or squeezed between two boxes and
  the line joining them, looks like.

The wall colour and the empty colour are read off the picture, the frames from
the gaps they enclose; yellow is the task's own answer colour.
"""

from typing import List, Tuple

EMPTY = 0
FILL = 4

Rect = Tuple[int, int, int, int]  # top, left, bottom, right


def _rectangular_gaps(grid: List[List[int]], empty: int) -> List[Rect]:
    """Return the bounding boxes of the empty regions that are solid rectangles."""
    h, w = len(grid), len(grid[0])
    seen = [[False] * w for _ in range(h)]
    out: List[Rect] = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0][c0] or grid[r0][c0] != empty:
                continue
            stack = [(r0, c0)]
            seen[r0][c0] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] == empty:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            top = min(r for r, _ in cells)
            bottom = max(r for r, _ in cells)
            left = min(c for _, c in cells)
            right = max(c for _, c in cells)
            if len(cells) == (bottom - top + 1) * (right - left + 1):
                out.append((top, left, bottom, right))
    return out


def _is_box(grid: List[List[int]], frame: Rect, wall: int) -> bool:
    """True if *frame* can be the outline of a drawn box."""
    h, w = len(grid), len(grid[0])
    top, left, bottom, right = frame

    def is_wall(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w and grid[r][c] == wall

    def off_picture(r: int, c: int) -> bool:
        return not (0 <= r < h and 0 <= c < w)

    # Every part of the outline that is on the picture must be wall.
    for c in range(left, right + 1):
        for r in (top, bottom):
            if not is_wall(r, c) and not off_picture(r, c):
                return False
    for r in range(top, bottom + 1):
        for c in (left, right):
            if not is_wall(r, c) and not off_picture(r, c):
                return False

    # Nothing may touch the outline at a corner, from outside it.
    for r, dr in ((top, -1), (bottom, 1)):
        for c, dc in ((left, -1), (right, 1)):
            if is_wall(r + dr, c) or is_wall(r, c + dc) or is_wall(r + dr, c + dc):
                return False
    return True


def solve(grid: List[List[int]]) -> List[List[int]]:
    out = [list(row) for row in grid]
    if not grid or not grid[0]:
        return out

    h, w = len(grid), len(grid[0])
    colours = {v for row in grid for v in row}
    walls = colours - {EMPTY}
    if EMPTY not in colours or len(walls) != 1:
        return out
    wall = walls.pop()

    for top, left, bottom, right in _rectangular_gaps(grid, EMPTY):
        frame = (top - 1, left - 1, bottom + 1, right + 1)
        if frame[0] < 0 or frame[2] >= h:
            continue  # a box is never cut off by the top or bottom border
        if not _is_box(grid, frame, wall):
            continue
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                out[r][c] = FILL
    return out
