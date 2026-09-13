"""a78176bb: each grey wedge props up one more copy of the diagonal.

The picture holds one full diagonal line of colour plus one or two solid grey
right-triangles, each resting against that line with its right-angle corner
pointing away from it. The answer keeps the original line, rubs out the grey,
and adds one parallel line per wedge: it runs through the cell just beyond the
wedge's outer corner, i.e. two diagonals further out than the corner itself.

Everything is measured off the picture. The line is the colour whose cells
exactly fill one complete diagonal, the field is whatever is left once grey and
the line are set aside, and each wedge's reach is the largest distance from the
line that its grey cells attain -- no grid size, offset or wedge size is
assumed. Grey itself is named because it is part of the task's own vocabulary:
the props are always grey and the line never is.
"""

from typing import Dict, List, Optional, Tuple

from framework.grids import Grid

GREY = 5


def _find_line(grid: Grid) -> Optional[Tuple[int, int]]:
    """Return (colour, diagonal) of the drawn line, or None if unreadable."""
    h = len(grid)
    w = len(grid[0])

    cells: Dict[int, List[Tuple[int, int]]] = {}
    for r in range(h):
        for c in range(w):
            if grid[r][c] != GREY:
                cells.setdefault(grid[r][c], []).append((r, c))

    candidates: List[Tuple[int, int, int]] = []
    for colour, positions in cells.items():
        diagonals = {r - c for r, c in positions}
        if len(diagonals) != 1:
            continue
        diagonal = diagonals.pop()
        length = sum(1 for r in range(h) for c in range(w) if r - c == diagonal)
        # The line is drawn last in the picture, so it is never interrupted:
        # a colour that only partly covers its diagonal is background, not line.
        if len(positions) == length:
            candidates.append((len(positions), colour, diagonal))
    if not candidates:
        return None
    # If the field happens to look like a diagonal too, the line is the sparser
    # of the two: the field fills whatever the line and the wedges leave over.
    _, colour, diagonal = min(candidates)
    return colour, diagonal


def solve(grid: Grid) -> Grid:
    """Erase the grey props and draw the extra diagonals they hold up."""
    try:
        h = len(grid)
        w = len(grid[0]) if h else 0
        if h == 0 or w == 0:
            return [list(row) for row in grid]

        line = _find_line(grid)
        if line is None:
            return [list(row) for row in grid]
        colour, diagonal = line

        field: Dict[int, int] = {}
        for r in range(h):
            for c in range(w):
                value = grid[r][c]
                if value != GREY and value != colour:
                    field[value] = field.get(value, 0) + 1
        background = max(field, key=lambda v: (field[v], -v)) if field else 0

        out = [[background if value == GREY else value for value in row] for row in grid]

        # One wedge per side of the line; its outer corner is simply the grey cell
        # lying on the diagonal furthest from the line on that side.
        for side in (1, -1):
            reach = 0
            for r in range(h):
                for c in range(w):
                    if grid[r][c] != GREY:
                        continue
                    offset = (r - c) - diagonal
                    if offset * side > 0:
                        reach = max(reach, offset * side)
            if reach == 0:
                continue
            extra = diagonal + side * (reach + 2)
            for r in range(h):
                for c in range(w):
                    if r - c == extra:
                        out[r][c] = colour
        return out
    except Exception:
        return [list(row) for row in grid]
