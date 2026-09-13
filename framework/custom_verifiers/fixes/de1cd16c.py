"""de1cd16c: name the panel that collects the most speckles.

The picture is a rectangular patchwork of solid panels, every panel a different
colour, and one extra colour is sprinkled over them as single stray pixels. The
answer is a 1x1 grid holding the colour of the panel that carries the most
strays.

The panels are found from the picture: for a colour to be a panel colour its
cells must fill a rectangle in which nothing appears but that colour and the
speckle colour, and the panel rectangles must tile the grid exactly. Only one
candidate speckle colour can satisfy that, so nothing about the number of
panels, their sizes or the palette needs to be assumed.
"""

from typing import Dict, List, Optional, Tuple

from framework.grids import Grid

# (colour, top, left, bottom, right) for one panel.
_Panel = Tuple[int, int, int, int, int]


def _panels_for_speckle(grid: Grid, speckle: int) -> Optional[List[_Panel]]:
    """Return the panel rectangles if *speckle* is the stray colour, else None."""
    h = len(grid)
    w = len(grid[0])

    boxes: Dict[int, List[int]] = {}
    for r in range(h):
        for c in range(w):
            value = grid[r][c]
            if value == speckle:
                continue
            box = boxes.get(value)
            if box is None:
                boxes[value] = [r, c, r, c]
            else:
                box[0] = min(box[0], r)
                box[1] = min(box[1], c)
                box[2] = max(box[2], r)
                box[3] = max(box[3], c)
    if not boxes:
        return None

    # Each rectangle must be pure -- its own colour plus strays, nothing else --
    # and the rectangles together must cover every cell exactly once. A panel
    # colour mistaken for the speckle colour fails this at once, because the real
    # speckle colour is scattered over the whole picture and its bounding box
    # then swallows several panels.
    owner = [[-1] * w for _ in range(h)]
    panels: List[_Panel] = []
    for colour, (top, left, bottom, right) in boxes.items():
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                if grid[r][c] not in (colour, speckle):
                    return None
                if owner[r][c] != -1:
                    return None
                owner[r][c] = colour
        panels.append((colour, top, left, bottom, right))

    if any(owner[r][c] == -1 for r in range(h) for c in range(w)):
        return None
    return panels


def solve(grid: Grid) -> Grid:
    """Return a 1x1 grid with the colour of the most heavily speckled panel."""
    try:
        h = len(grid)
        w = len(grid[0]) if h else 0
        if h == 0 or w == 0:
            return [list(row) for row in grid]

        palette = {grid[r][c] for r in range(h) for c in range(w)}
        for speckle in sorted(palette):
            panels = _panels_for_speckle(grid, speckle)
            if panels is None:
                continue
            best_colour = None
            best_count = -1
            # Scan order breaks a tie deterministically; the task never poses one.
            for colour, top, left, bottom, right in sorted(
                panels, key=lambda p: (p[1], p[2])
            ):
                count = sum(
                    1
                    for r in range(top, bottom + 1)
                    for c in range(left, right + 1)
                    if grid[r][c] == speckle
                )
                if count > best_count:
                    best_colour, best_count = colour, count
            if best_colour is not None:
                return [[best_colour]]
        return [list(row) for row in grid]
    except Exception:
        return [list(row) for row in grid]
