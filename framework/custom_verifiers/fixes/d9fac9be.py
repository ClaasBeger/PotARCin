"""d9fac9be: report the colour trapped inside the ring.

Two colours are sprinkled at random over an empty field, and somewhere among
the noise one of them is laid down as a solid 3x3 ring with a single cell of
the other colour in its middle. The answer is a 1x1 grid holding that middle
colour.

The ring is found by looking for a 3x3 window whose eight border cells are all
the same colour and whose centre differs. Lone noise pixels sitting in an empty
patch match that description too, so a window only counts when neither its ring
nor its centre is the background, and when ring, centre and background are the
whole palette -- which is what the ring the task means always looks like. The
background is read as the commonest colour outside the window under test, so a
picture small enough for the ring itself to outnumber the field is still
handled.
"""

from typing import Dict, List, Optional

from framework.grids import Grid

_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _centre_of_ring(grid: Grid) -> Optional[int]:
    h = len(grid)
    w = len(grid[0])

    counts: Dict[int, int] = {}
    for row in grid:
        for value in row:
            counts[value] = counts.get(value, 0) + 1
    palette = set(counts)

    found: List[int] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            ring = grid[r - 1][c - 1]
            if any(grid[r + dr][c + dc] != ring for dr, dc in _OFFSETS):
                continue
            centre = grid[r][c]
            if centre == ring:
                continue

            # Judge the background on the rest of the picture: inside a tiny grid
            # the ring's own eight cells can otherwise outvote the field.
            outside = dict(counts)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    value = grid[r + dr][c + dc]
                    outside[value] -= 1
            background = max(outside, key=lambda colour: (outside[colour], -colour))

            if ring == background or centre == background:
                continue
            if palette != {background, ring, centre}:
                continue
            found.append(centre)

    if found and len(set(found)) == 1:
        return found[0]
    return None


def solve(grid: Grid) -> Grid:
    """Return a 1x1 grid with the colour inside the uniform 3x3 ring."""
    try:
        if len(grid) < 3 or len(grid[0]) < 3:
            return [list(row) for row in grid]
        centre = _centre_of_ring(grid)
        if centre is None:
            return [list(row) for row in grid]
        return [[centre]]
    except Exception:
        return [list(row) for row in grid]
