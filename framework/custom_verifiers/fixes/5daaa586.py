"""5daaa586: crop to the frame and let the matching pixels streak into it.

Two full rows and two full columns of four different colours cut a rectangular
frame out of the picture, and loose pixels are scattered about in a fifth
role: they all share the colour of one of the four lines. The answer is the
frame and its inside, with every loose pixel inside the frame drawn out into a
ray that runs towards the line it is coloured after and stops just short of it.

The frame is read off the picture: a line row is a row made of a single non
background colour apart from at most the two cells where the line columns cross
it (the four lines are drawn in a random order, so whichever is drawn last owns
the corner). Copying the frame out of the input therefore reproduces the corners
correctly without having to know that order. The ray colour is whatever
non-background colour appears strictly inside the frame, and its direction is
the side whose line wears the same colour -- so no grid size, frame position or
palette is assumed.
"""

from typing import Dict, List, Optional, Tuple

from framework.grids import Grid


def _line_colour(cells: List[int], background: int) -> Optional[int]:
    """Return the colour of a line running along *cells*, or None if it is not one."""
    counts: Dict[int, int] = {}
    for value in cells:
        counts[value] = counts.get(value, 0) + 1
    colour = max(counts, key=lambda v: (counts[v], -v))
    # A line loses at most its two crossings with the perpendicular lines; an
    # ordinary row or column is mostly background and never gets this close.
    if colour == background or counts[colour] < len(cells) - 2:
        return None
    return colour


def solve(grid: Grid) -> Grid:
    """Return the framed region with each loose pixel extended into a ray."""
    fallback = [list(row) for row in grid]
    try:
        h = len(grid)
        w = len(grid[0]) if h else 0
        if h == 0 or w == 0:
            return fallback

        counts: Dict[int, int] = {}
        for row in grid:
            for value in row:
                counts[value] = counts.get(value, 0) + 1
        background = max(counts, key=lambda v: (counts[v], -v))

        rows: List[Tuple[int, int]] = []
        for r in range(h):
            colour = _line_colour(list(grid[r]), background)
            if colour is not None:
                rows.append((r, colour))
        cols: List[Tuple[int, int]] = []
        for c in range(w):
            colour = _line_colour([grid[r][c] for r in range(h)], background)
            if colour is not None:
                cols.append((c, colour))
        if len(rows) != 2 or len(cols) != 2:
            return fallback

        (up, up_colour), (down, down_colour) = rows
        (left, left_colour), (right, right_colour) = cols
        if down - up < 2 or right - left < 2:
            return fallback

        out = [list(row[left : right + 1]) for row in grid[up : down + 1]]

        inside = {
            grid[r][c]
            for r in range(up + 1, down)
            for c in range(left + 1, right)
            if grid[r][c] != background
        }
        if len(inside) != 1:
            return out
        ray = inside.pop()

        steps = [
            (ray == left_colour, (0, -1)),
            (ray == right_colour, (0, 1)),
            (ray == up_colour, (-1, 0)),
            (ray == down_colour, (1, 0)),
        ]
        matches = [step for hit, step in steps if hit]
        if len(matches) != 1:
            return out
        dr, dc = matches[0]

        for r in range(up + 1, down):
            for c in range(left + 1, right):
                if grid[r][c] != ray:
                    continue
                rr, cc = r, c
                # The ray stops before the frame, which keeps its own colour.
                while up < rr < down and left < cc < right:
                    out[rr - up][cc - left] = ray
                    rr, cc = rr + dr, cc + dc
        return out
    except Exception:
        return fallback
