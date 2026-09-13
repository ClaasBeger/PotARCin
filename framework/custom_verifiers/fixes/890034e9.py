"""890034e9: frame every cut-out the way the sample cut-out is framed.

The picture is coloured noise with rectangular holes punched out of it.  One
hole already wears a frame -- a one-cell-thick rectangle of a colour that is
used nowhere else.  Every other hole of the same size gets the same frame drawn
around it; nothing else changes.

Both the frame colour and the hole colour are read off the picture: the frame is
the colour whose cells are *exactly* the outline of their own bounding box, and
the hole colour is whatever uniformly fills the inside of that outline.  The
hole size likewise comes from that inside, so this works for any frame size and
any palette.  Frames that would run past the border are clipped.
"""

from typing import Dict, List, Optional, Tuple

Cells = List[Tuple[int, int]]


def _outline(top: int, left: int, bottom: int, right: int) -> set:
    ring = set()
    for c in range(left, right + 1):
        ring.add((top, c))
        ring.add((bottom, c))
    for r in range(top, bottom + 1):
        ring.add((r, left))
        ring.add((r, right))
    return ring


def _frame(grid: List[List[int]]) -> Optional[Tuple[int, int, int, int, int, int]]:
    """Return (colour, top, left, bottom, right, hole colour) of the sample frame."""
    h, w = len(grid), len(grid[0])
    cells: Dict[int, Cells] = {}
    for r in range(h):
        for c in range(w):
            cells.setdefault(grid[r][c], []).append((r, c))

    candidates = []
    for colour, pts in cells.items():
        top = min(r for r, _ in pts)
        bottom = max(r for r, _ in pts)
        left = min(c for _, c in pts)
        right = max(c for _, c in pts)
        # A frame uses its colour for the outline and for nothing else, which is
        # what separates it from noise that happens to form a ring somewhere.
        if bottom - top < 2 or right - left < 2:
            continue
        if set(pts) != _outline(top, left, bottom, right):
            continue
        inside = {grid[r][c] for r in range(top + 1, bottom) for c in range(left + 1, right)}
        if len(inside) != 1:
            continue
        candidates.append((len(pts), colour, top, left, bottom, right, inside.pop()))

    if not candidates:
        return None
    # Noise is not supposed to produce a second such colour; picking the
    # smallest ring just keeps the choice deterministic if it ever does.
    _, colour, top, left, bottom, right, hole = min(candidates)
    return colour, top, left, bottom, right, hole


def solve(grid: List[List[int]]) -> List[List[int]]:
    out = [list(row) for row in grid]
    if not grid or not grid[0]:
        return out

    found = _frame(grid)
    if found is None:
        return out
    colour, top, left, bottom, right, hole = found

    hole_h = bottom - top - 1
    hole_w = right - left - 1
    h, w = len(grid), len(grid[0])

    for r in range(h - hole_h + 1):
        for c in range(w - hole_w + 1):
            if any(grid[r + dr][c + dc] != hole
                   for dr in range(hole_h) for dc in range(hole_w)):
                continue
            for rr, cc in _outline(r - 1, c - 1, r + hole_h, c + hole_w):
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr][cc] = colour
    return out
