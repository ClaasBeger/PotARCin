"""a48eeaf7: every loose pixel slides in until it touches the block.

The picture holds one solid rectangular block plus a scatter of loose pixels of
another colour.  Each loose pixel is pulled straight onto the ring of cells that
hugs the block: its row is clamped into the ring's row span and its column into
the ring's column span, which is the same thing as moving it to the nearest ring
cell.  The pixel leaves no trace behind.

Nothing about the block's size, its position or the number of loose pixels is
assumed: the block is found as the one colour whose cells fill their own
bounding box, and the ring is derived from that box.  A picture with no loose
pixels at all -- which the generator does produce, rarely -- asks for no
movement, so the answer is the picture itself.
"""

from typing import Dict, List, Tuple


def _colour_cells(grid: List[List[int]]) -> Dict[int, List[Tuple[int, int]]]:
    cells: Dict[int, List[Tuple[int, int]]] = {}
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            cells.setdefault(v, []).append((r, c))
    return cells


def _is_filled_rectangle(cells: List[Tuple[int, int]]) -> bool:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    height = max(rows) - min(rows) + 1
    width = max(cols) - min(cols) + 1
    return height * width == len(cells)


def solve(grid: List[List[int]]) -> List[List[int]]:
    out = [list(row) for row in grid]
    if not grid or not grid[0]:
        return out

    by_colour = _colour_cells(grid)
    # The field colour is whatever covers most of the picture; everything else
    # is either the block or a loose pixel.
    background = max(by_colour, key=lambda v: len(by_colour[v]))

    blocks = [
        (colour, cells)
        for colour, cells in by_colour.items()
        if colour != background and _is_filled_rectangle(cells)
    ]
    if not blocks:
        return out
    # The loose pixels are a scatter, so the solid block is the largest such
    # group; ties (there should be none) fall to the topmost, leftmost one.
    _, _, block = min((-len(cells), min(cells), cells) for _, cells in blocks)
    block_cells = set(block)

    movers = [
        (r, c, v)
        for r, row in enumerate(grid)
        for c, v in enumerate(row)
        if v != background and (r, c) not in block_cells
    ]
    if not movers:
        return out

    top = min(r for r, _ in block) - 1
    bottom = max(r for r, _ in block) + 1
    left = min(c for _, c in block) - 1
    right = max(c for _, c in block) + 1

    for r, c, _ in movers:
        out[r][c] = background
    for r, c, colour in movers:
        nr = min(max(r, top), bottom)
        nc = min(max(c, left), right)
        if 0 <= nr < len(out) and 0 <= nc < len(out[0]):
            out[nr][nc] = colour
    return out
