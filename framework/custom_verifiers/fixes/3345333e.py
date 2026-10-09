"""ARC 3345333e: lift the block off a symmetric shape and mend the shape.

On a plain background sit a shape in one colour, mirror-symmetric about a
vertical or horizontal axis, and a solid rectangle in another colour that hides
part of it.  The rectangle is removed, and the hidden part of the shape is
restored by mirroring the visible part across the shape's axis.

The golf verifiers are tuned to the task's 16x16 layout and fail most RE-ARC
grids; the re_arc verifier misjudges the axis on about one generator draw in
twenty-five.  This one tries every axis and keeps the one that the visible cells
support best.
"""


def _background(grid):
    counts = {}
    for row in grid:
        for value in row:
            counts[value] = counts.get(value, 0) + 1
    return max(counts, key=lambda v: counts[v])


def _cells(grid, colour):
    return {(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == colour}


def _is_rectangle(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return len(cells) == (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)


def _best_mirror(shape, block, height, width):
    """(score, mirrored cells) for the axis the visible shape supports best.

    An axis is allowed when every visible shape cell mirrors to a shape cell or
    to a cell under the block, never onto open background or off the grid.  The
    block covers one side of the shape, so an axis it does not straddle beats
    one it does: a rectangular shape is symmetric about both of its axes, and
    only the one away from the block restores all of the hidden part.  After
    that, more visible cells mirroring onto visible cells is better.
    """
    best = None
    for vertical in (True, False):
        length = width if vertical else height
        for twice_axis in range(0, 2 * length - 1):
            mirrored = set()
            for r, c in shape:
                cell = (r, twice_axis - c) if vertical else (twice_axis - r, c)
                if not (0 <= cell[0] < height and 0 <= cell[1] < width):
                    break
                if cell not in shape and cell not in block:
                    break
                mirrored.add(cell)
            else:
                sides = {(2 * (c if vertical else r) > twice_axis) - (2 * (c if vertical else r) < twice_axis)
                         for r, c in block}
                score = (len(sides) == 1 and 0 not in sides, len(mirrored & shape))
                if best is None or score > best[0]:
                    best = (score, mirrored)
    return best


def solve(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return grid
    height, width = len(grid), len(grid[0])
    background = _background(grid)
    colours = sorted({value for row in grid for value in row} - {background})
    if len(colours) != 2:
        return [list(row) for row in grid]
    answers = []
    for block_colour in colours:
        block = _cells(grid, block_colour)
        if not _is_rectangle(block):
            continue
        shape_colour = colours[0] if colours[1] == block_colour else colours[1]
        shape = _cells(grid, shape_colour)
        best = _best_mirror(shape, block, height, width)
        if best is not None:
            restored = len(best[1] & block)
            top = min(r for r, _ in block)
            one_side, overlap = best[0]
            answers.append(((one_side, restored, top if not restored else 0, overlap), block, shape_colour, shape | best[1]))
    if not answers:
        return [list(row) for row in grid]
    # If both colours are solid rectangles, the block is the one whose removal
    # leaves the better-supported symmetry, then the one hiding more of the
    # shape.  When it hides nothing either way, the block is the lower one: it
    # is laid over the shape from below its top row.
    _, block, shape_colour, shape = max(answers, key=lambda a: a[0])
    out = [list(row) for row in grid]
    for r, c in block:
        out[r][c] = background
    for r, c in shape:
        out[r][c] = shape_colour
    return out
