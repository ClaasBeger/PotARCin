"""ARC 760b3cac: mirror the cyan shape to the side the yellow signpost names.

Only the cyan shape is ever copied.  The yellow shape never moves; it is a
signpost whose bounding box has exactly one side carrying more than a single
yellow cell, and that side is the direction.  The cyan shape is then reflected
across that same side of *its own* bounding box, so the copy lands flush against
the original.

The old verifier picked the signpost by shape statistics instead of by colour;
a cyan shape that happened to match those statistics was chosen, and the yellow
signpost got mirrored instead.
"""

CYAN = 8
YELLOW = 4


def _cells(grid: list[list[int]], colour: int) -> list[tuple[int, int]]:
    return [
        (row, col)
        for row, line in enumerate(grid)
        for col, value in enumerate(line)
        if value == colour
    ]


def solve(grid: list[list[int]]) -> list[list[int]]:
    try:
        height = len(grid)
        width = len(grid[0]) if height else 0
        cyan = _cells(grid, CYAN)
        yellow = _cells(grid, YELLOW)
        if not cyan or not yellow:
            return grid

        y_top = min(r for r, _ in yellow)
        y_bottom = max(r for r, _ in yellow)
        y_left = min(c for _, c in yellow)
        y_right = max(c for _, c in yellow)
        sides = {
            "up": sum(1 for r, _ in yellow if r == y_top),
            "down": sum(1 for r, _ in yellow if r == y_bottom),
            "left": sum(1 for _, c in yellow if c == y_left),
            "right": sum(1 for _, c in yellow if c == y_right),
        }
        heaviest = max(sides.values())
        pointing = [name for name, count in sides.items() if count == heaviest]
        if len(pointing) != 1:
            # An ambiguous signpost is off-distribution; leave the grid alone.
            return grid
        direction = pointing[0]

        c_top = min(r for r, _ in cyan)
        c_bottom = max(r for r, _ in cyan)
        c_left = min(c for _, c in cyan)
        c_right = max(c for _, c in cyan)

        out = [row[:] for row in grid]
        for row, col in cyan:
            if direction == "left":
                # Reflect across the line just outside the bounding box edge.
                target = (row, 2 * c_left - 1 - col)
            elif direction == "right":
                target = (row, 2 * c_right + 1 - col)
            elif direction == "up":
                target = (2 * c_top - 1 - row, col)
            else:
                target = (2 * c_bottom + 1 - row, col)
            t_row, t_col = target
            if 0 <= t_row < height and 0 <= t_col < width:
                out[t_row][t_col] = CYAN
        return out
    except Exception:
        return grid
