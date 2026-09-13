"""ARC 00d62c1b: paint yellow every pocket that the green shape encloses.

A cell is *inside* the shape exactly when it cannot walk off the grid through
non-green cells using 4-connected steps.  So flood inwards from the border
through everything that is not green: whatever the flood fails to reach is
enclosed and becomes yellow.

The walls are recognised by their colour.  Deciding "wall vs. field" by which
colour is the most frequent is what the old verifier did, and it inverts the
picture on dense grids where the shape outnumbers the background - the flood
then starts from green cells and nothing is ever reported as enclosed.
"""

GREEN = 3
YELLOW = 4


def solve(grid: list[list[int]]) -> list[list[int]]:
    try:
        height = len(grid)
        width = len(grid[0]) if height else 0
        if not height or not width:
            return grid

        outside = [[False] * width for _ in range(height)]
        frontier = []
        # Seed with every non-green border cell: those are trivially outside.
        for row in range(height):
            for col in (0, width - 1):
                if grid[row][col] != GREEN and not outside[row][col]:
                    outside[row][col] = True
                    frontier.append((row, col))
        for col in range(width):
            for row in (0, height - 1):
                if grid[row][col] != GREEN and not outside[row][col]:
                    outside[row][col] = True
                    frontier.append((row, col))

        while frontier:
            row, col = frontier.pop()
            for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n_row, n_col = row + d_row, col + d_col
                if not (0 <= n_row < height and 0 <= n_col < width):
                    continue
                if outside[n_row][n_col] or grid[n_row][n_col] == GREEN:
                    continue
                outside[n_row][n_col] = True
                frontier.append((n_row, n_col))

        out = [row[:] for row in grid]
        for row in range(height):
            for col in range(width):
                if grid[row][col] != GREEN and not outside[row][col]:
                    out[row][col] = YELLOW
        return out
    except Exception:
        return grid
