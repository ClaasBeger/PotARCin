"""ARC 2281f1f4: mark red every crossing of the grey markers' rows and columns.

The grey markers sit on two borders: one edge *row* nominates a set of columns,
one edge *column* nominates a set of rows.  Every (row, column) crossing turns
red; the markers themselves stay grey.  When one of the two border lines carries
no marker there is nothing to cross and the grid is returned untouched - the old
verifier instead blanked out a corner marker on such grids.
"""

GREY = 5
RED = 2


def solve(grid: list[list[int]]) -> list[list[int]]:
    try:
        height = len(grid)
        width = len(grid[0]) if height else 0
        if not height or not width:
            return grid

        markers = [
            (row, col)
            for row in range(height)
            for col in range(width)
            if grid[row][col] == GREY
        ]
        if not markers:
            return grid

        # Which border holds which line is read off the picture rather than
        # assumed: try every (edge row, edge column) pairing and keep the one
        # that actually yields crossings.  A marker on the corner the two lines
        # share belongs to both, so it is left out of each side's tally -
        # counting it would only regenerate cells that are already markers.
        best_rows: list[int] = []
        best_cols: list[int] = []
        best_score = -1
        for edge_row in {0, height - 1}:
            for edge_col in {0, width - 1}:
                cols = [c for (r, c) in markers if r == edge_row and c != edge_col]
                rows = [r for (r, c) in markers if c == edge_col and r != edge_row]
                score = len(rows) * len(cols)
                if score > best_score:
                    best_score, best_rows, best_cols = score, rows, cols

        out = [row[:] for row in grid]
        for row in best_rows:
            for col in best_cols:
                if out[row][col] != GREY:
                    out[row][col] = RED
        return out
    except Exception:
        return grid
