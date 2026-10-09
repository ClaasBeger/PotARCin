"""ARC 29ec7d0e: fill the holes in a repeating pattern.

The grid is a pattern that repeats with some row period and some column period,
with patches painted over in one noise colour that the pattern itself never
uses.  Every noise cell takes the value its position has elsewhere in the
pattern.  In the task the pattern is a multiplication table modulo n and the
noise is black; RE-ARC draws arbitrary tiles and noise colours.

The google, keymoon and neurips verifiers rebuild the multiplication table over
a hard-coded 18x18 range, so they answer every other size with an 18x18 grid.
This one rebuilds the table at the grid's own size when the visible cells fit
it, and otherwise finds the period from the grid itself.
"""


def _period(grid, noise, along_rows):
    """Smallest period, along rows or columns, consistent with the non-noise cells."""
    height, width = len(grid), len(grid[0])
    length = height if along_rows else width
    for period in range(1, length + 1):
        ok = True
        for start in range(period):
            for other in range(width if along_rows else height):
                seen = None
                for pos in range(start, length, period):
                    value = grid[pos][other] if along_rows else grid[other][pos]
                    if value == noise:
                        continue
                    if seen is None:
                        seen = value
                    elif value != seen:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return period
    return length


def _tile(grid, noise):
    """The repeating tile with noise removed, or None if some tile cell is never visible."""
    row_period = _period(grid, noise, along_rows=True)
    col_period = _period(grid, noise, along_rows=False)
    tile = [[None] * col_period for _ in range(row_period)]
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value == noise:
                continue
            slot = tile[r % row_period][c % col_period]
            if slot is None:
                tile[r % row_period][c % col_period] = value
            elif slot != value:
                return None
    if any(value is None for line in tile for value in line):
        return None
    return tile


def _table(grid, noise):
    """Cell (r, c) = (r * c mod n) + 1 if that fits every non-noise cell, else None."""
    visible = [value for row in grid for value in row if value != noise]
    if not visible:
        return None
    modulus = max(visible)
    table = [[(r * c) % modulus + 1 for c in range(len(grid[0]))] for r in range(len(grid))]
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value != noise and value != table[r][c]:
                return None
    return table


def solve(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return grid
    counts = {}
    for row in grid:
        for value in row:
            counts[value] = counts.get(value, 0) + 1
    # The task's own pattern first: a multiplication table modulo n, with n its
    # largest value.  This also covers tile cells the noise hides in every copy.
    for noise in sorted(counts, key=lambda v: counts[v]):
        table = _table(grid, noise)
        if table is not None:
            return table
    # Otherwise any repeating tile.  The noise colour is the one whose removal
    # leaves a pattern that covers every tile cell; a pattern colour usually
    # fails this because its own cells vanish.
    best = None
    for noise in sorted(counts, key=lambda v: counts[v]):
        tile = _tile(grid, noise)
        if tile is None:
            continue
        key = (len(tile) * len(tile[0]), counts[noise])
        if best is None or key < best[0]:
            best = (key, tile)
    if best is None:
        return [list(row) for row in grid]
    tile = best[1]
    return [[tile[r % len(tile)][c % len(tile[0])] for c in range(len(grid[0]))] for r in range(len(grid))]
