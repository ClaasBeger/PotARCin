"""ARC 3631a71a: restore a symmetric picture from under its masks.

The grid is square.  Ignoring its first two rows and columns, the picture is
mirror-symmetric left to right and top to bottom, and the whole grid is
symmetric about its main diagonal.  Patches are painted over in one mask colour
that the picture never uses (maroon in the task); each masked cell takes the
value of a symmetric partner that is not masked.

The re_arc verifier picks the mask colour by asking which colour, removed, leaves
a diagonally symmetric picture.  When the masks hide only black cells, black
itself passes that test, and the mask colour is copied across the picture.  The
golf verifiers hard-code the task's 30x30 layout.  This one works at any size
and checks each candidate mask colour against every symmetric partner.
"""

OFFSET = 2  # rows and columns before the mirror-symmetric block starts


def _partners(r, c, size):
    """Cells that must equal (r, c): its orbit under the three symmetries."""
    seen = {(r, c)}
    stack = [(r, c)]
    while stack:
        i, j = stack.pop()
        moves = [(j, i)]
        if j >= OFFSET:
            moves.append((i, size - 1 + OFFSET - j))
        if i >= OFFSET:
            moves.append((size - 1 + OFFSET - i, j))
        for cell in moves:
            if 0 <= cell[0] < size and 0 <= cell[1] < size and cell not in seen:
                seen.add(cell)
                stack.append(cell)
    return seen


def _restore(grid, mask, orbits):
    """Picture with *mask* cells filled from their partners, or None if impossible."""
    out = [list(row) for row in grid]
    for orbit in orbits:
        values = {grid[i][j] for i, j in orbit if grid[i][j] != mask}
        if len(values) > 1:
            return None
        has_mask = any(grid[i][j] == mask for i, j in orbit)
        if has_mask and not values:
            return None
        if has_mask:
            value = values.pop()
            for i, j in orbit:
                out[i][j] = value
    return out


def solve(grid: list[list[int]]) -> list[list[int]]:
    size = len(grid)
    if not size or any(len(row) != size for row in grid):
        return [list(row) for row in grid]
    orbits, assigned = [], set()
    for r in range(size):
        for c in range(size):
            if (r, c) not in assigned:
                orbit = _partners(r, c, size)
                assigned |= orbit
                orbits.append(orbit)
    counts = {}
    for row in grid:
        for value in row:
            counts[value] = counts.get(value, 0) + 1
    # A picture colour fails as the mask: some of its cells have only partners
    # of the same colour, which leaves them nothing to be filled from.  Among
    # colours that pass, the mask is the one painted over the fewest cells.
    for mask in sorted(counts, key=lambda v: counts[v]):
        restored = _restore(grid, mask, orbits)
        if restored is not None and restored != grid:
            return restored
    return [list(row) for row in grid]
