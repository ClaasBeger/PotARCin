"""b2862040: recolour every shape that closes a loop around empty ground.

The picture is a field of boxes drawn in one colour, some with a complete rim
and some with a bite taken out of it, plus stray pixels stuck to their corners.
A box whose rim is complete traps ground inside it and is redrawn in cyan,
strays and all; a box with a gap in its rim is left as it is.

Each shape is judged on its own: only its own cells count as wall, and every
other cell -- including the cells of neighbouring shapes -- counts as open
ground. Otherwise a pixel pinched between the strays of two different boxes
would read as trapped and would wrongly close both of them.
"""

CLOSED = 8  # the colour this task uses for a shape that holds ground inside

_NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _background(grid):
    counts = {}
    for row in grid:
        for value in row:
            counts[value] = counts.get(value, 0) + 1
    return max(counts, key=lambda colour: (counts[colour], -colour))


def _shapes(grid, h, w, background):
    """4-connected groups of same-coloured non-background cells."""
    seen = [[False] * w for _ in range(h)]
    groups = []
    for r in range(h):
        for c in range(w):
            if seen[r][c] or grid[r][c] == background:
                continue
            colour = grid[r][c]
            seen[r][c] = True
            group = [(r, c)]
            stack = [(r, c)]
            while stack:
                y, x = stack.pop()
                for dy, dx in _NEIGHBOURS:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny][nx] \
                            and grid[ny][nx] == colour:
                        seen[ny][nx] = True
                        group.append((ny, nx))
                        stack.append((ny, nx))
            groups.append(group)
    return groups


def _traps_ground(h, w, wall):
    """True if some cell cannot reach the outside without crossing the wall."""
    blocked = set(wall)
    reached = set()
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if (r, c) not in blocked and (r, c) not in reached:
                reached.add((r, c))
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if (r, c) not in blocked and (r, c) not in reached:
                reached.add((r, c))
                stack.append((r, c))
    while stack:
        y, x = stack.pop()
        for dy, dx in _NEIGHBOURS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in blocked \
                    and (ny, nx) not in reached:
                reached.add((ny, nx))
                stack.append((ny, nx))
    return len(reached) + len(blocked) < h * w


def solve(grid: list[list[int]]) -> list[list[int]]:
    try:
        h = len(grid)
        w = len(grid[0]) if h else 0
        if not h or not w:
            return [list(row) for row in grid]

        background = _background(grid)
        out = [list(row) for row in grid]
        for group in _shapes(grid, h, w, background):
            if _traps_ground(h, w, group):
                for y, x in group:
                    out[y][x] = CLOSED
        return out
    except Exception:
        return [list(row) for row in grid]
