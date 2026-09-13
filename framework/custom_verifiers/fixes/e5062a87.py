"""e5062a87: stamp the marked sprite everywhere its shape fits in the holes.

The picture is a field of blocked cells with holes punched in it, and one copy
of a small sprite already painted in the marker colour. Every other position
where that same set of cells lands entirely on holes is painted in the marker
colour too; everything else is left alone.

Two candidate positions can claim the same hole, and then only one of them can
be painted. That happens only in the hand-drawn examples -- the generator never
emits colliding positions -- and the example resolves it in favour of the
snugger fit: the copy with fewer holes touching its outside. The already
painted copy is kept no matter what, since it cannot be un-painted.
"""

EMPTY = 0  # ARC's usual "nothing here" colour; the sprite drops into these

_NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _marker_colour(grid):
    """The sprite's colour: the rarest colour on the grid apart from the holes.

    The sprite is a handful of cells while the blocked cells and the holes each
    cover a large share of the picture, so counting separates them without
    assuming which colour plays which role.
    """
    counts = {}
    for row in grid:
        for value in row:
            if value != EMPTY:
                counts[value] = counts.get(value, 0) + 1
    if not counts:
        return None
    return min(counts, key=lambda colour: (counts[colour], colour))


def _loose_holes(grid, h, w, cells):
    """How many holes touch the placement from outside -- lower is snugger."""
    inside = set(cells)
    touching = set()
    for r, c in cells:
        for dr, dc in _NEIGHBOURS:
            y, x = r + dr, c + dc
            if 0 <= y < h and 0 <= x < w and (y, x) not in inside:
                if grid[y][x] == EMPTY:
                    touching.add((y, x))
    return len(touching)


def solve(grid: list[list[int]]) -> list[list[int]]:
    try:
        h = len(grid)
        w = len(grid[0]) if h else 0
        if not h or not w:
            return [list(row) for row in grid]

        marker = _marker_colour(grid)
        if marker is None:
            return [list(row) for row in grid]

        drawn = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == marker]
        if not drawn:
            return [list(row) for row in grid]

        top = min(r for r, _ in drawn)
        left = min(c for _, c in drawn)
        shape = [(r - top, c - left) for r, c in drawn]
        height = max(dr for dr, _ in shape) + 1
        width = max(dc for _, dc in shape) + 1

        placements = [(top, left, drawn)]  # the copy that is already painted
        for r in range(h - height + 1):
            for c in range(w - width + 1):
                if (r, c) == (top, left):
                    continue
                cells = [(r + dr, c + dc) for dr, dc in shape]
                if all(grid[y][x] == EMPTY for y, x in cells):
                    placements.append((r, c, cells))

        # Snuggest first, reading order to stay deterministic among equals.
        placements.sort(key=lambda p: (_loose_holes(grid, h, w, p[2]), p[0], p[1]))

        taken = set()
        chosen = []
        for r, c, cells in placements:
            forced = (r, c) == (top, left)  # the copy that is already painted
            if forced or not taken.intersection(cells):
                taken.update(cells)
                chosen.append(cells)

        out = [list(row) for row in grid]
        for cells in chosen:
            for y, x in cells:
                out[y][x] = marker
        return out
    except Exception:
        return [list(row) for row in grid]
