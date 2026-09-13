"""6855a6e4: fold each loose piece over its gripper and into the box.

Two grippers face each other across an empty box: each is a straight bar with a
short lip turned in at both ends, and the bar is the side away from the box. A
loose piece lies outside each gripper. The answer mirrors every loose piece
across its gripper's bar, which lands both pieces inside the box, and clears
the ground they came from.

Mirroring across the bar -- rather than sliding the piece in until it touches --
is what keeps the gap between piece and bar the same on the way in as it was on
the way out.
"""

EMPTY = 0

_NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(cells):
    """4-connected groups of the given cells."""
    remaining = set(cells)
    groups = []
    while remaining:
        seed = remaining.pop()
        group = [seed]
        stack = [seed]
        while stack:
            r, c = stack.pop()
            for dr, dc in _NEIGHBOURS:
                nxt = (r + dr, c + dc)
                if nxt in remaining:
                    remaining.discard(nxt)
                    group.append(nxt)
                    stack.append(nxt)
        groups.append(group)
    return groups


def _longest_run(cells):
    """Length of the longest straight (horizontal or vertical) run of cells."""
    best = 0
    present = set(cells)
    for r, c in cells:
        for dr, dc in ((0, 1), (1, 0)):
            if (r - dr, c - dc) in present:
                continue  # only measure from the start of a run
            length = 1
            while (r + dr * length, c + dc * length) in present:
                length += 1
            best = max(best, length)
    return best


def _gripper_bar(cells):
    """The bar of one gripper: which edge of its box is solid, and its axis.

    Returns (axis, line, inward) where axis is 'row' or 'col', line is the index
    of the solid edge and inward is the step from the bar towards the box.
    """
    present = set(cells)
    top = min(r for r, _ in cells)
    bottom = max(r for r, _ in cells)
    left = min(c for _, c in cells)
    right = max(c for _, c in cells)
    candidates = []
    for line, inward in ((top, 1), (bottom, -1)):
        if all((line, c) in present for c in range(left, right + 1)):
            candidates.append((right - left + 1, 'row', line, inward))
    for line, inward in ((left, 1), (right, -1)):
        if all((r, line) in present for r in range(top, bottom + 1)):
            candidates.append((bottom - top + 1, 'col', line, inward))
    if not candidates:
        return None
    # A short side of the box can be solid by accident, so take the long one.
    _, axis, line, inward = max(candidates)
    return axis, line, inward


def solve(grid: list[list[int]]) -> list[list[int]]:
    try:
        h = len(grid)
        w = len(grid[0]) if h else 0
        if not h or not w:
            return [list(row) for row in grid]

        by_colour = {}
        for r in range(h):
            for c in range(w):
                value = grid[r][c]
                if value != EMPTY:
                    by_colour.setdefault(value, []).append((r, c))
        if len(by_colour) != 2:
            return [list(row) for row in grid]

        # The bars run the full width of the box while a loose piece only ever
        # spans the box's inside, so the longest straight run is a bar.
        gripper_colour = max(by_colour, key=lambda colour: (_longest_run(by_colour[colour]), colour))
        piece_colour = next(colour for colour in by_colour if colour != gripper_colour)

        bars = []
        for group in _components(by_colour[gripper_colour]):
            bar = _gripper_bar(group)
            if bar is None:
                return [list(row) for row in grid]
            axis, line, inward = bar
            if axis == 'row':
                span = (min(c for _, c in group), max(c for _, c in group))
            else:
                span = (min(r for r, _ in group), max(r for r, _ in group))
            bars.append((axis, line, inward, span))
        if not bars:
            return [list(row) for row in grid]

        out = [list(row) for row in grid]
        for r, c in by_colour[piece_colour]:
            landed = None
            for axis, line, inward, (lo, hi) in bars:
                across = r if axis == 'row' else c
                along = c if axis == 'row' else r
                # The piece sits outside the box, on the far side of the bar.
                if not lo <= along <= hi:
                    continue
                if (across - line) * inward >= 0:
                    continue
                mirrored = 2 * line - across
                landed = (mirrored, c) if axis == 'row' else (r, mirrored)
                break
            if landed is None:
                continue  # nothing claims it; leave it where it is
            y, x = landed
            if not (0 <= y < h and 0 <= x < w):
                continue
            out[r][c] = EMPTY
            out[y][x] = piece_colour
        return out
    except Exception:
        return [list(row) for row in grid]
