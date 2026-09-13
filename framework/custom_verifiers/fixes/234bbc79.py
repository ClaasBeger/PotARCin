"""ARC 234bbc79: stitch the coloured segments into one strip at the grey joints.

Each segment is a little path whose end cells are grey wherever it meets a
neighbour: the left-hand grey is a socket, the right-hand grey a plug.  Working
left to right, every segment is slid so that its socket sits immediately to the
right of the previous segment's plug, and each grey cell is repainted in the
colour of the segment it belongs to.  The stitched strip is redrawn on a grid as
tall as the input and only as wide as the strip itself, which drops the blank
spacer columns the input used to keep the segments apart.

The old verifier first had to guess which colour was the connector, by looking
for a colour present in every segment at most twice.  When a segment colour
happened to satisfy that too and tie on frequency it could pick the wrong one,
run out of joints part way through the chain and raise.  Grey is the connector
by the task's definition, so it is simply named here.
"""

GREY = 5
BACKGROUND = 0


def _segments(grid: list[list[int]], height: int, width: int) -> list[list[tuple[int, int]]]:
    """4-connected components of non-background cells."""
    seen = [[False] * width for _ in range(height)]
    found = []
    for row in range(height):
        for col in range(width):
            if grid[row][col] == BACKGROUND or seen[row][col]:
                continue
            seen[row][col] = True
            stack = [(row, col)]
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for d_r, d_c in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n_r, n_c = r + d_r, c + d_c
                    if not (0 <= n_r < height and 0 <= n_c < width):
                        continue
                    if seen[n_r][n_c] or grid[n_r][n_c] == BACKGROUND:
                        continue
                    seen[n_r][n_c] = True
                    stack.append((n_r, n_c))
            found.append(cells)
    return found


def solve(grid: list[list[int]]) -> list[list[int]]:
    try:
        height = len(grid)
        width = len(grid[0]) if height else 0
        if not height or not width:
            return grid

        segments = _segments(grid, height, width)
        if len(segments) < 2:
            return grid
        segments.sort(key=lambda cells: min(c for _, c in cells))

        placed: dict[tuple[int, int], int] = {}
        join: tuple[int, int] | None = None  # where the next socket must land
        for index, cells in enumerate(segments):
            # Sorted by column, so the first grey is the socket and the last the
            # plug; a segment spans at least two columns, so they never coincide.
            greys = sorted(((c, r) for r, c in cells if grid[r][c] == GREY))
            body = {grid[r][c] for r, c in cells} - {GREY}
            if len(body) != 1 or not greys:
                return grid
            needs_plug = index + 1 < len(segments)
            if index and needs_plug and len(greys) < 2:
                return grid
            colour = body.pop()

            if index == 0:
                d_row = d_col = 0
            else:
                socket_col, socket_row = greys[0]
                d_row = join[0] - socket_row
                d_col = join[1] + 1 - socket_col
            for r, c in cells:
                placed[(r + d_row, c + d_col)] = colour
            if needs_plug:
                plug_col, plug_row = greys[-1]
                join = (plug_row + d_row, plug_col + d_col)

        left = min(c for _, c in placed)
        right = max(c for _, c in placed)
        out = [[BACKGROUND] * (right - left + 1) for _ in range(height)]
        for (r, c), colour in placed.items():
            if 0 <= r < height:
                out[r][c - left] = colour
        return out
    except Exception:
        return grid
