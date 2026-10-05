"""ARC e76a88a6: stamp the pattern onto every solid block of its size.

One object carries the pattern; the others are solid one-colour rectangles of
the same size (grey in the task), and each block is overwritten with a copy of
the pattern.  A pattern with several colours is the only such object; a pattern
of one colour is the one that is not grey.  The re_arc verifier took a grey block
as the pattern whenever the real one had a single colour, and the google verifier
miscopied patterns that are not square.
"""

GREY = 5
BACKGROUND = 0


def _objects(grid: list[list[int]]) -> list[list[tuple[int, int]]]:
    """Cells of each 4-connected group of non-background cells."""
    height, width = len(grid), len(grid[0])
    seen = [[False] * width for _ in range(height)]
    found = []
    for row in range(height):
        for col in range(width):
            if seen[row][col] or grid[row][col] == BACKGROUND:
                continue
            stack, cells = [(row, col)], []
            seen[row][col] = True
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr][nc] and grid[nr][nc] != BACKGROUND:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            found.append(cells)
    return found


def _box(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def solve(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return grid
    out = [list(row) for row in grid]
    objects = _objects(grid)
    solid = [cells for cells in objects if len({grid[r][c] for r, c in cells}) == 1 and len(cells) == _box(cells)[2] * _box(cells)[3]]
    patterns = [cells for cells in objects if not any(cells is s for s in solid)]
    if not patterns:
        patterns = [cells for cells in solid if grid[cells[0][0]][cells[0][1]] != GREY]
    if len(patterns) != 1:
        return out
    pattern = patterns[0]
    top, left, height, width = _box(pattern)
    for cells in solid:
        if cells is pattern:
            continue
        b_top, b_left, b_height, b_width = _box(cells)
        if (b_height, b_width) != (height, width) or len(cells) != height * width:
            continue
        for dr in range(height):
            for dc in range(width):
                out[b_top + dr][b_left + dc] = grid[top + dr][left + dc]
    return out
