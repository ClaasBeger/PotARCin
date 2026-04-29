"""Generator for Test2/t46: 4-color sudoku over 2x2 quadrants.

Rule:
- The grid is always 5x5.
- The middle row and middle column are gray (5), splitting the grid into four
  2x2 quadrants.
- The four non-gray colors must fill the sixteen non-gray cells.
- No row, column, or 2x2 quadrant may contain the same color twice.
- Inputs reveal exactly four colored cells, one of each color; the output is
  the unique valid completion.

Generation constraints used to keep examples well-formed and unambiguous:
- the gray cross is fixed at row 2 and column 2;
- exactly four distinct non-gray/non-black colors are used;
- exactly four givens are shown, one per color;
- the partial puzzle has exactly one valid completion;
- generated givens are spread across at least three quadrants when possible.
"""

from __future__ import annotations

import itertools
import json
import random
from typing import Dict, List, Sequence, Tuple

Grid = List[List[int]]
Cell = Tuple[int, int]

BG = 0
GRAY = 5
NON_GRAY_CELLS: list[Cell] = [
    (r, c)
    for r in (0, 1, 3, 4)
    for c in (0, 1, 3, 4)
]
PALETTE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _empty() -> Grid:
    grid = [[BG for _ in range(5)] for _ in range(5)]
    for i in range(5):
        grid[2][i] = GRAY
        grid[i][2] = GRAY
    return grid


def _quadrant_index(r: int, c: int) -> int:
    if r < 2 and c < 2:
        return 0
    if r < 2 and c > 2:
        return 1
    if r > 2 and c < 2:
        return 2
    if r > 2 and c > 2:
        return 3
    raise ValueError("gray-cross cell has no quadrant")


def _valid_grid_shape(grid: Grid) -> bool:
    return len(grid) == 5 and all(len(row) == 5 for row in grid)


def _check_gray_cross(grid: Grid) -> None:
    if not _valid_grid_shape(grid):
        raise ValueError("grid must be 5x5")
    for i in range(5):
        if grid[2][i] != GRAY or grid[i][2] != GRAY:
            raise ValueError("grid must contain a gray center row and column")


def _colors_from_grid(grid: Grid) -> list[int]:
    colors = sorted({grid[r][c] for r, c in NON_GRAY_CELLS if grid[r][c] not in (BG, GRAY)})
    if len(colors) != 4:
        raise ValueError("expected exactly four non-gray colors")
    return colors


def _is_consistent_assignment(assign: Dict[Cell, int]) -> bool:
    # Rows.
    for r in (0, 1, 3, 4):
        vals = [assign[(r, c)] for c in (0, 1, 3, 4) if (r, c) in assign]
        if len(vals) != len(set(vals)):
            return False

    # Columns.
    for c in (0, 1, 3, 4):
        vals = [assign[(r, c)] for r in (0, 1, 3, 4) if (r, c) in assign]
        if len(vals) != len(set(vals)):
            return False

    # Quadrants.
    for q in range(4):
        vals = [
            v
            for (r, c), v in assign.items()
            if _quadrant_index(r, c) == q
        ]
        if len(vals) != len(set(vals)):
            return False

    return True


def _solve(colors: Sequence[int], givens: Dict[Cell, int], max_solutions: int = 2) -> list[Dict[Cell, int]]:
    cells = NON_GRAY_CELLS[:]
    solutions: list[Dict[Cell, int]] = []

    if not _is_consistent_assignment(givens):
        return []

    rows = {r: set() for r in (0, 1, 3, 4)}
    cols = {c: set() for c in (0, 1, 3, 4)}
    quads = {q: set() for q in range(4)}
    assign: Dict[Cell, int] = {}

    for cell, val in givens.items():
        r, c = cell
        q = _quadrant_index(r, c)
        if val not in colors or val in rows[r] or val in cols[c] or val in quads[q]:
            return []
        rows[r].add(val)
        cols[c].add(val)
        quads[q].add(val)
        assign[cell] = val

    def candidates(cell: Cell) -> list[int]:
        r, c = cell
        q = _quadrant_index(r, c)
        return [v for v in colors if v not in rows[r] and v not in cols[c] and v not in quads[q]]

    def backtrack() -> None:
        if len(solutions) >= max_solutions:
            return
        unfilled = [cell for cell in cells if cell not in assign]
        if not unfilled:
            solutions.append(dict(assign))
            return

        cell = min(unfilled, key=lambda x: (len(candidates(x)), x))
        for val in candidates(cell):
            r, c = cell
            q = _quadrant_index(r, c)
            assign[cell] = val
            rows[r].add(val)
            cols[c].add(val)
            quads[q].add(val)
            backtrack()
            quads[q].remove(val)
            cols[c].remove(val)
            rows[r].remove(val)
            del assign[cell]
            if len(solutions) >= max_solutions:
                return

    backtrack()
    return solutions


def _assignment_to_grid(assign: Dict[Cell, int]) -> Grid:
    grid = _empty()
    for (r, c), val in assign.items():
        grid[r][c] = val
    return grid


def transform_t46(grid: Grid) -> Grid:
    _check_gray_cross(grid)
    colors = _colors_from_grid(grid)
    givens = {(r, c): grid[r][c] for r, c in NON_GRAY_CELLS if grid[r][c] != BG}
    sols = _solve(colors, givens, max_solutions=2)
    if len(sols) != 1:
        raise ValueError("input does not have a unique valid completion")
    return _assignment_to_grid(sols[0])


def _canonical_solution(symbols: Sequence[int]) -> Dict[Cell, int]:
    """Return a random-symbolized 4x4 sudoku solution.

    Base pattern is a valid 4x4 Latin sudoku with 2x2 boxes:
        0 1 | 2 3
        2 3 | 0 1
        ---------
        1 0 | 3 2
        3 2 | 1 0
    """
    pattern = [
        [0, 1, 2, 3],
        [2, 3, 0, 1],
        [1, 0, 3, 2],
        [3, 2, 1, 0],
    ]
    real_rows = [0, 1, 3, 4]
    real_cols = [0, 1, 3, 4]
    return {
        (real_rows[i], real_cols[j]): symbols[pattern[i][j]]
        for i in range(4)
        for j in range(4)
    }


def _permute_solution(rng: random.Random, assign: Dict[Cell, int]) -> Dict[Cell, int]:
    # Permutations that preserve rows/columns/quadrants:
    # swap rows within top/bottom bands, swap bands, same for columns.
    row_bands = [[0, 1], [3, 4]]
    col_bands = [[0, 1], [3, 4]]
    for band in row_bands:
        if rng.random() < 0.5:
            band.reverse()
    for band in col_bands:
        if rng.random() < 0.5:
            band.reverse()
    if rng.random() < 0.5:
        row_bands.reverse()
    if rng.random() < 0.5:
        col_bands.reverse()
    row_map = dict(zip([0, 1, 3, 4], row_bands[0] + row_bands[1]))
    col_map = dict(zip([0, 1, 3, 4], col_bands[0] + col_bands[1]))
    return {(row_map[r], col_map[c]): v for (r, c), v in assign.items()}


def _choose_unique_givens(
    rng: random.Random,
    solution: Dict[Cell, int],
    colors: Sequence[int],
) -> Dict[Cell, int] | None:
    cells_by_color: dict[int, list[Cell]] = {c: [] for c in colors}
    for cell, val in solution.items():
        cells_by_color[val].append(cell)

    # One given per color. Try random choices until the completion is unique.
    for _ in range(2000):
        chosen: Dict[Cell, int] = {}
        for color in colors:
            cell = rng.choice(cells_by_color[color])
            chosen[cell] = color
        if len(chosen) != 4:
            continue
        if len({_quadrant_index(*cell) for cell in chosen}) < 3:
            continue
        sols = _solve(colors, chosen, max_solutions=2)
        if len(sols) == 1 and sols[0] == solution:
            return chosen
    return None


def generate(
    rng: random.Random | None = None,
    colors: Sequence[int] | None = None,
) -> dict[str, Grid]:
    rng = rng or random.Random()
    for _attempt in range(1000):
        cols = list(colors) if colors is not None else rng.sample(PALETTE_COLORS, 4)
        if len(cols) != 4 or len(set(cols)) != 4 or GRAY in cols or BG in cols:
            continue

        rng.shuffle(cols)
        solution = _canonical_solution(cols)
        solution = _permute_solution(rng, solution)

        givens = _choose_unique_givens(rng, solution, cols)
        if givens is None:
            continue

        inp = _empty()
        for cell, val in givens.items():
            r, c = cell
            inp[r][c] = val
        out = _assignment_to_grid(solution)
        if transform_t46(inp) == out:
            return {"input": inp, "output": out}

    raise RuntimeError("failed to generate a valid t46 sample")


if __name__ == "__main__":
    print(json.dumps(generate(rng=random.Random(0)), indent=2))
