from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


Color = int


Grid = List[List[Color]]


@dataclass(frozen=True)
class GridPair:
    """Simple container for an input/output grid pair."""

    input: Grid
    output: Grid


def is_equal_grid(a: Grid, b: Grid) -> bool:
    """Return True if two grids are structurally and element-wise equal."""
    if len(a) != len(b):
        return False
    return all(row_a == row_b for row_a, row_b in zip(a, b))


def pretty_grid(grid: Grid) -> str:
    """Return a human-readable string representation of a grid."""
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def validate_grid(grid: Grid) -> None:
    """Validate that a grid is a non-empty rectangular matrix of ints.

    Raises:
        ValueError: If the grid is malformed.
    """
    if not grid:
        raise ValueError("Grid must be non-empty")
    row_length = len(grid[0])
    if row_length == 0:
        raise ValueError("Grid rows must be non-empty")
    for row in grid:
        if len(row) != row_length:
            raise ValueError("Grid must be rectangular (all rows same length)")
        for cell in row:
            if not isinstance(cell, int):
                raise ValueError("Grid cells must be integers")


def clone_grid(grid: Grid) -> Grid:
    """Return a deep copy of a grid."""
    return [list(row) for row in grid]


def batch_equal(
    a: Sequence[Grid],
    b: Sequence[Grid],
) -> bool:
    """Return True if two sequences of grids are all pairwise equal."""
    if len(a) != len(b):
        return False
    return all(is_equal_grid(ga, gb) for ga, gb in zip(a, b))


def iter_cells(grid: Grid) -> Iterable[tuple[int, int, Color]]:
    """Yield (row_idx, col_idx, color) triples for all cells in a grid."""
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            yield r, c, color

