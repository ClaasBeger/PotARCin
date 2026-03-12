"""
Core package for the ARC unified assessment framework.

This package is intentionally lightweight at this stage and only exposes
the most central types and helper functions so that experiments can
import them without depending on internal layout details.
"""

from .grids import Grid, GridPair, is_equal_grid, pretty_grid  # noqa: F401

__all__ = [
    "Grid",
    "GridPair",
    "is_equal_grid",
    "pretty_grid",
]

