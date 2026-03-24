from __future__ import annotations

from typing import Callable, Optional

from framework.grids import Grid


def _solve_6cf79266(grid: Grid) -> Grid:
    """Explicit one-pass 3x3 fill mirroring ARC-GEN task162 behavior."""
    out = [list(row) for row in grid]
    h = len(out)
    w = len(out[0]) if h else 0

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            neighborhood_total = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    neighborhood_total += out[r + dr][c + dc]
            if neighborhood_total:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    out[r + dr][c + dc] = 1
    return out


def get_custom_verifier(task_id: str) -> Optional[Callable[[Grid], Grid]]:
    """Return optional project-local verifier overrides/additions."""
    if task_id == "6cf79266":
        return _solve_6cf79266
    return None
