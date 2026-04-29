"""Generator for Test2/t44: count rotations needed to align outline gaps.

Rule:
- The input has two cells of one marker color: one at the center of a nest of
  concentric square outlines, and one on the grid edge aligned horizontally or
  vertically with the center.
- Each square outline has exactly one gap at the midpoint of one side.
- Starting from the innermost outline and moving outward, count how many
  clockwise 90-degree rotations are needed to move that outline's gap to the
  side facing the edge marker / clear path.
- The output has width 3 and one row per outline. A count n in {0,1,2,3} is
  encoded by filling the first n cells of the row with that outline's color.

Generation constraints used to keep examples well-formed and unambiguous:
- exactly one marker color appears exactly twice, once at the center and once
  on an aligned edge;
- every outline is a square border centered on the center marker;
- every outline has exactly one midpoint-side gap;
- outlines are sorted by radius from inner to outer;
- same-colored outlines are not adjacent/merged into one component.
"""

from __future__ import annotations

import json
import random
from collections import deque
from typing import Iterable, List, Sequence, Set, Tuple

Grid = List[List[int]]
Cell = Tuple[int, int]

BG = 0
COLORS = list(range(1, 10))
NEI4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

SIDE_TO_INDEX = {"top": 0, "right": 1, "bottom": 2, "left": 3}
INDEX_TO_SIDE = {v: k for k, v in SIDE_TO_INDEX.items()}


def _empty(h: int, w: int) -> Grid:
    return [[BG for _ in range(w)] for _ in range(h)]


def _rectangular(grid: Grid) -> bool:
    return bool(grid) and bool(grid[0]) and all(len(row) == len(grid[0]) for row in grid)


def _is_edge(cell: Cell, h: int, w: int) -> bool:
    r, c = cell
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _find_markers(grid: Grid) -> tuple[int, Cell, Cell, str]:
    """Return marker color, center cell, edge cell, and target side."""
    h, w = len(grid), len(grid[0])
    by_color: dict[int, list[Cell]] = {}
    for r in range(h):
        for c in range(w):
            v = grid[r][c]
            if v != BG:
                by_color.setdefault(v, []).append((r, c))

    candidates: list[tuple[int, Cell, Cell, str]] = []
    for color, cells in by_color.items():
        if len(cells) != 2:
            continue
        edge_cells = [p for p in cells if _is_edge(p, h, w)]
        center_cells = [p for p in cells if not _is_edge(p, h, w)]
        if len(edge_cells) != 1 or len(center_cells) != 1:
            continue
        edge = edge_cells[0]
        center = center_cells[0]
        if edge[0] == center[0]:
            target = "left" if edge[1] < center[1] else "right"
        elif edge[1] == center[1]:
            target = "top" if edge[0] < center[0] else "bottom"
        else:
            continue
        candidates.append((color, center, edge, target))

    if len(candidates) != 1:
        raise ValueError("expected exactly one aligned center/edge marker pair")
    return candidates[0]


def _components(grid: Grid, exclude_color: int) -> list[tuple[int, set[Cell]]]:
    h, w = len(grid), len(grid[0])
    seen: set[Cell] = set()
    comps: list[tuple[int, set[Cell]]] = []
    for r in range(h):
        for c in range(w):
            color = grid[r][c]
            if color in (BG, exclude_color) or (r, c) in seen:
                continue
            q = deque([(r, c)])
            seen.add((r, c))
            cells: set[Cell] = set()
            while q:
                rr, cc = q.popleft()
                cells.add((rr, cc))
                for dr, dc in NEI4:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and (nr, nc) not in seen
                        and grid[nr][nc] == color
                    ):
                        seen.add((nr, nc))
                        q.append((nr, nc))
            comps.append((color, cells))
    return comps


def _border_cells(center: Cell, radius: int) -> set[Cell]:
    cr, cc = center
    cells: set[Cell] = set()
    for c in range(cc - radius, cc + radius + 1):
        cells.add((cr - radius, c))
        cells.add((cr + radius, c))
    for r in range(cr - radius, cr + radius + 1):
        cells.add((r, cc - radius))
        cells.add((r, cc + radius))
    return cells


def _gap_side(center: Cell, gap: Cell, radius: int) -> str:
    cr, cc = center
    r, c = gap
    if r == cr - radius and c == cc:
        return "top"
    if r == cr and c == cc + radius:
        return "right"
    if r == cr + radius and c == cc:
        return "bottom"
    if r == cr and c == cc - radius:
        return "left"
    raise ValueError("gap is not at the midpoint of a side")


def _parse_outlines(grid: Grid, marker_color: int, center: Cell) -> list[tuple[int, int, str]]:
    """Return outline tuples (radius, color, gap_side), sorted inner to outer."""
    outlines: list[tuple[int, int, str]] = []
    for color, cells in _components(grid, marker_color):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        cr, cc = center
        if (min_r + max_r) != 2 * cr or (min_c + max_c) != 2 * cc:
            raise ValueError("outline is not centered on marker")
        radius = max_r - cr
        if radius <= 0 or radius != cr - min_r or radius != max_c - cc or radius != cc - min_c:
            raise ValueError("outline bounding box is not a centered square")

        border = _border_cells(center, radius)
        if not cells <= border:
            raise ValueError("outline contains cells away from its square border")
        missing = border - cells
        if len(missing) != 1:
            raise ValueError("each outline must have exactly one gap")
        side = _gap_side(center, next(iter(missing)), radius)
        outlines.append((radius, color, side))

    if not outlines:
        raise ValueError("expected at least one outline")
    outlines.sort(key=lambda x: x[0])
    if len({r for r, _, _ in outlines}) != len(outlines):
        raise ValueError("each outline must have a distinct radius")
    return outlines


def _count_rotations(gap_side: str, target_side: str) -> int:
    return (SIDE_TO_INDEX[target_side] - SIDE_TO_INDEX[gap_side]) % 4


def transform_t44(grid: Grid) -> Grid:
    if not _rectangular(grid):
        raise ValueError("input grid must be non-empty and rectangular")
    marker_color, center, _edge, target_side = _find_markers(grid)
    outlines = _parse_outlines(grid, marker_color, center)

    out: Grid = []
    for _radius, color, gap in outlines:
        n = _count_rotations(gap, target_side)
        out.append([color if c < n else BG for c in range(3)])
    return out


def _draw_outline(grid: Grid, center: Cell, radius: int, color: int, gap_side: str) -> None:
    cells = _border_cells(center, radius)
    cr, cc = center
    gap_by_side = {
        "top": (cr - radius, cc),
        "right": (cr, cc + radius),
        "bottom": (cr + radius, cc),
        "left": (cr, cc - radius),
    }
    gap = gap_by_side[gap_side]
    for r, c in cells:
        if (r, c) != gap:
            grid[r][c] = color


def _edge_for_direction(center: Cell, h: int, w: int, target_side: str) -> Cell:
    r, c = center
    if target_side == "top":
        return (0, c)
    if target_side == "bottom":
        return (h - 1, c)
    if target_side == "left":
        return (r, 0)
    if target_side == "right":
        return (r, w - 1)
    raise ValueError(target_side)


def generate(
    rng: random.Random | None = None,
    num_outlines: int | None = None,
    target_side: str | None = None,
) -> dict[str, Grid]:
    rng = rng or random.Random()

    for _attempt in range(1000):
        n = num_outlines if num_outlines is not None else rng.randint(3, 6)
        target = target_side if target_side is not None else rng.choice(["top", "right", "bottom", "left"])
        if target not in SIDE_TO_INDEX:
            raise ValueError("target_side must be top/right/bottom/left")

        # Use step 1 or 2 between radii; step 2 gives visually separated rings.
        step = rng.choice([1, 2])
        start_radius = rng.randint(1, 2)
        radii = [start_radius + step * i for i in range(n)]
        max_r = radii[-1]

        margin_before = rng.randint(1, 3)
        margin_after = rng.randint(1, 3)
        if target in ("top", "bottom"):
            h = max_r + 1 + max_r + margin_before + margin_after + 1
            center_r = margin_before + max_r
            w = 2 * max_r + 1 + rng.randint(2, 5)
            center_c = rng.randint(max_r, w - max_r - 1)
        else:
            w = max_r + 1 + max_r + margin_before + margin_after + 1
            center_c = margin_before + max_r
            h = 2 * max_r + 1 + rng.randint(2, 5)
            center_r = rng.randint(max_r, h - max_r - 1)

        center = (center_r, center_c)
        edge = _edge_for_direction(center, h, w, target)
        marker_color = rng.choice(COLORS)
        outline_palette = [c for c in COLORS if c != marker_color]
        if len(outline_palette) < 2:
            continue

        grid = _empty(h, w)
        prev_color = None
        outline_specs: list[tuple[int, int, str]] = []
        for radius in radii:
            color_choices = [c for c in outline_palette if c != prev_color]
            color = rng.choice(color_choices)
            gap = rng.choice(["top", "right", "bottom", "left"])
            outline_specs.append((radius, color, gap))
            prev_color = color

        for radius, color, gap in outline_specs:
            _draw_outline(grid, center, radius, color, gap)

        grid[center_r][center_c] = marker_color
        grid[edge[0]][edge[1]] = marker_color

        try:
            out = transform_t44(grid)
        except Exception:
            continue

        # Prefer varied encodings, including at least one zero and one nonzero row.
        counts = [sum(1 for v in row if v != BG) for row in out]
        if len(set(counts)) < 2:
            continue
        if not any(c == 0 for c in counts):
            continue
        if not any(c == 3 for c in counts):
            continue
        return {"input": grid, "output": out}

    raise RuntimeError("failed to generate a valid t44 sample")


if __name__ == "__main__":
    print(json.dumps(generate(rng=random.Random(0)), indent=2))
