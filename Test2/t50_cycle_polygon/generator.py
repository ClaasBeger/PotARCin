"""
Generator for t50: Functional-graph cycle detection → polygon drawing.

Layout
------
Nodes occupy a regular (R_n × C_n) sub-grid inside the pixel grid.
Each node sits at pixel position

    (r0 + ri * s,  c0 + ci * s)

where (ri, ci) is its 0-based node-grid index, r0/c0 are the top-left
offsets, and s is the node spacing (s >= 3).

Each node has exactly one outgoing direction (up/down/left/right) stored as
a unit vector (dr, dc).  The orange (7) indicator is placed one pixel from
the node in that direction.  The next node is s pixels away in the same
direction.

Well-formedness constraints enforced at generation time
-------------------------------------------------------
  1. Node spacing s >= 3  →  orange cells never coincide with other nodes.
  2. Directions only point to valid in-grid neighbours  →  no out-of-bounds
     arrows; orange cells are always at least 1 pixel inside the grid border.
  3. Cycle length >= 4.
  4. The highlighted node is NOT in the cycle.
  5. The cycle spans >= 2 distinct pixel rows AND >= 2 distinct pixel columns
     (guarantees a visible polygon, not a degenerate line segment).
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

Grid = List[List[int]]


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _empty(rows: int, cols: int) -> Grid:
    return [[0] * cols for _ in range(rows)]


def _node_pixel(r0: int, c0: int, s: int, ri: int, ci: int) -> Tuple[int, int]:
    """Return the pixel coordinate of node-grid position (ri, ci)."""
    return (r0 + ri * s, c0 + ci * s)


def _valid_dirs(ri: int, ci: int, R_n: int, C_n: int) -> List[Tuple[int, int]]:
    """
    Return the list of valid unit-vector directions for the node at node-grid
    position (ri, ci).  A direction is valid if it leads to an in-grid neighbour.
    """
    result: List[Tuple[int, int]] = []
    if ri > 0:       result.append((-1,  0))   # up
    if ri < R_n - 1: result.append(( 1,  0))   # down
    if ci > 0:       result.append(( 0, -1))   # left
    if ci < C_n - 1: result.append(( 0,  1))   # right
    return result


def _detect_cycle(
    start: Tuple[int, int],
    graph: Dict[Tuple[int, int], Tuple[int, int]],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Follow the functional graph from *start* until a node is revisited.

    Returns (full_path, cycle) where *cycle* is the list of nodes in the
    detected cycle, in traversal order, beginning with the first repeated node.
    Returns an empty cycle list if no cycle is found (malformed graph).
    """
    path: List[Tuple[int, int]] = [start]
    visited: Dict[Tuple[int, int], int] = {start: 0}
    current = start
    while True:
        nxt = graph.get(current)
        if nxt is None:
            return path, []
        if nxt in visited:
            return path, path[visited[nxt]:]
        visited[nxt] = len(path)
        path.append(nxt)
        current = nxt


def _draw_cycle(out: Grid, cycle: List[Tuple[int, int]], color: int) -> None:
    """
    Draw straight lines between every consecutive pair of cycle nodes
    (including the wrap-around pair last → first) using *color*.
    Lines are axis-aligned; both endpoints are included.
    """
    n = len(cycle)
    for i in range(n):
        r1, c1 = cycle[i]
        r2, c2 = cycle[(i + 1) % n]
        if r1 == r2:                            # horizontal
            for col in range(min(c1, c2), max(c1, c2) + 1):
                out[r1][col] = color
        else:                                   # vertical  (c1 == c2)
            for row in range(min(r1, r2), max(r1, r2) + 1):
                out[row][c1] = color


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def generate(
    R_n: Optional[int] = None,
    C_n: Optional[int] = None,
    s: Optional[int] = None,
    r0: Optional[int] = None,
    c0: Optional[int] = None,
    grid_rows: Optional[int] = None,
    grid_cols: Optional[int] = None,
    dirs: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None,
    highlight_pos: Optional[Tuple[int, int]] = None,
    highlight_color: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> dict:
    """
    Generate one t50 ARC puzzle pair  {"input": Grid, "output": Grid}.

    Parameters (all optional — omit for random choices)
    ---------------------------------------------------
    R_n, C_n        : Node-grid dimensions (both >= 2).
    s               : Node spacing in pixels (>= 3; 3 or 4 recommended).
    r0, c0          : Pixel offset of the top-left node (>= 1).
    grid_rows/cols  : Total pixel-grid dimensions (auto-sized if omitted).
    dirs            : Mapping  (ri, ci) -> (dr, dc)  for every node.
                      (dr, dc) must be a valid unit direction for that node.
    highlight_pos   : Node-grid (ri, ci) of the highlighted start node.
    highlight_color : Integer colour for the highlighted node (1–9, ≠ 5, 7).
    rng             : Random source (uses random.Random() if omitted).

    Raises
    ------
    RuntimeError if no valid sample is found after 500 attempts.
    ValueError   if fixed parameters produce a constraint violation.
    """
    rr = rng if rng is not None else random.Random()
    _FIXED = dirs is not None and highlight_pos is not None

    for attempt in range(500):

        # ---------------------------------------------------------------- #
        # Choose layout                                                     #
        # ---------------------------------------------------------------- #
        rn = R_n if R_n is not None else rr.randint(2, 5)
        cn = C_n if C_n is not None else rr.randint(2, 5)
        sp = s if s is not None else rr.choice([3, 4])
        row_off = r0 if r0 is not None else rr.randint(1, 3)
        col_off = c0 if c0 is not None else rr.randint(1, 3)

        # Grid size: enough room for nodes + at least 1 border pixel on each side
        min_gr = row_off + (rn - 1) * sp + 2
        min_gc = col_off + (cn - 1) * sp + 2
        gr = max(grid_rows or 0, min_gr + rr.randint(0, 2))
        gc = max(grid_cols or 0, min_gc + rr.randint(0, 2))

        # ---------------------------------------------------------------- #
        # Assign directions                                                 #
        # ---------------------------------------------------------------- #
        if dirs is not None:
            chosen_dirs: Dict[Tuple[int, int], Tuple[int, int]] = dirs
        else:
            chosen_dirs = {}
            for ri in range(rn):
                for ci in range(cn):
                    valid = _valid_dirs(ri, ci, rn, cn)
                    chosen_dirs[(ri, ci)] = rr.choice(valid)

        # ---------------------------------------------------------------- #
        # Build the pixel-level functional graph                           #
        # ---------------------------------------------------------------- #
        pixel_graph: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for (ri, ci), (dr, dc) in chosen_dirs.items():
            src = _node_pixel(row_off, col_off, sp, ri,      ci)
            dst = _node_pixel(row_off, col_off, sp, ri + dr, ci + dc)
            pixel_graph[src] = dst

        # ---------------------------------------------------------------- #
        # Choose highlighted node                                          #
        # ---------------------------------------------------------------- #
        if highlight_pos is not None:
            hl_ri, hl_ci = highlight_pos
            hl_pixel = _node_pixel(row_off, col_off, sp, hl_ri, hl_ci)
        else:
            hl_ri = rr.randint(0, rn - 1)
            hl_ci = rr.randint(0, cn - 1)
            hl_pixel = _node_pixel(row_off, col_off, sp, hl_ri, hl_ci)

        # ---------------------------------------------------------------- #
        # Detect cycle and check constraints                               #
        # ---------------------------------------------------------------- #
        _, cycle = _detect_cycle(hl_pixel, pixel_graph)

        def _fail(msg: str) -> None:
            if _FIXED:
                raise ValueError(msg)

        if len(cycle) < 4:
            _fail("Cycle length < 4.")
            continue

        if hl_pixel in cycle:
            _fail("Highlighted node is in the cycle.")
            continue

        cycle_rows = {r for r, _ in cycle}
        cycle_cols = {c for _, c in cycle}
        if len(cycle_rows) < 2 or len(cycle_cols) < 2:
            _fail("Cycle does not span >=2 rows and >=2 cols.")
            continue

        # ---------------------------------------------------------------- #
        # Choose highlight colour                                          #
        # ---------------------------------------------------------------- #
        if highlight_color is not None:
            hl_color = highlight_color
        else:
            hl_color = rr.choice([v for v in range(1, 10) if v not in (5, 7)])

        # ---------------------------------------------------------------- #
        # Build input grid                                                 #
        # ---------------------------------------------------------------- #
        inp = _empty(gr, gc)
        for ri in range(rn):
            for ci in range(cn):
                r_px, c_px = _node_pixel(row_off, col_off, sp, ri, ci)
                inp[r_px][c_px] = hl_color if (r_px, c_px) == hl_pixel else 5
                dr, dc = chosen_dirs[(ri, ci)]
                inp[r_px + dr][c_px + dc] = 7      # orange arrow

        # ---------------------------------------------------------------- #
        # Build output grid                                                #
        # ---------------------------------------------------------------- #
        out = [row[:] for row in inp]
        out[hl_pixel[0]][hl_pixel[1]] = 5          # restore start to gray
        _draw_cycle(out, cycle, hl_color)

        return {"input": inp, "output": out}

    raise RuntimeError("Could not generate a valid t50 sample after 500 attempts.")
