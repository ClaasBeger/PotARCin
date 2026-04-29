"""
Verifier for t50: Functional-graph cycle detection → polygon drawing.

Rule
----
The input is a pixel grid containing:
  • Gray (5) nodes arranged on a regular sub-grid.
  • Exactly one highlighted node (any colour ≠ 0, 5, 7) — the start.
  • For every node (gray or highlighted) exactly one adjacent (4-connected)
    orange (7) cell indicating the direction of travel to the next node.

Starting from the highlighted node, follow arrows node-to-node until a node
is visited for the second time.  That revisited node is the first element of
the cycle; all nodes from that point up to (but not including) the revisit
form the cycle.

Output construction
-------------------
  1. Copy input.
  2. Restore the highlighted node to gray (5); its orange arrow is kept.
  3. For each consecutive pair of cycle nodes (including the wrap-around pair
     last → first), draw a straight horizontal or vertical line through every
     pixel between them (inclusive) in the highlight colour.  This overwrites
     any content already on those pixels (cycle node markers and their
     orange arrows are naturally covered by the lines that pass through them).

Well-formedness assumptions (the verifier does not validate these):
  • Every node has exactly one adjacent orange.
  • Every orange points toward another node that is in-bounds.
  • The highlighted node is NOT itself a member of the cycle.
  • The cycle spans at least 4 nodes, at least 2 distinct rows, at least 2
    distinct columns — guaranteeing a visible, non-degenerate polygon.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

Grid = List[List[int]]


def verify_t50(grid: Grid) -> Grid:
    H = len(grid)
    if H == 0:
        return []
    W = len(grid[0])
    if W == 0:
        return [[] for _ in range(H)]

    # ------------------------------------------------------------------ #
    # 1. Locate every node and the unique highlighted node.               #
    # ------------------------------------------------------------------ #
    node_set: set[Tuple[int, int]] = set()
    highlight_pos: Optional[Tuple[int, int]] = None
    highlight_color: Optional[int] = None

    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            if v == 5:
                node_set.add((r, c))
            elif v not in (0, 7):
                node_set.add((r, c))
                highlight_pos = (r, c)
                highlight_color = v

    if highlight_pos is None:
        return [row[:] for row in grid]

    # ------------------------------------------------------------------ #
    # 2. For each node find its direction from the adjacent orange cell.  #
    # ------------------------------------------------------------------ #
    directions: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for r, c in node_set:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and grid[rr][cc] == 7:
                directions[(r, c)] = (dr, dc)
                break  # exactly one orange per node

    # ------------------------------------------------------------------ #
    # 3. Build the functional graph: each node maps to its successor.     #
    #    Walk pixel-by-pixel in the indicated direction until the next    #
    #    node is found; this handles any node spacing >= 3 automatically. #
    # ------------------------------------------------------------------ #
    graph: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for (r, c), (dr, dc) in directions.items():
        nr, nc = r + dr, c + dc
        while 0 <= nr < H and 0 <= nc < W and (nr, nc) not in node_set:
            nr += dr
            nc += dc
        if 0 <= nr < H and 0 <= nc < W and (nr, nc) in node_set:
            graph[(r, c)] = (nr, nc)

    # ------------------------------------------------------------------ #
    # 4. Detect the cycle by following the path from the highlighted node.#
    # ------------------------------------------------------------------ #
    path: List[Tuple[int, int]] = [highlight_pos]
    visited: Dict[Tuple[int, int], int] = {highlight_pos: 0}
    current = highlight_pos
    cycle: Optional[List[Tuple[int, int]]] = None

    while True:
        nxt = graph.get(current)
        if nxt is None:
            break  # dead end — malformed input
        if nxt in visited:
            # nxt is the first repeated node; the cycle starts there
            cycle = path[visited[nxt]:]
            break
        visited[nxt] = len(path)
        path.append(nxt)
        current = nxt

    if cycle is None:
        return [row[:] for row in grid]

    # ------------------------------------------------------------------ #
    # 5. Build the output.                                                #
    # ------------------------------------------------------------------ #
    out = [row[:] for row in grid]

    # Restore the highlighted start node to plain gray
    out[highlight_pos[0]][highlight_pos[1]] = 5

    # Draw the cycle polygon: connect consecutive nodes with straight lines
    n = len(cycle)
    for i in range(n):
        r1, c1 = cycle[i]
        r2, c2 = cycle[(i + 1) % n]
        if r1 == r2:                          # horizontal segment
            for col in range(min(c1, c2), max(c1, c2) + 1):
                out[r1][col] = highlight_color
        else:                                 # vertical segment  (c1 == c2)
            for row in range(min(r1, r2), max(r1, r2) + 1):
                out[row][c1] = highlight_color

    return out
