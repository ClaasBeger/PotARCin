"""Generator for Test2/t39: shortest directed path across cannon-islands.

Rule:
- The input contains several 3x3 hollow square "islands" laid out on a grid.
- The island borders use one outline color.
- A red cell (color 2) at the midpoint of one side of an island is a one-way
  cannon, allowing movement from that island to the adjacent island in that
  direction.
- One island center contains a yellow cell (color 4), the source.
- One island center contains a cyan cell (color 8), the target.
- Follow the unique shortest directed path from source to target.
- In the output, every red cannon used on that path is recolored gray (color 5),
  the source center is cleared to background, and the target center is recolored
  yellow.

Generation constraints used to keep examples well-formed and unambiguous:
- islands form a complete rectangular lattice of 2..5 rows and 2..5 columns;
- only valid in-bounds one-step directional cannons are used;
- there is exactly one source and one target;
- a path from source to target exists and the shortest directed path is unique;
- generated tasks prefer nontrivial paths of length at least 2;
- most islands are given at least one outgoing cannon so red markers are spread
  more uniformly across the lattice, making the true path less visually obvious;
- most generated examples also contain at least one longer alternative directed
  route from source to target, so the task is not just following the only
  apparent non-cycling path.
"""

from __future__ import annotations

import json
import random
from collections import deque
from typing import Dict, List, Tuple

Grid = List[List[int]]
Node = Tuple[int, int]

BG = 0
RED = 2
YELLOW = 4
GRAY = 5
CYAN = 8
OUTLINE_COLORS = [1, 3, 6, 7, 9]
DIRS = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1),
}
OPPOSITE = {'U': 'D', 'D': 'U', 'L': 'R', 'R': 'L'}
MID_POS = {
    'U': (0, 1),
    'D': (2, 1),
    'L': (1, 0),
    'R': (1, 2),
}


class Island:
    def __init__(self, top: int, left: int, center: int, mids: Dict[str, int]):
        self.top = top
        self.left = left
        self.center = center
        self.mids = dict(mids)


def _empty(h: int, w: int) -> Grid:
    return [[BG for _ in range(w)] for _ in range(h)]


def _rectangular(grid: Grid) -> bool:
    return bool(grid) and bool(grid[0]) and all(len(row) == len(grid[0]) for row in grid)


def _is_island_at(grid: Grid, r: int, c: int) -> bool:
    h, w = len(grid), len(grid[0])
    if r + 2 >= h or c + 2 >= w:
        return False
    border_positions = [(0,0),(0,1),(0,2),(1,0),(1,2),(2,0),(2,1),(2,2)]
    for dr, dc in border_positions:
        if grid[r + dr][c + dc] == BG:
            return False
    center = grid[r + 1][c + 1]
    if center not in (BG, YELLOW, CYAN):
        return False
    return True


def _parse_islands(grid: Grid) -> tuple[dict[Node, Island], list[int], list[int]]:
    if not _rectangular(grid):
        raise ValueError("input grid must be non-empty and rectangular")
    h, w = len(grid), len(grid[0])
    found: list[tuple[int, int]] = []
    for r in range(h - 2):
        for c in range(w - 2):
            if _is_island_at(grid, r, c):
                found.append((r, c))
    if not found:
        raise ValueError("no islands found")

    # islands are separated, so every valid top-left is distinct.
    rows = sorted({r for r, _ in found})
    cols = sorted({c for _, c in found})
    islands: dict[Node, Island] = {}
    for r, c in found:
        ri = rows.index(r)
        ci = cols.index(c)
        sub = [grid[r + rr][c:c + 3] for rr in range(3)]
        center = sub[1][1]
        mids = {d: sub[dr][dc] for d, (dr, dc) in MID_POS.items()}
        islands[(ri, ci)] = Island(r, c, center, mids)

    # Validate a full rectangular lattice.
    for ri in range(len(rows)):
        for ci in range(len(cols)):
            if (ri, ci) not in islands:
                raise ValueError("islands must form a complete rectangular lattice")
    return islands, rows, cols


def _graph_from_islands(islands: dict[Node, Island], nrows: int, ncols: int) -> dict[Node, list[tuple[Node, str]]]:
    graph: dict[Node, list[tuple[Node, str]]] = {node: [] for node in islands}
    for (r, c), isl in islands.items():
        for d, val in isl.mids.items():
            if val != RED:
                continue
            dr, dc = DIRS[d]
            nb = (r + dr, c + dc)
            if 0 <= nb[0] < nrows and 0 <= nb[1] < ncols:
                graph[(r, c)].append((nb, d))
    return graph


def _find_source_target(islands: dict[Node, Island]) -> tuple[Node, Node]:
    src = [node for node, isl in islands.items() if isl.center == YELLOW]
    tgt = [node for node, isl in islands.items() if isl.center == CYAN]
    if len(src) != 1 or len(tgt) != 1:
        raise ValueError("expected exactly one yellow source and one cyan target")
    return src[0], tgt[0]


def _shortest_path_unique(graph: dict[Node, list[tuple[Node, str]]], src: Node, tgt: Node) -> tuple[list[tuple[Node, str]], bool]:
    dist: dict[Node, int] = {src: 0}
    count: dict[Node, int] = {src: 1}
    parent: dict[Node, tuple[Node, str]] = {}
    q = deque([src])
    while q:
        u = q.popleft()
        for v, d in graph[u]:
            nd = dist[u] + 1
            if v not in dist:
                dist[v] = nd
                count[v] = count[u]
                parent[v] = (u, d)
                q.append(v)
            elif nd == dist[v]:
                count[v] = min(2, count[v] + count[u])
                # keep existing parent for deterministic reconstruction if needed
    if tgt not in dist:
        return [], False
    path: list[tuple[Node, str]] = []
    cur = tgt
    while cur != src:
        prev, d = parent[cur]
        path.append((prev, d))
        cur = prev
    path.reverse()
    return path, count[tgt] == 1


def _has_longer_alternative_path(
    graph: dict[Node, list[tuple[Node, str]]],
    src: Node,
    tgt: Node,
    shortest_len: int,
    max_extra: int = 6,
) -> bool:
    """Return whether there exists a simple source->target path longer than the shortest.

    We only search up to shortest_len + max_extra to keep generation fast.
    That is sufficient for generating plausible decoy routes.
    """
    max_len = shortest_len + max_extra

    def dfs(node: Node, depth: int, seen: set[Node]) -> bool:
        if depth > max_len:
            return False
        if node == tgt:
            return depth > shortest_len
        # Even in the best case, we can only increase length from here. If we have
        # already exceeded the cap, stop.
        for nxt, _ in graph[node]:
            if nxt in seen:
                continue
            seen.add(nxt)
            if dfs(nxt, depth + 1, seen):
                return True
            seen.remove(nxt)
        return False

    return dfs(src, 0, {src})


def transform_t39(grid: Grid) -> Grid:
    islands, rows, cols = _parse_islands(grid)
    nrows, ncols = len(rows), len(cols)
    graph = _graph_from_islands(islands, nrows, ncols)
    src, tgt = _find_source_target(islands)
    path, reachable_unique = _shortest_path_unique(graph, src, tgt)
    if not path:
        raise ValueError("no directed path from source to target")
    # Provided tasks / generated tasks are unique; if not unique, the generator avoids it.
    out = [row[:] for row in grid]
    # Clear source center.
    src_isl = islands[src]
    out[src_isl.top + 1][src_isl.left + 1] = BG
    # Recolor used cannons gray.
    for node, d in path:
        isl = islands[node]
        dr, dc = MID_POS[d]
        out[isl.top + dr][isl.left + dc] = GRAY
    # Recolor target center yellow.
    tgt_isl = islands[tgt]
    out[tgt_isl.top + 1][tgt_isl.left + 1] = YELLOW
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _grid_graph_neighbors(node: Node, nrows: int, ncols: int) -> list[tuple[Node, str]]:
    r, c = node
    out = []
    for d, (dr, dc) in DIRS.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < nrows and 0 <= nc < ncols:
            out.append(((nr, nc), d))
    return out


def _add_uniform_background_edges(edges: dict[Node, set[str]], nodes: list[Node], nrows: int, ncols: int, rng: random.Random) -> None:
    """Densify outgoing edges across the whole lattice for visual uniformity."""
    shuffled = nodes[:]
    rng.shuffle(shuffled)

    # First pass: give most islands at least one outgoing cannon.
    for node in shuffled:
        neighbors = _grid_graph_neighbors(node, nrows, ncols)
        if not neighbors:
            continue
        if len(edges[node]) == 0 and rng.random() < 0.92:
            # Prefer directions not directly heading toward the target path structure, to create decoys.
            _, d = rng.choice(neighbors)
            edges[node].add(d)

    # Second pass: many islands get a second cannon as well.
    rng.shuffle(shuffled)
    for node in shuffled:
        neighbors = _grid_graph_neighbors(node, nrows, ncols)
        available = [d for _, d in neighbors if d not in edges[node]]
        if not available:
            continue
        min_desired = 1
        max_desired = min(3, len(neighbors))
        desired = min_desired
        if len(neighbors) >= 2 and rng.random() < 0.68:
            desired = 2
        if len(neighbors) >= 3 and rng.random() < 0.18:
            desired = 3
        desired = min(desired, max_desired)
        while len(edges[node]) < desired and available:
            d = rng.choice(available)
            edges[node].add(d)
            available.remove(d)


def _edge_distribution_ok(edges: dict[Node, set[str]], nrows: int, ncols: int) -> bool:
    nodes = [(r, c) for r in range(nrows) for c in range(ncols)]
    active_nodes = sum(1 for node in nodes if edges[node])
    total_nodes = len(nodes)
    min_active_ratio = 0.60 if total_nodes <= 9 else 0.72
    if active_nodes < int((total_nodes * min_active_ratio) + 0.999999):
        return False

    # Avoid visually empty rows/columns in larger lattices.
    if nrows >= 3:
        row_active = [sum(1 for c in range(ncols) if edges[(r, c)]) for r in range(nrows)]
        if min(row_active) < max(1, ncols // 3):
            return False
    if ncols >= 3:
        col_active = [sum(1 for r in range(nrows) if edges[(r, c)]) for c in range(ncols)]
        if min(col_active) < max(1, nrows // 3):
            return False
    return True


def _render_instance(nrows: int, ncols: int, gap_r: int, gap_c: int, margin_t: int, margin_l: int,
                     outline: int, edges: dict[Node, set[str]], src: Node, tgt: Node) -> Grid:
    h = margin_t * 2 + nrows * 3 + (nrows - 1) * gap_r
    w = margin_l * 2 + ncols * 3 + (ncols - 1) * gap_c
    grid = _empty(h, w)
    for r in range(nrows):
        for c in range(ncols):
            top = margin_t + r * (3 + gap_r)
            left = margin_l + c * (3 + gap_c)
            # outline
            for dc in range(3):
                grid[top][left + dc] = outline
                grid[top + 2][left + dc] = outline
            for dr in range(3):
                grid[top + dr][left] = outline
                grid[top + dr][left + 2] = outline
            # cannons
            for d in edges[(r, c)]:
                dr, dc = MID_POS[d]
                grid[top + dr][left + dc] = RED
            # centers
            if (r, c) == src:
                grid[top + 1][left + 1] = YELLOW
            elif (r, c) == tgt:
                grid[top + 1][left + 1] = CYAN
    return grid


def generate(
    rng: random.Random | None = None,
    nrows: int | None = None,
    ncols: int | None = None,
) -> dict[str, Grid]:
    rng = rng or random.Random()

    for _attempt in range(5000):
        rr = nrows if nrows is not None else rng.randint(2, 5)
        cc = ncols if ncols is not None else rng.randint(2, 5)
        if rr * cc < 4:
            # still allow 2x2 minimum; no special rejection here
            pass
        gap_r = rng.randint(1, 2)
        gap_c = rng.randint(1, 2)
        margin_t = rng.randint(1, 2)
        margin_l = rng.randint(1, 2)
        h = margin_t * 2 + rr * 3 + (rr - 1) * gap_r
        w = margin_l * 2 + cc * 3 + (cc - 1) * gap_c
        if h > 30 or w > 30:
            continue

        nodes = [(r, c) for r in range(rr) for c in range(cc)]
        src, tgt = rng.sample(nodes, 2)
        edges: dict[Node, set[str]] = {node: set() for node in nodes}

        # Add a guaranteed random simple path first.
        cur = src
        visited = {src}
        forced_path: list[tuple[Node, str]] = []
        # Bias a wandering path toward tgt.
        for _ in range(rr * cc * 3):
            if cur == tgt:
                break
            choices = _grid_graph_neighbors(cur, rr, cc)
            # Prefer unvisited neighbors; with some probability prefer moves reducing manhattan distance.
            unvisited = [x for x in choices if x[0] not in visited]
            if not unvisited:
                unvisited = choices
            def score(item):
                (nr, nc), _ = item
                return abs(nr - tgt[0]) + abs(nc - tgt[1]) + rng.random() * 0.35
            if rng.random() < 0.7:
                nxt, d = min(unvisited, key=score)
            else:
                nxt, d = rng.choice(unvisited)
            forced_path.append((cur, d))
            edges[cur].add(d)
            cur = nxt
            visited.add(cur)
        if cur != tgt:
            # Finish with a direct Manhattan walk.
            while cur != tgt:
                cr, cc2 = cur
                if cr < tgt[0]:
                    d = 'D'
                elif cr > tgt[0]:
                    d = 'U'
                elif cc2 < tgt[1]:
                    d = 'R'
                else:
                    d = 'L'
                edges[cur].add(d)
                forced_path.append((cur, d))
                dr, dc = DIRS[d]
                cur = (cur[0] + dr, cur[1] + dc)
        # Add extra background edges so the red markers are visually spread around the whole grid.
        _add_uniform_background_edges(edges, nodes, rr, cc, rng)
        if not _edge_distribution_ok(edges, rr, cc):
            continue

        graph = {node: [((node[0] + DIRS[d][0], node[1] + DIRS[d][1]), d) for d in sorted(edges[node])] for node in nodes}
        path, unique = _shortest_path_unique(graph, src, tgt)
        if not path or not unique:
            continue
        if len(path) < 2:
            continue

        has_alt = _has_longer_alternative_path(graph, src, tgt, len(path), max_extra=max(4, rr + cc))
        # Prefer samples with at least one longer decoy path, but still allow a
        # minority of simpler cases for variety.
        if not has_alt and rng.random() < 0.8:
            continue

        outline = rng.choice(OUTLINE_COLORS)
        inp = _render_instance(rr, cc, gap_r, gap_c, margin_t, margin_l, outline, edges, src, tgt)
        try:
            out = transform_t39(inp)
        except Exception:
            continue
        if out == inp:
            continue
        return {"input": inp, "output": out}

    raise RuntimeError("failed to generate a valid t39 sample")


if __name__ == "__main__":
    print(json.dumps(generate(rng=random.Random(0)), indent=2))
