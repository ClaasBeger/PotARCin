"""Generator for Test2/t49: aligned 2x2 nodes connected by an MST.

Rule:
- The input contains several 2x2 nodes, all in one non-black/non-gray color.
- Two nodes can be connected only when their 2x2 blocks have the same top row
  or the same left column.
- The weight of a valid connection is the gap distance between the two blocks,
  measured in cells between their facing edges.
- Draw a minimal spanning tree of the alignment graph using gray (5) cells.
  Corridors are two cells wide, matching the width/height of the 2x2 nodes.

Generation constraints used to keep examples unambiguous:
- every colored component is exactly a 2x2 node;
- the alignment graph is connected;
- the MST is unique;
- selected MST corridors do not pass through another colored node;
- at least one horizontal and one vertical MST edge appear when possible;
- default generation prefers richer multi-row / multi-column topologies rather
  than simple single-row / single-column chains.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import List, Sequence, Tuple

Grid = List[List[int]]
Cell = Tuple[int, int]
Node = Tuple[int, int]  # top-left row, top-left col

BG = 0
GRAY = 5
NODE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
NEI4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


@dataclass(frozen=True)
class Edge:
    weight: int
    a: int
    b: int


class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1
        return True


def _empty(h: int, w: int) -> Grid:
    return [[BG for _ in range(w)] for _ in range(h)]


def _rectangular(grid: Grid) -> bool:
    return bool(grid) and bool(grid[0]) and all(len(row) == len(grid[0]) for row in grid)


def _node_cells(node: Node) -> set[Cell]:
    r, c = node
    return {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}


def _touch_or_overlap(a: Node, b: Node) -> bool:
    # True when the two 2x2 blocks overlap or touch by a 4-neighbor edge.
    ar, ac = a
    br, bc = b
    return not (ar + 1 < br - 1 or br + 1 < ar - 1 or ac + 1 < bc - 1 or bc + 1 < ac - 1)


def _find_nodes(grid: Grid) -> tuple[int, list[Node]]:
    """Return (foreground color, sorted 2x2 node top-lefts)."""
    if not _rectangular(grid):
        raise ValueError("input grid must be non-empty and rectangular")
    colors = sorted({v for row in grid for v in row if v not in (BG, GRAY)})
    if len(colors) != 1:
        raise ValueError("expected exactly one non-gray foreground color")
    color = colors[0]
    h, w = len(grid), len(grid[0])
    seen: set[Cell] = set()
    nodes: list[Node] = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] != color or (r, c) in seen:
                continue
            stack = [(r, c)]
            comp: set[Cell] = set()
            while stack:
                rr, cc = stack.pop()
                if (rr, cc) in comp or grid[rr][cc] != color:
                    continue
                comp.add((rr, cc))
                for dr, dc in NEI4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == color and (nr, nc) not in comp:
                        stack.append((nr, nc))
            seen |= comp
            if len(comp) != 4:
                raise ValueError("each foreground component must be a 2x2 node")
            rs = [rr for rr, _ in comp]
            cs = [cc for _, cc in comp]
            r0, c0 = min(rs), min(cs)
            if comp != _node_cells((r0, c0)):
                raise ValueError("each foreground component must be a filled 2x2 node")
            nodes.append((r0, c0))
    if not nodes:
        raise ValueError("expected at least one node")
    return color, sorted(nodes)


def _edge_weight(a: Node, b: Node) -> int | None:
    if a[0] == b[0]:
        return abs(a[1] - b[1]) - 2
    if a[1] == b[1]:
        return abs(a[0] - b[0]) - 2
    return None


def _edges(nodes: Sequence[Node]) -> list[Edge]:
    out: list[Edge] = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            wt = _edge_weight(nodes[i], nodes[j])
            if wt is not None and wt > 0:
                out.append(Edge(wt, i, j))
    return sorted(out, key=lambda e: (e.weight, nodes[e.a], nodes[e.b]))


def _mst_edges(nodes: Sequence[Node]) -> list[Edge]:
    dsu = DSU(len(nodes))
    chosen: list[Edge] = []
    for e in _edges(nodes):
        if dsu.union(e.a, e.b):
            chosen.append(e)
            if len(chosen) == len(nodes) - 1:
                break
    if len(nodes) > 1 and len(chosen) != len(nodes) - 1:
        raise ValueError("alignment graph is disconnected")
    return chosen


def _has_unique_mst(nodes: Sequence[Node]) -> bool:
    """Kruskal uniqueness test: each equal-weight group must be acyclic after contraction."""
    edges = _edges(nodes)
    if len(nodes) <= 1:
        return True
    if len(_mst_edges(nodes)) != len(nodes) - 1:
        return False
    dsu = DSU(len(nodes))
    i = 0
    while i < len(edges):
        wt = edges[i].weight
        group: list[Edge] = []
        while i < len(edges) and edges[i].weight == wt:
            group.append(edges[i])
            i += 1
        temp = DSU(len(nodes))
        for e in group:
            a, b = dsu.find(e.a), dsu.find(e.b)
            if a == b:
                continue
            if not temp.union(a, b):
                return False
        for e in group:
            dsu.union(e.a, e.b)
    return True


def _corridor_cells(a: Node, b: Node) -> list[Cell]:
    ar, ac = a
    br, bc = b
    cells: list[Cell] = []
    if ar == br:
        lo, hi = sorted((ac, bc))
        for c in range(lo + 2, hi):
            cells.append((ar, c))
            cells.append((ar + 1, c))
    elif ac == bc:
        lo, hi = sorted((ar, br))
        for r in range(lo + 2, hi):
            cells.append((r, ac))
            cells.append((r, ac + 1))
    else:
        raise ValueError("nodes are not aligned")
    return cells


def _corridor_clear(nodes: Sequence[Node], e: Edge) -> bool:
    occupied: set[Cell] = set()
    for k, n in enumerate(nodes):
        if k not in (e.a, e.b):
            occupied |= _node_cells(n)
    return all(cell not in occupied for cell in _corridor_cells(nodes[e.a], nodes[e.b]))


def transform_t49(grid: Grid) -> Grid:
    """Draw the gray minimal-spanning tree between aligned 2x2 nodes."""
    _color, nodes = _find_nodes(grid)
    h, w = len(grid), len(grid[0])
    out = [list(row) for row in grid]
    for e in _mst_edges(nodes):
        for r, c in _corridor_cells(nodes[e.a], nodes[e.b]):
            if 0 <= r < h and 0 <= c < w and out[r][c] == BG:
                out[r][c] = GRAY
    return out


def _alignment_connected(nodes: Sequence[Node]) -> bool:
    if not nodes:
        return False
    dsu = DSU(len(nodes))
    for e in _edges(nodes):
        dsu.union(e.a, e.b)
    root = dsu.find(0)
    return all(dsu.find(i) == root for i in range(len(nodes)))


def _alignment_edge_count(nodes: Sequence[Node]) -> int:
    return len(_edges(nodes))


def _rich_structure(nodes: Sequence[Node]) -> bool:
    """Prefer nontrivial multi-row / multi-column graphs for default generation."""
    n = len(nodes)
    rows = Counter(r for r, _ in nodes)
    cols = Counter(c for _, c in nodes)
    dr = len(rows)
    dc = len(cols)
    repeated_rows = sum(v >= 2 for v in rows.values())
    repeated_cols = sum(v >= 2 for v in cols.values())
    ecount = _alignment_edge_count(nodes)

    if n >= 6:
        return dr >= 3 and dc >= 3 and repeated_rows >= 2 and repeated_cols >= 2 and ecount >= n
    if n == 5:
        return dr >= 3 and dc >= 3 and repeated_rows >= 2 and repeated_cols >= 2 and ecount >= n - 1
    if n == 4:
        return dr >= 2 and dc >= 2 and repeated_rows >= 1 and repeated_cols >= 1 and ecount >= n - 1
    return dr >= 2 and dc >= 2


def _valid_nodes(nodes: Sequence[Node], h: int, w: int, *, prefer_rich: bool = False) -> bool:
    if len(set(nodes)) != len(nodes):
        return False
    for r, c in nodes:
        if r < 0 or c < 0 or r + 1 >= h or c + 1 >= w:
            return False
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if _touch_or_overlap(nodes[i], nodes[j]):
                return False
    if not _alignment_connected(nodes):
        return False
    try:
        mst = _mst_edges(nodes)
    except ValueError:
        return False
    if not _has_unique_mst(nodes):
        return False
    if not all(_corridor_clear(nodes, e) for e in mst):
        return False
    if len(nodes) >= 4:
        has_h = any(nodes[e.a][0] == nodes[e.b][0] for e in mst)
        has_v = any(nodes[e.a][1] == nodes[e.b][1] for e in mst)
        if not (has_h and has_v):
            return False
    if prefer_rich and not _rich_structure(nodes):
        return False
    return True


def _render(h: int, w: int, nodes: Sequence[Node], color: int) -> Grid:
    grid = _empty(h, w)
    for r, c in nodes:
        for rr, cc in _node_cells((r, c)):
            grid[rr][cc] = color
    return grid


def _sample_axis_positions(rng: random.Random, limit: int, count: int) -> list[int] | None:
    """Sample top-left coordinates with >=3 spacing so same-line nodes never touch."""
    vals = list(range(0, max(0, limit - 1)))
    rng.shuffle(vals)
    chosen: list[int] = []
    for v in vals:
        if all(abs(v - u) >= 3 for u in chosen):
            chosen.append(v)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _grow_nodes_on_lattice(
    rng: random.Random,
    rows: Sequence[int],
    cols: Sequence[int],
    n: int,
) -> list[Node] | None:
    """Grow a connected set of intersections, biased toward richer topology."""
    if not rows or not cols or n <= 0:
        return None

    nodes: set[Node] = set()
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    # Seed with a 2x2 rectangle whenever possible to discourage simple chains.
    if n >= 4 and len(rows) >= 2 and len(cols) >= 2:
        rs = rng.sample(list(rows), 2)
        cs = rng.sample(list(cols), 2)
        for r in rs:
            for c in cs:
                nodes.add((r, c))
        used_rows.update(rs)
        used_cols.update(cs)
    else:
        start = (rng.choice(list(rows)), rng.choice(list(cols)))
        nodes.add(start)
        used_rows.add(start[0])
        used_cols.add(start[1])

    while len(nodes) < n:
        candidates: list[tuple[int, Node]] = []
        for r in rows:
            for c in cols:
                cand = (r, c)
                if cand in nodes:
                    continue
                if r not in used_rows and c not in used_cols:
                    continue  # would disconnect the alignment graph
                row_count = sum(rr == r for rr, _ in nodes)
                col_count = sum(cc == c for _, cc in nodes)
                introduces = int(r not in used_rows) + int(c not in used_cols)

                # Weighted preference:
                # - new row xor new col expands the graph,
                # - existing row and existing col densifies it and adds extra alignments,
                # - candidates on already-populated lines are favored for richer graphs.
                if introduces == 1:
                    score = 4 + row_count + col_count
                else:
                    score = 6 + 2 * row_count + 2 * col_count
                candidates.append((score, cand))
        if not candidates:
            return None
        total = sum(score for score, _ in candidates)
        pick = rng.randrange(total)
        acc = 0
        chosen = candidates[-1][1]
        for score, cand in candidates:
            acc += score
            if pick < acc:
                chosen = cand
                break
        nodes.add(chosen)
        used_rows.add(chosen[0])
        used_cols.add(chosen[1])

    return sorted(nodes)


def _sample_rich_nodes(rng: random.Random, h: int, w: int, n: int) -> list[Node] | None:
    max_rows = max(1, (h + 1) // 3)
    max_cols = max(1, (w + 1) // 3)
    if n >= 6:
        row_count = min(max_rows, rng.randint(3, min(4, n - 1, max_rows)))
        col_count = min(max_cols, rng.randint(3, min(4, n - 1, max_cols)))
    elif n == 5:
        row_count = min(max_rows, rng.randint(3, min(4, max_rows)))
        col_count = min(max_cols, rng.randint(3, min(4, max_cols)))
    elif n == 4:
        row_count = min(max_rows, rng.randint(2, min(3, max_rows)))
        col_count = min(max_cols, rng.randint(2, min(3, max_cols)))
    else:
        row_count = min(max_rows, rng.randint(2, min(3, max_rows)))
        col_count = min(max_cols, rng.randint(2, min(3, max_cols)))

    if row_count < 2 or col_count < 2:
        return None
    rows = _sample_axis_positions(rng, h, row_count)
    cols = _sample_axis_positions(rng, w, col_count)
    if rows is None or cols is None:
        return None
    if row_count * col_count < n:
        return None
    return _grow_nodes_on_lattice(rng, rows, cols, n)


def _sample_basic_nodes(rng: random.Random, h: int, w: int, n: int) -> list[Node] | None:
    start = (rng.randint(0, h - 2), rng.randint(0, w - 2))
    nodes = [start]
    for _ in range(n - 1):
        base = rng.choice(nodes)
        br, bc = base
        dirs = ["left", "right", "up", "down"]
        rng.shuffle(dirs)
        cand: Node | None = None
        for direction in dirs:
            if direction in ("left", "right"):
                max_gap = (bc - 2) if direction == "left" else (w - (bc + 2) - 2)
            else:
                max_gap = (br - 2) if direction == "up" else (h - (br + 2) - 2)
            if max_gap < 1:
                continue
            gaps = list(range(1, max_gap + 1))
            rng.shuffle(gaps)
            for gap in gaps:
                if direction == "left":
                    trial = (br, bc - gap - 2)
                elif direction == "right":
                    trial = (br, bc + gap + 2)
                elif direction == "up":
                    trial = (br - gap - 2, bc)
                else:
                    trial = (br + gap + 2, bc)
                if all(not _touch_or_overlap(trial, existing) for existing in nodes):
                    cand = trial
                    break
            if cand is not None:
                break
        if cand is None:
            return None
        nodes.append(cand)
    return sorted(nodes)


def generate(
    rng: random.Random | None = None,
    h: int | None = None,
    w: int | None = None,
    num_nodes: int | None = None,
    color: int | None = None,
) -> dict[str, Grid]:
    """Generate one valid ARC-style example."""
    rng = rng or random.Random()
    for attempt in range(6000):
        hh = h if h is not None else rng.randint(11, 30)
        ww = w if w is not None else rng.randint(11, 30)
        if hh < 5 or ww < 5:
            continue

        max_nodes = min(10, max(4, (hh * ww) // 28))
        min_nodes = 4 if min(hh, ww) < 13 else 5
        n = num_nodes if num_nodes is not None else rng.randint(min_nodes, max_nodes)
        col = color if color is not None else rng.choice(NODE_COLORS)

        # Prefer richer lattice-grown structures; occasionally fall back to a
        # simpler construction so the task family still covers a range of cases.
        use_rich = attempt < 5000 and (num_nodes is None or n >= 4)
        nodes = _sample_rich_nodes(rng, hh, ww, n) if use_rich else _sample_basic_nodes(rng, hh, ww, n)
        if nodes is None:
            continue
        if not _valid_nodes(nodes, hh, ww, prefer_rich=use_rich):
            continue

        inp = _render(hh, ww, nodes, col)
        out = transform_t49(inp)
        return {"input": inp, "output": out}
    raise RuntimeError("failed to generate a valid t49 sample")


if __name__ == "__main__":
    import json

    print(json.dumps(generate(rng=random.Random(0)), indent=2))
