"""Generator for Test2/t36: follow the GPS-like relative-turn instructions.

Rule:
- The top row contains a yellow start token, then a sequence of (location color,
  turn color) pairs, then a cyan end token.
- The lower part of the grid is a lattice of gray 3x3 node-tiles separated by
  blank spacer rows/cols. Each node has a colored center.
- Start at the unique yellow node center (4). Move to a neighboring node whose
  color matches the first location token. There is no direction token for the
  start node.
- At each visited internal node, use the paired direction token to turn relative
  to the direction you were just traveling:
    * blue (1) = straight,
    * red (2) = right,
    * green (3) = left.
- Move one edge in that new direction to the next node. The next node must match
  the next location token, and after the final internal node, the final turn
  must take you to the cyan end node (8).
- In the output, the top-row instructions are deleted and the taken route is
  drawn in yellow as a 1-cell-thick polyline through the visited nodes and
  corridors. The cyan destination center remains cyan.

Generation constraints used to keep examples well-formed and unambiguous:
- the full instruction sequence (location + turn) yields exactly one valid path;
- the first move is unambiguous: exactly one neighbor of the yellow start
  matches the first location color;
- after that, the sequence of location colors alone yields at least two valid
  candidate paths, so the turn colors are actually needed;
- the intended path and at least one alternative color-only path have the same
  start, end, and length, but differ in their turn pattern;
- node colors avoid gray/yellow/cyan except for the unique start/end nodes.
"""

from __future__ import annotations

import json
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Grid = List[List[int]]
Node = Tuple[int, int]

BG = 0
BLUE = 1   # straight
RED = 2    # right
GREEN = 3  # left
YELLOW = 4
GRAY = 5
CYAN = 8
LOC_COLORS = [1, 2, 3, 6, 7, 9]
DIRS = {
    'U': (-1, 0),
    'R': (0, 1),
    'D': (1, 0),
    'L': (0, -1),
}
DIR_ORDER = ['U', 'R', 'D', 'L']
TURN_COLOR = {'S': BLUE, 'R': RED, 'L': GREEN}


def _empty(h: int, w: int, fill: int = 0) -> Grid:
    return [[fill for _ in range(w)] for _ in range(h)]


def _rectangular(grid: Grid) -> bool:
    return bool(grid) and bool(grid[0]) and all(len(row) == len(grid[0]) for row in grid)


def _neighbors(node: Node, nrows: int, ncols: int) -> list[Node]:
    r, c = node
    out: list[Node] = []
    for dr, dc in DIRS.values():
        nr, nc = r + dr, c + dc
        if 0 <= nr < nrows and 0 <= nc < ncols:
            out.append((nr, nc))
    return out


def _dir_between(a: Node, b: Node) -> str:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    for d, (rr, cc) in DIRS.items():
        if (dr, dc) == (rr, cc):
            return d
    raise ValueError('nodes are not orthogonally adjacent')


def _turn_from(prev_dir: str, next_dir: str) -> str:
    i = DIR_ORDER.index(prev_dir)
    j = DIR_ORDER.index(next_dir)
    diff = (j - i) % 4
    if diff == 0:
        return 'S'
    if diff == 1:
        return 'R'
    if diff == 3:
        return 'L'
    raise ValueError('U-turns are not allowed in generated paths')


def _apply_turn(prev_dir: str, turn_color: int) -> str:
    i = DIR_ORDER.index(prev_dir)
    if turn_color == BLUE:
        return prev_dir
    if turn_color == RED:
        return DIR_ORDER[(i + 1) % 4]
    if turn_color == GREEN:
        return DIR_ORDER[(i - 1) % 4]
    raise ValueError('invalid turn color')


def _node_center(node: Node) -> tuple[int, int]:
    r, c = node
    return 3 + 4 * r, 2 + 4 * c


def _parse_nodes(grid: Grid) -> tuple[int, int, dict[Node, int]]:
    h, w = len(grid), len(grid[0])
    if (h - 2) % 4 != 0 or (w - 1) % 4 != 0:
        raise ValueError('grid dimensions do not match the node lattice layout')
    nrows = (h - 2) // 4
    ncols = (w - 1) // 4
    colors: dict[Node, int] = {}
    for r in range(nrows):
        for c in range(ncols):
            rr, cc = _node_center((r, c))
            colors[(r, c)] = grid[rr][cc]
    return nrows, ncols, colors


def _instruction_tokens(grid: Grid) -> list[int]:
    return [v for v in grid[0] if v != BG]


def _find_unique(colors: dict[Node, int], color: int) -> Node:
    nodes = [node for node, v in colors.items() if v == color]
    if len(nodes) != 1:
        raise ValueError(f'expected exactly one node of color {color}, got {len(nodes)}')
    return nodes[0]


def _count_color_only_paths(colors: dict[Node, int], nrows: int, ncols: int, start: Node, end: Node, locs: list[int]) -> int:
    k = len(locs)
    count = 0
    used = {start}

    def dfs(node: Node, idx: int) -> None:
        nonlocal count
        if count >= 50:
            return
        if idx == k:
            if end in _neighbors(node, nrows, ncols) and end not in used:
                count += 1
            return
        target_color = locs[idx]
        for nb in _neighbors(node, nrows, ncols):
            if nb in used or colors[nb] != target_color:
                continue
            used.add(nb)
            dfs(nb, idx + 1)
            used.remove(nb)

    if k == 0:
        return 1 if end in _neighbors(start, nrows, ncols) else 0
    dfs(start, 0)
    return count


def _first_step_is_unambiguous(colors: dict[Node, int], nrows: int, ncols: int, start: Node, first_loc: int) -> bool:
    """Exactly one adjacent node to the start matches the first location token."""
    return sum(1 for nb in _neighbors(start, nrows, ncols) if colors[nb] == first_loc) == 1


def _full_instruction_paths(colors: dict[Node, int], nrows: int, ncols: int, start: Node, end: Node, locs: list[int], turns: list[int]) -> list[list[Node]]:
    k = len(locs)
    out: list[list[Node]] = []
    if k != len(turns):
        raise ValueError('locations and turn counts must match')
    if k == 0:
        return [[start, end]] if end in _neighbors(start, nrows, ncols) else []
    for first in _neighbors(start, nrows, ncols):
        if colors[first] != locs[0]:
            continue
        prev_dir = _dir_between(start, first)
        path = [start, first]
        used = {start, first}
        cur = first
        ok = True
        for i in range(k):
            out_dir = _apply_turn(prev_dir, turns[i])
            dr, dc = DIRS[out_dir]
            nxt = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nxt[0] < nrows and 0 <= nxt[1] < ncols):
                ok = False
                break
            if i == k - 1:
                if nxt != end or nxt in used:
                    ok = False
                    break
                path.append(end)
            else:
                if nxt in used or colors[nxt] != locs[i + 1]:
                    ok = False
                    break
                path.append(nxt)
                used.add(nxt)
                prev_dir = out_dir
                cur = nxt
        if ok:
            out.append(path)
    return out


def transform_t36(grid: Grid) -> Grid:
    if not _rectangular(grid):
        raise ValueError('input grid must be non-empty and rectangular')
    tokens = _instruction_tokens(grid)
    if len(tokens) < 4 or len(tokens) % 2 != 0:
        raise ValueError('instruction strip must have start, end, and one or more pairs')
    if tokens[0] != YELLOW or tokens[-1] != CYAN:
        raise ValueError('instruction strip must start with yellow and end with cyan')
    locs = tokens[1:-1:2]
    turns = tokens[2:-1:2]
    nrows, ncols, colors = _parse_nodes(grid)
    start = _find_unique(colors, YELLOW)
    end = _find_unique(colors, CYAN)
    candidates = _full_instruction_paths(colors, nrows, ncols, start, end, locs, turns)
    if len(candidates) != 1:
        raise ValueError(f'expected exactly one full-instruction path, got {len(candidates)}')
    path = candidates[0]

    out = [row[:] for row in grid]
    # Delete the instruction strip.
    out[0] = [BG] * len(out[0])

    # Draw all but the last segment inclusive of both ends.
    for i in range(len(path) - 2):
        a = path[i]
        b = path[i + 1]
        r1, c1 = _node_center(a)
        r2, c2 = _node_center(b)
        if r1 == r2:
            step = 1 if c2 > c1 else -1
            for c in range(c1, c2 + step, step):
                out[r1][c] = YELLOW
        else:
            step = 1 if r2 > r1 else -1
            for r in range(r1, r2 + step, step):
                out[r][c1] = YELLOW

    # Final segment: stop just before the cyan center.
    if len(path) >= 2:
        a = path[-2]
        b = path[-1]
        r1, c1 = _node_center(a)
        r2, c2 = _node_center(b)
        if r1 == r2:
            step = 1 if c2 > c1 else -1
            for c in range(c1, c2, step):
                out[r1][c] = YELLOW
        else:
            step = 1 if r2 > r1 else -1
            for r in range(r1, r2, step):
                out[r][c1] = YELLOW
        out[r2][c2] = CYAN
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _random_simple_path(rng: random.Random, nrows: int, ncols: int, length_edges: int) -> Optional[list[Node]]:
    nodes = [(r, c) for r in range(nrows) for c in range(ncols)]
    start = rng.choice(nodes)
    path = [start]
    used = {start}
    for _ in range(length_edges):
        cur = path[-1]
        options = [nb for nb in _neighbors(cur, nrows, ncols) if nb not in used]
        if not options:
            return None
        nxt = rng.choice(options)
        path.append(nxt)
        used.add(nxt)
    return path


def _all_simple_paths_of_length(start: Node, end: Node, nrows: int, ncols: int, length_edges: int, limit: int = 200) -> list[list[Node]]:
    out: list[list[Node]] = []
    used = {start}
    path = [start]

    def dfs(node: Node, steps: int) -> None:
        if len(out) >= limit:
            return
        if steps == length_edges:
            if node == end:
                out.append(path[:])
            return
        remaining = length_edges - steps
        # Cheap pruning by Manhattan distance.
        md = abs(node[0] - end[0]) + abs(node[1] - end[1])
        if md > remaining or (remaining - md) % 2 != 0:
            return
        for nb in _neighbors(node, nrows, ncols):
            if nb in used:
                continue
            used.add(nb)
            path.append(nb)
            dfs(nb, steps + 1)
            path.pop()
            used.remove(nb)

    dfs(start, 0)
    return out


def _instruction_from_path(path: list[Node], node_colors: dict[Node, int]) -> tuple[list[int], list[int], list[int]]:
    # returns tokens, loc sequence, turns
    locs = [node_colors[path[i]] for i in range(1, len(path) - 1)]
    turns: list[int] = []
    for i in range(1, len(path) - 1):
        prev_dir = _dir_between(path[i - 1], path[i])
        next_dir = _dir_between(path[i], path[i + 1])
        turns.append(TURN_COLOR[_turn_from(prev_dir, next_dir)])
    tokens = [YELLOW]
    for loc, turn in zip(locs, turns):
        tokens.extend([loc, turn])
    tokens.append(CYAN)
    return tokens, locs, turns


def _render_instance(nrows: int, ncols: int, colors: dict[Node, int], tokens: list[int]) -> Grid:
    h = 2 + 4 * nrows
    w = 1 + 4 * ncols
    grid = _empty(h, w, BG)

    # Top-row instruction strip.
    grid[0][0] = tokens[0]
    k = (len(tokens) - 2) // 2
    for i in range(k):
        grid[0][3 + 4 * i] = tokens[1 + 2 * i]
        grid[0][4 + 4 * i] = tokens[2 + 2 * i]
    grid[0][3 + 4 * k] = tokens[-1]

    # Node lattice.
    for r in range(nrows):
        for c in range(ncols):
            top = 2 + 4 * r
            left = 1 + 4 * c
            for dc in range(3):
                grid[top][left + dc] = GRAY
                grid[top + 2][left + dc] = GRAY
            for dr in range(3):
                grid[top + dr][left] = GRAY
                grid[top + dr][left + 2] = GRAY
            grid[top + 1][left + 1] = colors[(r, c)]
    return grid


def generate(rng: random.Random | None = None, nrows: int | None = None, ncols: int | None = None) -> dict[str, Grid]:
    rng = rng or random.Random()
    for _attempt in range(10000):
        rr = nrows if nrows is not None else rng.randint(3, 5)
        cc = ncols if ncols is not None else rng.randint(3, 5)
        length_edges = cc

        # Create intended path P.
        P = _random_simple_path(rng, rr, cc, length_edges)
        if P is None:
            continue
        start, end = P[0], P[-1]
        # Need at least one alternative path Q of the same length between same endpoints.
        all_paths = _all_simple_paths_of_length(start, end, rr, cc, length_edges, limit=500)
        alts = [q for q in all_paths if q != P]
        if not alts:
            continue
        # Prefer an alternative that is internally disjoint from P for easier color assignment.
        disjoint_alts = [q for q in alts if set(q[1:-1]).isdisjoint(set(P[1:-1]))]
        Q = rng.choice(disjoint_alts if disjoint_alts else alts)

        # Assign node colors.
        colors: dict[Node, int] = {}
        for r in range(rr):
            for c in range(cc):
                colors[(r, c)] = rng.choice(LOC_COLORS)
        colors[start] = YELLOW
        colors[end] = CYAN

        internal_colors = [rng.choice(LOC_COLORS) for _ in range(1, len(P) - 1)]
        for i, color in enumerate(internal_colors, start=1):
            colors[P[i]] = color
            colors[Q[i]] = color

        tokens, locs, turns = _instruction_from_path(P, colors)
        # Extra safety: top strip must fit the width pattern, which is ensured by length_edges == ncols.
        if len(tokens) != 2 * cc:
            continue

        # Check ambiguity / uniqueness constraints.
        if not _first_step_is_unambiguous(colors, rr, cc, start, locs[0]):
            continue
        color_only_count = _count_color_only_paths(colors, rr, cc, start, end, locs)
        if color_only_count < 2:
            continue
        full_paths = _full_instruction_paths(colors, rr, cc, start, end, locs, turns)
        if len(full_paths) != 1 or full_paths[0] != P:
            continue

        inp = _render_instance(rr, cc, colors, tokens)
        try:
            out = transform_t36(inp)
        except Exception:
            continue
        if out == inp:
            continue
        return {'input': inp, 'output': out}
    raise RuntimeError('failed to generate a valid t36 sample')


if __name__ == '__main__':
    print(json.dumps(generate(rng=random.Random(0)), indent=2))
