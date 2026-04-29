"""Generator for Test2/t48: uniquely completable four-color map coloring.

Rule:
- The last row begins with four palette colors.
- In the map area above, one boundary color (not in the palette) forms the map
  walls, enclosing several regions.
- Some regions are already colored with palette colors; at least one region is
  left uncolored (0).
- Complete the empty regions so that adjacent regions never share a color.

Generation constraints used to keep examples well-formed and unambiguous:
- exactly four distinct palette colors are shown in the bottom-left legend;
- exactly one distinct non-palette color is used as the boundary color;
- every interior region is a 4-connected component separated by 1-cell-thick
  boundary walls;
- the map contains at least one hidden region and at least two revealed regions;
- the completed map uses all four palette colors at least once;
- the partial coloring has exactly one valid completion;
- generated maps prefer irregular outer silhouettes rather than always using a
  full rectangular enclosure.
"""

from __future__ import annotations

import itertools
import json
import random
from typing import List, Sequence, Set, Tuple

Grid = List[List[int]]
Cell = Tuple[int, int]
CoarseCell = Tuple[int, int]

BG = 0
ALL_COLORS = list(range(1, 10))
NEI4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# ---------------------------------------------------------------------------
# Generic transform / solver
# ---------------------------------------------------------------------------


def _empty(h: int, w: int) -> Grid:
    return [[0 for _ in range(w)] for _ in range(h)]


def _find_palette(grid: Grid) -> list[int]:
    if not grid or not grid[0]:
        raise ValueError("grid must be non-empty")
    palette = grid[-1][:4]
    if len(palette) != 4 or any(v == 0 for v in palette) or len(set(palette)) != 4:
        raise ValueError("bottom-left legend must contain four distinct nonzero colors")
    return list(palette)


def _find_boundary_color(grid: Grid, palette: Sequence[int]) -> int:
    colors = {v for row in grid[:-1] for v in row if v != 0}
    candidates = sorted(colors - set(palette))
    if len(candidates) != 1:
        raise ValueError("expected exactly one non-palette boundary color")
    return candidates[0]


def _components_non_boundary(grid: Grid, boundary_color: int) -> tuple[list[list[int]], list[set[Cell]]]:
    """Return component-id map and component cell sets for cells != boundary_color.

    Components include both interior regions and outer background; callers can
    decide which ones to use.
    """
    h, w = len(grid), len(grid[0])
    comp_id = [[-1 for _ in range(w)] for _ in range(h)]
    comps: list[set[Cell]] = []
    cid = 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] == boundary_color or comp_id[r][c] != -1:
                continue
            stack = [(r, c)]
            cells: set[Cell] = set()
            comp_id[r][c] = cid
            while stack:
                rr, cc = stack.pop()
                cells.add((rr, cc))
                for dr, dc in NEI4:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and grid[nr][nc] != boundary_color
                        and comp_id[nr][nc] == -1
                    ):
                        comp_id[nr][nc] = cid
                        stack.append((nr, nc))
            comps.append(cells)
            cid += 1
    return comp_id, comps


class ParsedMap:
    def __init__(
        self,
        palette: Sequence[int],
        boundary_color: int,
        regions: list[set[Cell]],
        region_colors: list[int],
        adjacency: list[set[int]],
    ) -> None:
        self.palette = list(palette)
        self.boundary_color = boundary_color
        self.regions = regions
        self.region_colors = region_colors
        self.adjacency = adjacency


def _parse_map(grid: Grid) -> ParsedMap:
    palette = _find_palette(grid)
    boundary = _find_boundary_color(grid, palette)
    map_grid = [row[:] for row in grid[:-1]]
    h, w = len(map_grid), len(map_grid[0])

    comp_id, comps = _components_non_boundary(map_grid, boundary)

    # Interior regions are the non-boundary components that do not touch the outer frame.
    region_ids: list[int] = []
    for cid, cells in enumerate(comps):
        touches_border = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)
        if not touches_border:
            region_ids.append(cid)

    if not region_ids:
        raise ValueError("no interior map regions found")

    id_map = {cid: i for i, cid in enumerate(region_ids)}
    regions = [comps[cid] for cid in region_ids]
    region_colors: list[int] = []
    for cells in regions:
        vals = {map_grid[r][c] for r, c in cells}
        if len(vals) != 1:
            raise ValueError("each region must be monochrome or uncolored")
        v = next(iter(vals))
        if v != 0 and v not in palette:
            raise ValueError("region color must be 0 or a legend color")
        region_colors.append(v)

    adjacency: list[set[int]] = [set() for _ in regions]
    for r in range(h):
        for c in range(w):
            if map_grid[r][c] != boundary:
                continue
            neigh_ids: set[int] = set()
            for dr, dc in NEI4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    cid = comp_id[nr][nc]
                    if cid in id_map:
                        neigh_ids.add(id_map[cid])
            neigh = sorted(neigh_ids)
            for i in range(len(neigh)):
                for j in range(i + 1, len(neigh)):
                    a, b = neigh[i], neigh[j]
                    adjacency[a].add(b)
                    adjacency[b].add(a)

    return ParsedMap(palette, boundary, regions, region_colors, adjacency)


def _solve_region_coloring(
    palette: Sequence[int],
    adjacency: Sequence[Set[int]],
    preset: Sequence[int],
    max_solutions: int = 2,
) -> list[list[int]]:
    n = len(adjacency)
    domains = [set(palette) for _ in range(n)]
    for i, v in enumerate(preset):
        if v != 0:
            if v not in palette:
                return []
            domains[i] = {v}

    for i in range(n):
        if len(domains[i]) == 1:
            vi = next(iter(domains[i]))
            for j in adjacency[i]:
                if i < j and len(domains[j]) == 1 and next(iter(domains[j])) == vi:
                    return []

    solutions: list[list[int]] = []
    assign = [0] * n

    def backtrack() -> None:
        if len(solutions) >= max_solutions:
            return
        unassigned = [i for i in range(n) if assign[i] == 0]
        if not unassigned:
            solutions.append(assign[:])
            return
        var = min(unassigned, key=lambda i: (len(domains[i]), -len(adjacency[i]), i))
        forbidden = {assign[j] for j in adjacency[var] if assign[j] != 0}
        choices = sorted(v for v in domains[var] if v not in forbidden)
        for v in choices:
            assign[var] = v
            backtrack()
            assign[var] = 0
            if len(solutions) >= max_solutions:
                return

    for i in range(n):
        if len(domains[i]) == 1:
            assign[i] = next(iter(domains[i]))

    for i in range(n):
        if assign[i] == 0:
            continue
        for j in adjacency[i]:
            if i < j and assign[j] == assign[i] != 0:
                return []

    backtrack()
    return solutions


def transform_t48(grid: Grid) -> Grid:
    parsed = _parse_map(grid)
    solutions = _solve_region_coloring(parsed.palette, parsed.adjacency, parsed.region_colors, max_solutions=2)
    if len(solutions) != 1:
        raise ValueError("input does not have a unique valid completion")
    sol = solutions[0]
    out = [row[:] for row in grid]
    for ridx, cells in enumerate(parsed.regions):
        color = sol[ridx]
        for r, c in cells:
            out[r][c] = color
    return out


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def _shape_connected(shape: set[CoarseCell]) -> bool:
    if not shape:
        return False
    start = next(iter(shape))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in NEI4:
            nb = (r + dr, c + dc)
            if nb in shape and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(shape)


def _shape_hole_free(shape: set[CoarseCell], coarse_h: int, coarse_w: int) -> bool:
    inactive = {(r, c) for r in range(coarse_h) for c in range(coarse_w) if (r, c) not in shape}
    if not inactive:
        return True
    stack = [cell for cell in inactive if cell[0] in (0, coarse_h - 1) or cell[1] in (0, coarse_w - 1)]
    seen = set(stack)
    while stack:
        r, c = stack.pop()
        for dr, dc in NEI4:
            nb = (r + dr, c + dc)
            if 0 <= nb[0] < coarse_h and 0 <= nb[1] < coarse_w and nb in inactive and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return seen == inactive


def _generate_shape(rng: random.Random, coarse_h: int, coarse_w: int, target_cells: int) -> set[CoarseCell] | None:
    all_cells = {(r, c) for r in range(coarse_h) for c in range(coarse_w)}
    start = (rng.randrange(coarse_h), rng.randrange(coarse_w))
    shape = {start}
    frontier = set()
    for dr, dc in NEI4:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < coarse_h and 0 <= nc < coarse_w:
            frontier.add((nr, nc))

    while len(shape) < target_cells and frontier:
        candidates = list(frontier)
        weighted: list[tuple[int, CoarseCell]] = []
        for cell in candidates:
            r, c = cell
            deg = sum((r + dr, c + dc) in shape for dr, dc in NEI4)
            score = 1 + 2 * deg
            # slight bias away from filling the whole bounding rectangle too early
            min_r = min(x for x, _ in shape | {cell})
            max_r = max(x for x, _ in shape | {cell})
            min_c = min(y for _, y in shape | {cell})
            max_c = max(y for _, y in shape | {cell})
            bbox_area = (max_r - min_r + 1) * (max_c - min_c + 1)
            if bbox_area > len(shape) + 1:
                score += 1
            weighted.append((score, cell))
        total = sum(score for score, _ in weighted)
        pick = rng.randrange(total)
        acc = 0
        chosen = weighted[-1][1]
        for score, cell in weighted:
            acc += score
            if pick < acc:
                chosen = cell
                break
        shape.add(chosen)
        frontier.discard(chosen)
        r, c = chosen
        for dr, dc in NEI4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < coarse_h and 0 <= nc < coarse_w and (nr, nc) not in shape:
                frontier.add((nr, nc))

    if len(shape) != target_cells:
        return None
    if not _shape_connected(shape):
        return None
    if not _shape_hole_free(shape, coarse_h, coarse_w):
        return None
    # Prefer irregular silhouettes when possible.
    min_r = min(r for r, _ in shape)
    max_r = max(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    max_c = max(c for _, c in shape)
    bbox_area = (max_r - min_r + 1) * (max_c - min_c + 1)
    if target_cells >= 5 and bbox_area == len(shape):
        return None
    return shape


def _partition_shape(
    rng: random.Random,
    coarse_h: int,
    coarse_w: int,
    shape: set[CoarseCell],
    target_regions: int,
) -> tuple[list[int], list[int], set[CoarseCell], list[list[int]], list[set[int]]]:
    """Create a merged partition on an irregular connected coarse shape."""
    active_cells = sorted(shape)
    idx_of = {cell: i for i, cell in enumerate(active_cells)}
    parent = list(range(len(active_cells)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    edges: list[tuple[int, int]] = []
    for r, c in active_cells:
        a = idx_of[(r, c)]
        for dr, dc in ((1, 0), (0, 1)):
            nb = (r + dr, c + dc)
            if nb in idx_of:
                edges.append((a, idx_of[nb]))
    rng.shuffle(edges)

    region_count = len(active_cells)
    for a, b in edges:
        if region_count <= target_regions:
            break
        if union(a, b):
            region_count -= 1

    root_to_region: dict[int, int] = {}
    region_of_cell = [[-1 for _ in range(coarse_w)] for _ in range(coarse_h)]
    next_id = 0
    for r, c in active_cells:
        root = find(idx_of[(r, c)])
        if root not in root_to_region:
            root_to_region[root] = next_id
            next_id += 1
        region_of_cell[r][c] = root_to_region[root]

    nregions = next_id
    adjacency = [set() for _ in range(nregions)]
    for r, c in active_cells:
        a = region_of_cell[r][c]
        for dr, dc in ((1, 0), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < coarse_h and 0 <= nc < coarse_w and region_of_cell[nr][nc] != -1:
                b = region_of_cell[nr][nc]
                if a != b:
                    adjacency[a].add(b)
                    adjacency[b].add(a)

    row_sizes = [rng.randint(2, 4) for _ in range(coarse_h)]
    col_sizes = [rng.randint(2, 4) for _ in range(coarse_w)]
    return row_sizes, col_sizes, shape, region_of_cell, adjacency


def _enumerate_full_colorings(
    palette: Sequence[int], adjacency: Sequence[Set[int]], max_solutions: int = 10000
) -> list[list[int]]:
    return _solve_region_coloring(palette, adjacency, [0] * len(adjacency), max_solutions=max_solutions)


def _choose_target_and_revealed(
    rng: random.Random,
    palette: Sequence[int],
    adjacency: Sequence[Set[int]],
) -> tuple[list[int], set[int]] | None:
    """Pick a full coloring and a reveal set with a unique completion."""
    n = len(adjacency)
    all_colorings = _enumerate_full_colorings(palette, adjacency, max_solutions=10000)
    if not all_colorings:
        return None
    good_targets = [col for col in all_colorings if set(col) == set(palette)]
    if not good_targets:
        return None
    rng.shuffle(good_targets)

    region_indices = list(range(n))
    hidden_sizes = [2, 3, 1] if n >= 5 else [1, 2]

    for target in good_targets:
        for hidden_size in hidden_sizes:
            if hidden_size <= 0 or hidden_size >= n:
                continue
            candidates = list(itertools.combinations(region_indices, hidden_size))
            rng.shuffle(candidates)
            for hidden in candidates:
                hidden_set = set(hidden)
                revealed = [i for i in region_indices if i not in hidden_set]
                if len(revealed) < 2:
                    continue
                filtered = [col for col in all_colorings if all(col[i] == target[i] for i in revealed)]
                if len(filtered) == 1:
                    return list(target), set(revealed)
    return None


def _render_partition(
    row_sizes: Sequence[int],
    col_sizes: Sequence[int],
    shape: set[CoarseCell],
    region_of_cell: Sequence[Sequence[int]],
    full_colors: Sequence[int],
    revealed: Set[int],
    boundary_color: int,
    palette: Sequence[int],
    top_margin: int,
    left_margin: int,
    right_margin: int,
    gap_above_legend: int,
) -> tuple[Grid, Grid]:
    coarse_h = len(row_sizes)
    coarse_w = len(col_sizes)
    map_h = sum(row_sizes) + coarse_h + 1
    map_w = sum(col_sizes) + coarse_w + 1
    total_h = top_margin + map_h + gap_above_legend + 1
    total_w = max(left_margin + map_w + right_margin, 4)

    inp = _empty(total_h, total_w)
    out = _empty(total_h, total_w)

    for c, v in enumerate(palette):
        inp[-1][c] = v
        out[-1][c] = v

    r0 = top_margin
    c0 = left_margin
    row_starts = []
    cur = r0 + 1
    for hs in row_sizes:
        row_starts.append(cur)
        cur += hs + 1
    col_starts = []
    cur = c0 + 1
    for ws in col_sizes:
        col_starts.append(cur)
        cur += ws + 1

    # Draw each active coarse cell as a local rectangle with walls.
    for rr, cc in sorted(shape):
        rid = region_of_cell[rr][cc]
        fill_in = full_colors[rid] if rid in revealed else 0
        fill_out = full_colors[rid]
        rs = row_starts[rr]
        cs = col_starts[cc]
        hs = row_sizes[rr]
        ws = col_sizes[cc]

        # Outer local box first (walls + interior).
        for r in range(rs - 1, rs + hs + 1):
            for c in range(cs - 1, cs + ws + 1):
                inp[r][c] = boundary_color
                out[r][c] = boundary_color

        # Interior.
        for r in range(rs, rs + hs):
            for c in range(cs, cs + ws):
                inp[r][c] = fill_in
                out[r][c] = fill_out

    # Remove walls between same-region neighboring active cells.
    for rr, cc in sorted(shape):
        rid = region_of_cell[rr][cc]
        fill_in = full_colors[rid] if rid in revealed else 0
        fill_out = full_colors[rid]
        rs = row_starts[rr]
        cs = col_starts[cc]
        hs = row_sizes[rr]
        ws = col_sizes[cc]
        if (rr + 1, cc) in shape and region_of_cell[rr + 1][cc] == rid:
            wall_r = rs + hs
            for c in range(cs, cs + ws):
                inp[wall_r][c] = fill_in
                out[wall_r][c] = fill_out
        if (rr, cc + 1) in shape and region_of_cell[rr][cc + 1] == rid:
            wall_c = cs + ws
            for r in range(rs, rs + hs):
                inp[r][wall_c] = fill_in
                out[r][wall_c] = fill_out

    return inp, out


def generate(
    rng: random.Random | None = None,
    h: int | None = None,
    w: int | None = None,
) -> dict[str, Grid]:
    rng = rng or random.Random()
    for _attempt in range(6000):
        boundary_color = rng.choice(ALL_COLORS)
        palette = rng.sample([c for c in ALL_COLORS if c != boundary_color], 4)

        coarse_h = rng.randint(3, 5)
        coarse_w = rng.randint(3, 5)
        max_cells = coarse_h * coarse_w
        min_cells = min(max_cells, 5)
        target_cells = rng.randint(min_cells, max_cells - 1 if max_cells >= 6 else max_cells)
        shape = _generate_shape(rng, coarse_h, coarse_w, target_cells)
        if shape is None:
            continue

        target_regions = rng.randint(4, min(len(shape), 8))
        row_sizes, col_sizes, shape, region_of_cell, adjacency = _partition_shape(
            rng, coarse_h, coarse_w, shape, target_regions
        )
        nregions = len(adjacency)
        if nregions < 4:
            continue

        chosen = _choose_target_and_revealed(rng, palette, adjacency)
        if chosen is None:
            continue
        full_colors, revealed = chosen
        if len(revealed) >= nregions or len(revealed) < 2:
            continue

        top_margin = rng.randint(0, 3)
        left_margin = rng.randint(0, 3)
        right_margin = rng.randint(0, 3)
        gap_above_legend = rng.randint(1, 2)

        inp, out = _render_partition(
            row_sizes,
            col_sizes,
            shape,
            region_of_cell,
            full_colors,
            revealed,
            boundary_color,
            palette,
            top_margin,
            left_margin,
            right_margin,
            gap_above_legend,
        )

        if h is not None or w is not None:
            hh = len(inp)
            ww = len(inp[0])
            target_h = h if h is not None else hh
            target_w = w if w is not None else ww
            if hh > target_h or ww > target_w:
                continue
            pad_top = rng.randint(0, target_h - hh)
            pad_left = rng.randint(0, target_w - ww)
            new_inp = _empty(target_h, target_w)
            new_out = _empty(target_h, target_w)
            for r in range(hh):
                for c in range(ww):
                    new_inp[pad_top + r][pad_left + c] = inp[r][c]
                    new_out[pad_top + r][pad_left + c] = out[r][c]
            inp, out = new_inp, new_out
            for c, v in enumerate(palette):
                inp[-1][c] = v
                out[-1][c] = v

        try:
            check = transform_t48(inp)
        except Exception:
            continue
        if check != out:
            continue
        return {"input": inp, "output": out}

    raise RuntimeError("failed to generate a valid t48 sample")


if __name__ == "__main__":
    print(json.dumps(generate(rng=random.Random(0)), indent=2))
