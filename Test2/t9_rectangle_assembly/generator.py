"""Generator for Test2/t9 unique rectangle assembly task."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional

Grid = List[List[int]]

PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def _empty(h: int, w: int) -> Grid:
    return [[0 for _ in range(w)] for _ in range(h)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


@dataclass(frozen=True)
class Piece:
    color: int
    cells: tuple[tuple[int, int], ...]  # normalized offsets
    h: int
    w: int
    area: int


def _extract_pieces(inp: Grid) -> list[Piece]:
    h = len(inp)
    w = len(inp[0]) if h else 0
    seen: set[tuple[int, int]] = set()
    pieces: list[Piece] = []
    for r in range(h):
        for c in range(w):
            col = inp[r][c]
            if col == 0 or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            comp: set[tuple[int, int]] = set()
            while stack:
                cr, cc = stack.pop()
                comp.add((cr, cc))
                for nr, nc in _neighbors8(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and inp[nr][nc] == col:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            min_r = min(rr for rr, _ in comp)
            min_c = min(cc for _, cc in comp)
            norm = sorted((rr - min_r, cc - min_c) for rr, cc in comp)
            ph = max(rr for rr, _ in norm) + 1
            pw = max(cc for _, cc in norm) + 1
            pieces.append(Piece(color=col, cells=tuple(norm), h=ph, w=pw, area=len(norm)))
    pieces.sort(key=lambda p: p.area, reverse=True)
    return pieces


def _factor_dims(area: int, min_h: int, min_w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for h in range(1, area + 1):
        if area % h != 0:
            continue
        w = area // h
        if h >= min_h and w >= min_w:
            out.append((h, w))
    out.sort(key=lambda hw: (abs(hw[0] - hw[1]), hw[0] * hw[1]))
    return out


def _enumerate_rectangle_solutions(
    pieces: list[Piece], *, max_solutions: int = 2
) -> list[Grid]:
    total = sum(p.area for p in pieces)
    min_h = max(p.h for p in pieces)
    min_w = max(p.w for p in pieces)
    dims = _factor_dims(total, min_h, min_w)
    if not dims:
        return []

    solutions: list[Grid] = []
    seen_solution_keys: set[tuple[tuple[int, ...], ...]] = set()

    for oh, ow in dims:
        board = _empty(oh, ow)
        used = [False] * len(pieces)

        def first_empty() -> tuple[int, int] | None:
            for rr in range(oh):
                for cc in range(ow):
                    if board[rr][cc] == 0:
                        return rr, cc
            return None

        def place(idx: int, dr: int, dc: int) -> bool:
            p = pieces[idx]
            for pr, pc in p.cells:
                rr, cc = pr + dr, pc + dc
                if rr < 0 or rr >= oh or cc < 0 or cc >= ow:
                    return False
                if board[rr][cc] != 0:
                    return False
            for pr, pc in p.cells:
                board[pr + dr][pc + dc] = p.color
            return True

        def unplace(idx: int, dr: int, dc: int) -> None:
            p = pieces[idx]
            for pr, pc in p.cells:
                board[pr + dr][pc + dc] = 0

        def dfs() -> None:
            if len(solutions) >= max_solutions:
                return
            spot = first_empty()
            if spot is None:
                key = tuple(tuple(v for v in row) for row in board)
                if key not in seen_solution_keys:
                    seen_solution_keys.add(key)
                    solutions.append(copy.deepcopy(board))
                return
            er, ec = spot
            for i, p in enumerate(pieces):
                if used[i]:
                    continue
                # Anchor one piece-cell onto the first empty cell.
                for pr, pc in p.cells:
                    dr = er - pr
                    dc = ec - pc
                    if not place(i, dr, dc):
                        continue
                    used[i] = True
                    dfs()
                    used[i] = False
                    unplace(i, dr, dc)
                    if len(solutions) >= max_solutions:
                        return

        dfs()
        if len(solutions) >= max_solutions:
            break

    return solutions


def _best_unique_subset_solution(all_pieces: list[Piece]) -> Optional[Grid]:
    """Find unique rectangle assembly from a subset of pieces.

    Preference order:
    1) maximize assembled area
    2) maximize number of used pieces
    3) lexicographically smallest output grid
    """
    n = len(all_pieces)
    best_area = -1
    best_k = -1
    best_grid: Optional[Grid] = None

    # Try larger subsets first, but scoring is primarily by assembled area.
    idxs = list(range(n))
    for k in range(n, 1, -1):
        for comb in combinations(idxs, k):
            pieces = [all_pieces[i] for i in comb]
            area = sum(p.area for p in pieces)
            # Lower-bound pruning.
            if area < best_area:
                continue
            sols = _enumerate_rectangle_solutions(pieces, max_solutions=2)
            if len(sols) != 1:
                continue
            grid = sols[0]
            if (
                area > best_area
                or (area == best_area and k > best_k)
                or (
                    area == best_area
                    and k == best_k
                    and best_grid is not None
                    and tuple(tuple(v for v in row) for row in grid)
                    < tuple(tuple(v for v in row) for row in best_grid)
                )
                or best_grid is None
            ):
                best_area = area
                best_k = k
                best_grid = grid
    return best_grid


def transform_t9(inp: Grid) -> Grid:
    h = len(inp)
    w = len(inp[0]) if h else 0
    if h == 0 or w == 0 or any(len(row) != w for row in inp):
        raise ValueError("input must be rectangular non-empty")
    pieces = _extract_pieces(inp)
    if len(pieces) < 2:
        raise ValueError("need at least two shapes")
    solved = _best_unique_subset_solution(pieces)
    if solved is None:
        raise ValueError("assembly is not solvable")
    return solved


def _build_output_partition(oh: int, ow: int, k: int, rng: random.Random) -> Grid:
    # Partition full rectangle into k connected regions by seeded growth.
    labels = [[-1 for _ in range(ow)] for _ in range(oh)]
    cells = [(r, c) for r in range(oh) for c in range(ow)]
    rng.shuffle(cells)
    seeds = cells[:k]
    frontiers: list[list[tuple[int, int]]] = [[] for _ in range(k)]
    for i, (r, c) in enumerate(seeds):
        labels[r][c] = i
        frontiers[i].append((r, c))
    unassigned = {(r, c) for r in range(oh) for c in range(ow) if labels[r][c] == -1}

    while unassigned:
        progress = False
        frontier_ids = [i for i in range(k) if frontiers[i]]
        rng.shuffle(frontier_ids)
        for i in frontier_ids:
            if not frontiers[i]:
                continue
            r, c = frontiers[i][rng.randrange(len(frontiers[i]))]
            nbrs = [
                (nr, nc)
                for nr, nc in _neighbors8(r, c)
                if 0 <= nr < oh and 0 <= nc < ow and (nr, nc) in unassigned
            ]
            if not nbrs:
                continue
            rng.shuffle(nbrs)
            take = rng.randint(1, min(2, len(nbrs)))
            for nr, nc in nbrs[:take]:
                labels[nr][nc] = i
                frontiers[i].append((nr, nc))
                unassigned.remove((nr, nc))
            progress = True
        if not progress:
            # Attach one remaining cell to any adjacent labeled region.
            r, c = rng.choice(list(unassigned))
            nbr_ids = [
                labels[nr][nc]
                for nr, nc in _neighbors8(r, c)
                if 0 <= nr < oh and 0 <= nc < ow and labels[nr][nc] >= 0
            ]
            if not nbr_ids:
                i = rng.randrange(k)
            else:
                i = rng.choice(nbr_ids)
            labels[r][c] = i
            frontiers[i].append((r, c))
            unassigned.remove((r, c))

    colors = rng.sample(PALETTE, k=k)
    out = _empty(oh, ow)
    for r in range(oh):
        for c in range(ow):
            out[r][c] = colors[labels[r][c]]
    return out


def _piece_has_diagonal_adjacency(piece: Piece) -> bool:
    cells = set(piece.cells)
    for r, c in cells:
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            if (r + dr, c + dc) in cells:
                return True
    return False


def _scatter_output_as_input(out: Grid, rng: random.Random) -> Grid | None:
    oh = len(out)
    ow = len(out[0]) if oh else 0
    pieces = _extract_pieces(out)
    ih = oh + rng.randint(4, 8)
    iw = ow + rng.randint(4, 8)
    ih = max(ih, 9)
    iw = max(iw, 9)
    inp = _empty(ih, iw)

    placed_boxes: list[tuple[int, int, int, int]] = []
    order = list(range(len(pieces)))
    rng.shuffle(order)
    for idx in order:
        p = pieces[idx]
        ok = False
        for _ in range(100):
            r0 = rng.randint(0, ih - p.h)
            c0 = rng.randint(0, iw - p.w)
            # bbox separation to avoid accidental touching/merging clutter
            cand = (r0, c0, r0 + p.h - 1, c0 + p.w - 1)
            conflict = False
            for a0, b0, a1, b1 in placed_boxes:
                if not (cand[2] + 1 < a0 or a1 + 1 < cand[0] or cand[3] + 1 < b0 or b1 + 1 < cand[1]):
                    conflict = True
                    break
            if conflict:
                continue
            overlap = False
            for pr, pc in p.cells:
                rr, cc = r0 + pr, c0 + pc
                if inp[rr][cc] != 0:
                    overlap = True
                    break
            if overlap:
                continue
            for pr, pc in p.cells:
                inp[r0 + pr][c0 + pc] = p.color
            placed_boxes.append(cand)
            ok = True
            break
        if not ok:
            return None
    return inp


def generate(
    width: int | None = None,
    height: int | None = None,
    n_shapes: int | None = None,
    rng: random.Random | None = None,
) -> dict[str, Grid]:
    rr = rng if rng is not None else random
    if width is not None and width < 3:
        raise ValueError("width must be >= 3")
    if height is not None and height < 3:
        raise ValueError("height must be >= 3")
    if n_shapes is not None and n_shapes < 2:
        raise ValueError("n_shapes must be >= 2")

    for _ in range(800):
        oh = height if height is not None else rr.randint(3, 6)
        ow = width if width is not None else rr.randint(3, 6)
        area = oh * ow
        k = n_shapes if n_shapes is not None else rr.randint(3, min(7, area - 1))
        out = _build_output_partition(oh, ow, k, rr)
        out_pieces = _extract_pieces(out)
        # Require at least one diagonally-connected shape.
        if not any(_piece_has_diagonal_adjacency(p) for p in out_pieces):
            continue
        inp = _scatter_output_as_input(out, rr)
        if inp is None:
            continue
        pieces = _extract_pieces(inp)  # same piece set, scattered
        sols = _enumerate_rectangle_solutions(pieces, max_solutions=2)
        # Generation requires all pieces used exactly once in a unique rectangle.
        if len(sols) != 1:
            continue
        solved = sols[0]
        if solved != out:
            # If a different unique arrangement is found, keep canonical solver result.
            out = solved
        return {"input": inp, "output": out}

    raise RuntimeError("failed to sample valid t9 instance")
