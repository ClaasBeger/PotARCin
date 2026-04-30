"""Generator for Test2/t12: demonstration-grounded container push.

Design:
- Container templates are grounded in the provided t12 demonstrations/test.
- We expand those templates with rotations/mirrors for directional variety.
- Container opening direction is therefore unambiguous by template identity.
- Support bar (9) is always a single connected full line.
- Container always touches (8-neighbor) the bar, does not overlap it, and lies on
  one strict side of it.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

Grid = List[List[int]]
Pos = Tuple[int, int]
Dir = Tuple[int, int]

LINE = 9
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8]


def _empty(h: int, w: int) -> Grid:
    return [[0 for _ in range(w)] for _ in range(h)]


def _normalize(cells: Set[Pos]) -> Set[Pos]:
    mr = min(r for r, _ in cells)
    mc = min(c for _, c in cells)
    return {(r - mr, c - mc) for r, c in cells}


def _rotate_cw_cells(cells: Set[Pos]) -> Set[Pos]:
    return _normalize({(c, -r) for r, c in cells})


def _mirror_h_cells(cells: Set[Pos]) -> Set[Pos]:
    return _normalize({(r, -c) for r, c in cells})


def _rotate_cw_dir(d: Dir) -> Dir:
    dr, dc = d
    return (dc, -dr)


def _mirror_h_dir(d: Dir) -> Dir:
    dr, dc = d
    return (dr, -dc)


def _neighbors8(r: int, c: int) -> list[Pos]:
    out: list[Pos] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _demo_template_library() -> Dict[frozenset[Pos], Dir]:
    """Template->opening direction, grounded in t12 demos (+ symmetries)."""
    bases: list[tuple[Set[Pos], Dir]] = []

    # Demo-like C container (train[0]) opening right.
    bases.append(
        (
            {
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (2, 0),
                (3, 0),
                (3, 1),
                (3, 2),
            },
            (0, 1),
        )
    )

    # Demo-like V/chevron (train[1]) opening up.
    bases.append(({(0, 0), (0, 4), (1, 1), (1, 3), (2, 2)}, (-1, 0)))

    # Demo-like diagonal container (train[2], color-8) opening up-left.
    # In the demonstration this shape moves along (-1, -1), not orthogonally.
    bases.append(({(0, 3), (1, 4), (2, 3), (3, 0), (3, 2), (4, 1)}, (-1, -1)))

    # Demo-like wide ring/diamond (train[2]/test-like) opening left.
    bases.append(
        (
            {
                (0, 2),
                (1, 1),
                (1, 3),
                (2, 0),
                (2, 4),
                (3, 4),
                (4, 0),
                (4, 4),
                (5, 1),
                (5, 3),
                (6, 2),
            },
            (0, -1),
        )
    )

    out: Dict[frozenset[Pos], Dir] = {}
    for cells0, dir0 in bases:
        cells = _normalize(cells0)
        d = dir0
        for _ in range(4):
            out.setdefault(frozenset(cells), d)
            mc = _mirror_h_cells(cells)
            md = _mirror_h_dir(d)
            out.setdefault(frozenset(mc), md)
            cells = _rotate_cw_cells(cells)
            d = _rotate_cw_dir(d)
    return out


TEMPLATE_TO_DIR = _demo_template_library()
TEMPLATES: list[Set[Pos]] = [set(k) for k in TEMPLATE_TO_DIR.keys()]


def _components8_by_color(grid: Grid) -> list[tuple[int, Set[Pos]]]:
    h = len(grid)
    w = len(grid[0]) if h else 0
    seen: Set[Pos] = set()
    comps: list[tuple[int, Set[Pos]]] = []
    for r in range(h):
        for c in range(w):
            col = grid[r][c]
            if col in (0, LINE) or (r, c) in seen:
                continue
            q = deque([(r, c)])
            seen.add((r, c))
            comp: Set[Pos] = set()
            while q:
                cr, cc = q.popleft()
                comp.add((cr, cc))
                for nr, nc in _neighbors8(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and grid[nr][nc] == col:
                        seen.add((nr, nc))
                        q.append((nr, nc))
            comps.append((col, comp))
    return comps


def _metric(orient: str, r: int, c: int) -> int:
    if orient == "h":
        return r
    if orient == "v":
        return c
    if orient == "d1":
        return r - c
    if orient == "d2":
        return r + c
    raise ValueError("bad orientation")


def _line_cells(h: int, w: int, orient: str, k: int) -> Set[Pos]:
    out: Set[Pos] = set()
    if orient == "h":
        if 0 <= k < h:
            for c in range(w):
                out.add((k, c))
        return out
    if orient == "v":
        if 0 <= k < w:
            for r in range(h):
                out.add((r, k))
        return out
    if orient == "d1":  # r-c=k
        for r in range(h):
            c = r - k
            if 0 <= c < w:
                out.add((r, c))
        return out
    if orient == "d2":  # r+c=k
        for r in range(h):
            c = k - r
            if 0 <= c < w:
                out.add((r, c))
        return out
    raise ValueError("bad orientation")


def _touching(a: Set[Pos], b: Set[Pos]) -> bool:
    for r, c in a:
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in b:
                return True
    return False


def _choose_line_for_container(h: int, w: int, container: Set[Pos], direction: Dir, rng: random.Random) -> Optional[Set[Pos]]:
    """Place a connected full bar touching container on exactly one side."""
    dr, dc = direction
    # For diagonal-opening containers, enforce bar axis parallel to movement/opening.
    if dr != 0 and dc != 0:
        orientations = ["d1"] if dr == dc else ["d2"]
    else:
        orientations = ["h", "v", "d1", "d2"]
        rng.shuffle(orientations)

    for orient in orientations:
        vals = [_metric(orient, r, c) for r, c in container]
        minv = min(vals)
        maxv = max(vals)
        for side in ("less", "greater"):
            k = maxv + 1 if side == "less" else minv - 1
            line = _line_cells(h, w, orient, k)
            if not line:
                continue
            if container & line:
                continue
            if not _touching(container, line):
                continue
            vals2 = [_metric(orient, r, c) for r, c in container]
            if side == "less" and not all(v < k for v in vals2):
                continue
            if side == "greater" and not all(v > k for v in vals2):
                continue
            return line
    return None


def _detect_container_and_direction(grid: Grid) -> tuple[int, Set[Pos], Dir]:
    """Find the unique template-matching component touching the support bar."""
    h = len(grid)
    w = len(grid[0]) if h else 0
    bar = {(r, c) for r in range(h) for c in range(w) if grid[r][c] == LINE}
    matches: list[tuple[int, Set[Pos], Dir]] = []
    for col, comp in _components8_by_color(grid):
        key = frozenset(_normalize(comp))
        d = TEMPLATE_TO_DIR.get(key)
        if d is None:
            continue
        if not _touching(comp, bar):
            continue
        matches.append((col, comp, d))
    if len(matches) != 1:
        raise ValueError("expected exactly one template-matching container touching bar")
    return matches[0]


def _sorted_for_move(cells: Set[Pos], dr: int, dc: int) -> list[Pos]:
    return sorted(cells, key=lambda rc: rc[0] * dr + rc[1] * dc, reverse=True)


def _encountered_cells(r: int, c: int, dr: int, dc: int) -> list[Pos]:
    """Cells a moving block face can encounter during one step.

    For diagonal motion, also include the two orthogonal front-adjacent cells.
    This prevents corner slip-through where payload can bypass via 8-neighborhood.
    """
    cells = [(r + dr, c + dc)]
    if dr != 0 and dc != 0:
        cells.append((r + dr, c))
        cells.append((r, c + dc))
    return cells


def _touches8_cell(p: Pos, cells: Set[Pos]) -> bool:
    pr, pc = p
    for nr, nc in _neighbors8(pr, pc):
        if (nr, nc) in cells:
            return True
    return False


def _diagonal_seam_flanks(container: Set[Pos], dr: int, dc: int) -> Set[Pos]:
    """Flank cells blocked by diagonal container-to-container seams.

    Any pair of diagonally adjacent container cells forms a closed seam between
    the two orthogonal in-between cells. This seam rule is container-specific.
    """
    if dr == 0 or dc == 0:
        return set()
    blocked: Set[Pos] = set()
    for r, c in container:
        # Consider both diagonal orientations; use only "downward" pairs to
        # avoid duplicating the same seam from both endpoints.
        for ddr, ddc in ((1, 1), (1, -1)):
            r2, c2 = r + ddr, c + ddc
            if (r2, c2) in container:
                blocked.add((r2, c))
                blocked.add((r, c2))
    return blocked


def _step_push(grid: Grid, container: Set[Pos], dr: int, dc: int) -> tuple[bool, Set[Pos]]:
    h = len(grid)
    w = len(grid[0]) if h else 0
    moving: Set[Pos] = set(container)
    container_next: Set[Pos] = {(r + dr, c + dc) for r, c in container}
    seam_flanks = _diagonal_seam_flanks(container, dr, dc)
    q = deque(container)
    blocked = False

    while q and not blocked:
        r, c = q.popleft()
        for i, (nr, nc) in enumerate(_encountered_cells(r, c, dr, dc)):
            # Primary forward cell enforces hard collision/bounds.
            if i == 0:
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    blocked = True
                    break
                if grid[nr][nc] == LINE and (nr, nc) not in moving:
                    blocked = True
                    break
                if grid[nr][nc] != 0 and (nr, nc) not in moving:
                    moving.add((nr, nc))
                    q.append((nr, nc))
                continue

            # Diagonal flank contacts are only forced when they cross a blocked
            # seam formed by two diagonally connected container cells. This
            # seam-blocking property is specific to container cells.
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                continue
            if grid[nr][nc] in (0, LINE) or (nr, nc) in moving:
                continue
            if (nr, nc) not in seam_flanks:
                continue
            moving.add((nr, nc))
            # Flank-contact payload is pushed by the container shell directly.
            # Do not recurse push-through from this flank payload in the same
            # step; this avoids unrelated long chains turning the whole step
            # into a hard block.

    if blocked:
        return False, container

    old = {(r, c): grid[r][c] for r, c in moving}
    for r, c in moving:
        grid[r][c] = 0
    for r, c in _sorted_for_move(moving, dr, dc):
        grid[r + dr][c + dc] = old[(r, c)]
    return True, {(r + dr, c + dc) for r, c in container}


def transform_t12(inp: Grid) -> Grid:
    h = len(inp)
    w = len(inp[0]) if h else 0
    if h == 0 or w == 0 or any(len(row) != w for row in inp):
        raise ValueError("input must be a non-empty rectangular grid")

    out = [row[:] for row in inp]
    _col, container, (dr, dc) = _detect_container_and_direction(out)
    while True:
        moved, container = _step_push(out, container, dr, dc)
        if not moved:
            break
    return out


def _scatter_payload(grid: Grid, container_color: int, rng: random.Random, n: int) -> None:
    h = len(grid)
    w = len(grid[0]) if h else 0
    colors = [c for c in PALETTE if c != container_color]
    cells = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 0]
    rng.shuffle(cells)
    for r, c in cells[:n]:
        grid[r][c] = rng.choice(colors)


def generate(
    width: int | None = None,
    height: int | None = None,
    rng: random.Random | None = None,
) -> dict[str, Grid]:
    rr = rng if rng is not None else random
    w = width if width is not None else rr.randint(10, 14)
    h = height if height is not None else rr.randint(10, 14)
    if w < 8 or h < 8:
        raise ValueError("width and height must be >= 8")

    for _ in range(1500):
        inp = _empty(h, w)
        tmpl = rr.choice(TEMPLATES)
        direction = TEMPLATE_TO_DIR[frozenset(tmpl)]
        th = max(r for r, _ in tmpl) + 1
        tw = max(c for _, c in tmpl) + 1
        if th >= h - 2 or tw >= w - 2:
            continue
        ar = rr.randint(1, h - th - 1)
        ac = rr.randint(1, w - tw - 1)
        container_cells = {(ar + r, ac + c) for r, c in tmpl}
        container_color = rr.choice(PALETTE)
        for r, c in container_cells:
            inp[r][c] = container_color

        line_cells = _choose_line_for_container(h, w, container_cells, direction, rr)
        if line_cells is None:
            continue
        for r, c in line_cells:
            inp[r][c] = LINE

        _scatter_payload(inp, container_color, rr, n=rr.randint(7, 18))

        try:
            out = transform_t12(inp)
        except Exception:
            continue
        if out == inp:
            continue
        return {"input": inp, "output": out}

    raise RuntimeError("failed to sample valid t12 instance")

