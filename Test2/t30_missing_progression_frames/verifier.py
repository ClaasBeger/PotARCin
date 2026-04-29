"""Verifier for t30: fill missing frames in a simple progression.

The grid is a vertical strip of equal-size frames separated by a full row of 6s.
Blank frames are missing. Supported progression families are deliberately
simple and natural: expanding diamond rings, growing V outlines, constant-speed
border motion, cumulative path drawing, moving rigid shapes, sweeping lines,
and growing square outlines / filled squares.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, Tuple

Grid = List[List[int]]
Cell = Tuple[int, int]

SEP = 6


# ------------------------------ generic helpers ------------------------------

def _empty(h: int, w: int) -> Grid:
    return [[0 for _ in range(w)] for _ in range(h)]


def _split_frames(grid: Grid) -> tuple[list[Grid], int] | None:
    h = len(grid)
    w = len(grid[0]) if h else 0
    if h == 0 or w == 0 or any(len(row) != w for row in grid):
        return None
    sep_rows = [i for i, row in enumerate(grid) if all(v == SEP for v in row)]
    if not sep_rows:
        return None
    starts = [0] + [r + 1 for r in sep_rows]
    ends = sep_rows + [h]
    frames = [grid[s:e] for s, e in zip(starts, ends) if e > s]
    if not frames:
        return None
    fh = len(frames[0])
    if fh == 0 or any(len(f) != fh for f in frames):
        return None
    return frames, w


def _join_frames(frames: Sequence[Grid], w: int) -> Grid:
    out: Grid = []
    for i, f in enumerate(frames):
        out.extend([row[:] for row in f])
        if i != len(frames) - 1:
            out.append([SEP] * w)
    return out


def _cells(frame: Grid) -> Set[Cell]:
    return {(r, c) for r, row in enumerate(frame) for c, v in enumerate(row) if v and v != SEP}


def _colored(frame: Grid) -> Dict[Cell, int]:
    return {(r, c): v for r, row in enumerate(frame) for c, v in enumerate(row) if v and v != SEP}


def _colors(frames: Sequence[Grid]) -> Set[int]:
    return {v for f in frames for row in f for v in row if v and v != SEP}


def _single_color(frames: Sequence[Grid]) -> int | None:
    vals = list(_colors(frames))
    return vals[0] if len(vals) == 1 else None


def _frame_from_cells(h: int, w: int, cells: Iterable[Cell], color: int) -> Grid:
    g = _empty(h, w)
    for r, c in cells:
        if 0 <= r < h and 0 <= c < w:
            g[r][c] = color
    return g


def _frame_from_colored(h: int, w: int, cells: Dict[Cell, int]) -> Grid:
    g = _empty(h, w)
    for (r, c), color in cells.items():
        if 0 <= r < h and 0 <= c < w:
            g[r][c] = color
    return g


def _visible(frames: Sequence[Grid]) -> list[tuple[int, Set[Cell]]]:
    return [(i, _cells(f)) for i, f in enumerate(frames) if _cells(f)]


def _visible_colored(frames: Sequence[Grid]) -> list[tuple[int, Dict[Cell, int]]]:
    return [(i, _colored(f)) for i, f in enumerate(frames) if _colored(f)]


def _fit_line(vals: list[tuple[int, int]]) -> tuple[int, int] | None:
    if not vals:
        return None
    if len(vals) == 1:
        i, y = vals[0]
        return y - i, 1
    (i0, y0), (i1, y1) = vals[0], vals[1]
    den = i1 - i0
    if den == 0 or (y1 - y0) % den != 0:
        return None
    b = (y1 - y0) // den
    a = y0 - b * i0
    if all(a + b * i == y for i, y in vals):
        return a, b
    return None


def _check_preserve(frames: Sequence[Grid], filled: Sequence[Grid]) -> bool:
    return all((not _cells(frames[i])) or frames[i] == filled[i] for i in range(len(frames)))


# ---------------------------- expanding ring family ----------------------------

def _ring_cells(h: int, w: int, cr: int, cc: int, radius: int) -> Set[Cell]:
    return {(r, c) for r in range(h) for c in range(w) if abs(r - cr) + abs(c - cc) == radius}


def _try_diamond_rings(frames: Sequence[Grid]) -> list[Grid] | None:
    h, w, n = len(frames[0]), len(frames[0][0]), len(frames)
    vis = _visible(frames)
    if len(vis) < 2:
        return None
    col = _single_color(frames)
    if col is None:
        return None
    for cr in range(h):
        for cc in range(w):
            vals: list[tuple[int, int]] = []
            ok = True
            for i, cells in vis:
                ds = {abs(r - cr) + abs(c - cc) for r, c in cells}
                if len(ds) != 1:
                    ok = False
                    break
                rad = next(iter(ds))
                if cells != _ring_cells(h, w, cr, cc, rad):
                    ok = False
                    break
                vals.append((i, rad))
            if not ok:
                continue
            line = _fit_line(vals)
            if line is None:
                continue
            a, b = line
            if b <= 0:
                continue
            out: list[Grid] = []
            valid = True
            for i in range(n):
                rad = a + b * i
                if rad < 0:
                    valid = False
                    break
                out.append(_frame_from_cells(h, w, _ring_cells(h, w, cr, cc, rad), col))
            if valid:
                return out
    return None


# --------------------------- growing V / triangle family --------------------------

def _v_cells(h: int, w: int, cc: int, level: int) -> Set[Cell]:
    cells: Set[Cell] = set()
    for r in range(0, min(h, level + 1)):
        for c in (cc - r, cc + r):
            if 0 <= c < w:
                cells.add((r, c))
    return cells


def _try_v_growth(frames: Sequence[Grid]) -> list[Grid] | None:
    h, w, n = len(frames[0]), len(frames[0][0]), len(frames)
    vis = _visible(frames)
    if len(vis) < 2:
        return None
    col = _single_color(frames)
    if col is None:
        return None
    for cc in range(w):
        vals: list[tuple[int, int]] = []
        ok = True
        for i, cells in vis:
            matching = [lev for lev in range(h) if cells == _v_cells(h, w, cc, lev)]
            if len(matching) != 1:
                ok = False
                break
            vals.append((i, matching[0]))
        if not ok:
            continue
        line = _fit_line(vals)
        if line is None:
            continue
        a, b = line
        if b <= 0:
            continue
        out: list[Grid] = []
        valid = True
        for i in range(n):
            lev = a + b * i
            if lev < 0 or lev >= h:
                valid = False
                break
            out.append(_frame_from_cells(h, w, _v_cells(h, w, cc, lev), col))
        if valid:
            return out
    return None


# ------------------------- growing square outline / filled ------------------------

def _square_outline_cells(h: int, w: int, cr: int, cc: int, rad: int) -> Set[Cell]:
    return {(r, c) for r in range(h) for c in range(w) if max(abs(r - cr), abs(c - cc)) == rad}


def _square_filled_cells(h: int, w: int, cr: int, cc: int, rad: int) -> Set[Cell]:
    return {(r, c) for r in range(h) for c in range(w) if max(abs(r - cr), abs(c - cc)) <= rad}


def _try_square_growth(frames: Sequence[Grid], filled_mode: bool) -> list[Grid] | None:
    h, w, n = len(frames[0]), len(frames[0][0]), len(frames)
    vis = _visible(frames)
    if len(vis) < 2:
        return None
    col = _single_color(frames)
    if col is None:
        return None
    cell_fn = _square_filled_cells if filled_mode else _square_outline_cells
    for cr in range(h):
        for cc in range(w):
            vals: list[tuple[int, int]] = []
            ok = True
            for i, cells in vis:
                matching = [rad for rad in range(max(h, w)) if cells == cell_fn(h, w, cr, cc, rad)]
                if len(matching) != 1:
                    ok = False
                    break
                vals.append((i, matching[0]))
            if not ok:
                continue
            line = _fit_line(vals)
            if line is None:
                continue
            a, b = line
            if b <= 0:
                continue
            out: list[Grid] = []
            valid = True
            for i in range(n):
                rad = a + b * i
                if rad < 0:
                    valid = False
                    break
                out.append(_frame_from_cells(h, w, cell_fn(h, w, cr, cc, rad), col))
            if valid:
                return out
    return None


def _try_square_outline_growth(frames: Sequence[Grid]) -> list[Grid] | None:
    return _try_square_growth(frames, filled_mode=False)


def _try_square_filled_growth(frames: Sequence[Grid]) -> list[Grid] | None:
    return _try_square_growth(frames, filled_mode=True)


# ------------------------------ line sweep family ------------------------------

def _row_cells(h: int, w: int, rr: int) -> Set[Cell]:
    return {(rr, c) for c in range(w)} if 0 <= rr < h else set()


def _col_cells(h: int, w: int, cc: int) -> Set[Cell]:
    return {(r, cc) for r in range(h)} if 0 <= cc < w else set()


def _diag_cells(h: int, w: int, d: int) -> Set[Cell]:
    return {(r, c) for r in range(h) for c in range(w) if c - r == d}


def _anti_diag_cells(h: int, w: int, s: int) -> Set[Cell]:
    return {(r, c) for r in range(h) for c in range(w) if r + c == s}


def _try_line_sweep(frames: Sequence[Grid]) -> list[Grid] | None:
    h, w, n = len(frames[0]), len(frames[0][0]), len(frames)
    vis = _visible(frames)
    if len(vis) < 2:
        return None
    col = _single_color(frames)
    if col is None:
        return None
    families = [
        (range(h), _row_cells),
        (range(w), _col_cells),
        (range(-(h - 1), w), _diag_cells),
        (range(h + w - 1), _anti_diag_cells),
    ]
    for domain, fn in families:
        vals: list[tuple[int, int]] = []
        ok = True
        for i, cells in vis:
            matches = [p for p in domain if cells == fn(h, w, p)]
            if len(matches) != 1:
                ok = False
                break
            vals.append((i, matches[0]))
        if not ok:
            continue
        line = _fit_line(vals)
        if line is None:
            continue
        a, b = line
        if b == 0:
            continue
        out: list[Grid] = []
        valid = True
        params = list(domain)
        lo, hi = min(params), max(params)
        for i in range(n):
            p = a + b * i
            if p < lo or p > hi:
                valid = False
                break
            cells = fn(h, w, p)
            if not cells:
                valid = False
                break
            out.append(_frame_from_cells(h, w, cells, col))
        if valid:
            return out
    return None


# --------------------------- rigid translation family --------------------------

def _shift_cells(cells: Set[Cell], dr: int, dc: int) -> Set[Cell]:
    return {(r + dr, c + dc) for r, c in cells}


def _shift_colored(cells: Dict[Cell, int], dr: int, dc: int) -> Dict[Cell, int]:
    return {(r + dr, c + dc): col for (r, c), col in cells.items()}


def _candidate_shifts(base: Set[Cell], target: Set[Cell]) -> Set[Tuple[int, int]]:
    if len(base) != len(target) or not base:
        return set()
    out: Set[Tuple[int, int]] = set()
    for r0, c0 in base:
        for r1, c1 in target:
            dr, dc = r1 - r0, c1 - c0
            if _shift_cells(base, dr, dc) == target:
                out.add((dr, dc))
    return out


def _try_translation(frames: Sequence[Grid]) -> list[Grid] | None:
    h, w, n = len(frames[0]), len(frames[0][0]), len(frames)
    vis = _visible_colored(frames)
    if len(vis) < 2:
        return None
    # Same color multiset across visible frames.
    color_sets = [sorted(c.values()) for _, c in vis]
    if any(cs != color_sets[0] for cs in color_sets[1:]):
        return None
    base_i, base = vis[0]
    base_positions = set(base.keys())
    deltas_r: list[tuple[int, int]] = []
    deltas_c: list[tuple[int, int]] = []
    chosen: Dict[int, Tuple[int, int]] = {base_i: (0, 0)}
    for i, cur in vis[1:]:
        cur_pos = set(cur.keys())
        shifts = []
        for dr, dc in _candidate_shifts(base_positions, cur_pos):
            shifted = _shift_colored(base, dr, dc)
            if shifted == cur:
                shifts.append((dr, dc))
        if len(shifts) != 1:
            return None
        dr, dc = shifts[0]
        chosen[i] = (dr, dc)
        deltas_r.append((i, dr))
        deltas_c.append((i, dc))
    line_r = _fit_line(deltas_r)
    line_c = _fit_line(deltas_c)
    if line_r is None or line_c is None:
        return None
    ar, br = line_r
    ac, bc = line_c
    if br == 0 and bc == 0:
        return None
    out: list[Grid] = []
    valid = True
    for i in range(n):
        dr = ar + br * i
        dc = ac + bc * i
        shifted = _shift_colored(base, dr, dc)
        if not all(0 <= r < h and 0 <= c < w for r, c in shifted):
            valid = False
            break
        out.append(_frame_from_colored(h, w, shifted))
    if valid:
        return out
    return None


# ------------------------------ moving border family -----------------------------

def _border_path(h: int, w: int) -> list[Cell]:
    if h == 1:
        return [(0, c) for c in range(w)]
    if w == 1:
        return [(r, 0) for r in range(h)]
    path: list[Cell] = []
    path += [(0, c) for c in range(w)]
    path += [(r, w - 1) for r in range(1, h)]
    path += [(h - 1, c) for c in range(w - 2, -1, -1)]
    path += [(r, 0) for r in range(h - 2, 0, -1)]
    return path


def _try_border_motion(frames: Sequence[Grid]) -> list[Grid] | None:
    # Original monochrome family retained for compatibility with provided examples.
    h, w, n = len(frames[0]), len(frames[0][0]), len(frames)
    vis = _visible(frames)
    if len(vis) < 3:
        return None
    sizes = {len(c) for _, c in vis}
    if len(sizes) != 1 or next(iter(sizes)) == 0:
        return None
    path = _border_path(h, w)
    pos = {cell: i for i, cell in enumerate(path)}
    L = len(path)
    if any(any(cell not in pos for cell in cells) for _, cells in vis):
        return None
    col = _single_color(frames)
    if col is None:
        return None
    base_i, base_cells = vis[0]
    base_indices = {pos[c] for c in base_cells}
    for step in range(1, L):
        ok = True
        for i, cells in vis[1:]:
            pred = {(p + step * (i - base_i)) % L for p in base_indices}
            got = {pos[c] for c in cells}
            if pred != got:
                ok = False
                break
        if ok:
            out: list[Grid] = []
            for i in range(n):
                inds = {(p + step * (i - base_i)) % L for p in base_indices}
                out.append(_frame_from_cells(h, w, [path[j] for j in inds], col))
            return out
    return None


def _try_border_motion_multicolor(frames: Sequence[Grid]) -> list[Grid] | None:
    h, w, n = len(frames[0]), len(frames[0][0]), len(frames)
    vis = _visible_colored(frames)
    if len(vis) < 3:
        return None
    path = _border_path(h, w)
    pos = {cell: i for i, cell in enumerate(path)}
    L = len(path)
    # Same colors visible in every nonblank frame and exactly one cell per color.
    color_set = set(vis[0][1].values())
    if len(color_set) != len(vis[0][1]) or len(color_set) < 2:
        return None
    for _, cur in vis:
        if set(cur.values()) != color_set:
            return None
        if any(cell not in pos for cell in cur):
            return None
    first_i, first = vis[0]
    base_by_color = {color: cell for cell, color in first.items()}
    base_idx = {color: pos[cell] for color, cell in base_by_color.items()}
    for step in range(1, L):
        ok = True
        for i, cur in vis[1:]:
            expected = {}
            for color, p0 in base_idx.items():
                p = (p0 + step * (i - first_i)) % L
                expected[path[p]] = color
            if expected != cur:
                ok = False
                break
        if ok:
            out: list[Grid] = []
            for i in range(n):
                cells: Dict[Cell, int] = {}
                for color, p0 in base_idx.items():
                    p = (p0 + step * (i - first_i)) % L
                    cells[path[p]] = color
                out.append(_frame_from_colored(h, w, cells))
            return out
    return None


# ---------------------------- cumulative path family ----------------------------

def _neighbors(cell: Cell) -> list[Cell]:
    r, c = cell
    return [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]


def _order_path(cells: Set[Cell], first_cells: Set[Cell]) -> list[Cell] | None:
    if not cells or not first_cells <= cells:
        return None
    adj = {p: [q for q in _neighbors(p) if q in cells] for p in cells}
    if any(len(v) > 2 for v in adj.values()):
        return None
    endpoints = [p for p, v in adj.items() if len(v) <= 1]
    if not endpoints:
        return None
    for start in endpoints:
        order: list[Cell] = []
        prev: Cell | None = None
        cur: Cell | None = start
        seen: Set[Cell] = set()
        while cur is not None:
            order.append(cur)
            seen.add(cur)
            nxts = [q for q in adj[cur] if q != prev]
            nxt = nxts[0] if nxts else None
            prev, cur = cur, nxt
            if cur in seen:
                return None
        if len(order) != len(cells):
            continue
        m = len(first_cells)
        if set(order[:m]) == first_cells:
            return order
        rev = list(reversed(order))
        if set(rev[:m]) == first_cells:
            return rev
    return None


def _try_path_prefix(frames: Sequence[Grid]) -> list[Grid] | None:
    h, w, n = len(frames[0]), len(frames[0][0]), len(frames)
    vis = _visible(frames)
    if len(vis) < 2:
        return None
    for (_, a), (_, b) in zip(vis, vis[1:]):
        if not a <= b:
            return None
    final_i, final = vis[-1]
    first_i, first_cells = vis[0]
    if first_i != 0 or final_i != n - 1:
        return None
    order = _order_path(final, first_cells)
    if order is None:
        return None
    length_vals = [(i, len(cells)) for i, cells in vis]
    best_step = None
    for step in range(1, max(2, len(final) + 1)):
        ok = True
        start_len = len(first_cells)
        for i, cells in vis:
            pred_len = min(len(final), start_len + step * i)
            if pred_len != len(cells) or set(order[:pred_len]) != cells:
                ok = False
                break
        if ok:
            best_step = step
            break
    if best_step is None:
        return None
    col = _single_color(frames)
    if col is None:
        return None
    out: list[Grid] = []
    for i in range(n):
        pred_len = min(len(final), len(first_cells) + best_step * i)
        out.append(_frame_from_cells(h, w, order[:pred_len], col))
    return out


# --------------------------------- main verifier --------------------------------

def verify_t30(grid: Grid) -> Grid:
    split = _split_frames(grid)
    if split is None:
        return []
    frames, w = split
    if not any(not _cells(f) for f in frames):
        return [row[:] for row in grid]

    solvers = (
        _try_v_growth,
        _try_diamond_rings,
        _try_square_outline_growth,
        _try_square_filled_growth,
        _try_line_sweep,
        _try_border_motion_multicolor,
        _try_border_motion,
        _try_translation,
        _try_path_prefix,
    )
    for solver in solvers:
        filled = solver(frames)
        if filled is None:
            continue
        if _check_preserve(frames, filled):
            return _join_frames(filled, w)
    return [row[:] for row in grid]
