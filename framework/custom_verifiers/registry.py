from __future__ import annotations

from collections import Counter, deque
from typing import Callable, Deque, Dict, List, Optional, Set, Tuple

from framework.grids import Grid


def _solve_6cf79266(grid: Grid) -> Grid:
    """Explicit one-pass 3x3 fill mirroring ARC-GEN task162 behavior."""
    out = [list(row) for row in grid]
    h = len(out)
    w = len(out[0]) if h else 0

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            neighborhood_total = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    neighborhood_total += out[r + dr][c + dc]
            if neighborhood_total:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    out[r + dr][c + dc] = 1
    return out


def _solve_e6721834(grid: Grid) -> Grid:
    """Custom verifier for ARC task e6721834.

    This fixes the *root cause symptom* in existing verifiers: occasionally
    degenerating to returning one input half unchanged (missing forecolor-filled
    boxes) when dot evidence is too sparse to satisfy their object-matching
    predicate.

    Approach:
    - Use the existing re_arc verifier as the baseline.
    - If (and only if) its output equals one of the two input halves, treat that
      as the degeneration case and reconstruct the missing filled boxes by
      dot-color alignment + bbox recovery via flood-fill on the opposite half.
    """

    from framework.integrations.re_arc_adapter import get_re_arc_verifier
    from framework.grids import is_equal_grid

    H = len(grid)
    W = len(grid[0]) if H else 0
    if H == 0 or W == 0:
        return []

    base = getattr(_solve_e6721834, "_base_verifier", None)
    if base is None:
        base = get_re_arc_verifier("e6721834")
        setattr(_solve_e6721834, "_base_verifier", base)
    if base is None:
        return [list(row) for row in grid]

    out = base([list(row) for row in grid])

    def _mode_color(g: Grid) -> int:
        c = Counter(x for row in g for x in row)
        return c.most_common(1)[0][0]

    # Only handle the ARC-GEN-style 2-half composites (the source of the mismatch).
    horiz = W > H
    if horiz:
        if W % 2 != 0:
            return out
        half_w = W // 2
        half0 = [row[:half_w] for row in grid]
        half1 = [row[half_w:] for row in grid]
    else:
        if H % 2 != 0:
            return out
        half_h = H // 2
        half0 = grid[:half_h]
        half1 = grid[half_h:]

    out_is_half0 = is_equal_grid(out, half0)
    out_is_half1 = is_equal_grid(out, half1)
    if not (out_is_half0 or out_is_half1):
        return out

    dot_half = half0 if out_is_half0 else half1
    box_half = half1 if out_is_half0 else half0

    dot_bg = _mode_color(dot_half)

    # On the "box half", the filled-rectangle color (forecolor) is typically
    # the most frequent value, while the background is usually the runner-up.
    box_counts_all = Counter(x for row in box_half for x in row)
    forecolor = box_counts_all.most_common(1)[0][0]
    box_counts_all.pop(forecolor, None)
    if not box_counts_all:
        return out
    box_bg = box_counts_all.most_common(1)[0][0]

    # Collect dot colors and their positions on the dot half.
    dot_color_to_positions: Dict[int, List[Tuple[int, int]]] = {}
    for r, row in enumerate(dot_half):
        for c, v in enumerate(row):
            if v == dot_bg:
                continue
            dot_color_to_positions.setdefault(v, []).append((r, c))

    def _flood_fill_non_bg(half: Grid, bg: int, start: Tuple[int, int]) -> List[Tuple[int, int]]:
        hh = len(half)
        ww = len(half[0]) if hh else 0
        sr, sc = start
        if not (0 <= sr < hh and 0 <= sc < ww):
            return []
        if half[sr][sc] == bg:
            return []
        q: Deque[Tuple[int, int]] = deque([(sr, sc)])
        seen: Set[Tuple[int, int]] = {start}
        cells: List[Tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if nr < 0 or nc < 0 or nr >= hh or nc >= ww:
                    continue
                if (nr, nc) in seen:
                    continue
                if half[nr][nc] == bg:
                    continue
                seen.add((nr, nc))
                q.append((nr, nc))
        return cells

    def _bbox(cells: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        return min(rs), min(cs), max(rs), max(cs)

    out_h = len(dot_half)
    out_w = len(dot_half[0]) if out_h else 0
    patched: Grid = [[dot_bg for _ in range(out_w)] for _ in range(out_h)]

    # Reconstruct each box using dot-color alignment.
    for dot_color, out_positions in dot_color_to_positions.items():
        box_positions = [
            (r, c)
            for r, row in enumerate(box_half)
            for c, v in enumerate(row)
            if v == dot_color
        ]
        if not box_positions:
            # Paint dots only.
            for r, c in out_positions:
                patched[r][c] = dot_color
            continue

        region = _flood_fill_non_bg(box_half, box_bg, box_positions[0])
        if not region:
            for r, c in out_positions:
                patched[r][c] = dot_color
            continue

        r0, c0, r1, c1 = _bbox(region)
        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1

        internal_offsets = [(r - r0, c - c0) for r, c in box_positions]
        internal_set = set(internal_offsets)
        out_set = set(out_positions)
        out_dot0 = out_positions[0]

        chosen: Optional[Tuple[int, int]] = None
        for off_r, off_c in internal_set:
            tr = out_dot0[0] - off_r
            tc = out_dot0[1] - off_c
            predicted = set((tr + rr, tc + cc) for rr, cc in internal_set)
            if predicted == out_set:
                chosen = (tr, tc)
                break
        if chosen is None:
            off_r, off_c = internal_offsets[0]
            chosen = (out_dot0[0] - off_r, out_dot0[1] - off_c)

        tr, tc = chosen

        for rr in range(tr, tr + box_h):
            if rr < 0 or rr >= out_h:
                continue
            row = patched[rr]
            for cc in range(tc, tc + box_w):
                if 0 <= cc < out_w:
                    row[cc] = forecolor

        for r, c in out_positions:
            patched[r][c] = dot_color

    return patched


def _solve_ac0a08a4(grid: Grid) -> Grid:
    """Upscale by non-black pixel count for ARC-GEN task269/ac0a08a4.

    Root cause in v1: it infers background via `mostcolor`, which fails when
    the 3x3 input has no black cells (all 9 cells colored), yielding scale=8.
    ARC-GEN generator uses black explicitly as background, so scale should be
    total_cells - black_count.
    """

    h = len(grid)
    w = len(grid[0]) if h else 0
    if h == 0 or w == 0:
        return []

    black_count = sum(1 for row in grid for val in row if val == 0)
    scale = h * w - black_count
    if scale <= 0:
        return []

    out: Grid = []
    for row in grid:
        expanded_rows = [[] for _ in range(scale)]
        for val in row:
            block = [val] * scale
            for rr in range(scale):
                expanded_rows[rr].extend(block)
        out.extend(expanded_rows)
    return out


def _solve_90f3ed37(grid: Grid) -> Grid:
    """v5 for 90f3ed37: latent enumeration + NeurIPS baseline; aligned with task219.generate."""

    import importlib.util
    from pathlib import Path

    h = len(grid)
    w = len(grid[0]) if h else 0
    if h == 0 or w == 0:
        return []

    BLACK, BLUE, CYAN = 0, 1, 8

    # Shared latent enumeration (same geometry as custom ARC-GEN filter). When the
    # input uniquely determines the output, return it without calling v4 (avoids
    # rare NeurIPS solver crashes and matches unambiguous dynamic samples).
    from framework.tasks.arc_dataset import (
        _arc_gen_90f3ed37_analyze_latent_fits,
        _arc_gen_90f3ed37_cols_match_global_awide,
    )

    _probe = [[0] * w for _ in range(h)]
    _keys, _ = _arc_gen_90f3ed37_analyze_latent_fits(
        {"input": grid, "output": _probe}, apply_col_lexmax=False
    )
    if len(_keys) == 1:
        _only = next(iter(_keys))
        return [list(row) for row in _only]

    # Load/cached v4 solver.
    base = getattr(_solve_90f3ed37, "_v4_verifier", None)
    if base is None:
        root = Path(__file__).resolve().parents[2]
        path = root / "external" / "NeurIPS-Code-Golf-2025" / "solutions" / "task219.py"
        if not path.exists():
            return [list(row) for row in grid]
        spec = importlib.util.spec_from_file_location("neurips_task219_v4", path)
        if spec is None or spec.loader is None:
            return [list(row) for row in grid]
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        solve = getattr(module, "solve", None)
        if callable(solve):
            base = solve
        else:
            base = None
            for attr_name in dir(module):
                if attr_name.startswith("_"):
                    continue
                attr = getattr(module, attr_name)
                if callable(attr):
                    base = attr
                    break
            if base is None:
                return [list(row) for row in grid]
        setattr(_solve_90f3ed37, "_v4_verifier", base)

    base_out = [list(row) for row in base([list(r) for r in grid])]

    active = [any(v == CYAN for v in row) for row in grid]
    bands: List[Tuple[int, int]] = []
    r = 0
    while r < h:
        if not active[r]:
            r += 1
            continue
        s = r
        while r + 1 < h and active[r + 1]:
            r += 1
        bands.append((s, r))
        r += 1
    if not bands:
        return base_out

    tall_candidates = sorted({(e - s + 1) for s, e in bands if 1 <= (e - s + 1) <= 3}) or [1, 2, 3]

    def _pack(vals: List[List[int]], wide: int, tall: int) -> int:
        m = 0
        for rr in range(tall):
            for cc in range(wide):
                if vals[rr][cc]:
                    m |= 1 << (rr * wide + cc)
        return m

    def _render(tall: int, aw: int, bw: int, cw: int, cols: List[int], ma: int, mb: int, mc: int, output_mode: bool) -> Grid:
        out = [[BLACK for _ in range(w)] for _ in range(h)]
        for bi, (s, _e) in enumerate(bands):
            col = cols[bi]
            for rr in range(tall):
                row = s + rr
                for a0 in range(0, col, aw):
                    for lc in range(aw):
                        if ((ma >> (rr * aw + lc)) & 1):
                            c = a0 + lc
                            if 0 <= c < w:
                                out[row][c] = CYAN
                for lc in range(bw):
                    if ((mb >> (rr * bw + lc)) & 1):
                        c = col + lc
                        if 0 <= c < w:
                            out[row][c] = CYAN
                if output_mode:
                    c_color = CYAN if bi == 0 else BLUE
                    for c0 in range(col + bw, w, cw):
                        for lc in range(cw):
                            if ((mc >> (rr * cw + lc)) & 1):
                                c = c0 + lc
                                if 0 <= c < w:
                                    out[row][c] = c_color
                elif bi == 0:
                    for c0 in range(col + bw, w, cw):
                        for lc in range(cw):
                            if ((mc >> (rr * cw + lc)) & 1):
                                c = c0 + lc
                                if 0 <= c < w:
                                    out[row][c] = CYAN
        return out

    candidates: List[Grid] = []
    for tall in tall_candidates:
        if any((e - s + 1) != tall for s, e in bands):
            continue
        n = len(bands)
        for aw in (1, 2):
            for bw in (1, 2):
                for cw in (1, 2):
                    opts = [aw, 2 * aw] + ([4 * aw] if aw == 1 else [])
                    for cm in range(len(opts) ** n):
                        cols: List[int] = []
                        tmp = cm
                        for _ in range(n):
                            cols.append(opts[tmp % len(opts)])
                            tmp //= len(opts)
                        if not _arc_gen_90f3ed37_cols_match_global_awide(cols):
                            continue
                        a = [[-1] * aw for _ in range(tall)]
                        b = [[-1] * bw for _ in range(tall)]
                        c = [[-1] * cw for _ in range(tall)]
                        valid = True

                        def _set(arr: List[List[int]], rr: int, cc: int, val: int) -> None:
                            nonlocal valid
                            old = arr[rr][cc]
                            if old == -1:
                                arr[rr][cc] = val
                            elif old != val:
                                valid = False

                        for bi, (s, _e) in enumerate(bands):
                            col = cols[bi]
                            for rr in range(tall):
                                row = s + rr
                                for a0 in range(0, col, aw):
                                    for lc in range(aw):
                                        _set(a, rr, lc, 1 if grid[row][a0 + lc] == CYAN else 0)
                                        if not valid:
                                            break
                                    if not valid:
                                        break
                                if not valid:
                                    break
                                for lc in range(bw):
                                    cc = col + lc
                                    if 0 <= cc < w:
                                        _set(b, rr, lc, 1 if grid[row][cc] == CYAN else 0)
                                if not valid:
                                    break
                                if bi == 0:
                                    for c0 in range(col + bw, w, cw):
                                        for lc in range(cw):
                                            cc = c0 + lc
                                            if 0 <= cc < w:
                                                _set(c, rr, lc, 1 if grid[row][cc] == CYAN else 0)
                                        if not valid:
                                            break
                                if not valid:
                                    break
                            if not valid:
                                break

                        if not valid:
                            continue

                        for rr in range(tall):
                            for cc in range(aw):
                                if a[rr][cc] == -1:
                                    a[rr][cc] = 0
                            for cc in range(bw):
                                if b[rr][cc] == -1:
                                    b[rr][cc] = 0
                            for cc in range(cw):
                                if c[rr][cc] == -1:
                                    c[rr][cc] = 0

                        ma, mb, mc = _pack(a, aw, tall), _pack(b, bw, tall), _pack(c, cw, tall)
                        if ma == 0 or mb == 0 or mc == 0:
                            continue
                        if _render(tall, aw, bw, cw, cols, ma, mb, mc, output_mode=False) != grid:
                            continue
                        candidates.append(_render(tall, aw, bw, cw, cols, ma, mb, mc, output_mode=True))

    if not candidates:
        return base_out

    def _score(cand: Grid) -> Tuple[int, int]:
        diff = 0
        penalty = 0
        for rr in range(h):
            for cc in range(w):
                if cand[rr][cc] != base_out[rr][cc]:
                    diff += 1
        for bi, (s, e) in enumerate(bands):
            if bi == 0:
                continue
            for rr in range(s, e + 1):
                for cc in range(w):
                    if grid[rr][cc] == CYAN and cand[rr][cc] == BLUE:
                        penalty += 1
        return diff, penalty

    return min(candidates, key=_score)


def get_custom_verifier(task_id: str) -> Optional[Callable[[Grid], Grid]]:
    """Return optional project-local verifier overrides/additions.

    These are attached as ``quinary_verifier`` in :func:`framework.tasks.arc_dataset.load_task`.
    For ``90f3ed37`` (ARC-GEN task219), v5 agrees with ``task219.generate``'s latent-unambiguous
    filter and the stable pair corpus.
    """
    if task_id == "6cf79266":
        return _solve_6cf79266
    if task_id == "e6721834":
        return _solve_e6721834
    if task_id == "ac0a08a4":
        return _solve_ac0a08a4
    if task_id == "90f3ed37":
        return _solve_90f3ed37
    return None
