# Modified generator for ARC-GEN task255 (excavation).
#
# Rule implemented:
# - Whenever the green artery touches the left or right grid edge in a row,
#   expand it horizontally to fill the available width, but do not overwrite
#   any non-green non-black (the "other" colored) cells.
#
# This file intentionally does NOT modify the original task255.py.

import copy
from pathlib import Path
import importlib.util

import common


def _load_base_task255():
    base_path = Path(__file__).resolve().parent / "task255.py"
    spec = importlib.util.spec_from_file_location("arcgen_task255_base", base_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load base task255 at {base_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE = None


def _get_base():
    global _BASE
    if _BASE is None:
        _BASE = _load_base_task255()
    return _BASE


def _edgefill_output(output, GREEN):
    h = len(output)
    w = len(output[0]) if h else 0
    if h == 0 or w == 0:
        return output

    # Any non-zero and non-green cell is considered "other color" and is
    # protected from overwriting.
    def _is_other(v: int) -> bool:
        return v != 0 and v != GREEN

    # Component-based expansion to preserve rectangle-ness:
    # - find 4-connected GREEN components in the base output.
    # - if a component touches left/right border, expand it horizontally as a
    #   rectangle across the component's full row-span.
    # - never place a new GREEN cell adjacent (8-neighborhood) to an OTHER-color
    #   cell from the base output.
    src = output
    out = [list(row) for row in src]

    visited = [[False for _ in range(w)] for _ in range(h)]

    def _can_place_at(rr: int, cc: int) -> bool:
        # Hard block: cannot be OTHER-color.
        if _is_other(src[rr][cc]):
            return False
        # No adjacency in the full 9-neighborhood (incl. diagonals).
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and _is_other(src[nr][nc]):
                    return False
        return True

    from collections import deque

    def _contiguous_runs(rows):
        if not rows:
            return []
        rows = sorted(set(rows))
        runs = []
        start = rows[0]
        prev = rows[0]
        for r in rows[1:]:
            if r == prev + 1:
                prev = r
                continue
            runs.append((start, prev))
            start = r
            prev = r
        runs.append((start, prev))
        return runs

    def _prune_non_edge_additions(src_grid, out_grid):
        """Remove edgefill additions that do not belong to edge-reaching runs.

        We only remove cells that were newly added by edgefill (src!=GREEN, out==GREEN).
        For each 4-connected GREEN component in out, consider contiguous row-runs
        within that component. If a run has no left/right edge contact in out,
        drop all newly added cells in that run.
        """
        from collections import deque

        visited2 = [[False for _ in range(w)] for _ in range(h)]
        for sr2 in range(h):
            for sc2 in range(w):
                if visited2[sr2][sc2] or out_grid[sr2][sc2] != GREEN:
                    continue
                q2 = deque([(sr2, sc2)])
                visited2[sr2][sc2] = True
                comp2 = []
                while q2:
                    r2, c2 = q2.popleft()
                    comp2.append((r2, c2))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr2, nc2 = r2 + dr, c2 + dc
                        if 0 <= nr2 < h and 0 <= nc2 < w and not visited2[nr2][nc2] and out_grid[nr2][nc2] == GREEN:
                            visited2[nr2][nc2] = True
                            q2.append((nr2, nc2))

                rows = sorted({r2 for (r2, _c2) in comp2})

                for a2, b2 in _contiguous_runs(rows):
                    run_rows = list(range(a2, b2 + 1))
                    touches_side = False
                    for rr2 in run_rows:
                        cols = [c2 for (r2, c2) in comp2 if r2 == rr2]
                        if not cols:
                            continue
                        if min(cols) == 0 or max(cols) == w - 1:
                            touches_side = True
                            break
                    if touches_side:
                        continue
                    # No edge contact in this run -> remove only newly added cells.
                    for rr2 in run_rows:
                        for cc2 in range(w):
                            if out_grid[rr2][cc2] == GREEN and src_grid[rr2][cc2] != GREEN:
                                out_grid[rr2][cc2] = src_grid[rr2][cc2]

    def _prune_non_edge_branches_from_mainbar(src_grid, out_grid):
        """Narrow directional prune based on opposite-side edge reach.

        For off-axis rows (width differs from main bar width):
        - if left overhang does NOT reach left edge, but right side reaches right edge,
          trim the left overhang.
        - symmetric for right overhang.
        """
        from collections import Counter, deque

        def _num_cc(cells):
            if not cells:
                return 0
            cells = set(cells)
            seen = set()
            ncc = 0
            for cell in list(cells):
                if cell in seen:
                    continue
                ncc += 1
                stack = [cell]
                seen.add(cell)
                while stack:
                    rr, cc = stack.pop()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nn = (rr + dr, cc + dc)
                        if nn in cells and nn not in seen:
                            seen.add(nn)
                            stack.append(nn)
            return ncc

        while True:
            changed = False
            visited3 = [[False for _ in range(w)] for _ in range(h)]
            for sr3 in range(h):
                for sc3 in range(w):
                    if visited3[sr3][sc3] or out_grid[sr3][sc3] != GREEN:
                        continue
                    q3 = deque([(sr3, sc3)])
                    visited3[sr3][sc3] = True
                    comp3 = []
                    while q3:
                        r3, c3 = q3.popleft()
                        comp3.append((r3, c3))
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr3, nc3 = r3 + dr, c3 + dc
                            if 0 <= nr3 < h and 0 <= nc3 < w and not visited3[nr3][nc3] and out_grid[nr3][nc3] == GREEN:
                                visited3[nr3][nc3] = True
                                q3.append((nr3, nc3))

                    rows = sorted({r3 for (r3, _c3) in comp3})
                    if not rows:
                        continue
                    row_min = {}
                    row_max = {}
                    row_w = {}
                    for rr3 in rows:
                        cs = [cc3 for (r3, cc3) in comp3 if r3 == rr3]
                        row_min[rr3] = min(cs)
                        row_max[rr3] = max(cs)
                        row_w[rr3] = row_max[rr3] - row_min[rr3] + 1

                    main_w = Counter(row_w.values()).most_common(1)[0][0]
                    main_rows = [r for r in rows if row_w[r] == main_w]
                    if not main_rows:
                        continue
                    core_l = Counter([row_min[r] for r in main_rows]).most_common(1)[0][0]
                    core_r = core_l + main_w - 1
                    comp_set = set(comp3)

                    for rr3 in rows:
                        if row_w[rr3] == main_w:
                            continue
                        l = row_min[rr3]
                        r = row_max[rr3]
                        left_nonedge = l < core_l and l > 0
                        right_nonedge = r > core_r and r < w - 1
                        reaches_left = l == 0
                        reaches_right = r == w - 1

                        if left_nonedge and reaches_right:
                            cand = {
                                (rr3, cc3)
                                for cc3 in range(l, core_l)
                                if out_grid[rr3][cc3] == GREEN and src_grid[rr3][cc3] != GREEN
                            }
                            # Keep component connected after removal.
                            remain = comp_set - cand
                            if _num_cc(remain) <= 1:
                                for rrn, ccn in cand:
                                    out_grid[rrn][ccn] = 0
                                    changed = True
                                comp_set = remain
                        if right_nonedge and reaches_left:
                            cand = {
                                (rr3, cc3)
                                for cc3 in range(core_r + 1, r + 1)
                                if out_grid[rr3][cc3] == GREEN and src_grid[rr3][cc3] != GREEN
                            }
                            remain = comp_set - cand
                            if _num_cc(remain) <= 1:
                                for rrn, ccn in cand:
                                    out_grid[rrn][ccn] = 0
                                    changed = True
                                comp_set = remain

            if not changed:
                break

    for sr in range(h):
        for sc in range(w):
            if visited[sr][sc] or src[sr][sc] != GREEN:
                continue

            q = deque([(sr, sc)])
            visited[sr][sc] = True
            comp = []
            rmin = sr
            rmax = sr
            touches_left = sc == 0
            touches_right = sc == w - 1

            while q:
                r, c = q.popleft()
                comp.append((r, c))
                rmin = min(rmin, r)
                rmax = max(rmax, r)
                if c == 0:
                    touches_left = True
                if c == w - 1:
                    touches_right = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and src[nr][nc] == GREEN:
                        visited[nr][nc] = True
                        q.append((nr, nc))

            if not (touches_left or touches_right):
                continue

            # Only rows that already touch the respective edge are candidates
            # for expansion in that direction. If a row is blocked right at
            # the edge by OTHER-color adjacency, it gets no expansion.
            if touches_left:
                left_rows = [r for (r, c) in comp if c == 0]
                for a, b in _contiguous_runs(left_rows):
                    row_span = list(range(a, b + 1))
                    eligible_rows = [rr for rr in row_span if _can_place_at(rr, 0)]
                    for ea, eb in _contiguous_runs(eligible_rows):
                        run_rows = list(range(ea, eb + 1))
                        end = -1
                        for cc in range(w):
                            if all(_can_place_at(rr, cc) for rr in run_rows):
                                end = cc
                            else:
                                break
                        if end >= 0:
                            for rr in run_rows:
                                for cc in range(0, end + 1):
                                    out[rr][cc] = GREEN

            if touches_right:
                right_rows = [r for (r, c) in comp if c == w - 1]
                for a, b in _contiguous_runs(right_rows):
                    row_span = list(range(a, b + 1))
                    eligible_rows = [rr for rr in row_span if _can_place_at(rr, w - 1)]
                    for ea, eb in _contiguous_runs(eligible_rows):
                        run_rows = list(range(ea, eb + 1))
                        start = w
                        for cc in range(w - 1, -1, -1):
                            if all(_can_place_at(rr, cc) for rr in run_rows):
                                start = cc
                            else:
                                break
                        if start < w:
                            for rr in run_rows:
                                for cc in range(start, w):
                                    out[rr][cc] = GREEN

                # Conservative edge-closure:
                # if a row in the same contiguous band is exactly one column
                # short of the right edge, and the edge cell is safe, fill only
                # that final edge column cell. This targets underfill like
                # dynamic[33] without broad run bridging.
                comp_rows = sorted({r for (r, _c) in comp})
                row_to_max = {}
                for rr in comp_rows:
                    cols = [cc for (r2, cc) in comp if r2 == rr]
                    if cols:
                        row_to_max[rr] = max(cols)
                right_row_set = set(right_rows)
                for a, b in _contiguous_runs(comp_rows):
                    run = list(range(a, b + 1))
                    if not any(r in right_row_set for r in run):
                        continue
                    for rr in run:
                        if row_to_max.get(rr, -1) == w - 2 and _can_place_at(rr, w - 1):
                            out[rr][w - 1] = GREEN


    _prune_non_edge_additions(src, out)
    _prune_non_edge_branches_from_mainbar(src, out)
    return out


def _edge_runs_are_consistent(output, GREEN):
    """Reject partial side-expansion runs.

    For any 4-connected GREEN component, if a contiguous run of rows contains
    at least one row touching a side edge (left/right), then every row in that
    run must touch that side. Otherwise we treat it as a blocked-row artifact
    and reject/resample.
    """
    h = len(output)
    w = len(output[0]) if h else 0
    if h == 0 or w == 0:
        return True

    from collections import deque

    visited = [[False for _ in range(w)] for _ in range(h)]

    def _contiguous_runs(rows):
        if not rows:
            return []
        rows = sorted(set(rows))
        runs = []
        a = rows[0]
        b = rows[0]
        for r in rows[1:]:
            if r == b + 1:
                b = r
            else:
                runs.append((a, b))
                a = r
                b = r
        runs.append((a, b))
        return runs

    for sr in range(h):
        for sc in range(w):
            if visited[sr][sc] or output[sr][sc] != GREEN:
                continue
            q = deque([(sr, sc)])
            visited[sr][sc] = True
            comp = []
            while q:
                r, c = q.popleft()
                comp.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and output[nr][nc] == GREEN:
                        visited[nr][nc] = True
                        q.append((nr, nc))

            rows_in_comp = sorted({r for (r, _c) in comp})
            if not rows_in_comp:
                continue

            row_min = {}
            row_max = {}
            for r in rows_in_comp:
                cols = [c for (rr, c) in comp if rr == r]
                row_min[r] = min(cols)
                row_max[r] = max(cols)

            # Left side consistency.
            for a, b in _contiguous_runs(rows_in_comp):
                run = list(range(a, b + 1))
                touch = [r for r in run if row_min.get(r, 1) == 0]
                if touch and len(touch) != len(run):
                    return False

            # Right side consistency.
            for a, b in _contiguous_runs(rows_in_comp):
                run = list(range(a, b + 1))
                touch = [r for r in run if row_max.get(r, -1) == w - 1]
                if touch and len(touch) != len(run):
                    return False

    return True


def generate(colors=None, size=30):
    base = _get_base()
    GREEN = common.green()
    ex = base.generate(colors=colors, size=size)
    out = _edgefill_output(ex["output"], GREEN)
    inp = [list(row) for row in out]
    for r in range(len(inp)):
        for c in range(len(inp[0])):
            if inp[r][c] == GREEN:
                inp[r][c] = 0
    return {"input": inp, "output": out}


def validate():
    # Best-effort: defer to base task255.validate (behavioral equivalence isn't guaranteed).
    base = _get_base()
    return base.validate()

