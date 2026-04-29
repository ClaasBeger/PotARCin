"""Generator for Test2/t26: fill symmetry-supported gaps in a colored outline.

Rule:
- The input shows a colored hollow outline plus one or two central cyan lines.
- Each cyan line is a line of symmetry (vertical, horizontal, main diagonal, or
  anti-diagonal).
- In the output, any gap position is filled if some other position in its
  symmetry orbit already contains a non-cyan colored cell. The copied color is
  the orbit's color.
- If every non-line position in an orbit is empty, the orbit remains empty; the
  task does not complete the shape from global gestalt alone.

Generation constraints used to keep examples well-formed and unambiguous:
- the final underlying outline is symmetric across the shown line(s);
- each symmetry orbit contains at most one non-cyan color in the input/output;
- some samples include fully missing orbits (unfillable gaps) so those remain
  empty in the output;
- all generated shapes keep the cyan line markers inside the hollow interior;
- both one-line and two-line symmetry examples are generated, including
  mixed orthogonal+diagonal two-line pairs;
- cyan symmetry-line markers are always preserved in the output.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Sequence, Set, Tuple

Grid = List[List[int]]
Coord = Tuple[int, int]
Line = Tuple[str, int]  # ('h', row), ('v', col), ('d', diff=r-c), ('a', sum=r+c)

BG = 0
CYAN = 8
PALETTE = [1, 2, 3, 4, 5, 6, 7, 9]


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------


def _empty(h: int, w: int, fill: int = BG) -> Grid:
    return [[fill for _ in range(w)] for _ in range(h)]


def _rectangular(grid: Grid) -> bool:
    return bool(grid) and bool(grid[0]) and all(len(row) == len(grid[0]) for row in grid)


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[Coord]:
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _boundary_of_mask(mask: List[List[int]]) -> Set[Coord]:
    h, w = len(mask), len(mask[0])
    out: Set[Coord] = set()
    for r in range(h):
        for c in range(w):
            if not mask[r][c]:
                continue
            if any(mask[nr][nc] == 0 for nr, nc in _neighbors4(r, c, h, w)) or r in (0, h - 1) or c in (0, w - 1):
                out.add((r, c))
    return out


# ---------------------------------------------------------------------------
# Symmetry line parsing and reflections
# ---------------------------------------------------------------------------


def _reflect(line: Line, pos: Coord) -> Coord:
    r, c = pos
    typ, k = line
    if typ == 'h':
        return (2 * k - r, c)
    if typ == 'v':
        return (r, 2 * k - c)
    if typ == 'd':  # r - c = k
        return (c + k, r - k)
    if typ == 'a':  # r + c = k
        return (k - c, k - r)
    raise ValueError(f'unknown line type: {typ}')


def _on_line(line: Line, pos: Coord) -> bool:
    r, c = pos
    typ, k = line
    if typ == 'h':
        return r == k
    if typ == 'v':
        return c == k
    if typ == 'd':
        return r - c == k
    if typ == 'a':
        return r + c == k
    raise ValueError(line)


def _candidate_lines_from_cyan(cyan: Set[Coord]) -> List[Line]:
    if not cyan:
        return []
    rows = defaultdict(int)
    cols = defaultdict(int)
    diffs = defaultdict(int)
    sums = defaultdict(int)
    for r, c in cyan:
        rows[r] += 1
        cols[c] += 1
        diffs[r - c] += 1
        sums[r + c] += 1
    cands: list[Line] = []
    for r, cnt in rows.items():
        if cnt >= 3:
            cands.append(('h', r))
    for c, cnt in cols.items():
        if cnt >= 3:
            cands.append(('v', c))
    for d, cnt in diffs.items():
        if cnt >= 3:
            cands.append(('d', d))
    for s, cnt in sums.items():
        if cnt >= 3:
            cands.append(('a', s))
    return cands


def _parse_lines(grid: Grid) -> List[Line]:
    h, w = len(grid), len(grid[0])
    cyan = {(r, c) for r in range(h) for c in range(w) if grid[r][c] == CYAN}
    cands = _candidate_lines_from_cyan(cyan)
    # Try exact cover by one line, then two lines.
    for line in cands:
        cover = {p for p in cyan if _on_line(line, p)}
        if cover == cyan:
            return [line]
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            l1, l2 = cands[i], cands[j]
            cover = {p for p in cyan if _on_line(l1, p) or _on_line(l2, p)}
            if cover == cyan:
                return [l1, l2]
    raise ValueError('could not parse cyan symmetry line(s)')


def _orbit(pos: Coord, lines: Sequence[Line], h: int, w: int) -> Set[Coord]:
    q = deque([pos])
    seen = {pos}
    while q:
        cur = q.popleft()
        for line in lines:
            nxt = _reflect(line, cur)
            if 0 <= nxt[0] < h and 0 <= nxt[1] < w and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


def transform_t26(grid: Grid) -> Grid:
    if not _rectangular(grid):
        raise ValueError('input grid must be non-empty and rectangular')
    h, w = len(grid), len(grid[0])
    lines = _parse_lines(grid)
    out = _empty(h, w, BG)

    visited: Set[Coord] = set()
    for r in range(h):
        for c in range(w):
            if grid[r][c] == CYAN or (r, c) in visited:
                continue
            orb = _orbit((r, c), lines, h, w)
            visited |= orb
            colors = {grid[rr][cc] for rr, cc in orb if grid[rr][cc] not in (BG, CYAN)}
            if len(colors) > 1:
                raise ValueError(f'conflicting colors in orbit {orb}: {colors}')
            if len(colors) == 1:
                color = next(iter(colors))
                for rr, cc in orb:
                    if grid[rr][cc] != CYAN:
                        out[rr][cc] = color

    # Preserve all cyan symmetry-line markers in the output.
    for r in range(h):
        for c in range(w):
            if grid[r][c] == CYAN:
                out[r][c] = CYAN
    return out


# ---------------------------------------------------------------------------
# Generator helpers
# ---------------------------------------------------------------------------


def _line_cells(line: Line, h: int, w: int, length: int = 5) -> Set[Coord]:
    typ, k = line
    cells: Set[Coord] = set()
    half = length // 2
    if typ == 'h':
        r = k
        c0 = w // 2
        for dc in range(-half, half + 1):
            c = c0 + dc
            if 0 <= c < w:
                cells.add((r, c))
    elif typ == 'v':
        c = k
        r0 = h // 2
        for dr in range(-half, half + 1):
            r = r0 + dr
            if 0 <= r < h:
                cells.add((r, c))
    elif typ == 'd':
        # parameterize around center with equal row/col changes
        r0, c0 = h // 2, w // 2
        # project center to the line by adjusting row to c + k
        r0 = c0 + k
        if not (0 <= r0 < h):
            r0 = h // 2
            c0 = r0 - k
        for t in range(-half, half + 1):
            r, c = r0 + t, c0 + t
            if 0 <= r < h and 0 <= c < w and r - c == k:
                cells.add((r, c))
    elif typ == 'a':
        r0, c0 = h // 2, w // 2
        r0 = k - c0
        if not (0 <= r0 < h):
            r0 = h // 2
            c0 = k - r0
        for t in range(-half, half + 1):
            r, c = r0 + t, c0 - t
            if 0 <= r < h and 0 <= c < w and r + c == k:
                cells.add((r, c))
    else:
        raise ValueError(line)
    return cells


def _filled_rectangle(h: int, w: int, top: int, bottom: int, left: int, right: int) -> List[List[int]]:
    mask = _empty(h, w, 0)
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            mask[r][c] = 1
    return mask


def _filled_diamond(h: int, w: int, radius: int) -> List[List[int]]:
    mask = _empty(h, w, 0)
    cr, cc = h // 2, w // 2
    for r in range(h):
        for c in range(w):
            if abs(r - cr) + abs(c - cc) <= radius:
                mask[r][c] = 1
    return mask


def _filled_octagon(h: int, w: int, R: int, S: int) -> List[List[int]]:
    mask = _empty(h, w, 0)
    cr, cc = h // 2, w // 2
    for r in range(h):
        for c in range(w):
            dr, dc = abs(r - cr), abs(c - cc)
            if max(dr, dc) <= R and dr + dc <= S:
                mask[r][c] = 1
    return mask


def _filled_vert_blob(h: int, w: int, rng: random.Random, mirror_h: bool) -> List[List[int]]:
    mask = _empty(h, w, 0)
    cc = w // 2
    top = rng.randint(1, h // 3)
    bottom = rng.randint(2 * h // 3, h - 2)
    if mirror_h:
        top = h - 1 - bottom
    rows = list(range(top, bottom + 1))
    if mirror_h:
        half_rows = [r for r in rows if r <= h // 2]
    else:
        half_rows = rows
    widths: Dict[int, int] = {}
    cur = rng.randint(max(2, w // 6), max(3, w // 4))
    for r in half_rows:
        cur += rng.choice([-1, 0, 1])
        cur = max(2, min((w // 2) - 2, cur))
        widths[r] = cur
        if mirror_h:
            widths[h - 1 - r] = cur
    for r in rows:
        width = widths[r]
        for c in range(cc - width, cc + width + 1):
            mask[r][c] = 1
    return mask


def _outline_from_family(h: int, w: int, lines: Sequence[Line], rng: random.Random) -> Set[Coord]:
    line_types = {t for t, _ in lines}
    use_d4 = any(t in ('d', 'a') for t in line_types)
    for _ in range(500):
        if use_d4:
            fam = rng.choice(['square', 'diamond', 'octagon'])
            if fam == 'square':
                span = rng.randint(min(h, w) // 3, (min(h, w) // 2) - 1)
                mask = _filled_rectangle(h, w, h // 2 - span, h // 2 + span, w // 2 - span, w // 2 + span)
            elif fam == 'diamond':
                radius = rng.randint(min(h, w) // 3, (min(h, w) // 2) - 1)
                mask = _filled_diamond(h, w, radius)
            else:
                R = rng.randint(min(h, w) // 3, (min(h, w) // 2) - 1)
                S = R + rng.randint(max(2, R // 2), R)
                mask = _filled_octagon(h, w, R, S)
        else:
            fam = rng.choice(['rectangle', 'diamond', 'vblob'])
            if fam == 'rectangle':
                hh = rng.randint(h // 2, h - 3)
                ww = rng.randint(w // 2, w - 3)
                top = (h - hh) // 2 if len(lines) == 2 else rng.randint(1, h - hh - 1)
                bottom = top + hh - 1
                left = (w - ww) // 2
                right = left + ww - 1
                mask = _filled_rectangle(h, w, top, bottom, left, right)
            elif fam == 'diamond':
                radius = rng.randint(min(h, w) // 3, (min(h, w) // 2) - 1)
                mask = _filled_diamond(h, w, radius)
            else:
                mask = _filled_vert_blob(h, w, rng, mirror_h=(len(lines) == 2))
        outline = _boundary_of_mask(mask)
        # Remove any cells that lie directly on a cyan symmetry marker. This lets
        # one- and two-line examples both keep the marker inside the hollow shape
        # rather than overwriting it with outline cells.
        outline = {p for p in outline if not any(_on_line(line, p) for line in lines)}
        if len(outline) < 12:
            continue
        # Keep only full symmetry orbits that stay entirely off the cyan lines.
        # This is especially important for mixed two-line cases such as
        # vertical+diagonal: some boundary cells may have reflected partners that
        # land on a symmetry line, and those incomplete orbits must be excluded.
        filtered: Set[Coord] = set()
        seen_orbit_cells: Set[Coord] = set()
        for p in sorted(outline):
            if p in seen_orbit_cells:
                continue
            orb = _orbit(p, lines, h, w)
            seen_orbit_cells |= orb
            if orb <= outline:
                filtered |= orb
        outline = filtered
        if len(outline) < 12:
            continue
        return outline
    raise RuntimeError('failed to generate symmetric outline')


def _partition_orbits(cells: Set[Coord], lines: Sequence[Line], h: int, w: int) -> List[Set[Coord]]:
    out: List[Set[Coord]] = []
    seen: Set[Coord] = set()
    for p in sorted(cells):
        if p in seen:
            continue
        orb = _orbit(p, lines, h, w) & cells
        seen |= orb
        out.append(orb)
    return out


def _assign_colors(orbits: Sequence[Set[Coord]], rng: random.Random) -> Dict[Coord, int]:
    n_colors = rng.randint(2, min(5, len(PALETTE)))
    palette = rng.sample(PALETTE, n_colors)
    out: Dict[Coord, int] = {}
    for i, orb in enumerate(orbits):
        color = palette[i % n_colors] if rng.random() < 0.35 else rng.choice(palette)
        for p in orb:
            out[p] = color
    return out


def _line_setup(rng: random.Random, h: int, w: int) -> List[Line]:
    cr, cc = h // 2, w // 2
    # Include both one-line and two-line examples. Two-line cases are weighted
    # deliberately so the generated sample set reflects the original demos.
    one_line: List[List[Line]] = [
        [('v', cc)],
        [('h', cr)],
        [('d', cr - cc)],
        [('a', cr + cc)],
    ]
    two_line: List[List[Line]] = [
        [('v', cc), ('h', cr)],
        [('d', cr - cc), ('a', cr + cc)],
        [('v', cc), ('d', cr - cc)],
        [('v', cc), ('a', cr + cc)],
        [('h', cr), ('d', cr - cc)],
        [('h', cr), ('a', cr + cc)],
    ]
    return rng.choice(two_line if rng.random() < 0.45 else one_line)


def generate(rng: random.Random | None = None) -> Dict[str, Grid]:
    rng = rng or random.Random()
    for _ in range(4000):
        # Odd sizes keep all symmetry lines centered cleanly.
        h = rng.choice([11, 13, 15, 17])
        w = h if rng.random() < 0.8 else rng.choice([11, 13, 15, 17])
        lines = _line_setup(rng, h, w)
        try:
            outline = _outline_from_family(h, w, lines, rng)
        except RuntimeError:
            continue
        orbits = [orb for orb in _partition_orbits(outline, lines, h, w) if len(orb) >= 2]
        if len(orbits) < 4:
            continue
        colors = _assign_colors(orbits, rng)

        want_unfillable = rng.random() < 0.5
        n_partial = rng.randint(1, max(1, min(5, len(orbits) - 1)))
        partial_orbits = rng.sample(orbits, n_partial)
        remaining = [orb for orb in orbits if orb not in partial_orbits]
        full_orbits: List[Set[Coord]] = []
        if want_unfillable and remaining:
            n_full = rng.randint(1, min(3, len(remaining)))
            full_orbits = rng.sample(remaining, n_full)

        input_cells: Dict[Coord, int] = {}
        # Start with the full final outline.
        for p, color in colors.items():
            input_cells[p] = color
        # Remove full orbits entirely (these remain empty in the output too).
        for orb in full_orbits:
            for p in orb:
                input_cells.pop(p, None)
        # Remove some but not all cells in selected partial orbits.
        for orb in partial_orbits:
            k = rng.randint(1, len(orb) - 1)
            removed = rng.sample(sorted(orb), k)
            # Ensure at least one cell remains.
            for p in removed:
                input_cells.pop(p, None)

        # Build input grid.
        inp = _empty(h, w, BG)
        line_cells: Set[Coord] = set()
        for line in lines:
            line_cells |= _line_cells(line, h, w, length=5)
        for r, c in line_cells:
            inp[r][c] = CYAN
        for (r, c), color in input_cells.items():
            if inp[r][c] == CYAN:
                raise RuntimeError('shape overlapped line marker unexpectedly')
            inp[r][c] = color
        out = transform_t26(inp)

        # Quality checks.
        if out == inp:
            continue
        # Verify at least one fillable gap got restored.
        restored = sum(1 for r in range(h) for c in range(w) if inp[r][c] == BG and out[r][c] not in (BG, CYAN))
        if restored == 0:
            continue
        # Ensure requested unfillable case exists when desired.
        if want_unfillable:
            still_empty = 0
            for orb in full_orbits:
                for p in orb:
                    if out[p[0]][p[1]] == BG:
                        still_empty += 1
            if still_empty == 0:
                continue
        return {'input': inp, 'output': out}
    raise RuntimeError('failed to generate a valid t26 sample')


if __name__ == '__main__':
    print(json.dumps(generate(rng=random.Random(0)), indent=2))
