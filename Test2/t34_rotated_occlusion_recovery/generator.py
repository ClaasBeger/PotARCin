"""Generator for t34: recover an occluded pattern, then rotate by the red-corner cue.

Task rule:
- The input contains a cyan square occluder with exactly one red cell at one corner.
- The output is the pattern hidden beneath the square, rotated so that the red corner
  would move to the top-left of the square.
  * red already top-left     -> no rotation
  * red top-right            -> rotate hidden patch 90° counterclockwise
  * red bottom-left          -> rotate hidden patch 90° clockwise
  * red bottom-right         -> rotate hidden patch 180°

Solver model classes supported:
1) nested concentric rectangular rings,
2) row-constant stripe fields recoverable by bridging across the square, and
3) diagonal-repeat fields with step (s, s) for small s.

Generator constraints / ambiguity controls:
- Underlying colors never use cyan (8) or red (2), reserving those for the occluder.
- Generated examples come from one of the three supported pattern families.
- Examples are filtered so the intended family is the one selected by the verifier.
- Generated backgrounds are visually structured rather than noisy: diagonal examples
  use regular stripe/band patterns, row examples use coherent horizontal bands,
  and ring examples use concentric frames.
- The occluded patch is non-trivial (contains at least one non-black cell and some
  visible context exists around it).
"""

from __future__ import annotations

import json
import random
from typing import Dict, List, Optional, Tuple

Grid = List[List[int]]
Cell = Tuple[int, int]
PALETTE = [0, 1, 3, 4, 5, 6, 7, 9]
NONZERO_PALETTE = [1, 3, 4, 5, 6, 7, 9]
CYAN = 8
RED = 2


def _empty(h: int, w: int, fill: int = 0) -> Grid:
    return [[fill for _ in range(w)] for _ in range(h)]


def _rot_ccw(m: Grid) -> Grid:
    return [list(row) for row in zip(*m)][::-1]


def _rot_cw(m: Grid) -> Grid:
    return [list(row) for row in zip(*m[::-1])]


def _rot_180(m: Grid) -> Grid:
    return [row[::-1] for row in m[::-1]]


def _copy_grid(g: Grid) -> Grid:
    return [row[:] for row in g]


def _find_square(grid: Grid) -> tuple[int, int, int, int, str]:
    pts = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v in (CYAN, RED)]
    if not pts:
        raise ValueError('No cyan/red occluder found')
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    if r1 - r0 != c1 - c0:
        raise ValueError('Occluder must be a square')
    size = r1 - r0 + 1
    if size < 2:
        raise ValueError('Occluder square too small')
    red_cells = [(r, c) for r, c in pts if grid[r][c] == RED]
    if len(red_cells) != 1:
        raise ValueError('Expected exactly one red direction cell')
    red = red_cells[0]
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if grid[r][c] not in (CYAN, RED):
                raise ValueError('Occluder square must contain only cyan/red cells')
    corner_map = {
        (r0, c0): 'tl',
        (r0, c1): 'tr',
        (r1, c0): 'bl',
        (r1, c1): 'br',
    }
    if red not in corner_map:
        raise ValueError('Red direction cell must be at a corner of the cyan square')
    return r0, r1, c0, c1, corner_map[red]


def _orient_hidden(hidden: Grid, corner: str) -> Grid:
    if corner == 'tl':
        return hidden
    if corner == 'tr':
        return _rot_ccw(hidden)
    if corner == 'bl':
        return _rot_cw(hidden)
    if corner == 'br':
        return _rot_180(hidden)
    raise ValueError(f'Unknown corner {corner}')


def _ring_candidate(grid: Grid) -> Optional[Grid]:
    h, w = len(grid), len(grid[0])
    r0, r1, c0, c1, _ = _find_square(grid)
    depth_color: Dict[int, int] = {}
    for r in range(h):
        for c in range(w):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            d = min(r, c, h - 1 - r, w - 1 - c)
            v = grid[r][c]
            if d in depth_color and depth_color[d] != v:
                return None
            depth_color[d] = v
    hidden: Grid = []
    for r in range(r0, r1 + 1):
        row: List[int] = []
        for c in range(c0, c1 + 1):
            d = min(r, c, h - 1 - r, w - 1 - c)
            if d not in depth_color:
                return None
            row.append(depth_color[d])
        hidden.append(row)
    return hidden


def _row_bridge_candidate(grid: Grid) -> Optional[Grid]:
    h, w = len(grid), len(grid[0])
    r0, r1, c0, c1, _ = _find_square(grid)
    if c0 == 0 or c1 == w - 1:
        return None
    hidden: Grid = []
    for r in range(r0, r1 + 1):
        lv = grid[r][c0 - 1]
        rv = grid[r][c1 + 1]
        if lv != rv:
            return None
        hidden.append([lv] * (c1 - c0 + 1))
    return hidden


def _diag_candidate(grid: Grid, step: int) -> Optional[Grid]:
    h, w = len(grid), len(grid[0])
    r0, r1, c0, c1, _ = _find_square(grid)
    groups: Dict[Tuple[int, int], int] = {}
    for r in range(h):
        for c in range(w):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            key = (r - c, r % step)
            v = grid[r][c]
            if key in groups and groups[key] != v:
                return None
            groups[key] = v
    hidden: Grid = []
    for r in range(r0, r1 + 1):
        row: List[int] = []
        for c in range(c0, c1 + 1):
            key = (r - c, r % step)
            if key not in groups:
                return None
            row.append(groups[key])
        hidden.append(row)
    return hidden


def _recover_hidden(grid: Grid) -> tuple[Grid, str]:
    # Preference order chosen to match the supplied task references.
    cand = _ring_candidate(grid)
    if cand is not None:
        return cand, 'ring'
    cand = _row_bridge_candidate(grid)
    if cand is not None:
        return cand, 'row'
    for step in (1, 2, 3, 4):
        cand = _diag_candidate(grid, step)
        if cand is not None:
            return cand, f'diag{step}'
    raise ValueError('Could not infer hidden pattern from supported families')


def transform_t34(grid: Grid) -> Grid:
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError('Input grid must be non-empty and rectangular')
    hidden, _family = _recover_hidden(grid)
    _r0, _r1, _c0, _c1, corner = _find_square(grid)
    return _orient_hidden(hidden, corner)


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def _overlay_occluder(full: Grid, r0: int, c0: int, size: int, corner: str) -> tuple[Grid, Grid]:
    hidden = [row[c0:c0 + size] for row in full[r0:r0 + size]]
    inp = _copy_grid(full)
    for r in range(r0, r0 + size):
        for c in range(c0, c0 + size):
            inp[r][c] = CYAN
    red_pos = {
        'tl': (r0, c0),
        'tr': (r0, c0 + size - 1),
        'bl': (r0 + size - 1, c0),
        'br': (r0 + size - 1, c0 + size - 1),
    }[corner]
    inp[red_pos[0]][red_pos[1]] = RED
    return inp, _orient_hidden(hidden, corner)


def _gen_diag(rng: random.Random) -> Optional[tuple[Grid, Grid, str]]:
    """Generate a clear diagonal stripe/band field, not random sparse noise."""
    h = rng.randint(10, 18)
    w = rng.randint(10, 18)
    size = rng.randint(4, min(6, h - 2, w - 2))
    r0 = rng.randint(1, h - size - 1)
    c0 = rng.randint(1, w - size - 1)
    # Step 1 is the clearest; step 2 appears occasionally to match the reference variety.
    step = rng.choice([1, 1, 1, 2])

    period = rng.choice([3, 4])
    active_width = rng.choice([1, 1, 2])
    stripe_colors = rng.sample(NONZERO_PALETTE, rng.randint(2, 4))
    phase = rng.randrange(period)
    full = _empty(h, w)
    for r in range(h):
        for c in range(w):
            d = r - c + phase
            if d % period < active_width:
                # Color changes only by diagonal band and the optional step lane,
                # producing a regular readable pattern rather than per-cell noise.
                full[r][c] = stripe_colors[((d // period) + (r % step)) % len(stripe_colors)]
            else:
                full[r][c] = 0

    hidden = [row[c0:c0 + size] for row in full[r0:r0 + size]]
    hidden_vals = [v for row in hidden for v in row]
    if all(v == 0 for v in hidden_vals) or len(set(hidden_vals)) < 2:
        return None
    # Ensure row-bridge candidate fails so the intended family is diagonal.
    if all(full[r][c0 - 1] == full[r][c0 + size] for r in range(r0, r0 + size)):
        return None
    corner = rng.choice(['tl', 'tr', 'bl', 'br'])
    inp, out = _overlay_occluder(full, r0, c0, size, corner)
    return inp, out, f'diag{step}'


def _gen_row(rng: random.Random) -> Optional[tuple[Grid, Grid, str]]:
    """Generate coherent horizontal color bands, not independent random rows."""
    size = rng.randint(3, 6)
    h = rng.randint(size + 4, size + 10)
    w = rng.randint(size + 6, size + 14)
    r0 = rng.randint(1, h - size - 1)
    c0 = rng.randint(1, w - size - 1)
    full = _empty(h, w)
    band_colors = rng.sample(NONZERO_PALETTE, rng.randint(2, 4))
    band_height = rng.choice([1, 2, 2, 3])
    include_black_gaps = rng.random() < 0.35
    phase = rng.randrange(len(band_colors))
    for r in range(h):
        band = (r // band_height + phase) % (len(band_colors) + (1 if include_black_gaps else 0))
        color = 0 if include_black_gaps and band == len(band_colors) else band_colors[band]
        for c in range(w):
            full[r][c] = color
    hidden = [row[c0:c0 + size] for row in full[r0:r0 + size]]
    vals = [hidden[i][0] for i in range(size)]
    if len(set(vals)) < 2 or all(v == 0 for v in vals):
        return None
    corner = rng.choice(['tl', 'tr', 'bl', 'br'])
    inp, out = _overlay_occluder(full, r0, c0, size, corner)
    return inp, out, 'row'


def _gen_ring(rng: random.Random) -> Optional[tuple[Grid, Grid, str]]:
    h = rng.randint(9, 17)
    w = rng.randint(9, 17)
    max_depth = min(h, w) // 2
    if max_depth < 3:
        return None
    ring_colors = [rng.choice(NONZERO_PALETTE)]
    while len(ring_colors) < max_depth + 1:
        prev = ring_colors[-1]
        choices = [c for c in NONZERO_PALETTE if c != prev]
        ring_colors.append(rng.choice(choices))
    full = _empty(h, w)
    for r in range(h):
        for c in range(w):
            d = min(r, c, h - 1 - r, w - 1 - c)
            full[r][c] = ring_colors[d]

    size = rng.randint(3, min(6, h - 2, w - 2))
    # Keep the square away from the border so there is visible context all around.
    r0 = rng.randint(1, h - size - 1)
    c0 = rng.randint(1, w - size - 1)
    hidden = [row[c0:c0 + size] for row in full[r0:r0 + size]]
    if len(set(v for row in hidden for v in row)) < 2:
        return None
    corner = rng.choice(['tl', 'tr', 'bl', 'br'])
    inp, out = _overlay_occluder(full, r0, c0, size, corner)
    return inp, out, 'ring'


def generate(rng: random.Random | None = None) -> Dict[str, Grid]:
    rng = rng or random.Random()
    families = ['diag', 'row', 'ring']
    for _ in range(5000):
        family = rng.choices(families, weights=[0.25, 0.30, 0.45], k=1)[0]
        made: Optional[tuple[Grid, Grid, str]]
        if family == 'diag':
            made = _gen_diag(rng)
        elif family == 'row':
            made = _gen_row(rng)
        else:
            made = _gen_ring(rng)
        if made is None:
            continue
        inp, out, intended = made
        try:
            recovered_hidden, recovered_family = _recover_hidden(inp)
            _r0, _r1, _c0, _c1, corner = _find_square(inp)
            predicted = _orient_hidden(recovered_hidden, corner)
        except Exception:
            continue
        if predicted != out:
            continue
        # Ensure the verifier selects the intended family type.
        if intended == 'row' and recovered_family != 'row':
            continue
        if intended == 'ring' and recovered_family != 'ring':
            continue
        if intended.startswith('diag') and not recovered_family.startswith('diag'):
            continue
        return {'input': inp, 'output': out}
    raise RuntimeError('Failed to generate valid t34 example')


if __name__ == '__main__':
    print(json.dumps(generate(rng=random.Random(0)), indent=2))
