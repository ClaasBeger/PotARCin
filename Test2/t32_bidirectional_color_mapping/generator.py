"""Generator for t32: bidirectional color mapping across a horizontal separator.

Rule:
- The grid is divided into a top and bottom section by a full grey row.
- Above the grey row, each row encodes a color correspondence: the uniform color on
  the left half maps to the uniform color on the right half. The mapping is two-way.
- Below the grey row are colored blocks placed entirely on either the left half or
  the right half.
- In the output, every bottom block is preserved in place and also recreated on the
  opposite half using the mapped color while keeping the same shape and row positions.

Generator constraints / ambiguity controls:
- The grid width is odd, with equal-sized left/right halves and a black center column.
- The top mapping is a bijection: each left-side mapping color is unique and each
  right-side mapping color is unique.
- The mapping rows all have the same thickness within a given example, making the
  upper section read more clearly as a code/table rather than an object layout.
- Bottom blocks are solid rectangles that span the full width of one half, closely
  matching the reference distribution.
- Every generated example contains at least one block on each side so both mapping
  directions are demonstrated.
- Block colors on the two sides are chosen from mostly disjoint mapping pairs when
  possible, reducing outputs with unnecessarily large same-colored regions. Within
  each side, colors are distributed to increase diversity and reduce repetition.
- No copied block overlaps any original or copied block on the opposite side.
"""

from __future__ import annotations

import json
import random
from typing import Dict, List, Tuple

Grid = List[List[int]]
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GREY = 5
BLACK = 0


def _empty(h: int, w: int, fill: int = 0) -> Grid:
    return [[fill for _ in range(w)] for _ in range(h)]


def _rectangular(grid: Grid) -> bool:
    return bool(grid) and bool(grid[0]) and all(len(row) == len(grid[0]) for row in grid)


def _find_separator_row(grid: Grid) -> int:
    rows = [r for r, row in enumerate(grid) if set(row) == {GREY}]
    if len(rows) != 1:
        raise ValueError('Expected exactly one full grey separator row')
    return rows[0]


def _uniform_nonzero(vals: List[int]) -> int:
    s = set(vals)
    if len(s) != 1:
        raise ValueError('Expected a uniform color block')
    v = vals[0]
    if v in (BLACK, GREY):
        raise ValueError('Expected a non-black, non-grey color')
    return v


def _extract_mapping(grid: Grid) -> Tuple[int, int, Dict[int, int], Dict[int, int]]:
    if not _rectangular(grid):
        raise ValueError('Input grid must be non-empty and rectangular')
    h, w = len(grid), len(grid[0])
    if w % 2 != 1:
        raise ValueError('Grid width must be odd')
    mid = w // 2
    sep = _find_separator_row(grid)
    if sep < 1 or sep >= h - 1:
        raise ValueError('Separator must split non-empty top and bottom sections')

    l2r: Dict[int, int] = {}
    r2l: Dict[int, int] = {}
    for r in range(sep):
        if grid[r][mid] != BLACK:
            raise ValueError('Center divider above separator must be black')
        left = _uniform_nonzero(grid[r][:mid])
        right = _uniform_nonzero(grid[r][mid + 1:])
        if left in l2r and l2r[left] != right:
            raise ValueError('Inconsistent left-to-right mapping')
        if right in r2l and r2l[right] != left:
            raise ValueError('Inconsistent right-to-left mapping')
        l2r[left] = right
        r2l[right] = left
    if len(l2r) != len(r2l):
        raise ValueError('Mapping must be bijective')
    return sep, mid, l2r, r2l


def transform_t32(grid: Grid) -> Grid:
    sep, mid, l2r, r2l = _extract_mapping(grid)
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    for r in range(sep + 1, h):
        if grid[r][mid] != BLACK:
            raise ValueError('Center divider below separator must be black')
        # left -> right copies
        for c in range(mid):
            color = grid[r][c]
            if color in (BLACK, GREY):
                continue
            if color not in l2r:
                raise ValueError(f'Bottom left color {color} not in mapping')
            tc = c + mid + 1
            new_color = l2r[color]
            existing = out[r][tc]
            if existing not in (BLACK, new_color):
                raise ValueError('Copied block would collide on the right side')
            out[r][tc] = new_color
        # right -> left copies
        for c in range(mid + 1, w):
            color = grid[r][c]
            if color in (BLACK, GREY):
                continue
            if color not in r2l:
                raise ValueError(f'Bottom right color {color} not in mapping')
            tc = c - (mid + 1)
            new_color = r2l[color]
            existing = out[r][tc]
            if existing not in (BLACK, new_color):
                raise ValueError('Copied block would collide on the left side')
            out[r][tc] = new_color
    return out


def _place_block(grid: Grid, start_r: int, height: int, side: str, color: int, mid: int) -> None:
    cols = range(0, mid) if side == 'L' else range(mid + 1, len(grid[0]))
    for r in range(start_r, start_r + height):
        for c in cols:
            grid[r][c] = color


def _diverse_color_sequence(n: int, pool: List[int], rng: random.Random) -> List[int]:
    """Return n colors from pool with good diversity and limited repetition.

    Tries to cover as many distinct colors as possible on a side before reusing,
    and avoids immediate repeats when the pool has size > 1.
    """
    if n <= 0:
        return []
    if not pool:
        raise ValueError('color pool must be non-empty')
    seq: List[int] = []
    counts = {c: 0 for c in pool}
    remaining_unseen = set(pool)
    for _ in range(n):
        # Prefer colors not used yet on this side; otherwise prefer currently least-used.
        if remaining_unseen:
            candidates = list(remaining_unseen)
        else:
            min_count = min(counts.values())
            candidates = [c for c in pool if counts[c] == min_count]
        # Avoid immediate repetition when possible.
        if seq and len(candidates) > 1 and seq[-1] in candidates:
            candidates = [c for c in candidates if c != seq[-1]]
        color = rng.choice(candidates)
        seq.append(color)
        counts[color] += 1
        remaining_unseen.discard(color)
    return seq


def generate(rng: random.Random | None = None) -> Dict[str, Grid]:
    rng = rng or random.Random()
    for _ in range(5000):
        half_w = rng.randint(2, 4)
        w = 2 * half_w + 1
        # 4-7 mapping pairs. All mapping rows use the same height within an example.
        n_pairs = rng.randint(4, min(7, len(COLORS)))
        left_colors = rng.sample(COLORS, n_pairs)
        remaining = [c for c in COLORS if c not in left_colors]
        # choose right colors distinct from left side when possible; if not enough,
        # allow overlap across sides but keep uniqueness within each side.
        if len(remaining) >= n_pairs:
            right_colors = rng.sample(remaining, n_pairs)
        else:
            right_colors = rng.sample(COLORS, n_pairs)
        map_row_height = rng.choice([1, 2])
        pair_heights = [map_row_height for _ in range(n_pairs)]
        top_h = sum(pair_heights)

        # bottom blocks: 4-9 rectangles, each full half-width and height 1-3.
        # Require at least one left block and one right block.
        n_blocks = rng.randint(4, 9)
        sides = ['L', 'R'] + [rng.choice(['L', 'R']) for _ in range(n_blocks - 2)]
        rng.shuffle(sides)
        block_heights = [rng.choice([1, 1, 2, 2, 3]) for _ in range(n_blocks)]
        gap_after = [rng.choice([0, 1]) for _ in range(n_blocks - 1)]
        bottom_h = sum(block_heights) + sum(gap_after)

        h = top_h + 1 + bottom_h
        if not (10 <= h <= 30):
            continue
        grid = _empty(h, w, BLACK)
        mid = w // 2

        # top mapping rows
        l2r: Dict[int, int] = {}
        r2l: Dict[int, int] = {}
        rr = 0
        for lc, rc, ph in zip(left_colors, right_colors, pair_heights):
            if lc in l2r or rc in r2l:
                # enforce uniqueness within each side; regenerate on rare failure
                break
            l2r[lc] = rc
            r2l[rc] = lc
            for _ in range(ph):
                for c in range(0, mid):
                    grid[rr][c] = lc
                for c in range(mid + 1, w):
                    grid[rr][c] = rc
                rr += 1
        else:
            sep = rr
            for c in range(w):
                grid[sep][c] = GREY

            # Bias block-color choices to use disjoint mapping-pair subsets on the two
            # sides when possible, reducing outputs with large same-colored regions.
            pair_indices = list(range(n_pairs))
            rng.shuffle(pair_indices)
            # Keep both side-specific pools reasonably large when possible.
            if n_pairs >= 4:
                split = rng.randint(2, n_pairs - 2)
            else:
                split = rng.randint(1, n_pairs - 1)
            left_pool_idx = pair_indices[:split]
            right_pool_idx = pair_indices[split:]
            left_pool = [left_colors[i] for i in left_pool_idx]
            right_pool = [right_colors[i] for i in right_pool_idx]

            # Pre-assign colors with stronger within-side diversity.
            left_block_count = sum(1 for s in sides if s == 'L')
            right_block_count = sum(1 for s in sides if s == 'R')
            left_seq = _diverse_color_sequence(left_block_count, left_pool if left_pool else list(l2r.keys()), rng)
            right_seq = _diverse_color_sequence(right_block_count, right_pool if right_pool else list(r2l.keys()), rng)
            li = ri = 0

            # place bottom blocks while avoiding row overlap across sides, so copies do not collide.
            curr = sep + 1
            ok = True
            for i, (side, bh) in enumerate(zip(sides, block_heights)):
                if side == 'L':
                    color = left_seq[li]
                    li += 1
                else:
                    color = right_seq[ri]
                    ri += 1
                _place_block(grid, curr, bh, side, color, mid)
                curr += bh
                if i < len(gap_after):
                    curr += gap_after[i]
            # basic check that both sides are present
            has_left = any(any(v not in (BLACK, GREY) for v in row[:mid]) for row in grid[sep + 1:])
            has_right = any(any(v not in (BLACK, GREY) for v in row[mid + 1:]) for row in grid[sep + 1:])
            if not (has_left and has_right):
                ok = False
            if not ok:
                continue
            try:
                out = transform_t32(grid)
            except Exception:
                continue
            if out == grid:
                continue
            return {'input': grid, 'output': out}
    raise RuntimeError('Failed to generate valid t32 example')


if __name__ == '__main__':
    print(json.dumps(generate(rng=random.Random(0)), indent=2))
