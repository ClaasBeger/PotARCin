"""Generator for Test2/t37: row-wise shift with top-row recoloring markers.

Rule:
- The input contains a large single-color shape away from the top row and left
  column.
- Cells in the left column mark row shifts:
    * red (2) means shift that row's shape cells right by 1,
    * green (3) means shift that row's shape cells left by 1.
- Cells in the top row mark highlight colors for columns.
- For a row that is shifted, every shape cell in that row moves one step in the
  row's marked direction. Before moving, if the cell's column has a nonzero
  top-row marker, its color is changed to that top-row marker color.
- Shape cells on unmarked rows do not move and are not recolored.
- All markers in the top row and left column are removed in the output.

Generation constraints used to keep examples well-formed and unambiguous:
- the shape is a connected interior blob, usually symmetric and often left
  clean, with a smaller fraction using light symmetric edge erosion and a
  minority being irregular; it is kept at least one cell away from all borders
  so row-shifts stay in bounds;
- top-row highlight colors are chosen distinct from the shift-marker colors to
  reduce confusion;
- every top-row highlight marker affects at least one moving shape cell;
- there is at least one left-shift row and one right-shift row in most samples;
- output rows remain collision-free because each marked row shifts rigidly as a
  whole.
"""

from __future__ import annotations

import json
import random
from collections import deque
from typing import List, Sequence, Tuple

Grid = List[List[int]]
Mask = List[List[int]]

BG = 0
RED = 2
GREEN = 3
HIGHLIGHT_COLORS = [1, 4, 5, 6, 7, 8, 9]
SHAPE_COLORS = [1, 4, 5, 6, 7, 8, 9]


def _empty(h: int, w: int, fill: int = 0) -> Grid:
    return [[fill for _ in range(w)] for _ in range(h)]


def _rectangular(grid: Grid) -> bool:
    return bool(grid) and bool(grid[0]) and all(len(row) == len(grid[0]) for row in grid)


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _is_connected(mask: Mask) -> bool:
    h, w = len(mask), len(mask[0])
    cells = [(r, c) for r in range(h) for c in range(w) if mask[r][c]]
    if not cells:
        return False
    q = deque([cells[0]])
    seen = {cells[0]}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr][nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _shape_rows(mask: Mask) -> list[int]:
    return [r for r, row in enumerate(mask) if any(row)]


def _shape_cols(mask: Mask) -> list[int]:
    h, w = len(mask), len(mask[0])
    return [c for c in range(w) if any(mask[r][c] for r in range(h))]


def _postprocess_blob_mask(mask: Mask, rng: random.Random, allow_irregular: bool, apply_erosion: bool) -> Mask | None:
    h, w = len(mask), len(mask[0])
    if sum(v for row in mask for v in row) < 12:
        return None

    # Optional light random edge erosion while preserving connectivity. For
    # symmetric masks, erode in mirrored pairs so the overall symmetry is preserved.
    if apply_erosion:
        candidates = []
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if not mask[r][c]:
                    continue
                nbrs = sum(mask[nr][nc] for nr, nc in _neighbors4(r, c, h, w))
                if 1 <= nbrs <= 3:
                    candidates.append((r, c))
        rng.shuffle(candidates)
        remove_budget = rng.randint(1, min(6 if allow_irregular else 4, len(candidates))) if candidates else 0
        used = set()
        for i in range(remove_budget):
            if i >= len(candidates):
                break
            r, c = candidates[i]
            if (r, c) in used:
                continue
            to_remove = [(r, c)]
            if not allow_irregular:
                mr = h - 1 - r
                mc = w - 1 - c
                mirrors = {(r, c), (mr, c), (r, mc), (mr, mc)}
                to_remove = [(rr, cc) for rr, cc in mirrors if 0 <= rr < h and 0 <= cc < w and mask[rr][cc]]
            for rr, cc in to_remove:
                mask[rr][cc] = 0
            if not _is_connected(mask) or sum(v for row in mask for v in row) < 12:
                for rr, cc in to_remove:
                    mask[rr][cc] = 1
            else:
                used.update(to_remove)

    rows = _shape_rows(mask)
    cols = _shape_cols(mask)
    if len(rows) < 4 or len(cols) < 4:
        return None
    # Avoid empty rows/cols inside the occupied span becoming too fragmented.
    if max(rows) - min(rows) + 1 > len(rows) + 3:
        return None
    if max(cols) - min(cols) + 1 > len(cols) + 3:
        return None
    return mask


def _generate_symmetric_blob_mask(rng: random.Random, h: int, w: int, apply_erosion: bool) -> Mask | None:
    bh = rng.randint(max(4, h // 3), max(5, min(h, h - 1)))
    bw = rng.randint(max(4, w // 3), max(5, min(w, w - 1)))
    top = rng.randint(0, h - bh)
    left = rng.randint(0, w - bw)
    cy = top + (bh - 1) / 2.0
    cx = left + (bw - 1) / 2.0
    ay = max(1.8, bh / rng.uniform(2.0, 2.7))
    ax = max(1.8, bw / rng.uniform(2.0, 2.7))
    mode = rng.choice(['ellipse', 'diamond', 'super'])

    mask = _empty(h, w, 0)
    for r in range(top, top + bh):
        for c in range(left, left + bw):
            dy = (r - cy) / ay
            dx = (c - cx) / ax
            if mode == 'ellipse':
                inside = dx * dx + dy * dy <= 1.0
            elif mode == 'diamond':
                inside = abs(dx) + abs(dy) <= 1.05
            else:
                inside = abs(dx) ** 1.6 + abs(dy) ** 1.6 <= 1.2
            if inside:
                mask[r][c] = 1
    return _postprocess_blob_mask(mask, rng, allow_irregular=False, apply_erosion=apply_erosion)


def _generate_irregular_blob_mask(rng: random.Random, h: int, w: int) -> Mask | None:
    bh = rng.randint(max(4, h // 3), max(5, min(h, h - 1)))
    bw = rng.randint(max(4, w // 3), max(5, min(w, w - 1)))
    top = rng.randint(0, h - bh)
    left = rng.randint(0, w - bw)
    cy = top + (bh - 1) / 2.0
    cx = left + (bw - 1) / 2.0
    ay = max(1.8, bh / rng.uniform(2.1, 2.8))
    ax = max(1.8, bw / rng.uniform(2.1, 2.8))
    mode = rng.choice(['ellipse', 'diamond', 'super'])

    mask = _empty(h, w, 0)
    for r in range(top, top + bh):
        for c in range(left, left + bw):
            dy = (r - cy) / ay
            dx = (c - cx) / ax
            if mode == 'ellipse':
                inside = dx * dx + dy * dy <= 1.0
            elif mode == 'diamond':
                inside = abs(dx) + abs(dy) <= 1.05
            else:
                inside = abs(dx) ** 1.6 + abs(dy) ** 1.6 <= 1.2
            if inside:
                mask[r][c] = 1
    return _postprocess_blob_mask(mask, rng, allow_irregular=True, apply_erosion=True)


def _generate_blob_mask(rng: random.Random, h: int, w: int) -> Mask:
    """Generate a connected blob inside an h x w interior canvas.

    About 50% of samples are clean symmetric shapes, 25% are symmetric with
    light edge erosion, and 25% are more irregular.
    """
    for _ in range(4000):
        p = rng.random()
        if p < 0.50:
            mask = _generate_symmetric_blob_mask(rng, h, w, apply_erosion=False)
        elif p < 0.75:
            mask = _generate_symmetric_blob_mask(rng, h, w, apply_erosion=True)
        else:
            mask = _generate_irregular_blob_mask(rng, h, w)
        if mask is not None:
            return mask
    raise RuntimeError('failed to generate blob mask')


def transform_t37(grid: Grid) -> Grid:
    if not _rectangular(grid):
        raise ValueError('input grid must be non-empty and rectangular')
    h, w = len(grid), len(grid[0])
    if h < 3 or w < 3:
        raise ValueError('grid too small')

    top_markers = {c: grid[0][c] for c in range(1, w) if grid[0][c] != BG}
    row_dirs = {}
    for r in range(1, h):
        if grid[r][0] == RED:
            row_dirs[r] = 1
        elif grid[r][0] == GREEN:
            row_dirs[r] = -1
        elif grid[r][0] != BG:
            raise ValueError('left-column markers must be red, green, or blank')

    out = _empty(h, w, BG)
    for r in range(1, h):
        delta = row_dirs.get(r, 0)
        for c in range(1, w):
            v = grid[r][c]
            if v == BG:
                continue
            # Treat every nonzero cell outside marker row/col as part of the shape.
            new_c = c + delta
            if not (1 <= new_c < w):
                raise ValueError('shape moved out of bounds')
            color = top_markers[c] if delta != 0 and c in top_markers else v
            out[r][new_c] = color
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _render_input(mask: Mask, shape_color: int, row_markers: dict[int, int], col_markers: dict[int, int]) -> Grid:
    ih, iw = len(mask), len(mask[0])
    h, w = ih + 1, iw + 1
    grid = _empty(h, w, BG)
    for c, color in col_markers.items():
        grid[0][c] = color
    for r, color in row_markers.items():
        grid[r][0] = color
    for r in range(ih):
        for c in range(iw):
            if mask[r][c]:
                grid[r + 1][c + 1] = shape_color
    return grid


def generate(
    rng: random.Random | None = None,
    h: int | None = None,
    w: int | None = None,
) -> dict[str, Grid]:
    rng = rng or random.Random()
    for _ in range(5000):
        hh = h if h is not None else rng.randint(11, 20)
        ww = w if w is not None else rng.randint(11, 20)
        if not (8 <= hh <= 30 and 8 <= ww <= 30):
            continue

        interior_h, interior_w = hh - 1, ww - 1
        # Blob generated in a cropped interior canvas to keep one-cell margin after shifting.
        small_mask = _generate_blob_mask(rng, interior_h - 2, interior_w - 2)
        mask = _empty(interior_h, interior_w, 0)
        for r in range(interior_h - 2):
            for c in range(interior_w - 2):
                mask[r + 1][c + 1] = small_mask[r][c]

        rows = _shape_rows(mask)
        cols = _shape_cols(mask)
        if len(rows) < 4 or len(cols) < 4:
            continue

        shape_color = rng.choice(SHAPE_COLORS)
        available_highlights = [c for c in HIGHLIGHT_COLORS if c != shape_color]

        # Choose marked rows among occupied rows. Prefer both directions.
        num_marked_rows = rng.randint(2, min(6, len(rows)))
        marked_rows0 = sorted(rng.sample(rows, num_marked_rows))
        row_markers: dict[int, int] = {}
        if num_marked_rows >= 2:
            dirs = [RED, GREEN]
            for _ in range(num_marked_rows - 2):
                dirs.append(rng.choice([RED, GREEN]))
            rng.shuffle(dirs)
        else:
            dirs = [rng.choice([RED, GREEN])]
        for rr0, marker in zip(marked_rows0, dirs):
            row_markers[rr0 + 1] = marker

        # Columns eligible for top markers are occupied by at least one moving cell.
        moving_cols0 = sorted({c for r0 in marked_rows0 for c, v in enumerate(mask[r0]) if v})
        if len(moving_cols0) < 1:
            continue
        num_col_markers = rng.randint(1, min(5, len(moving_cols0), len(available_highlights)))
        chosen_cols0 = sorted(rng.sample(moving_cols0, num_col_markers))
        chosen_colors = rng.sample(available_highlights, num_col_markers)
        col_markers = {c0 + 1: col for c0, col in zip(chosen_cols0, chosen_colors)}

        # Make sure each top marker affects at least one moving cell.
        if not all(any(mask[r0][c0] for r0 in marked_rows0) for c0 in chosen_cols0):
            continue

        inp = _render_input(mask, shape_color, row_markers, col_markers)
        try:
            out = transform_t37(inp)
        except Exception:
            continue
        if out == inp:
            continue

        # Ensure at least one recolor occurs and both directions are often represented.
        recolor_count = sum(
            1
            for r0 in marked_rows0
            for c0 in chosen_cols0
            if mask[r0][c0]
        )
        if recolor_count == 0:
            continue
        if RED not in row_markers.values() or GREEN not in row_markers.values():
            if rng.random() < 0.8:
                continue

        return {'input': inp, 'output': out}
    raise RuntimeError('failed to generate a valid t37 sample')


if __name__ == '__main__':
    print(json.dumps(generate(rng=random.Random(0)), indent=2))
