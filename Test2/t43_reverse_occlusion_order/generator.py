"""Generator for Test2/t43: reverse the front-to-back occlusion order of objects.

Rule:
- The input shows several recognizable objects stacked/occluding one another.
- Their front-to-back order is uniquely identifiable from the visible overlaps.
- Reconstruct each full object, then reverse the depth order: the previous
  backmost object becomes frontmost, and the previous frontmost object becomes
  backmost.

Generation constraints used to keep examples well-formed and unambiguous:
- objects come from recognizable families: filled rectangles, rectangle outlines,
  straight lines, filled diamonds, plus/cross shapes, and exact masks;
- hidden parts of an object occur only beneath another non-background object;
- the overlap graph induces a unique front-to-back order;
- there is at least one true overlap/occlusion;
- generated samples reject side-touching-only contacts that can look like mere adjacency rather than occlusion.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict, deque
from typing import List, Sequence, Set, Tuple

Grid = List[List[int]]
Cell = Tuple[int, int]
Mask = Set[Cell]

BG = 0
COLORS = list(range(1, 10))
NEI4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _empty(h: int, w: int) -> Grid:
    return [[BG for _ in range(w)] for _ in range(h)]


def _rectangular(grid: Grid) -> bool:
    return bool(grid) and bool(grid[0]) and all(len(row) == len(grid[0]) for row in grid)


def _visible_cells_by_color(grid: Grid) -> dict[int, Mask]:
    out: dict[int, Mask] = defaultdict(set)
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v != BG:
                out[v].add((r, c))
    return dict(out)


def _bbox(cells: Mask) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def _bbox_cells(min_r: int, max_r: int, min_c: int, max_c: int) -> Mask:
    return {(r, c) for r in range(min_r, max_r + 1) for c in range(min_c, max_c + 1)}


def _rect_outline_cells(min_r: int, max_r: int, min_c: int, max_c: int) -> Mask:
    cells: Mask = set()
    for c in range(min_c, max_c + 1):
        cells.add((min_r, c))
        cells.add((max_r, c))
    for r in range(min_r, max_r + 1):
        cells.add((r, min_c))
        cells.add((r, max_c))
    return cells


def _diamond_cells(min_r: int, max_r: int, min_c: int, max_c: int) -> Mask:
    if (max_r - min_r) % 2 != 0 or (max_c - min_c) % 2 != 0:
        return set()
    cr = (min_r + max_r) // 2
    cc = (min_c + max_c) // 2
    radius_r = max_r - cr
    radius_c = max_c - cc
    if radius_r != radius_c:
        return set()
    rad = radius_r
    return {
        (r, c)
        for r in range(min_r, max_r + 1)
        for c in range(min_c, max_c + 1)
        if abs(r - cr) + abs(c - cc) <= rad
    }


def _plus_cells(min_r: int, max_r: int, min_c: int, max_c: int) -> Mask:
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    if h < 3 or w < 3:
        return set()
    center_rows = {min_r + h // 2} if h % 2 else {min_r + h // 2 - 1, min_r + h // 2}
    center_cols = {min_c + w // 2} if w % 2 else {min_c + w // 2 - 1, min_c + w // 2}
    return {
        (r, c)
        for r in range(min_r, max_r + 1)
        for c in range(min_c, max_c + 1)
        if r in center_rows or c in center_cols
    }


def _right_triangle_outline_cells(min_r: int, max_r: int, min_c: int, max_c: int, orientation: str) -> Mask:
    """Return a square right-triangle outline.

    Orientations identify the right-angle corner: tl, tr, bl, br.
    """
    height = max_r - min_r
    width = max_c - min_c
    if height != width or height < 2:
        return set()
    cells: Mask = set()
    if orientation == "tl":
        for c in range(min_c, max_c + 1):
            cells.add((min_r, c))
        for r in range(min_r, max_r + 1):
            cells.add((r, min_c))
            cells.add((r, max_c - (r - min_r)))
    elif orientation == "tr":
        for c in range(min_c, max_c + 1):
            cells.add((min_r, c))
        for r in range(min_r, max_r + 1):
            cells.add((r, max_c))
            cells.add((r, min_c + (r - min_r)))
    elif orientation == "bl":
        for c in range(min_c, max_c + 1):
            cells.add((max_r, c))
        for r in range(min_r, max_r + 1):
            cells.add((r, min_c))
            cells.add((r, min_c + (max_r - r)))
    elif orientation == "br":
        for c in range(min_c, max_c + 1):
            cells.add((max_r, c))
        for r in range(min_r, max_r + 1):
            cells.add((r, max_c))
            cells.add((r, max_c - (max_r - r)))
    else:
        raise ValueError(orientation)
    return cells


def _extras_are_occluded(mask: Mask, visible: Mask, nonzero: Mask) -> bool:
    return visible <= mask and (mask - visible) <= (nonzero - visible)


def _touches_all_bbox_sides(visible: Mask, min_r: int, max_r: int, min_c: int, max_c: int) -> bool:
    return (
        any(r == min_r for r, _ in visible)
        and any(r == max_r for r, _ in visible)
        and any(c == min_c for _, c in visible)
        and any(c == max_c for _, c in visible)
    )


def _infer_mask_for_color(visible: Mask, nonzero: Mask) -> Mask:
    min_r, max_r, min_c, max_c = _bbox(visible)
    candidates: list[tuple[int, int, str, Mask]] = []

    def add(priority: int, name: str, mask: Mask) -> None:
        if mask and _extras_are_occluded(mask, visible, nonzero):
            candidates.append((priority, len(mask - visible), name, mask))

    if min_r == max_r or min_c == max_c:
        add(0, "line", _bbox_cells(min_r, max_r, min_c, max_c))

    rect = _bbox_cells(min_r, max_r, min_c, max_c)
    if _touches_all_bbox_sides(visible, min_r, max_r, min_c, max_c):
        add(1, "filled_rect", rect)

    outline = _rect_outline_cells(min_r, max_r, min_c, max_c)
    if visible <= outline and _touches_all_bbox_sides(visible, min_r, max_r, min_c, max_c):
        add(1, "rect_outline", outline)

    # Filled diamond. Try the visible bounding box and small one-cell expansions,
    # since an extremal diamond cell can be hidden by a front object.
    for er0 in range(max(0, min_r - 1), min_r + 1):
        for er1 in range(max_r, max_r + 2):
            for ec0 in range(max(0, min_c - 1), min_c + 1):
                for ec1 in range(max_c, max_c + 2):
                    diamond = _diamond_cells(er0, er1, ec0, ec1)
                    if diamond and visible <= diamond:
                        add(1, "diamond", diamond)

    plus = _plus_cells(min_r, max_r, min_c, max_c)
    if plus and visible <= plus and _touches_all_bbox_sides(visible, min_r, max_r, min_c, max_c):
        add(1, "plus", plus)

    # Right-triangle outlines. These cover examples where an object is a
    # recognizable triangular outline, with one leg and the diagonal partly
    # hidden by objects in front.
    for er0 in range(max(0, min_r - 1), min_r + 1):
        for er1 in range(max_r, max_r + 2):
            for ec0 in range(max(0, min_c - 1), min_c + 1):
                for ec1 in range(max_c, max_c + 2):
                    for orient in ("tl", "tr", "bl", "br"):
                        tri = _right_triangle_outline_cells(er0, er1, ec0, ec1, orient)
                        if tri and visible <= tri:
                            add(1, "right_triangle", tri)

    add(9, "exact", set(visible))

    if not candidates:
        raise ValueError("could not infer a recognizable object mask")
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    return set(candidates[0][3])


def _infer_masks(grid: Grid) -> dict[int, Mask]:
    by_color = _visible_cells_by_color(grid)
    nonzero = set().union(*by_color.values()) if by_color else set()
    return {color: _infer_mask_for_color(cells, nonzero) for color, cells in by_color.items()}


def _front_order(grid: Grid, masks: dict[int, Mask]) -> list[int]:
    visible = _visible_cells_by_color(grid)
    colors = sorted(masks)
    edges: dict[int, set[int]] = {c: set() for c in colors}
    indeg = {c: 0 for c in colors}

    for front_color, cells in visible.items():
        for cell in cells:
            for back_color, mask in masks.items():
                if back_color != front_color and cell in mask:
                    if back_color not in edges[front_color]:
                        edges[front_color].add(back_color)
                        indeg[back_color] += 1

    order: list[int] = []
    ready = sorted([c for c in colors if indeg[c] == 0])
    while ready:
        c = ready.pop(0)
        order.append(c)
        for nb in sorted(edges[c]):
            indeg[nb] -= 1
            if indeg[nb] == 0:
                ready.append(nb)
                ready.sort()

    if len(order) != len(colors):
        raise ValueError("occlusion order contains a cycle")
    return order




def _overlap_size(mask_a: Mask, mask_b: Mask) -> int:
    return len(mask_a & mask_b)


def _touching_nonoverlap(mask_a: Mask, mask_b: Mask) -> bool:
    """Return True if two masks side-touch without overlapping.

    Such contacts are visually ambiguous in this task because they can look like
    either simple adjacency or an occluded edge.
    """
    if mask_a & mask_b:
        return False
    for r, c in mask_a:
        for dr, dc in NEI4:
            if (r + dr, c + dc) in mask_b:
                return True
    return False


def _adjacent_visible_color_pairs(grid: Grid) -> set[tuple[int, int]]:
    h, w = len(grid), len(grid[0])
    pairs: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            a = grid[r][c]
            if a == BG:
                continue
            for dr, dc in ((1, 0), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    b = grid[nr][nc]
                    if b != BG and b != a:
                        pairs.add(tuple(sorted((a, b))))
    return pairs


def _has_ambiguous_contacts(inp: Grid, masks: dict[int, Mask]) -> bool:
    """Heuristic filter for ambiguity.

    Reject cases where two different objects are visibly side-adjacent but their
    full masks never overlap. Those are the examples most likely to read as
    mere touching rather than one object covering another. Also reject very
    weak overlaps (single-cell intersections) when the pair is visibly adjacent.
    """
    for a, b in _adjacent_visible_color_pairs(inp):
        ma, mb = masks[a], masks[b]
        ov = _overlap_size(ma, mb)
        if ov <= 1:
            return True
    for colors in [(a, b) for a in masks for b in masks if a < b]:
        a, b = colors
        if _touching_nonoverlap(masks[a], masks[b]):
            return True
    return False

def _render(h: int, w: int, masks: dict[int, Mask], paint_order: Sequence[int]) -> Grid:
    grid = _empty(h, w)
    for color in paint_order:
        for r, c in masks[color]:
            if 0 <= r < h and 0 <= c < w:
                grid[r][c] = color
    return grid


def transform_t43(grid: Grid) -> Grid:
    if not _rectangular(grid):
        raise ValueError("input grid must be non-empty and rectangular")
    h, w = len(grid), len(grid[0])
    masks = _infer_masks(grid)
    # `order` is front-to-back in the input. Painting in that order reverses
    # the stack because later-painted objects become frontmost.
    order = _front_order(grid, masks)
    return _render(h, w, masks, order)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _make_filled_rect(r0: int, c0: int, h: int, w: int) -> Mask:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _make_rect_outline(r0: int, c0: int, h: int, w: int) -> Mask:
    return _rect_outline_cells(r0, r0 + h - 1, c0, c0 + w - 1)


def _make_line_horizontal(r: int, c0: int, length: int) -> Mask:
    return {(r, c) for c in range(c0, c0 + length)}


def _make_line_vertical(c: int, r0: int, length: int) -> Mask:
    return {(r, c) for r in range(r0, r0 + length)}


def _make_diamond(cr: int, cc: int, radius: int) -> Mask:
    return {
        (r, c)
        for r in range(cr - radius, cr + radius + 1)
        for c in range(cc - radius, cc + radius + 1)
        if abs(r - cr) + abs(c - cc) <= radius
    }


def _random_mask(rng: random.Random, h: int, w: int, kind: str) -> Mask:
    if kind == "filled_rect":
        rh = rng.randint(3, max(3, min(7, h - 2)))
        rw = rng.randint(3, max(3, min(8, w - 2)))
        return _make_filled_rect(rng.randint(1, h - rh - 1), rng.randint(1, w - rw - 1), rh, rw)
    if kind == "rect_outline":
        rh = rng.randint(5, max(5, min(10, h - 2)))
        rw = rng.randint(5, max(5, min(10, w - 2)))
        return _make_rect_outline(rng.randint(1, h - rh - 1), rng.randint(1, w - rw - 1), rh, rw)
    if kind == "hline":
        length = rng.randint(6, max(6, w - 2))
        return _make_line_horizontal(rng.randint(1, h - 2), rng.randint(0, w - length), length)
    if kind == "vline":
        length = rng.randint(6, max(6, h - 2))
        return _make_line_vertical(rng.randint(1, w - 2), rng.randint(0, h - length), length)
    if kind == "diamond":
        rad = rng.randint(2, max(2, min(4, (h - 3) // 2, (w - 3) // 2)))
        return _make_diamond(rng.randint(rad + 1, h - rad - 2), rng.randint(rad + 1, w - rad - 2), rad)
    raise ValueError(kind)


def generate(
    rng: random.Random | None = None,
    h: int | None = None,
    w: int | None = None,
    num_objects: int | None = None,
) -> dict[str, Grid]:
    rng = rng or random.Random()
    for _attempt in range(8000):
        hh = h if h is not None else rng.randint(12, 22)
        ww = w if w is not None else rng.randint(12, 22)
        n = num_objects if num_objects is not None else rng.randint(2, 4)
        if n < 2:
            continue
        colors = rng.sample(COLORS, n)

        kind_pool = ["filled_rect", "rect_outline", "hline", "vline", "diamond"]
        kinds = [rng.choice(kind_pool) for _ in range(n)]
        masks: dict[int, Mask] = {}
        ok = True
        for color, kind in zip(colors, kinds):
            try:
                masks[color] = _random_mask(rng, hh, ww, kind)
            except Exception:
                ok = False
                break
        if not ok:
            continue

        back_to_front = colors[:]
        rng.shuffle(back_to_front)

        # Require overlaps and visibility for all objects.
        overlap_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                if masks[colors[i]] & masks[colors[j]]:
                    overlap_pairs += 1
        if overlap_pairs < n - 1:
            continue

        inp = _render(hh, ww, masks, back_to_front)
        if _has_ambiguous_contacts(inp, masks):
            continue
        visible = _visible_cells_by_color(inp)
        if set(visible) != set(colors):
            continue

        try:
            inferred_masks = _infer_masks(inp)
            if inferred_masks != masks:
                continue
            out = transform_t43(inp)
        except Exception:
            continue

        expected = _render(hh, ww, masks, list(reversed(back_to_front)))
        if out != expected:
            continue
        if out == inp:
            continue
        return {"input": inp, "output": out}

    raise RuntimeError("failed to generate a valid t43 sample")


if __name__ == "__main__":
    print(json.dumps(generate(rng=random.Random(0)), indent=2))
