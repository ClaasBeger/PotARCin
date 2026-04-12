"""Generator for Test2/t4 scaled-rectangle task."""

from __future__ import annotations

import random
from typing import List

Grid = List[List[int]]

RED = 2


def _empty(h: int, w: int) -> Grid:
    return [[0 for _ in range(w)] for _ in range(h)]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _find_sub_rectangles(cells: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    """Greedily decompose into mergeable 4-neighbor rectangles of size 9/6/4/2."""
    remaining = set(cells)
    groups: list[set[tuple[int, int]]] = []

    def all_rectangles_of_size(target: int) -> list[set[tuple[int, int]]]:
        out: list[set[tuple[int, int]]] = []
        rows = [r for r, _ in remaining]
        cols = [c for _, c in remaining]
        if not rows or not cols:
            return out
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        for r0 in range(min_r, max_r + 1):
            for c0 in range(min_c, max_c + 1):
                max_h = max_r - r0 + 1
                max_w = max_c - c0 + 1
                for h in range(1, max_h + 1):
                    for w in range(1, max_w + 1):
                        if h * w != target:
                            continue
                        cand = {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}
                        if cand.issubset(remaining):
                            out.append(cand)
        out.sort(key=lambda s: (min(r for r, _ in s), min(c for _, c in s)))
        return out

    while remaining:
        picked: set[tuple[int, int]] | None = None
        for size in (9, 6, 4, 2):
            candidates = all_rectangles_of_size(size)
            if candidates:
                picked = candidates[0]
                break
        if picked is None:
            # Unmergeable cell becomes a singleton rectangle.
            r, c = min(remaining)
            picked = {(r, c)}
        groups.append(picked)
        remaining -= picked
    return groups


def _color_groups_4n_rectangular(ref_cells: Grid) -> list[tuple[int, set[tuple[int, int]]]]:
    """Group same-color descriptor cells via 4-neighborhood, then rectangular merge rules."""
    h = len(ref_cells)
    w = len(ref_cells[0]) if h else 0
    seen: set[tuple[int, int]] = set()
    groups: list[tuple[int, set[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            color = ref_cells[r][c]
            if color == 0 or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            comp: set[tuple[int, int]] = set()
            while stack:
                cr, cc = stack.pop()
                comp.add((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen:
                        if ref_cells[nr][nc] == color:
                            seen.add((nr, nc))
                            stack.append((nr, nc))
            for part in _find_sub_rectangles(comp):
                groups.append((color, part))
    return groups


def _reference_corner(ref_top: int, ref_left: int, h: int, w: int) -> tuple[str, str]:
    vertical = "top" if ref_top == 0 else "bottom"
    horizontal = "left" if ref_left == 0 else "right"
    return vertical, horizontal


def _render_output(
    ref_cells: Grid,
    h: int,
    w: int,
    ref_top: int,
    ref_left: int,
) -> Grid:
    if h % 3 != 0 or w % 3 != 0:
        raise ValueError("grid dimensions must be divisible by 3")
    bh = h // 3
    bw = w // 3
    out = _empty(h, w)
    ref_vertical, ref_horizontal = _reference_corner(ref_top, ref_left, h, w)

    for color, comp in _color_groups_4n_rectangular(ref_cells):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        comp_h = r1 - r0 + 1
        comp_w = c1 - c0 + 1

        top = r0 * bh
        left = c0 * bw
        rect_h = comp_h * bh
        rect_w = comp_w * bw

        # Outline in component color.
        for cc in range(left, left + rect_w):
            out[top][cc] = color
            out[top + rect_h - 1][cc] = color
        for rr in range(top + 1, top + rect_h - 1):
            out[rr][left] = color
            out[rr][left + rect_w - 1] = color

        # Interior is split into 4 regions; red occupies the quadrant of ref corner.
        inner_top = top + 1
        inner_left = left + 1
        inner_h = rect_h - 2
        inner_w = rect_w - 2
        half_h = inner_h // 2
        half_w = inner_w // 2
        row_start = inner_top if ref_vertical == "top" else inner_top + half_h
        row_end = inner_top + half_h if ref_vertical == "top" else inner_top + inner_h
        col_start = inner_left if ref_horizontal == "left" else inner_left + half_w
        col_end = inner_left + half_w if ref_horizontal == "left" else inner_left + inner_w
        for rr in range(row_start, row_end):
            for cc in range(col_start, col_end):
                out[rr][cc] = RED

    return out


def _sample_reference_cells(rng: random.Random) -> Grid:
    # 3x3 descriptor inside the red 5x5 reference rectangle.
    cells = _empty(3, 3)
    palette = [1, 3, 4, 5, 6, 7, 8, 9]
    active_count = rng.randint(2, 5)
    coords = [(r, c) for r in range(3) for c in range(3)]
    rng.shuffle(coords)
    for i in range(active_count):
        r, c = coords[i]
        cells[r][c] = rng.choice(palette)

    # Encourage valid mergeable groups (2/4/6/9 cells, 4-neighborhood only).
    if rng.random() < 0.55:
        starts = [(r, c) for r in range(3) for c in range(3)]
        rng.shuffle(starts)
        r, c = starts[0]
        color = rng.choice(palette)
        cells[r][c] = color
        orient = rng.choice(["h", "v", "sq", "h6", "v6", "sq9"])
        if orient == "h" and c + 1 < 3:
            cells[r][c + 1] = color
        elif orient == "v" and r + 1 < 3:
            cells[r + 1][c] = color
        elif orient == "sq" and r + 1 < 3 and c + 1 < 3:
            cells[r + 1][c] = color
            cells[r][c + 1] = color
            cells[r + 1][c + 1] = color
        elif orient == "h6":
            rr = rng.choice([0, 1])
            for cc in range(3):
                cells[rr][cc] = color
                cells[rr + 1][cc] = color
        elif orient == "v6":
            cc = rng.choice([0, 1])
            for rr in range(3):
                cells[rr][cc] = color
                cells[rr][cc + 1] = color
        elif orient == "sq9":
            for rr in range(3):
                for cc in range(3):
                    cells[rr][cc] = color
    return cells


def generate(
    width: int | None = None,
    height: int | None = None,
    ref_top: int | None = None,
    ref_left: int | None = None,
    ref_cells: Grid | None = None,
    rng: random.Random | None = None,
) -> dict[str, Grid]:
    """Generate one t4 input/output pair."""
    rr = rng if rng is not None else random
    if width is None:
        width = rr.choice([12, 18])
    if height is None:
        height = width
    if width % 3 != 0 or height % 3 != 0:
        raise ValueError("width and height must be divisible by 3")
    if width < 9 or height < 9:
        raise ValueError("width and height must be >= 9")

    if ref_top is None or ref_left is None:
        corner = rr.choice(["ul", "ur", "bl", "br"])
        if ref_top is None:
            ref_top = 0 if corner in ("ul", "ur") else height - 5
        if ref_left is None:
            ref_left = 0 if corner in ("ul", "bl") else width - 5
    if not (0 <= ref_top <= height - 5 and 0 <= ref_left <= width - 5):
        raise ValueError("reference 5x5 rectangle does not fit")
    if ref_top not in (0, height - 5) or ref_left not in (0, width - 5):
        raise ValueError("reference rectangle must sit in a grid corner")

    if ref_cells is None:
        ref_cells = _sample_reference_cells(rr)
    if len(ref_cells) != 3 or any(len(row) != 3 for row in ref_cells):
        raise ValueError("ref_cells must be 3x3")
    if not any(ref_cells[r][c] != 0 for r in range(3) for c in range(3)):
        raise ValueError("ref_cells must contain at least one non-zero cell")
    for r in range(3):
        for c in range(3):
            if ref_cells[r][c] == RED:
                raise ValueError("ref_cells cannot use red (2)")

    inp = _empty(height, width)

    # Draw red 5x5 outline.
    for c in range(ref_left, ref_left + 5):
        inp[ref_top][c] = RED
        inp[ref_top + 4][c] = RED
    for r in range(ref_top + 1, ref_top + 4):
        inp[r][ref_left] = RED
        inp[r][ref_left + 4] = RED

    # Place descriptor cells in the 3x3 interior.
    for r in range(3):
        for c in range(3):
            inp[ref_top + 1 + r][ref_left + 1 + c] = ref_cells[r][c]

    out = _render_output(ref_cells, height, width, ref_top, ref_left)
    return {"input": inp, "output": out}
