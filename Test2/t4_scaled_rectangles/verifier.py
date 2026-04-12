"""Verifier for Test2/t4 scaled-rectangle task."""

from __future__ import annotations

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
            picked = {min(remaining)}
        groups.append(picked)
        remaining -= picked
    return groups


def _color_groups_4n_rectangular(ref_cells: Grid) -> list[tuple[int, set[tuple[int, int]]]]:
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


def _find_reference_top_left(inp: Grid) -> tuple[int, int]:
    h = len(inp)
    w = len(inp[0]) if h else 0
    candidates = [(0, 0), (0, w - 5), (h - 5, 0), (h - 5, w - 5)]
    for top, left in candidates:
        if top < 0 or left < 0:
            continue
        if top + 5 > h or left + 5 > w:
            continue
        ok = True
        for c in range(left, left + 5):
            if inp[top][c] != RED or inp[top + 4][c] != RED:
                ok = False
                break
        if not ok:
            continue
        for r in range(top + 1, top + 4):
            if inp[r][left] != RED or inp[r][left + 4] != RED:
                ok = False
                break
        if ok:
            return top, left
    raise ValueError("could not find red 5x5 reference rectangle")


def _reference_corner(ref_top: int, ref_left: int, h: int, w: int) -> tuple[str, str]:
    vertical = "top" if ref_top == 0 else "bottom"
    horizontal = "left" if ref_left == 0 else "right"
    return vertical, horizontal


def verify_t4(inp: Grid) -> Grid:
    if not inp or not inp[0]:
        raise ValueError("input must be non-empty")
    h = len(inp)
    w = len(inp[0])
    if any(len(row) != w for row in inp):
        raise ValueError("ragged input grid")
    if h % 3 != 0 or w % 3 != 0:
        raise ValueError("input dimensions must be divisible by 3")
    if h not in (12, 18) or w not in (12, 18):
        raise ValueError("this task uses size 12 or 18")

    ref_top, ref_left = _find_reference_top_left(inp)
    ref_cells = _empty(3, 3)
    for r in range(3):
        for c in range(3):
            ref_cells[r][c] = inp[ref_top + 1 + r][ref_left + 1 + c]

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

        for cc in range(left, left + rect_w):
            out[top][cc] = color
            out[top + rect_h - 1][cc] = color
        for rr in range(top + 1, top + rect_h - 1):
            out[rr][left] = color
            out[rr][left + rect_w - 1] = color

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
