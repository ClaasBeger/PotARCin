"""Verifier for Test2/t3 wrapped-rectangle task."""

from __future__ import annotations

from typing import List

Grid = List[List[int]]


def verify_t3(inp: Grid) -> Grid:
    if not inp or not inp[0]:
        raise ValueError("input grid must be non-empty")
    h = len(inp)
    w = len(inp[0])
    if any(len(row) != w for row in inp):
        raise ValueError("ragged grid")

    def cell(r: int, c: int) -> int:
        return inp[r % h][c % w]

    candidates: list[tuple[int, int, int, int, int]] = []
    for frame_color in range(1, 10):
        for frame_h in range(4, h + 1):
            for frame_w in range(4, w + 1):
                for top in range(h):
                    for left in range(w):
                        wraps_row = (top + frame_h - 1) >= h
                        wraps_col = (left + frame_w - 1) >= w
                        # Require exactly one wrapped axis.
                        if wraps_row == wraps_col:
                            continue
                        if wraps_row:
                            first_len = h - top
                            second_len = frame_h - first_len
                            if first_len < 2 or second_len < 2:
                                continue
                        if wraps_col:
                            first_len = w - left
                            second_len = frame_w - first_len
                            if first_len < 2 or second_len < 2:
                                continue
                        ok = True
                        for c in range(frame_w):
                            if cell(top, left + c) != frame_color:
                                ok = False
                                break
                            if cell(top + frame_h - 1, left + c) != frame_color:
                                ok = False
                                break
                        if not ok:
                            continue
                        for r in range(1, frame_h - 1):
                            if cell(top + r, left) != frame_color:
                                ok = False
                                break
                            if cell(top + r, left + frame_w - 1) != frame_color:
                                ok = False
                                break
                        if not ok:
                            continue
                        # Prefer larger frames; generator guarantees a unique best one.
                        area = frame_h * frame_w
                        candidates.append((area, top, left, frame_h, frame_w))

    if not candidates:
        raise ValueError("no wrapped outlined rectangle found")

    candidates.sort(reverse=True)
    _, top, left, frame_h, frame_w = candidates[0]
    content_h = frame_h - 2
    content_w = frame_w - 2
    out: Grid = []
    for r in range(content_h):
        out_row = []
        for c in range(content_w):
            out_row.append(cell(top + 1 + r, left + 1 + c))
        out.append(out_row)
    return out
