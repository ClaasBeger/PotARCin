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

    def map_local_to_global(
        *,
        r: int,
        c: int,
        top: int,
        left: int,
        frame_h: int,
        frame_w: int,
        shift: int,
        wraps_row: bool,
    ) -> tuple[int, int]:
        if wraps_row:
            base_r = top + r
            if base_r < h:
                return base_r, left + c
            return base_r - h, left + c + shift
        base_c = left + c
        if base_c < w:
            return top + r, base_c
        return top + r + shift, base_c - w

    def is_contained_shift(
        *,
        top: int,
        left: int,
        frame_h: int,
        frame_w: int,
        shift: int,
        wraps_row: bool,
    ) -> bool:
        if wraps_row:
            return 0 <= left + shift and left + shift + frame_w <= w
        return 0 <= top + shift and top + shift + frame_h <= h

    candidates: list[tuple[int, int, int, int, int, int, int, int]] = []
    for frame_h in range(4, h + 1):
        for frame_w in range(4, w + 1):
            for top in range(h):
                for left in range(w):
                    wraps_row = (top + frame_h - 1) >= h
                    wraps_col = (left + frame_w - 1) >= w
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
                    if wraps_row:
                        min_shift = -left
                        max_shift = w - frame_w - left
                    else:
                        min_shift = -top
                        max_shift = h - frame_h - top
                    for shift in range(min_shift, max_shift + 1):
                        if not is_contained_shift(
                            top=top,
                            left=left,
                            frame_h=frame_h,
                            frame_w=frame_w,
                            shift=shift,
                            wraps_row=wraps_row,
                        ):
                            continue
                        border_a: list[int] = []
                        border_b: list[int] = []
                        all_coords: set[tuple[int, int]] = set()
                        ok = True
                        for r in range(frame_h):
                            for c in range(frame_w):
                                gr, gc = map_local_to_global(
                                    r=r,
                                    c=c,
                                    top=top,
                                    left=left,
                                    frame_h=frame_h,
                                    frame_w=frame_w,
                                    shift=shift,
                                    wraps_row=wraps_row,
                                )
                                all_coords.add((gr, gc))
                                if not (r == 0 or c == 0 or r == frame_h - 1 or c == frame_w - 1):
                                    continue
                                v = inp[gr][gc]
                                if v == 0:
                                    ok = False
                                    break
                                in_first_half = (top + r) < h if wraps_row else (left + c) < w
                                if in_first_half:
                                    border_a.append(v)
                                else:
                                    border_b.append(v)
                            if not ok:
                                break
                        if not ok or not border_a or not border_b:
                            continue
                        # Avoid overlapping mapped frames: ambiguous extraction.
                        if len(all_coords) != frame_h * frame_w:
                            continue
                        color_a = border_a[0]
                        color_b = border_b[0]
                        if any(v != color_a for v in border_a):
                            continue
                        if any(v != color_b for v in border_b):
                            continue
                        area = frame_h * frame_w
                        bi = 1 if color_a != color_b else 0
                        candidates.append((area, bi, top, left, frame_h, frame_w, shift, int(wraps_row)))

    if not candidates:
        raise ValueError("no wrapped outlined rectangle found")

    candidates.sort(reverse=True)
    _, _bi, top, left, frame_h, frame_w, shift, wraps_row_flag = candidates[0]
    wraps_row = bool(wraps_row_flag)
    content_h = frame_h - 2
    content_w = frame_w - 2
    out: Grid = []
    for r in range(content_h):
        out_row = []
        for c in range(content_w):
            gr, gc = map_local_to_global(
                r=1 + r,
                c=1 + c,
                top=top,
                left=left,
                frame_h=frame_h,
                frame_w=frame_w,
                shift=shift,
                wraps_row=wraps_row,
            )
            out_row.append(inp[gr][gc])
        out.append(out_row)
    return out
