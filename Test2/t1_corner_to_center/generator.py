"""Corner-components to center-components generator for Test2/t1.

The rule matches ``Test2/t1.json``:
- up to four components, one per corner window
- each corner component is translated to the corresponding center window
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

Grid = List[List[int]]
PixelList = List[Tuple[int, int]]

_CORNERS = ("tl", "tr", "bl", "br")


def _empty(size: int) -> Grid:
    return [[0 for _ in range(size)] for _ in range(size)]


def _corner_span(size: int) -> int:
    # Matches examples: size=10 -> 3, size=12 -> 4.
    return size // 2 - 2


def _origins(size: int) -> Dict[str, Tuple[int, int]]:
    span = _corner_span(size)
    return {
        "tl": (0, 0),
        "tr": (0, size - span),
        "bl": (size - span, 0),
        "br": (size - span, size - span),
    }


def _random_component(rng: random.Random, span: int) -> PixelList:
    # 8-connected component without isolated cells.
    n = rng.randint(2, min(span * span, 8))
    all_cells = {(r, c) for r in range(span) for c in range(span)}
    start = rng.choice(list(all_cells))
    comp = {start}
    while len(comp) < n:
        frontier = set()
        for r, c in comp:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < span and 0 <= cc < span and (rr, cc) not in comp:
                        frontier.add((rr, cc))
        if not frontier:
            break
        comp.add(rng.choice(list(frontier)))
    return sorted(comp)


def _is_8_connected_and_non_isolated(pixels: PixelList) -> bool:
    if len(pixels) < 2:
        return False
    pix = set(pixels)
    seen = set()
    stack = [pixels[0]]
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        has_neighbor = False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                q = (r + dr, c + dc)
                if q in pix:
                    has_neighbor = True
                    if q not in seen:
                        stack.append(q)
        if not has_neighbor:
            return False
    return len(seen) == len(pix)


def generate(
    size: int | None = None,
    present: Sequence[str] | None = None,
    components: Dict[str, PixelList] | None = None,
    colors: Dict[str, int] | None = None,
    rng: random.Random | None = None,
) -> dict[str, Grid]:
    """Returns one ARC-style {"input": ..., "output": ...} pair.

    Args:
      size: square grid size (defaults to random even size in [8, 16]).
      present: subset of {"tl","tr","bl","br"} that contain components.
      components: local (row, col) coordinates per corner inside corner span.
      colors: color per corner.
    """
    rr = rng if rng is not None else random
    if size is None:
        size = rr.choice([8, 10, 12, 14, 16])
    if size < 8 or size % 2 != 0:
        raise ValueError("size must be even and >= 8")

    span = _corner_span(size)
    if span < 2:
        raise ValueError("size too small for this task layout")

    if present is None:
        k = rr.randint(1, 4)
        present = rr.sample(list(_CORNERS), k)
    present_set = set(present)
    if not present_set.issubset(set(_CORNERS)):
        raise ValueError("present must only use tl/tr/bl/br")
    if not present_set:
        raise ValueError("at least one component must be present")

    if components is None:
        components = {corner: _random_component(rr, span) for corner in present_set}
    if colors is None:
        available = [c for c in range(1, 10)]
        chosen = rr.sample(available, len(present_set))
        colors = {corner: col for corner, col in zip(sorted(present_set), chosen)}

    inp = _empty(size)
    out = _empty(size)
    src_o = _origins(size)
    half = size // 2

    by_corner: Dict[str, List[Tuple[int, int, int]]] = {c: [] for c in _CORNERS}
    for corner in sorted(present_set):
        pix = components.get(corner, [])
        if pix and not _is_8_connected_and_non_isolated(pix):
            raise ValueError(f"{corner} component must be 8-connected with no isolated cells")
        color = colors[corner]
        sr, sc = src_o[corner]
        for rr0, cc0 in pix:
            if rr0 < 0 or cc0 < 0 or rr0 >= span or cc0 >= span:
                raise ValueError(f"component pixel {(rr0, cc0)} outside corner span {span}")
            r, c = sr + rr0, sc + cc0
            inp[r][c] = color
            by_corner[corner].append((r, c, color))

    top_cells = by_corner["tl"] + by_corner["tr"]
    bot_cells = by_corner["bl"] + by_corner["br"]
    left_cells = by_corner["tl"] + by_corner["bl"]
    right_cells = by_corner["tr"] + by_corner["br"]

    top_max_r = max((r for r, _, _ in top_cells), default=half - 1)
    bot_min_r = min((r for r, _, _ in bot_cells), default=half)
    left_max_c = max((c for _, c, _ in left_cells), default=half - 1)
    right_min_c = min((c for _, c, _ in right_cells), default=half)

    dr_top = (half - 1) - top_max_r
    dr_bot = half - bot_min_r
    dc_left = (half - 1) - left_max_c
    dc_right = half - right_min_c

    for corner in _CORNERS:
        cells = by_corner[corner]
        if not cells:
            continue
        dr = dr_top if corner in ("tl", "tr") else dr_bot
        dc = dc_left if corner in ("tl", "bl") else dc_right
        for r, c, color in cells:
            out[r + dr][c + dc] = color

    return {"input": inp, "output": out}
