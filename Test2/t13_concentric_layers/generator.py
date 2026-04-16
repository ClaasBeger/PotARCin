"""Generator for Test2/t13: concentric rectangular layers + green marker + score stack.

Rule (matches Test2/t13.json demos):
- Input is a 0-padded grid with a solid axis-aligned rectangle of non-zero cells.
- Exactly one cell is green (3), marking a position inside the rectangle.
- Concentric layers are onion rings with configurable per-ring thickness; scoring follows
  the same logical-layer peel rule as ``verifier.transform_t13`` (deeper layer → higher score).
- Output equals input except:
  - the green cell becomes the true color of its layer;
  - in the rightmost column, the bottom ``score`` cells are set to brown/red (9), overwriting.
"""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

Grid = List[List[int]]

GREEN = 3
MARK = 9
BG = 0

_PALETTE = [1, 2, 4, 5, 6, 7, 8]


def _dist_to_bbox(r: int, c: int, r0: int, r1: int, c0: int, c1: int) -> int:
    return min(r - r0, r1 - r, c - c0, c1 - c)


def _layer_index_for_dist(d: int, thicknesses: Sequence[int]) -> int:
    """0 .. len(thicknesses)-1 for ring bands; len(thicknesses) for the center/core."""
    cum = 0
    for i, t in enumerate(thicknesses):
        if d < cum + t:
            return i
        cum += t
    return len(thicknesses)


def paint_concentric_rectangle(
    *,
    inner_h: int,
    inner_w: int,
    pad_top: int,
    pad_left: int,
    pad_bottom: int,
    pad_right: int,
    thicknesses: Sequence[int],
    colors: Sequence[int],
    center_color: int,
    rng: random.Random,
) -> Tuple[Grid, int, int, int]:
    """Return ``(grid, green_r, green_c, true_color_at_marker)``."""
    if inner_h < 3 or inner_w < 3:
        raise ValueError("inner rectangle too small")
    if len(thicknesses) < 1 or len(thicknesses) != len(colors):
        raise ValueError("thicknesses/colors mismatch")
    if center_color in (GREEN, MARK, BG):
        raise ValueError("invalid center color")
    if any(c in (GREEN, MARK) for c in colors):
        raise ValueError("invalid ring color")
    if any(t <= 0 for t in thicknesses):
        raise ValueError("thickness must be positive")

    H = pad_top + inner_h + pad_bottom
    W = pad_left + inner_w + pad_right
    g: Grid = [[BG for _ in range(W)] for _ in range(H)]

    r0, c0 = pad_top, pad_left
    r1, c1 = pad_top + inner_h - 1, pad_left + inner_w - 1

    def paint_cell(d: int) -> int:
        cum = 0
        for t, col in zip(thicknesses, colors):
            if d < cum + t:
                return col
            cum += t
        return center_color

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            d = _dist_to_bbox(r, c, r0, r1, c0, c1)
            g[r][c] = paint_cell(d)

    # Uniform over cells biases toward outer rings (they have more cells). Sample a
    # logical layer uniformly, then a cell uniformly within that layer.
    by_layer: dict[int, list[Tuple[int, int, int]]] = {}
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            d = _dist_to_bbox(r, c, r0, r1, c0, c1)
            li = _layer_index_for_dist(d, thicknesses)
            by_layer.setdefault(li, []).append((r, c, g[r][c]))

    layer_ids = list(by_layer.keys())
    chosen_layer = rng.choice(layer_ids)
    gr, gc, true_color = rng.choice(by_layer[chosen_layer])
    g[gr][gc] = GREEN
    return g, gr, gc, true_color


def generate(
    rng: random.Random | None = None,
    *,
    min_inner: int = 7,
    max_inner: int = 14,
) -> dict[str, Grid]:
    rng = rng or random.Random()
    inner_h = rng.randint(min_inner, max_inner)
    inner_w = rng.randint(min_inner, max_inner)

    max_d = (min(inner_h, inner_w) - 1) // 2
    if max_d < 2:
        raise ValueError("inner size too small for t13")

    n_rings = rng.randint(2, min(5, max_d))
    total = rng.randint(n_rings, max_d)  # total ring thickness budget (rest is thick core)

    cuts = sorted(rng.sample(range(1, total), n_rings - 1))
    thicknesses: list[int] = []
    prev = 0
    for cut in cuts + [total]:
        thicknesses.append(cut - prev)
        prev = cut

    colors = rng.sample(_PALETTE, k=n_rings)
    center_pool = [c for c in _PALETTE if c not in colors]
    if not center_pool:
        center_pool = [c for c in range(1, 10) if c not in colors and c not in (GREEN, MARK)]
    center_color = rng.choice(center_pool)

    pad_top = rng.randint(1, 3)
    pad_left = rng.randint(1, 3)
    pad_bottom = rng.randint(1, 3)
    pad_right = rng.randint(1, 3)

    inp, _gr, _gc, _tc = paint_concentric_rectangle(
        inner_h=inner_h,
        inner_w=inner_w,
        pad_top=pad_top,
        pad_left=pad_left,
        pad_bottom=pad_bottom,
        pad_right=pad_right,
        thicknesses=thicknesses,
        colors=colors,
        center_color=center_color,
        rng=rng,
    )

    from verifier import transform_t13

    out = transform_t13(inp)
    return {"input": inp, "output": out}
