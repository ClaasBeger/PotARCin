"""Verifier / forward transform for Test2/t13 (concentric layers + green marker)."""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from typing import List, Tuple

Grid = List[List[int]]

GREEN = 3
MARK = 9
BG = 0


def _bbox_nonzero(g: Grid) -> Tuple[int, int, int, int]:
    rows = [i for i, r in enumerate(g) for j, v in enumerate(r) if v != 0]
    cols = [j for i, r in enumerate(g) for j, v in enumerate(r) if v != 0]
    if not rows:
        raise ValueError("empty grid")
    return min(rows), max(rows), min(cols), max(cols)


def _find_green(g: Grid) -> Tuple[int, int]:
    found: list[Tuple[int, int]] = []
    for r, row in enumerate(g):
        for c, v in enumerate(row):
            if v == GREEN:
                found.append((r, c))
    if len(found) != 1:
        raise ValueError("expected exactly one green cell (3)")
    return found[0]


def _border_coords(r0: int, r1: int, c0: int, c1: int) -> List[Tuple[int, int]]:
    out: list[Tuple[int, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r == r0 or r == r1 or c == c0 or c == c1:
                out.append((r, c))
    return out


def _mode_non_green(vals: List[int]) -> int:
    xs = [v for v in vals if v not in (BG, GREEN)]
    if not xs:
        raise ValueError("empty border for mode")
    return Counter(xs).most_common(1)[0][0]


def _onion_bands_uniform(g: Grid, r0: int, r1: int, c0: int, c1: int) -> bool:
    """Each L1-distance band (inside bbox) has at most one non-zero color."""
    by_d: dict[int, set[int]] = defaultdict(set)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            v = g[r][c]
            if v == BG:
                continue
            d = min(r - r0, r1 - r, c - c0, c1 - c)
            by_d[d].add(v)
    for s in by_d.values():
        if len(s) > 1:
            return False
    return True


def _infer_center_color(
    inp: Grid, gr: int, gc: int, *, forbidden: set[int] | None = None
) -> int:
    """When green sits alone as the last 1×1 core, pick a consistent fill color.

    *forbidden* is the set of colors already used by outer logical layers (so we
    do not pick another ``4`` when the rim is ``4`` and only the core is missing).

    If several values keep L1-distance bands monochrome, break ties by closeness
    to the mean of the green cell's non-zero orthogonal neighbors.
    """
    ban = forbidden or set()
    nbr_vals: list[int] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        r, c = gr + dr, gc + dc
        if 0 <= r < len(inp) and 0 <= c < len(inp[0]):
            v = inp[r][c]
            if v not in (BG, GREEN):
                nbr_vals.append(v)
    target = sum(nbr_vals) / len(nbr_vals) if nbr_vals else 0.0

    candidates: list[int] = []
    for trial in range(1, 10):
        if trial in (GREEN, MARK) or trial in ban:
            continue
        t = copy.deepcopy(inp)
        t[gr][gc] = trial
        r0, r1, c0, c1 = _bbox_nonzero(t)
        if _onion_bands_uniform(t, r0, r1, c0, c1):
            candidates.append(trial)
    if not candidates:
        raise ValueError("could not infer center color")
    candidates.sort(key=lambda t: (abs(t - target), -t))
    return candidates[0]


def transform_t13(inp: Grid) -> Grid:
    """Apply t13 rule: replace green with layer color; stack score 9s in last column."""
    if not inp or any(len(row) != len(inp[0]) for row in inp):
        raise ValueError("non-rectangular grid")

    gr, gc = _find_green(inp)
    r0, r1, c0, c1 = _bbox_nonzero(inp)

    layers: list[int] = []
    prev_dom: int | None = None
    score: int | None = None
    rep: int | None = None

    br0, br1, bc0, bc1 = r0, r1, c0, c1
    while br0 <= br1 and bc0 <= bc1:
        if br0 == br1 and bc0 == bc1 and inp[br0][bc0] == GREEN:
            rep = _infer_center_color(inp, gr, gc, forbidden=set(layers))
            score = len(layers) + 1
            break

        coords = _border_coords(br0, br1, bc0, bc1)
        dom = _mode_non_green([inp[r][c] for r, c in coords])
        if prev_dom is None or dom != prev_dom:
            layers.append(dom)
            prev_dom = dom

        on_border = gr == br0 or gr == br1 or gc == bc0 or gc == bc1
        if on_border:
            rep = dom
            score = len(layers)
            break

        if br0 == br1 and bc0 == bc1:
            break
        br0, br1, bc0, bc1 = br0 + 1, br1 - 1, bc0 + 1, bc1 - 1

    if score is None or rep is None:
        raise ValueError("could not place green on a recognized layer")

    out = copy.deepcopy(inp)
    out[gr][gc] = rep
    h, w = len(out), len(out[0])
    last_col = w - 1
    for k in range(score):
        r = h - 1 - k
        if r < 0:
            raise ValueError("score exceeds grid height")
        out[r][last_col] = MARK
    return out


def verify_t13(inp: Grid, out: Grid) -> bool:
    try:
        return transform_t13(inp) == out
    except Exception:
        return False
