"""Generator for Test2/t38: recover the atomic template from an inverse fractal.

Rule:
- The input is a self-similar fractal built from a small atomic template.
- The atomic template is a binary shape of size HxW.
- Starting from a single filled cell, each recursion replaces every filled cell
  by a copy of the whole template, so after depth n the grid size is H^n x W^n.
- Most occupied cells in the final fractal use a base color; a small number of
  occupied cells use a distinct highlight color.
- The output is the atomic template itself, using the base color on occupied
  template cells, except that a template cell is recolored with the highlight
  color iff at least one highlighted input cell belongs to that template
  position's residue class (equivalently: has those coordinates modulo H and W).

Generation constraints used to keep examples well-formed and unambiguous:
- the atomic template is not all-empty or all-full, has at least two occupied
  cells, and has no empty rows or columns;
- the chosen template is rejected if it admits multiple valid fractal
  factorizations under the transform used here;
- highlight cells are a small minority of the occupied cells;
- highlight density is coordinated with template size and fractal size so that
  very small highlighted template sets do not produce many repeated highlights
  in the input;
- at least one template position is highlighted in the output;
- dimensions remain within the ARC 30x30 limit.
"""

from __future__ import annotations

import json
import random
from typing import Iterable, List, Optional, Sequence, Tuple

Grid = List[List[int]]
Mask = List[List[int]]

BG = 0
COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def _empty(h: int, w: int, fill: int = 0) -> Grid:
    return [[fill for _ in range(w)] for _ in range(h)]


def _rectangular(grid: Grid) -> bool:
    return bool(grid) and bool(grid[0]) and all(len(row) == len(grid[0]) for row in grid)


def _nonzero_mask(grid: Grid) -> Mask:
    return [[1 if v != BG else 0 for v in row] for row in grid]


def _mask_dims(mask: Mask) -> tuple[int, int]:
    return len(mask), len(mask[0])


def _mask_equal(a: Mask, b: Mask) -> bool:
    return a == b


def _mask_block(mask: Mask, r0: int, r1: int, c0: int, c1: int) -> Mask:
    return [row[c0:c1] for row in mask[r0:r1]]


def _mask_any(mask: Mask) -> bool:
    return any(v for row in mask for v in row)


def _mask_occupied_positions(mask: Mask) -> list[tuple[int, int]]:
    return [(r, c) for r, row in enumerate(mask) for c, v in enumerate(row) if v]


def _compose_fractal(template: Mask, depth: int) -> Mask:
    """Return the binary fractal mask generated from template at the given depth."""
    cur = [[1]]
    th, tw = _mask_dims(template)
    for _ in range(depth):
        ch, cw = _mask_dims(cur)
        out = _empty(ch * th, cw * tw, 0)
        for r in range(ch):
            for c in range(cw):
                if cur[r][c] == 0:
                    continue
                for tr in range(th):
                    for tc in range(tw):
                        if template[tr][tc]:
                            out[r * th + tr][c * tw + tc] = 1
        cur = out
    return cur


def _valid_decomposition_levels(mask: Mask, H: int, W: int) -> Optional[Mask]:
    """If mask is a fractal generated from some HxW template, return that template mask.

    More precisely, this checks whether repeated HxW decomposition yields the
    same top-level pattern at every level and eventually bottoms out at [[1]].
    """
    h, w = _mask_dims(mask)
    current = [row[:] for row in mask]
    top_pattern: Optional[Mask] = None

    while True:
        ch, cw = _mask_dims(current)
        if ch == 1 and cw == 1:
            return top_pattern if current[0][0] == 1 else None
        if ch % H != 0 or cw % W != 0:
            return None
        bh, bw = ch // H, cw // W
        if bh < 1 or bw < 1:
            return None

        pattern = _empty(H, W, 0)
        rep: Optional[Mask] = None
        for i in range(H):
            for j in range(W):
                block = _mask_block(current, i * bh, (i + 1) * bh, j * bw, (j + 1) * bw)
                has = _mask_any(block)
                if not has:
                    pattern[i][j] = 0
                else:
                    if rep is None:
                        rep = block
                    elif not _mask_equal(rep, block):
                        return None
                    pattern[i][j] = 1
        if rep is None:
            return None
        if top_pattern is None:
            top_pattern = pattern
        elif not _mask_equal(top_pattern, pattern):
            return None
        current = rep


def _infer_template_candidates(mask: Mask) -> list[tuple[int, int, Mask]]:
    h, w = _mask_dims(mask)
    out: list[tuple[int, int, Mask]] = []
    for H in range(2, h + 1):
        for W in range(2, w + 1):
            template = _valid_decomposition_levels(mask, H, W)
            if template is None:
                continue
            out.append((H, W, template))
    return out


def _choose_base_and_highlight(grid: Grid) -> tuple[int, int]:
    nonzero = [v for row in grid for v in row if v != BG]
    colors = sorted(set(nonzero))
    if len(colors) != 2:
        raise ValueError("expected exactly two nonzero colors")
    counts = {c: nonzero.count(c) for c in colors}
    highlight = min(colors, key=lambda c: counts[c])
    base = max(colors, key=lambda c: counts[c])
    if counts[highlight] >= counts[base]:
        raise ValueError("highlight color must be a strict minority")
    return base, highlight


def transform_t38(grid: Grid) -> Grid:
    if not _rectangular(grid):
        raise ValueError("input grid must be non-empty and rectangular")
    base, highlight = _choose_base_and_highlight(grid)
    mask = _nonzero_mask(grid)
    candidates = _infer_template_candidates(mask)
    if not candidates:
        raise ValueError("no valid template factorization found")
    # If multiple factorizations are mathematically possible, use the smallest
    # atomic template. This matches the intended inverse-fractal interpretation
    # and the provided task examples.
    H, W, template = min(candidates, key=lambda x: (x[0] * x[1], x[0], x[1]))

    out = _empty(H, W, BG)
    for r, c in _mask_occupied_positions(template):
        out[r][c] = base

    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v == highlight:
                rr, cc = r % H, c % W
                if template[rr][cc]:
                    out[rr][cc] = highlight
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _random_template(rng: random.Random, H: int, W: int) -> Mask:
    for _ in range(5000):
        mask = [[1 if rng.random() < 0.45 else 0 for _ in range(W)] for _ in range(H)]
        occ = sum(v for row in mask for v in row)
        if occ < 2 or occ == H * W:
            continue
        # Require every row and every column to contain at least one occupied cell,
        # so the atomic template has no empty rows or columns.
        if any(sum(mask[r][c] for c in range(W)) == 0 for r in range(H)):
            continue
        if any(sum(mask[r][c] for r in range(H)) == 0 for c in range(W)):
            continue
        # Avoid degenerate single-row/col occupancy only.
        occ_rows = {r for r, row in enumerate(mask) for c, v in enumerate(row) if v}
        occ_cols = {c for r, row in enumerate(mask) for c, v in enumerate(row) if v}
        if len(occ_rows) == 1 or len(occ_cols) == 1:
            continue
        return mask
    raise RuntimeError("failed to generate template")


def _render_colored(mask: Mask, base: int, highlight: int, highlighted_positions: set[tuple[int, int]]) -> Grid:
    h, w = _mask_dims(mask)
    out = _empty(h, w, BG)
    for r in range(h):
        for c in range(w):
            if not mask[r][c]:
                continue
            out[r][c] = highlight if (r, c) in highlighted_positions else base
    return out


def _choose_output_highlight_count(template_occ_count: int, fractal_occ_count: int, depth: int, rng: random.Random) -> int:
    """Choose how many template cells are highlighted in the output.

    Larger fractals tend to get slightly more highlighted template positions, so
    tiny highlighted-template sets are less common on big inputs.
    """
    max_k = min(template_occ_count, max(1, template_occ_count // 2 + 1))
    if fractal_occ_count >= 90 or depth >= 4:
        min_k = min(max_k, 3)
    elif fractal_occ_count >= 30 or depth >= 3:
        min_k = min(max_k, 2)
    else:
        min_k = 1
    return rng.randint(min_k, max_k)


def _highlight_leaf_positions(
    fractal_mask: Mask,
    H: int,
    W: int,
    template_occ: list[tuple[int, int]],
    desired_output_highlights: set[tuple[int, int]],
    rng: random.Random,
) -> set[tuple[int, int]]:
    """Choose a sparse set of highlighted leaf cells.

    Every highlighted template residue is represented at least once. For small
    highlighted template sets (especially 1-2 cells), avoid repeating the same
    residue many times across the input fractal, since that can make the template
    too easy to spot.
    """
    leaves_by_residue: dict[tuple[int, int], list[tuple[int, int]]] = {(r, c): [] for (r, c) in template_occ}
    fh, fw = _mask_dims(fractal_mask)
    for r in range(fh):
        for c in range(fw):
            if fractal_mask[r][c]:
                residue = (r % H, c % W)
                if residue in leaves_by_residue:
                    leaves_by_residue[residue].append((r, c))

    desired_list = sorted(desired_output_highlights)
    chosen: set[tuple[int, int]] = set()
    chosen_per_residue: dict[tuple[int, int], int] = {res: 0 for res in desired_list}

    # Ensure each desired template position gets at least one highlighted leaf.
    for residue in desired_list:
        pick = rng.choice(leaves_by_residue[residue])
        chosen.add(pick)
        chosen_per_residue[residue] += 1

    k = len(desired_list)
    occ_count = sum(v for row in fractal_mask for v in row)

    # Coordinate extra input highlights with template-highlight count and input size.
    if k <= 2:
        extras_budget = 0
        per_residue_cap = 1
    elif k == 3:
        extras_budget = rng.randint(0, 1 if occ_count < 80 else 2)
        per_residue_cap = 2
    else:
        size_factor = 0 if occ_count < 40 else (1 if occ_count < 120 else 2)
        extras_budget = rng.randint(0, min(k - 1 + size_factor, 2 * k))
        per_residue_cap = 2 if occ_count < 120 else 3

    selectable = [res for res in desired_list if len(leaves_by_residue[res]) > chosen_per_residue[res]]
    for _ in range(extras_budget):
        selectable = [
            res for res in desired_list
            if chosen_per_residue[res] < per_residue_cap
            and len(leaves_by_residue[res]) > chosen_per_residue[res]
        ]
        if not selectable:
            break
        residue = rng.choice(selectable)
        candidates = [p for p in leaves_by_residue[residue] if p not in chosen]
        if not candidates:
            continue
        pick = rng.choice(candidates)
        chosen.add(pick)
        chosen_per_residue[residue] += 1
    return chosen


def generate(
    rng: random.Random | None = None,
    template_h: int | None = None,
    template_w: int | None = None,
    depth: int | None = None,
) -> dict[str, Grid]:
    rng = rng or random.Random()

    for _attempt in range(5000):
        H = template_h if template_h is not None else rng.randint(2, 4)
        W = template_w if template_w is not None else rng.randint(2, 4)
        n = depth if depth is not None else rng.randint(2, 4)
        if H ** n > 30 or W ** n > 30:
            continue

        template = _random_template(rng, H, W)
        fractal_mask = _compose_fractal(template, n)
        occ_count = sum(v for row in fractal_mask for v in row)
        template_occ = _mask_occupied_positions(template)
        if occ_count <= len(template_occ):
            continue

        # Choose colors.
        base, highlight = rng.sample(COLORS, 2)
        # Choose a non-empty subset of template positions to appear highlighted in the output,
        # coordinating this with fractal size so larger inputs tend to have somewhat richer
        # highlight structure.
        k = _choose_output_highlight_count(len(template_occ), occ_count, n, rng)
        desired_output_highlights = set(rng.sample(template_occ, k))
        leaf_highlights = _highlight_leaf_positions(fractal_mask, H, W, template_occ, desired_output_highlights, rng)
        if not leaf_highlights:
            continue
        # Keep highlight color a clear minority.
        if len(leaf_highlights) * 3 >= occ_count * 2:  # < 2/3? stronger than necessary; effectively tiny minority here
            continue

        inp = _render_colored(fractal_mask, base, highlight, leaf_highlights)
        out = _render_colored(template, base, highlight, desired_output_highlights)

        # Reject ambiguous template factorizations under the verifier.
        try:
            got = transform_t38(inp)
        except Exception:
            continue
        if got != out:
            continue
        candidates = _infer_template_candidates(_nonzero_mask(inp))
        fh, fw = len(inp), len(inp[0])
        nontrivial = [c for c in candidates if c[0] * c[1] < fh * fw]
        if not nontrivial:
            continue
        best_area = min(c[0] * c[1] for c in nontrivial)
        best = [c for c in nontrivial if c[0] * c[1] == best_area]
        if len(best) != 1:
            continue
        bH, bW, bT = best[0]
        if bH != H or bW != W or bT != template:
            continue
        return {"input": inp, "output": out}

    raise RuntimeError("failed to generate a valid t38 sample")


if __name__ == "__main__":
    print(json.dumps(generate(rng=random.Random(0)), indent=2))
