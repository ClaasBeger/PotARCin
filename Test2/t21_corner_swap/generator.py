"""Generator for Test2/t21: extract corner blocks and swap diagonally.

Rule:
- Determine corner size k from input as the bbox side length of the largest
  corner-assigned 8-connected non-background component.
- Extract the four kxk corner patches from input.
- Rearrange diagonally:
    TL <- BR, TR <- BL, BL <- TR, BR <- TL
- Output is a 2k x 2k grid assembled from those four patches.

Diagonal connectivity is used for component detection.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Iterable, List, Tuple

Grid = List[List[int]]
Pos = Tuple[int, int]

BG = 0
NEI8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _empty(h: int, w: int) -> Grid:
    return [[BG for _ in range(w)] for _ in range(h)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _copy_patch(src: Grid, r0: int, c0: int, k: int) -> Grid:
    return [row[c0:c0 + k] for row in src[r0:r0 + k]]


def _paste_patch(dst: Grid, patch: Grid, r0: int, c0: int) -> None:
    for r in range(len(patch)):
        for c in range(len(patch[0])):
            dst[r0 + r][c0 + c] = patch[r][c]


def _bbox(cells: Iterable[Pos]) -> tuple[int, int]:
    xs = list(cells)
    rs = [r for r, _ in xs]
    cs = [c for _, c in xs]
    return max(rs) - min(rs) + 1, max(cs) - min(cs) + 1


def _corner_patches(grid: Grid, k: int) -> tuple[Grid, Grid, Grid, Grid]:
    h = len(grid)
    w = len(grid[0]) if h else 0
    tl = _copy_patch(grid, 0, 0, k)
    tr = _copy_patch(grid, 0, w - k, k)
    bl = _copy_patch(grid, h - k, 0, k)
    br = _copy_patch(grid, h - k, w - k, k)
    return tl, tr, bl, br


def _components8_all_nonzero(grid: Grid) -> list[set[Pos]]:
    h = len(grid)
    w = len(grid[0]) if h else 0
    rem = {(r, c) for r in range(h) for c in range(w) if grid[r][c] != BG}
    comps: list[set[Pos]] = []
    while rem:
        s = next(iter(rem))
        q = deque([s])
        rem.remove(s)
        comp = {s}
        while q:
            r, c = q.popleft()
            for dr, dc in NEI8:
                rr, cc = r + dr, c + dc
                if (rr, cc) in rem:
                    rem.remove((rr, cc))
                    comp.add((rr, cc))
                    q.append((rr, cc))
        comps.append(comp)
    return comps


def infer_corner_size_k(grid: Grid) -> int:
    """Infer k from largest corner-contained 8-connected component.

    A component contributes iff it lies fully in one of the four corner regions
    induced by half-splits of the grid. This matches demonstration behavior and
    avoids interior components influencing corner size.
    """
    h = len(grid)
    w = len(grid[0]) if h else 0
    hr = h // 2
    wc = w // 2
    regions = [
        (0, hr, 0, wc),          # TL
        (0, hr, w - wc, w),      # TR
        (h - hr, h, 0, wc),      # BL
        (h - hr, h, w - wc, w),  # BR
    ]

    best = 0
    for comp in _components8_all_nonzero(grid):
        for r0, r1, c0, c1 in regions:
            if all(r0 <= r < r1 and c0 <= c < c1 for r, c in comp):
                bh, bw = _bbox(comp)
                best = max(best, bh, bw)
                break
    if best < 2:
        raise ValueError("could not infer corner size")
    lim = min(h, w) // 2
    return min(best, lim)


def transform_t21(inp: Grid) -> Grid:
    if not inp or not inp[0]:
        raise ValueError("input grid must be non-empty")
    h = len(inp)
    w = len(inp[0])
    if any(len(row) != w for row in inp):
        raise ValueError("input grid must be rectangular")

    k = infer_corner_size_k(inp)
    tl, tr, bl, br = _corner_patches(inp, k)
    out = _empty(2 * k, 2 * k)
    # Diagonal swap.
    _paste_patch(out, br, 0, 0)      # TL <- BR
    _paste_patch(out, bl, 0, k)      # TR <- BL
    _paste_patch(out, tr, k, 0)      # BL <- TR
    _paste_patch(out, tl, k, k)      # BR <- TL
    return out


def _random_connected_shape(
    rr: random.Random, k: int, *, min_side: int, max_side: int
) -> set[Pos]:
    side = rr.randint(min_side, max_side)
    h = rr.randint(max(1, side - 2), side)
    w = rr.randint(max(1, side - 2), side)
    h = min(h, k)
    w = min(w, k)
    r0 = rr.randint(0, k - h)
    c0 = rr.randint(0, k - w)

    start = (r0 + rr.randrange(h), c0 + rr.randrange(w))
    cells = {start}
    target = rr.randint(max(2, (h * w) // 4), max(2, (h * w) // 2 + 2))
    while len(cells) < target:
        boundary: list[Pos] = []
        for r, c in cells:
            for dr, dc in NEI8:
                rr2, cc2 = r + dr, c + dc
                if r0 <= rr2 < r0 + h and c0 <= cc2 < c0 + w and (rr2, cc2) not in cells:
                    boundary.append((rr2, cc2))
        if not boundary:
            break
        p = rr.choice(boundary)
        cells.add(p)

    # Ensure bbox side at least side by pinning one far cell.
    if rr.random() < 0.5:
        cells.add((r0 + h - 1, c0))
    else:
        cells.add((r0, c0 + w - 1))
    return cells


def _expand_connected_cells(rr: random.Random, cells: set[Pos], k: int, extra_target: int) -> set[Pos]:
    """Expand an existing non-empty cell set while keeping it 8-connected."""
    out = set(cells)
    for _ in range(max(0, extra_target)):
        frontier: list[Pos] = []
        for r, c in out:
            for dr, dc in NEI8:
                rr2, cc2 = r + dr, c + dc
                if not _in_bounds(k, k, rr2, cc2):
                    continue
                if (rr2, cc2) in out:
                    continue
                frontier.append((rr2, cc2))
        if not frontier:
            break
        out.add(rr.choice(frontier))
    return out


def _paint_connected_subblob(
    rr: random.Random,
    patch: Grid,
    source_cells: set[Pos],
    color: int,
    size_target: int,
) -> None:
    """Recolor a connected subset of existing nonzero cells."""
    if not source_cells:
        return
    seed = rr.choice(list(source_cells))
    blob = {seed}
    while len(blob) < size_target:
        frontier: list[Pos] = []
        for r, c in blob:
            for dr, dc in NEI8:
                p = (r + dr, c + dc)
                if p in source_cells and p not in blob:
                    frontier.append(p)
        if not frontier:
            break
        blob.add(rr.choice(frontier))
    for r, c in blob:
        patch[r][c] = color


def _make_corner_patch(
    rr: random.Random,
    k: int,
    *,
    force_side_k: bool,
    allow_empty: bool,
) -> Grid:
    patch = _empty(k, k)
    if allow_empty and rr.random() < 0.10:
        return patch

    base_color = rr.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
    min_side = k if force_side_k else 1
    max_side = k if force_side_k else max(1, k - 1)
    cells = _random_connected_shape(rr, k, min_side=min_side, max_side=max_side)
    # Make corners crowded while preserving single connected component.
    extra = rr.randint(k // 2, k + 2) if force_side_k else rr.randint(1, k + 1)
    cells = _expand_connected_cells(rr, cells, k, extra)
    for r, c in cells:
        patch[r][c] = base_color

    # Multi-color within the SAME connected shape: recolor connected subblobs.
    n_colors = rr.choices([1, 2, 3], weights=[0.25, 0.50, 0.25], k=1)[0]
    for _ in range(n_colors - 1):
        c2 = rr.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
        if c2 == base_color:
            continue
        sub = rr.randint(2, max(2, min(6, len(cells) // 2 + 1)))
        _paint_connected_subblob(rr, patch, cells, c2, sub)
    return patch


def _paint_patch_to_corner(grid: Grid, patch: Grid, corner: str) -> None:
    h = len(grid)
    w = len(grid[0]) if h else 0
    k = len(patch)
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - k
    elif corner == "bl":
        r0, c0 = h - k, 0
    elif corner == "br":
        r0, c0 = h - k, w - k
    else:
        raise ValueError("unknown corner")
    _paste_patch(grid, patch, r0, c0)


def _add_connected_middle_noise(grid: Grid, rr: random.Random, k: int) -> None:
    """Add demo-like non-corner noise: many connected blobs, multi-color."""
    h = len(grid)
    w = len(grid[0]) if h else 0
    # Keep a one-cell moat around all corner windows to prevent touching.
    moat = 1

    def allowed(r: int, c: int) -> bool:
        if not _in_bounds(h, w, r, c):
            return False
        if grid[r][c] != BG:
            return False
        return not (
            (r < k + moat and c < k + moat)
            or (r < k + moat and c >= w - k - moat)
            or (r >= h - k - moat and c < k + moat)
            or (r >= h - k - moat and c >= w - k - moat)
        )

    free = [(r, c) for r in range(h) for c in range(w) if allowed(r, c)]
    if len(free) < 10:
        return

    # Aim near demo density: usually 12-28 noise cells.
    target_cells = rr.randint(10, min(28, len(free)))
    painted = 0
    attempts = 0
    while painted < target_cells and attempts < 80:
        attempts += 1
        color = rr.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
        size_goal = rr.randint(2, 6)
        seeds = [(r, c) for r, c in free if grid[r][c] == BG]
        if not seeds:
            break
        seed = rr.choice(seeds)
        blob = {seed}
        while len(blob) < size_goal:
            boundary: list[Pos] = []
            for br, bc in blob:
                for dr, dc in NEI8:
                    rr2, cc2 = br + dr, bc + dc
                    if (rr2, cc2) in blob:
                        continue
                    if allowed(rr2, cc2):
                        boundary.append((rr2, cc2))
            if not boundary:
                break
            blob.add(rr.choice(boundary))
        if len(blob) < 2:
            continue
        for br, bc in blob:
            grid[br][bc] = color
        painted += len(blob)


def _is_in_corner_window(r: int, c: int, h: int, w: int, k: int) -> bool:
    return (
        (r < k and c < k)
        or (r < k and c >= w - k)
        or (r >= h - k and c < k)
        or (r >= h - k and c >= w - k)
    )


def _has_corner_to_noise_bridge(grid: Grid, k: int) -> bool:
    """Return True if a nonzero component spans corner window and non-corner area."""
    h = len(grid)
    w = len(grid[0]) if h else 0
    comps = _components8_all_nonzero(grid)
    for comp in comps:
        in_corner = any(_is_in_corner_window(r, c, h, w, k) for r, c in comp)
        out_corner = any(not _is_in_corner_window(r, c, h, w, k) for r, c in comp)
        if in_corner and out_corner:
            return True
    return False


def _corner_windows_have_single_component(grid: Grid, k: int) -> bool:
    h = len(grid)
    w = len(grid[0]) if h else 0
    windows = [
        (0, k, 0, k),
        (0, k, w - k, w),
        (h - k, h, 0, k),
        (h - k, h, w - k, w),
    ]
    for r0, r1, c0, c1 in windows:
        cells = {(r, c) for r in range(r0, r1) for c in range(c0, c1) if grid[r][c] != BG}
        if not cells:
            continue
        rem = set(cells)
        comps = 0
        while rem:
            comps += 1
            s = next(iter(rem))
            q = deque([s])
            rem.remove(s)
            while q:
                r, c = q.popleft()
                for dr, dc in NEI8:
                    p = (r + dr, c + dc)
                    if p in rem:
                        rem.remove(p)
                        q.append(p)
            if comps > 1:
                return False
    return True


def generate(
    rng: random.Random | None = None,
    *,
    min_h: int = 10,
    max_h: int = 14,
    min_w: int = 10,
    max_w: int = 14,
) -> dict[str, Grid]:
    rr = rng or random.Random()
    corners = ["tl", "tr", "bl", "br"]

    for _ in range(2500):
        h = rr.randint(min_h, max_h)
        w = rr.randint(min_w, max_w)
        lim = min(h, w) // 2
        if lim < 3:
            continue
        # Keep some central area for non-corner noise.
        k_hi = max(3, lim - 1)
        k = rr.randint(3, k_hi)
        inp = _empty(h, w)

        force_corner = rr.choice(corners)
        empty_corner: str | None = None
        if rr.random() < 0.12:
            # Rare empty corner.
            empty_corner = rr.choice([c for c in corners if c != force_corner])

        for c in corners:
            if c == empty_corner:
                patch = _empty(k, k)
            else:
                patch = _make_corner_patch(
                    rr,
                    k,
                    force_side_k=(c == force_corner),
                    allow_empty=False,
                )
            _paint_patch_to_corner(inp, patch, c)

        _add_connected_middle_noise(inp, rr, k)

        # Require meaningful noise like demonstrations.
        h2 = len(inp)
        w2 = len(inp[0]) if h2 else 0
        noise_cells = sum(
            1
            for r in range(h2)
            for c in range(w2)
            if inp[r][c] != BG and not _is_in_corner_window(r, c, h2, w2, k)
        )
        if noise_cells < 10:
            continue

        # Enforce unambiguous corner sizing.
        try:
            if _has_corner_to_noise_bridge(inp, k):
                continue
            if not _corner_windows_have_single_component(inp, k):
                continue
            if infer_corner_size_k(inp) != k:
                continue
            out = transform_t21(inp)
        except ValueError:
            continue
        if out == inp:
            continue
        return {"input": inp, "output": out}

    raise RuntimeError("failed to sample valid t21 instance")
