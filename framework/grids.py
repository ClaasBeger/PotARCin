from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


Color = int


Grid = List[List[Color]]


@dataclass(frozen=True)
class GridPair:
    """Simple container for an input/output grid pair."""

    input: Grid
    output: Grid


def is_equal_grid(a: Grid, b: Grid) -> bool:
    """Return True if two grids are structurally and element-wise equal."""
    if len(a) != len(b):
        return False
    return all(row_a == row_b for row_a, row_b in zip(a, b))


def cell_edit_distance_same_shape(a: Grid, b: Grid) -> int:
    """Count cells where values differ. Same dimensions required.

    Each differing cell contributes cost 1 (value change), regardless of how far
    the integers are apart.
    """
    if len(a) != len(b):
        raise ValueError("cell_edit_distance_same_shape: height mismatch")
    if not a and not b:
        return 0
    w = len(a[0]) if a else 0
    for ra, rb in zip(a, b):
        if len(ra) != len(rb) or len(ra) != w:
            raise ValueError("cell_edit_distance_same_shape: width mismatch")
    return sum(
        1
        for ra, rb in zip(a, b)
        for ca, cb in zip(ra, rb)
        if ca != cb
    )


def normalized_cell_edit_distance_same_shape(a: Grid, b: Grid) -> float:
    """``cell_edit_distance_same_shape / (H * W)`` for non-empty same-shape grids."""
    if not a or not a[0]:
        if not b or not b[0]:
            return 0.0
        raise ValueError("normalized_cell_edit_distance_same_shape: shape mismatch")
    h, w = len(a), len(a[0])
    if len(b) != h or any(len(row) != w for row in b):
        raise ValueError("normalized_cell_edit_distance_same_shape: shape mismatch")
    d = cell_edit_distance_same_shape(a, b)
    return d / float(h * w)


def normalized_cell_edit_or_shape_mismatch(a: Grid, b: Grid) -> float:
    """Same as :func:`normalized_cell_edit_distance_same_shape` when shapes match; otherwise ``1.0``.

    Used when comparing verifier outputs: different output shapes are treated as maximally far.
    """
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    if len(a) != len(b):
        return 1.0
    w = len(a[0]) if a else 0
    if w == 0:
        return 0.0 if not any(len(row) != 0 for row in b) else 1.0
    if any(len(row) != w for row in a) or any(len(row) != w for row in b):
        return 1.0
    return normalized_cell_edit_distance_same_shape(a, b)


def cell_edit_distance_padded(
    a: Grid,
    b: Grid,
    *,
    fill: int = -1,
) -> int:
    """Pad both grids to a common bounding box with *fill*, then count differing cells.

    Used when comparing grids that may differ in size: padding positions count as
    edits relative to *fill*.
    """
    ha = len(a)
    wa = len(a[0]) if ha else 0
    hb = len(b)
    wb = len(b[0]) if hb else 0
    h, w = max(ha, hb), max(wa, wb)
    if h == 0 or w == 0:
        return 0
    n = 0
    for r in range(h):
        for c in range(w):
            va = a[r][c] if r < ha and c < wa else fill
            vb = b[r][c] if r < hb and c < wb else fill
            if va != vb:
                n += 1
    return n


def normalized_cell_edit_distance_padded(a: Grid, b: Grid, *, fill: int = -1) -> float:
    """Normalized edit distance on the padded bounding box."""
    ha = len(a)
    wa = len(a[0]) if ha else 0
    hb = len(b)
    wb = len(b[0]) if hb else 0
    h, w = max(ha, hb), max(wa, wb)
    if h == 0 or w == 0:
        return 0.0
    return cell_edit_distance_padded(a, b, fill=fill) / float(h * w)


def normalized_cell_edit_distance_padded_pair(
    a_in: Grid,
    a_out: Grid,
    b_in: Grid,
    b_out: Grid,
    *,
    fill: int = -1,
) -> tuple[float, float, float]:
    """Normalized padded distances for input and output, and their arithmetic mean."""
    ni = normalized_cell_edit_distance_padded(a_in, b_in, fill=fill)
    no = normalized_cell_edit_distance_padded(a_out, b_out, fill=fill)
    return ni, no, (ni + no) / 2.0


def normalized_cell_edit_between_outputs(
    gold: Grid,
    corrupt: Grid,
    *,
    fill: int = -1,
) -> float:
    """Verifier corruption similarity: padded normalized distance (handles different sizes)."""
    return normalized_cell_edit_distance_padded(gold, corrupt, fill=fill)


def _orthogonal_component_at(grid: Grid, sr: int, sc: int) -> List[tuple[int, int]]:
    """4-connected cells of the same color as ``grid[sr][sc]`` containing ``(sr, sc)``."""
    if not grid or not grid[0]:
        return []
    h, w = len(grid), len(grid[0])
    color = grid[sr][sc]
    stack = [(sr, sc)]
    seen = {(sr, sc)}
    comp: List[tuple[int, int]] = []
    while stack:
        cr, cc = stack.pop()
        comp.append((cr, cc))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and grid[nr][nc] == color:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return comp


def _pick_weighted_cell(
    inp: Grid,
    out: Grid,
    rng: random.Random,
    *,
    source_color_weight_black: float,
) -> tuple[Grid, int, int] | None:
    """Choose a random cell from input/output; cells with color 0 are less likely."""
    positions: List[tuple[Grid, int, int]] = []
    weights: List[float] = []
    for g in (inp, out):
        if not g or not g[0]:
            continue
        hh, ww = len(g), len(g[0])
        for r in range(hh):
            for c in range(ww):
                positions.append((g, r, c))
                col = g[r][c]
                weights.append(source_color_weight_black if col == 0 else 1.0)
    if not positions:
        return None
    idx = rng.choices(range(len(positions)), weights=weights, k=1)[0]
    return positions[idx]


def _flip_count_weights(max_flips: int) -> List[float]:
    """Relative weights for counts ``k`` in ``1 .. max_flips``.

    Base weight is ``(max_flips + 1 - k) ** 2`` (steeper than linear). Counts
    **1** and **2** get an extra **1.25×** boost; when ``max_flips >= 4``, counts
    **4** and **5** (any ``k >= 4``) get a **0.5×** cut so long runs are rarer.
    """
    if max_flips < 1:
        raise ValueError("max_flips must be >= 1")
    weights: List[float] = []
    for k in range(1, max_flips + 1):
        w = float((max_flips + 1 - k) ** 2)
        if k <= 2:
            w *= 1.25
        if max_flips >= 4 and k >= 4:
            w *= 0.5
        weights.append(w)
    return weights


def flip_count_probabilities(max_flips: int) -> dict[int, float]:
    """Normalized probabilities for :func:`sample_flip_count_favor_one` (same *max_flips*)."""
    counts = list(range(1, max_flips + 1))
    w = _flip_count_weights(max_flips)
    s = sum(w)
    return {k: w[i] / s for i, k in enumerate(counts)}


def sample_flip_count_favor_one(rng: random.Random, *, max_flips: int = 5) -> int:
    """Sample the number of recolor operations in ``{1, ..., max_flips}``.

    Uses :func:`_flip_count_weights` — favors **1** and **2** flips strongly and
    downweights **4** and **5** when ``max_flips >= 4``. See
    :func:`flip_count_probabilities` for exact masses.
    """
    counts = list(range(1, max_flips + 1))
    weights = _flip_count_weights(max_flips)
    (c,) = rng.choices(counts, weights=weights, k=1)
    return c


def _sample_new_color_weighted(
    rng: random.Random,
    old: int,
    *,
    new_color_weights: Sequence[float],
) -> int:
    """Pick a new color != *old*; *new_color_weights* biases how often 0 is chosen."""
    for _ in range(40):
        (c,) = rng.choices(range(10), weights=list(new_color_weights), k=1)
        if c != old:
            return c
    others = [c for c in range(10) if c != old]
    return rng.choice(others)


def connected_component_color_flips_on_pair(
    pair: GridPair,
    rng: random.Random,
    *,
    num_ops: int,
    source_color_weight_black: float = 0.22,
    new_color_weights: Sequence[float] = (
        0.18,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ),
) -> tuple[GridPair, int]:
    """Copy *pair* and apply *num_ops* random recolor operations.

    Each operation **first** picks a single cell on the input or output grid (cells
    with color 0 are less likely). It then takes the 4-connected monochromatic
    **neighborhood** containing that cell: if that neighborhood has **at least two**
    cells, the **whole** neighborhood is recolored to one new color; otherwise only
    that cell is recolored. New colors use *new_color_weights* so black (0) is less
    likely as the replacement.

    Returns ``(new_pair, n_operations_applied)``.
    """
    if num_ops < 1:
        raise ValueError("num_ops must be >= 1")
    inp = clone_grid(pair.input)
    out = clone_grid(pair.output)
    n_ops = num_ops
    applied = 0
    for _ in range(n_ops):
        pick = _pick_weighted_cell(
            inp, out, rng, source_color_weight_black=source_color_weight_black
        )
        if pick is None:
            continue
        g, r, c = pick
        comp = _orthogonal_component_at(g, r, c)
        old = g[r][c]
        new_c = _sample_new_color_weighted(rng, old, new_color_weights=new_color_weights)
        if len(comp) >= 2:
            for rr, cc in comp:
                g[rr][cc] = new_c
        else:
            g[r][c] = new_c
        applied += 1
    return GridPair(inp, out), applied


def random_color_flips_on_pair(
    pair: GridPair,
    rng: random.Random,
    *,
    max_flips: int = 5,
) -> tuple[GridPair, int]:
    """Sample flip count with :func:`sample_flip_count_favor_one` then recolor."""
    n = sample_flip_count_favor_one(rng, max_flips=max_flips)
    return connected_component_color_flips_on_pair(pair, rng, num_ops=n)


def pretty_grid(grid: Grid) -> str:
    """Return a human-readable string representation of a grid."""
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def validate_grid(grid: Grid) -> None:
    """Validate that a grid is a non-empty rectangular matrix of ints.

    Raises:
        ValueError: If the grid is malformed.
    """
    if not grid:
        raise ValueError("Grid must be non-empty")
    row_length = len(grid[0])
    if row_length == 0:
        raise ValueError("Grid rows must be non-empty")
    for row in grid:
        if len(row) != row_length:
            raise ValueError("Grid must be rectangular (all rows same length)")
        for cell in row:
            if not isinstance(cell, int):
                raise ValueError("Grid cells must be integers")


def clone_grid(grid: Grid) -> Grid:
    """Return a deep copy of a grid."""
    return [list(row) for row in grid]


def batch_equal(
    a: Sequence[Grid],
    b: Sequence[Grid],
) -> bool:
    """Return True if two sequences of grids are all pairwise equal."""
    if len(a) != len(b):
        return False
    return all(is_equal_grid(ga, gb) for ga, gb in zip(a, b))


def iter_cells(grid: Grid) -> Iterable[tuple[int, int, Color]]:
    """Yield (row_idx, col_idx, color) triples for all cells in a grid."""
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            yield r, c, color

