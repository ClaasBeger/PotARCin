"""Generator for Test2/t35: repair the order of frame sequence.

Rule:
- The input shows a horizontal sequence of same-sized "image" frames separated
  by vertical cyan lines.
- In the correct order, the frames show a natural visual progression.
- The output reorders the frames into that progression.

Generation constraints used to keep examples well-formed and unambiguous:
- all frames have the same size and are separated only by cyan columns;
- first and last frames are kept in their correct positions;
- at least two intermediate frames are out of order;
- generated sequences are from a diverse set of simple monotone progression
  families, so the intended ordering can be recovered deterministically from
  the frame content;
- the output contains exactly the same frames as the input, just reordered.
"""

from __future__ import annotations

import json
import random
from typing import List, Sequence, Tuple

Grid = List[List[int]]
Frame = List[List[int]]
Cell = Tuple[int, int]

BG = 0
CYAN = 8
COLORS = [1, 2, 3, 4, 5, 6, 7, 9]


def _empty(h: int, w: int, fill: int = BG) -> Grid:
    return [[fill for _ in range(w)] for _ in range(h)]


def _rectangular(grid: Grid) -> bool:
    return bool(grid) and bool(grid[0]) and all(len(row) == len(grid[0]) for row in grid)


def _occupied(frame: Frame) -> list[Cell]:
    return [(r, c) for r, row in enumerate(frame) for c, v in enumerate(row) if v != BG]


def _split_frames(grid: Grid) -> list[Frame]:
    if not _rectangular(grid):
        raise ValueError("input grid must be non-empty and rectangular")
    h, w = len(grid), len(grid[0])
    sep_cols = [c for c in range(w) if all(grid[r][c] == CYAN for r in range(h))]
    frames: list[Frame] = []
    start = 0
    for c in sep_cols + [w]:
        if c <= start:
            raise ValueError("empty frame")
        frames.append([row[start:c] for row in grid])
        start = c + 1
    if len(frames) < 3:
        raise ValueError("expected at least three frames")
    fh, fw = len(frames[0]), len(frames[0][0])
    if any(len(f) != fh or len(f[0]) != fw for f in frames):
        raise ValueError("all frames must share the same size")
    return frames


def _join_frames(frames: Sequence[Frame]) -> Grid:
    h = len(frames[0])
    out: Grid = []
    for r in range(h):
        row: list[int] = []
        for i, frame in enumerate(frames):
            if i:
                row.append(CYAN)
            row.extend(frame[r])
        out.append(row)
    return out


def _signature(frame: Frame) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(row) for row in frame)


def _mean_col(frame: Frame) -> float:
    occ = _occupied(frame)
    return sum(c for _, c in occ) / len(occ)


def _mean_row(frame: Frame) -> float:
    occ = _occupied(frame)
    return sum(r for r, _ in occ) / len(occ)


def _count(frame: Frame) -> int:
    return len(_occupied(frame))


def _two_dot_3x3_rank(frame: Frame) -> int | None:
    """Rank the two-dot 3x3 motif that appears in the reference examples."""
    if len(frame) != 3 or len(frame[0]) != 3:
        return None
    occ = sorted(_occupied(frame))
    rank = {
        ((0, 1), (2, 0)): 0,
        ((0, 0), (2, 1)): 1,
        ((1, 0), (2, 2)): 2,
        ((1, 2), (2, 0)): 3,
        ((0, 2), (2, 1)): 4,
        ((0, 1), (2, 2)): 5,
    }
    return rank.get(tuple(occ))


def _frame_sort_key(frame: Frame) -> tuple[float, float, float, float, tuple[tuple[int, ...], ...]]:
    occ = _occupied(frame)
    if not occ:
        raise ValueError("frames must not be empty")
    special = _two_dot_3x3_rank(frame)
    if special is not None:
        return (0, special, 0, 0, _signature(frame))
    return (1, _mean_col(frame), _count(frame), _mean_row(frame), _signature(frame))


def transform_t35(grid: Grid) -> Grid:
    frames = _split_frames(grid)
    ordered = sorted(frames, key=_frame_sort_key)
    return _join_frames(ordered)


# ---------------------------------------------------------------------------
# Generator families
# ---------------------------------------------------------------------------


def _frame_from_cells(h: int, w: int, cells: Sequence[Cell], color: int) -> Frame:
    f = _empty(h, w)
    for r, c in cells:
        if 0 <= r < h and 0 <= c < w:
            f[r][c] = color
    return f


def _single_dot_sequence(rng: random.Random) -> list[Frame]:
    h = rng.randint(3, 7)
    w = rng.randint(5, 7)
    color = rng.choice(COLORS)
    if rng.random() < 0.5:
        cells = [(round((h - 1) * (1 - c / (w - 1))), c) for c in range(w)]
    else:
        row = rng.randrange(h)
        cells = [(row, c) for c in range(w)]
    return [_frame_from_cells(h, w, [cell], color) for cell in cells]


def _growing_triangle_sequence(rng: random.Random) -> list[Frame]:
    h = rng.randint(4, 6)
    w = rng.randint(4, 6)
    n = rng.randint(4, min(6, w + 1))
    color = rng.choice(COLORS)
    frames: list[Frame] = []
    for k in range(1, n + 1):
        width = max(1, round((w * k) / n))
        cells: list[Cell] = []
        for r in range(h):
            row_fill = min(width, max(0, h - r))
            for c in range(row_fill):
                cells.append((r, c))
        frames.append(_frame_from_cells(h, w, cells, color))
    return frames


def _sliding_diagonal_sequence(rng: random.Random) -> list[Frame]:
    """A clean grow-to-full-diagonal-then-shrink sequence.

    The longest middle frame is symmetric and touches both opposite corners,
    avoiding the odd half-diagonal peak that looked visually arbitrary.
    """
    size = rng.choice([5, 7])
    color = rng.choice(COLORS)
    frames: list[Frame] = []
    for center in range(size):
        length = 2 * min(center, size - 1 - center) + 1
        if center == size // 2:
            length = size
        start = center - length // 2
        cols = range(start, start + length)
        cells = [(size - 1 - c, c) for c in cols]
        frames.append(_frame_from_cells(size, size, cells, color))
    return frames


def _vertical_bar_sweep_sequence(rng: random.Random) -> list[Frame]:
    """A vertical line propagates horizontally, perpendicular to its orientation."""
    h = rng.randint(4, 7)
    w = rng.randint(5, 7)
    color = rng.choice(COLORS)
    r0 = rng.randint(0, 1)
    r1 = rng.randint(h - 2, h - 1)
    cells_by_col = [[(r, c) for r in range(r0, r1 + 1)] for c in range(w)]
    return [_frame_from_cells(h, w, cells, color) for cells in cells_by_col]


def _horizontal_bar_sweep_sequence(rng: random.Random) -> list[Frame]:
    """A horizontal line propagates downward, perpendicular to its orientation."""
    h = rng.randint(5, 7)
    w = rng.randint(4, 7)
    color = rng.choice(COLORS)
    c0 = rng.randint(0, 1)
    c1 = rng.randint(w - 2, w - 1)
    cells_by_row = [[(r, c) for c in range(c0, c1 + 1)] for r in range(h)]
    return [_frame_from_cells(h, w, cells, color) for cells in cells_by_row]


def _center_growth_sequence(rng: random.Random) -> list[Frame]:
    """A centered shape grows outward, keeping one consistent shape family.

    Within a single example, all frames are either diamonds or squares; the
    shape type does not switch mid-sequence.
    """
    size = rng.choice([5, 7])
    color = rng.choice(COLORS)
    center = size // 2
    shape_kind = rng.choice(["diamond", "square"])
    frames: list[Frame] = []
    for radius in range(0, center + 1):
        if shape_kind == "diamond":
            cells = [
                (r, c)
                for r in range(size)
                for c in range(size)
                if abs(r - center) + abs(c - center) <= radius
            ]
        else:
            cells = [
                (r, c)
                for r in range(center - radius, center + radius + 1)
                for c in range(center - radius, center + radius + 1)
            ]
        frames.append(_frame_from_cells(size, size, cells, color))
    return frames


def _rightward_fill_sequence(rng: random.Random) -> list[Frame]:
    """A block grows by filling successive columns from left to right."""
    h = rng.randint(3, 6)
    w = rng.randint(5, 7)
    color = rng.choice(COLORS)
    top = rng.randint(0, 1)
    bottom = rng.randint(h - 2, h - 1)
    frames: list[Frame] = []
    for upto in range(w):
        cells = [(r, c) for r in range(top, bottom + 1) for c in range(upto + 1)]
        frames.append(_frame_from_cells(h, w, cells, color))
    return frames


def _two_dot_cycle_sequence(rng: random.Random) -> list[Frame]:
    color = rng.choice(COLORS)
    patterns = [
        [(0, 1), (2, 0)],
        [(0, 0), (2, 1)],
        [(1, 0), (2, 2)],
        [(1, 2), (2, 0)],
        [(0, 2), (2, 1)],
        [(0, 1), (2, 2)],
    ]
    return [_frame_from_cells(3, 3, cells, color) for cells in patterns]


def _make_disordered(frames: list[Frame], rng: random.Random) -> list[Frame]:
    if len(frames) < 4:
        raise ValueError("need at least four frames")
    middle = frames[1:-1]
    for _ in range(200):
        shuffled = middle[:]
        rng.shuffle(shuffled)
        if shuffled != middle and sum(1 for a, b in zip(shuffled, middle) if a != b) >= 2:
            return [frames[0]] + shuffled + [frames[-1]]
    raise RuntimeError("could not disorder frames")


def generate(rng: random.Random | None = None) -> dict[str, Grid]:
    rng = rng or random.Random()
    for _ in range(2000):
        family = rng.choices(
            [
                _single_dot_sequence,
                _growing_triangle_sequence,
                _sliding_diagonal_sequence,
                _two_dot_cycle_sequence,
                _vertical_bar_sweep_sequence,
                _horizontal_bar_sweep_sequence,
                _center_growth_sequence,
                _rightward_fill_sequence,
            ],
            # Reduce the old diagonal grow/move/shrink family and add several
            # clearer perpendicular-motion / growth families for diversity.
            weights=[2, 3, 1, 1, 4, 3, 3, 3],
        )[0]
        ordered = family(rng)
        if len({_signature(f) for f in ordered}) != len(ordered):
            continue
        if sorted(ordered, key=_frame_sort_key) != ordered:
            continue
        if len(ordered) < 4:
            continue
        inp_frames = _make_disordered(ordered, rng)
        inp = _join_frames(inp_frames)
        out = _join_frames(ordered)
        if transform_t35(inp) != out:
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("failed to generate a valid t35 sample")


if __name__ == "__main__":
    print(json.dumps(generate(rng=random.Random(0)), indent=2))
