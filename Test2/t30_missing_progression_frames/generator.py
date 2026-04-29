"""Generator for t30_missing_progression_frames.

Creates vertical frame sequences with blank missing frames. The output fills
those frames by continuing a simple visual progression.
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from verifier import (
    SEP,
    _anti_diag_cells,
    _border_path,
    _col_cells,
    _diag_cells,
    _frame_from_cells,
    _frame_from_colored,
    _join_frames,
    _row_cells,
    _square_filled_cells,
    _square_outline_cells,
    verify_t30,
)

Grid = List[List[int]]
Cell = Tuple[int, int]

MAX_DIM = 30


def _max_frames_for_height(frame_h: int) -> int:
    # total stacked height is n * frame_h + (n - 1) separators
    return max(1, (MAX_DIM + 1) // (frame_h + 1))


def _dims_ok_grid(grid: Grid) -> bool:
    h = len(grid)
    w = len(grid[0]) if h else 0
    return h <= MAX_DIM and w <= MAX_DIM


FAMILY_NAMES = [
    "diamond_rings",
    "v_growth",
    "border_motion",
    "path_prefix",
    "translation",
    "line_sweep",
]


def _empty(h: int, w: int) -> Grid:
    return [[0 for _ in range(w)] for _ in range(h)]


def _hide_some(
    rr: random.Random,
    frames: Sequence[Grid],
    min_visible: int = 2,
    exact_k: int | None = None,
    keep: Set[int] | None = None,
) -> list[Grid]:
    n = len(frames)
    keep = set() if keep is None else set(keep)
    candidates = [i for i in range(n) if i not in keep]
    rr.shuffle(candidates)
    max_hide = max(1, n - min_visible)
    if exact_k is None:
        k = rr.randint(1, min(3, max_hide))
    else:
        k = min(exact_k, max_hide)
    hide = set(candidates[:k])
    h, w = len(frames[0]), len(frames[0][0])
    return [_empty(h, w) if i in hide else [row[:] for row in f] for i, f in enumerate(frames)]


def _cells_to_frame(h: int, w: int, cells: Iterable[Cell], color: int) -> Grid:
    return _frame_from_cells(h, w, cells, color)


def _all_nonempty(frames: Sequence[Grid]) -> bool:
    return all(any(v != 0 for row in f for v in row) for f in frames)


def _input_frame_constraints_ok(inp_frames: Sequence[Grid]) -> bool:
    return len(inp_frames) >= 4 and sum(any(v != 0 for row in f for v in row) for f in inp_frames) >= 3


# ------------------------------ progression families ------------------------------

def _diamond_ring_cells(h: int, w: int, cr: int, cc: int, radius: int) -> Set[Cell]:
    return {(r, c) for r in range(h) for c in range(w) if abs(r - cr) + abs(c - cc) == radius}


def _make_diamond_rings(rr: random.Random) -> tuple[list[Grid], list[Grid]]:
    size = 5
    max_n = min(5, size, _max_frames_for_height(size))
    n = rr.randint(4, max_n)
    color = rr.choice([c for c in range(1, 10) if c != SEP])
    cr = cc = size // 2
    start = rr.randint(0, 1)
    frames = [_cells_to_frame(size, size, _diamond_ring_cells(size, size, cr, cc, start + i), color) for i in range(n)]
    inp = _hide_some(rr, frames, min_visible=3)
    return inp, frames


def _v_cells(h: int, w: int, cc: int, level: int) -> Set[Cell]:
    cells: Set[Cell] = set()
    for r in range(0, min(h, level + 1)):
        for c in (cc - r, cc + r):
            if 0 <= c < w:
                cells.add((r, c))
    return cells


def _make_v_growth(rr: random.Random) -> tuple[list[Grid], list[Grid]]:
    h = rr.choice([4, 5, 6])
    w = 2 * h - 1
    max_n = min(h, _max_frames_for_height(h))
    n = rr.randint(4, max_n)
    color = rr.choice([c for c in range(1, 10) if c != SEP])
    cc = w // 2
    frames = [_cells_to_frame(h, w, _v_cells(h, w, cc, i), color) for i in range(n)]
    inp = _hide_some(rr, frames, min_visible=3)
    return inp, frames


def _make_border_motion(rr: random.Random) -> tuple[list[Grid], list[Grid]]:
    """Multicolor border motion with exactly one missing frame."""
    h = rr.randint(4, 6)
    w = rr.randint(4, 7)
    max_n = min(6, _max_frames_for_height(h))
    n = rr.randint(4, max_n)
    path = _border_path(h, w)
    L = len(path)
    count = rr.randint(2, min(4, len(path) // 4))
    step = rr.choice([s for s in (1, 2) if s < L])
    base = rr.sample(range(L), count)
    colors = rr.sample([c for c in range(1, 10) if c != SEP], count)
    frames: list[Grid] = []
    for i in range(n):
        cells: Dict[Cell, int] = {}
        for color, p in zip(colors, base):
            cells[path[(p + step * i) % L]] = color
        frames.append(_frame_from_colored(h, w, cells))
    inp = _hide_some(rr, frames, min_visible=n - 1, exact_k=1)
    return inp, frames


def _random_path(rr: random.Random, h: int, w: int, length: int) -> list[Cell] | None:
    for _ in range(300):
        start = (rr.randrange(h), rr.randrange(w))
        path = [start]
        used = {start}
        while len(path) < length:
            r, c = path[-1]
            nbrs = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
            rr.shuffle(nbrs)
            found = False
            for q in nbrs:
                qr, qc = q
                if not (0 <= qr < h and 0 <= qc < w) or q in used:
                    continue
                near_old = 0
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    if (qr + dr, qc + dc) in used:
                        near_old += 1
                if near_old != 1:
                    continue
                path.append(q)
                used.add(q)
                found = True
                break
            if not found:
                break
        if len(path) == length:
            return path
    return None


def _make_path_prefix(rr: random.Random) -> tuple[list[Grid], list[Grid]]:
    h = rr.randint(4, 6)
    w = rr.randint(4, 6)
    max_n = min(5, _max_frames_for_height(h))
    n = rr.randint(4, max_n)
    step = rr.randint(1, 3)
    start_len = rr.randint(1, 2)
    total = min(h * w - 2, start_len + step * (n - 1))
    path = _random_path(rr, h, w, total)
    if path is None:
        return _make_v_growth(rr)
    color = rr.choice([c for c in range(1, 10) if c != SEP])
    frames: list[Grid] = []
    for i in range(n):
        m = min(total, start_len + step * i)
        frames.append(_cells_to_frame(h, w, path[:m], color))
    inp = _hide_some(rr, frames, min_visible=n - 1, exact_k=1, keep={0, n - 1})
    return inp, frames


def _make_translation(rr: random.Random) -> tuple[list[Grid], list[Grid]]:
    h = rr.randint(4, 6)
    w = rr.randint(4, 6)
    max_n = min(5, _max_frames_for_height(h))
    n = rr.randint(4, max_n)
    dr, dc = rr.choice([(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)])
    shape_opts = [
        {(0, 0)},
        {(0, 0), (0, 1)},
        {(0, 0), (1, 0)},
        {(0, 0), (1, 0), (1, 1)},
        {(0, 1), (1, 0), (1, 1)},
        {(0, 0), (0, 1), (1, 0)},
        {(0, 0), (1, 0), (2, 0), (2, 1)},
    ]
    shape = rr.choice(shape_opts)
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    rmin = max(0, -(n - 1) * dr)
    rmax = min(h - 1 - max_r, h - 1 - max_r - max(0, (n - 1) * dr))
    cmin = max(0, -(n - 1) * dc)
    cmax = min(w - 1 - max_c, w - 1 - max_c - max(0, (n - 1) * dc))
    if rmin > rmax or cmin > cmax:
        return _make_diamond_rings(rr)
    r0 = rr.randint(rmin, rmax)
    c0 = rr.randint(cmin, cmax)
    color = rr.choice([c for c in range(1, 10) if c != SEP])
    frames = []
    for i in range(n):
        cells = {(r0 + dr * i + r, c0 + dc * i + c) for r, c in shape}
        frames.append(_cells_to_frame(h, w, cells, color))
    inp = _hide_some(rr, frames, min_visible=3)
    return inp, frames


def _make_line_sweep(rr: random.Random) -> tuple[list[Grid], list[Grid]]:
    h = rr.randint(4, 6)
    w = rr.randint(4, 6)
    families = []
    if h >= 4:
        families.append((list(range(h)), lambda p: _row_cells(h, w, p)))
    if w >= 4:
        families.append((list(range(w)), lambda p: _col_cells(h, w, p)))
    families.append((list(range(-(h - 1), w)), lambda p: _diag_cells(h, w, p)))
    families.append((list(range(h + w - 1)), lambda p: _anti_diag_cells(h, w, p)))
    params, fn = rr.choice(families)
    if len(params) < 4:
        return _make_translation(rr)
    step = rr.choice([1, -1])
    max_n = min(5, len(params), _max_frames_for_height(h))
    n = rr.randint(4, max_n)
    if step == 1:
        start = rr.randint(params[0], params[-1] - (n - 1))
    else:
        start = rr.randint(params[0] + (n - 1), params[-1])
    color = rr.choice([c for c in range(1, 10) if c != SEP])
    frames = [_cells_to_frame(h, w, fn(start + step * i), color) for i in range(n)]
    inp = _hide_some(rr, frames, min_visible=3)
    return inp, frames


def _make_square_growth(rr: random.Random, filled: bool) -> tuple[list[Grid], list[Grid]]:
    # Keep the largest square strictly inside the frame so the progression never
    # saturates into repeated full-frame images.
    size = rr.choice([7, 9])
    max_rad = size // 2 - 1
    valid_starts = [start for start in (0, 1) if max_rad - start + 1 >= 3]
    start = rr.choice(valid_starts)
    max_n = min(6, max_rad - start + 1, _max_frames_for_height(size))
    n = rr.randint(3, max_n)
    color = rr.choice([c for c in range(1, 10) if c != SEP])
    cr = cc = size // 2
    fn = _square_filled_cells if filled else _square_outline_cells
    frames = [_cells_to_frame(size, size, fn(size, size, cr, cc, start + i), color) for i in range(n)]
    inp = _hide_some(rr, frames, min_visible=2)
    return inp, frames


def _make_square_outline_growth(rr: random.Random) -> tuple[list[Grid], list[Grid]]:
    return _make_square_growth(rr, filled=False)


def _make_square_filled_growth(rr: random.Random) -> tuple[list[Grid], list[Grid]]:
    return _make_square_growth(rr, filled=True)


FAMILY_MAKERS = {
    "diamond_rings": _make_diamond_rings,
    "v_growth": _make_v_growth,
    "border_motion": _make_border_motion,
    "path_prefix": _make_path_prefix,
    "translation": _make_translation,
    "line_sweep": _make_line_sweep,
    "square_outline_growth": _make_square_outline_growth,
    "square_filled_growth": _make_square_filled_growth,
}


def generate(rng: random.Random | None = None, family: str | None = None) -> dict[str, Grid]:
    rr = rng if rng is not None else random
    names = [family] if family is not None else FAMILY_NAMES[:]
    if family is None:
        rr.shuffle(names)
    for _ in range(120):
        for name in names:
            maker = FAMILY_MAKERS[name]
            inp_frames, out_frames = maker(rr)
            if not _all_nonempty(out_frames):
                continue
            if not _input_frame_constraints_ok(inp_frames):
                continue
            w = len(out_frames[0][0])
            inp = _join_frames(inp_frames, w)
            out = _join_frames(out_frames, w)
            if not (_dims_ok_grid(inp) and _dims_ok_grid(out)):
                continue
            if verify_t30(inp) == out:
                return {"input": inp, "output": out}
        if family is None:
            rr.shuffle(names)
    raise RuntimeError("failed to generate valid t30 sample")
