"""Generator for Test2/t16: edge fragments assemble to a target grey shape."""

from __future__ import annotations

import random
from collections import deque
from typing import Iterable, List, Sequence, Tuple

Grid = List[List[int]]
Pos = Tuple[int, int]
ShapeSig = Tuple[Pos, ...]

BG = 0
GREY = 5


def _empty(h: int, w: int) -> Grid:
    return [[BG for _ in range(w)] for _ in range(h)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> Iterable[Pos]:
    yield (r + 1, c)
    yield (r - 1, c)
    yield (r, c + 1)
    yield (r, c - 1)


def _normalize(cells: Iterable[Pos]) -> ShapeSig:
    cells_t = tuple(cells)
    if not cells_t:
        return ()
    min_r = min(r for r, _ in cells_t)
    min_c = min(c for _, c in cells_t)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells_t))


def _translate(sig: ShapeSig, dr: int, dc: int) -> set[Pos]:
    return {(r + dr, c + dc) for r, c in sig}


def _extract_components(grid: Grid, target_color: int) -> list[set[Pos]]:
    h = len(grid)
    w = len(grid[0]) if h else 0
    seen: set[Pos] = set()
    comps: list[set[Pos]] = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] != target_color or (r, c) in seen:
                continue
            q = deque([(r, c)])
            seen.add((r, c))
            comp: set[Pos] = set()
            while q:
                rr, cc = q.popleft()
                comp.add((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(h, w, nr, nc) and grid[nr][nc] == target_color and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_boundary(cells: set[Pos], h: int, w: int) -> bool:
    for r, c in cells:
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            return True
    return False


def _assemble_signatures(piece_sigs: Sequence[ShapeSig], *, max_states: int = 12000) -> set[ShapeSig]:
    """Enumerate connected unions achievable by translating all pieces."""
    if not piece_sigs:
        return set()

    ordered = sorted(piece_sigs, key=lambda s: (-len(s), s))
    first = set(ordered[0])
    rest = ordered[1:]
    out: set[ShapeSig] = set()
    seen_states: set[tuple[int, ShapeSig]] = set()
    budget = max_states

    def rec(idx: int, union: set[Pos]) -> None:
        nonlocal budget
        if budget <= 0:
            return
        budget -= 1

        key = (idx, _normalize(union))
        if key in seen_states:
            return
        seen_states.add(key)

        if idx == len(rest):
            out.add(_normalize(union))
            return

        piece = rest[idx]
        translations: set[Pos] = set()
        for ur, uc in union:
            for ar, ac in ((ur + 1, uc), (ur - 1, uc), (ur, uc + 1), (ur, uc - 1)):
                for pr, pc in piece:
                    translations.add((ar - pr, ac - pc))

        for tr, tc in sorted(translations):
            placed = _translate(piece, tr, tc)
            if union & placed:
                continue
            # Keep the final assembly connected by enforcing contact at each step.
            if not any((nr, nc) in union for r, c in placed for nr, nc in _neighbors4(r, c)):
                continue
            rec(idx + 1, union | placed)

    rec(0, first)
    return out


def _all_assemblies(piece_sigs: Sequence[ShapeSig]) -> set[ShapeSig]:
    sigs = _assemble_signatures(piece_sigs)
    if not sigs:
        raise ValueError("could not assemble colored edge fragments")
    return sigs


def _sample_mode_count(rr: random.Random, n_modes: int) -> int:
    """Bias toward 1-2 assembly modes, with all modes selected rarely."""
    if n_modes <= 1:
        return 1
    if n_modes == 2:
        return 1 if rr.random() < 0.7 else 2

    # n_modes >= 3: strongly favor 1 or 2; all modes are uncommon.
    choices: list[tuple[int, float]] = [
        (1, 0.45),
        (2, 0.35),
        (3, 0.12),
        (n_modes, 0.03),
    ]
    # Add medium counts (4..n-1) with low probability mass.
    middle = [k for k in range(4, n_modes) if k != n_modes]
    if middle:
        rem = 0.05 / len(middle)
        for k in middle:
            choices.append((k, rem))
    total = sum(w for _, w in choices)
    x = rr.random() * total
    acc = 0.0
    for k, w in choices:
        acc += w
        if x <= acc:
            return min(k, n_modes)
    return min(2, n_modes)


def transform_t16(inp: Grid) -> Grid:
    if not inp or not inp[0]:
        raise ValueError("input grid must be non-empty")
    w = len(inp[0])
    if any(len(row) != w for row in inp):
        raise ValueError("input grid must be rectangular")

    h = len(inp)
    colors = sorted({v for row in inp for v in row if v not in (BG, GREY)})
    if len(colors) != 1:
        raise ValueError("expected exactly one non-grey, non-background color")
    color = colors[0]

    color_comps = _extract_components(inp, color)
    edge_comps = [comp for comp in color_comps if _touches_boundary(comp, h, w)]
    if len(edge_comps) < 2:
        raise ValueError("expected at least two colored edge fragments")

    piece_sigs = [_normalize(comp) for comp in edge_comps]
    target_sigs = _all_assemblies(piece_sigs)

    out = [row[:] for row in inp]
    # Remove original colored fragments.
    for r in range(h):
        for c in range(w):
            if out[r][c] == color:
                out[r][c] = BG

    # Color every grey component matching any valid assembly.
    grey_comps = _extract_components(inp, GREY)
    for comp in grey_comps:
        if _normalize(comp) not in target_sigs:
            continue
        for r, c in comp:
            out[r][c] = color
    return out


def _random_polyomino(rr: random.Random, size: int) -> ShapeSig:
    cells: set[Pos] = {(0, 0)}
    while len(cells) < size:
        r, c = rr.choice(tuple(cells))
        nr, nc = rr.choice([(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)])
        cells.add((nr, nc))
    return _normalize(cells)


def _edge_offsets_for_piece(sig: ShapeSig, h: int, w: int, edge: str) -> list[Pos]:
    max_r = max(r for r, _ in sig)
    max_c = max(c for _, c in sig)
    out: list[Pos] = []
    if edge == "top":
        for dc in range(0, w - max_c):
            out.append((0, dc))
    elif edge == "bottom":
        base_r = h - 1 - max_r
        for dc in range(0, w - max_c):
            out.append((base_r, dc))
    elif edge == "left":
        for dr in range(0, h - max_r):
            out.append((dr, 0))
    elif edge == "right":
        base_c = w - 1 - max_c
        for dr in range(0, h - max_r):
            out.append((dr, base_c))
    return out


def _can_place(
    cells: set[Pos],
    occupied: set[Pos],
    h: int,
    w: int,
    *,
    forbid_adj4: bool,
) -> bool:
    for r, c in cells:
        if not _in_bounds(h, w, r, c):
            return False
        if (r, c) in occupied:
            return False
    if forbid_adj4:
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in occupied:
                    return False
    return True


def _place_piece_on_edge(
    rr: random.Random,
    sig: ShapeSig,
    edge: str,
    occupied: set[Pos],
    h: int,
    w: int,
) -> set[Pos] | None:
    offsets = _edge_offsets_for_piece(sig, h, w, edge)
    rr.shuffle(offsets)
    for dr, dc in offsets:
        cells = _translate(sig, dr, dc)
        if _can_place(cells, occupied, h, w, forbid_adj4=True):
            return cells
    return None


def _place_interior_shape(
    rr: random.Random,
    sig: ShapeSig,
    occupied: set[Pos],
    h: int,
    w: int,
) -> set[Pos] | None:
    max_r = max(r for r, _ in sig)
    max_c = max(c for _, c in sig)
    if h - 2 - max_r < 1 or w - 2 - max_c < 1:
        return None
    candidates: list[Pos] = []
    for dr in range(1, h - 1 - max_r):
        for dc in range(1, w - 1 - max_c):
            candidates.append((dr, dc))
    rr.shuffle(candidates)
    for dr, dc in candidates:
        cells = _translate(sig, dr, dc)
        if _can_place(cells, occupied, h, w, forbid_adj4=True):
            return cells
    return None


def generate(
    rng: random.Random | None = None,
    *,
    min_h: int = 11,
    max_h: int = 16,
    min_w: int = 11,
    max_w: int = 16,
) -> dict[str, Grid]:
    rr = rng or random.Random()

    for _ in range(1400):
        h = rr.randint(min_h, max_h)
        w = rr.randint(min_w, max_w)
        inp = _empty(h, w)

        color = rr.choice([1, 2, 3, 4, 6, 7, 8, 9])
        n_pieces = rr.randint(2, 3)
        piece_sigs = [_random_polyomino(rr, rr.randint(1, 4)) for _ in range(n_pieces)]
        assemblies = sorted(_all_assemblies(piece_sigs))
        if len(assemblies) < 2 or len(assemblies) > 6:
            continue
        if any(len(sig) < 4 or len(sig) > 12 for sig in assemblies):
            continue

        occupied: set[Pos] = set()
        edges = ["top", "bottom", "left", "right"]
        rr.shuffle(edges)
        piece_edges = [edges[i % len(edges)] for i in range(n_pieces)]
        rr.shuffle(piece_edges)

        edge_cells_all: list[set[Pos]] = []
        ok = True
        for sig, edge in zip(piece_sigs, piece_edges):
            placed = _place_piece_on_edge(rr, sig, edge, occupied, h, w)
            if placed is None:
                ok = False
                break
            edge_cells_all.append(placed)
            occupied |= placed
        if not ok:
            continue

        # Draw colored edge fragments.
        for cells in edge_cells_all:
            for r, c in cells:
                inp[r][c] = color

        # Place a sampled subset of assembly interpretations.
        n_target_modes = _sample_mode_count(rr, len(assemblies))
        selected = rr.sample(assemblies, n_target_modes)
        match_count = 0
        for sig in selected:
            placed = _place_interior_shape(rr, sig, occupied, h, w)
            if placed is None:
                continue
            for r, c in placed:
                inp[r][c] = GREY
            occupied |= placed
            match_count += 1
        if match_count < len(selected):
            continue
        # Optionally add a few duplicate matches from random assembly modes.
        extra_matches = rr.randint(0, 2)
        for _ in range(extra_matches):
            sig = rr.choice(selected)
            placed = _place_interior_shape(rr, sig, occupied, h, w)
            if placed is None:
                continue
            for r, c in placed:
                inp[r][c] = GREY
            occupied |= placed
            match_count += 1
        if match_count == 0:
            continue

        # Place non-matching grey distractors.
        n_distractors = rr.randint(2, 5)
        for _ in range(n_distractors):
            for _inner in range(20):
                cand = _random_polyomino(rr, rr.randint(1, 10))
                if _normalize(cand) in assemblies:
                    continue
                placed = _place_interior_shape(rr, cand, occupied, h, w)
                if placed is None:
                    continue
                for r, c in placed:
                    inp[r][c] = GREY
                occupied |= placed
                break

        out = transform_t16(inp)
        if out == inp:
            continue
        if not any(out[r][c] == color and inp[r][c] == GREY for r in range(h) for c in range(w)):
            continue
        return {"input": inp, "output": out}

    raise RuntimeError("failed to sample valid t16 instance")
