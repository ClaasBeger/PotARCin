"""Generator for Test2/t16: edge fragments assemble to a target grey shape.

Orthogonally-connected task-color islands are fused transitively whenever they king-touch,
so diagonal contact defines a single prototype (no special-case ridge split).

Interior grey is matched on 4-connectivity only. Generator rejects layouts where grey clues
would be ambiguous (king-only bridges between disjoint grey components, grey king-adjacent
to task colour).

Rigid assemblies union **every permutation** of prototype placement order (same translations,
orthogonal contacts only): anchoring lex-first signature alone skipped valid tilings reachable
only when another piece seeds the incremental placement chain.

"""

from __future__ import annotations

import math
import random
from collections import deque
from itertools import permutations
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


def _neighbors8(r: int, c: int) -> Iterable[Pos]:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield (r + dr, c + dc)


def _orthogonal_adjacent_cells(a: set[Pos], b: set[Pos]) -> bool:
    for r, c in a:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in b:
                return True
    return False


def _comps_king_touch(a: set[Pos], b: set[Pos]) -> bool:
    for r, c in a:
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in b:
                return True
    return False


def _merge_colored_fragments_king_linked(color_frags: Sequence[set[Pos]]) -> list[set[Pos]]:
    """Fuse pairwise king-touching orthogonal color components into rigid prototypes."""
    n = len(color_frags)
    if n <= 1:
        return [*color_frags]

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if _comps_king_touch(color_frags[i], color_frags[j]):
                union(i, j)

    buckets: dict[int, set[Pos]] = {}
    for i in range(n):
        r = find(i)
        buckets.setdefault(r, set()).update(color_frags[i])
    return list(buckets.values())


def _assemble_signatures_ordered(
    ordered: Sequence[ShapeSig],
    *,
    max_states: int,
    seen_states: set[tuple[int, ShapeSig]],
    out: set[ShapeSig],
) -> None:
    """DFS one placement order (ordered[0] anchors first). Mutates ``seen_states``, ``out``."""
    first = set(ordered[0])
    rest = ordered[1:]
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
            if not any((nr, nc) in union for r, c in placed for nr, nc in _neighbors4(r, c)):
                continue
            rec(idx + 1, union | placed)

    rec(0, first)


def _assemble_signatures(piece_sigs: Sequence[ShapeSig], *, max_states: int = 500_000) -> set[ShapeSig]:
    """Enumerate unions over translations and **all orders** of snapping pieces onto the union."""
    pieces = tuple(piece_sigs)
    if not pieces:
        return set()

    n = len(pieces)
    fac = math.factorial(n)
    per_budget = max(max_states // fac, min(80_000, max_states))

    out: set[ShapeSig] = set()
    for perm in permutations(pieces):
        seen_perm: set[tuple[int, ShapeSig]] = set()
        _assemble_signatures_ordered(perm, max_states=per_budget, seen_states=seen_perm, out=out)
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
    """Recolor interior orth-grey blobs whose shape equals a rigid orthogonal assembly."""
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

    raw_color = _extract_components(inp, color)
    prototypes = _merge_colored_fragments_king_linked(raw_color)
    if len(prototypes) < 2:
        raise ValueError("expected at least two colored prototypes after diagonal fusion")

    piece_sigs = [_normalize(comp) for comp in prototypes]
    target_sigs = _all_assemblies(piece_sigs)

    out = [row[:] for row in inp]
    for r in range(h):
        for c in range(w):
            if out[r][c] == color:
                out[r][c] = BG

    grey_comps = _extract_components(inp, GREY)
    for comp in grey_comps:
        if _normalize(comp) not in target_sigs:
            continue
        for r, c in comp:
            out[r][c] = color
    return out


def _grey_has_king_only_bridge(inp: Grid) -> bool:
    """True if two distinct orth-grey components touch at a diagonal only."""
    comps = _extract_components(inp, GREY)
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if _comps_king_touch(comps[i], comps[j]) and not _orthogonal_adjacent_cells(comps[i], comps[j]):
                return True
    return False


def _grey_king_adjacent_to_task_color(inp: Grid, task_color: int) -> bool:
    h, w = len(inp), len(inp[0])
    for r in range(h):
        for c in range(w):
            if inp[r][c] != GREY:
                continue
            for nr, nc in _neighbors8(r, c):
                if _in_bounds(h, w, nr, nc) and inp[nr][nc] == task_color:
                    return True
    return False


def _random_polyomino(rr: random.Random, size: int) -> ShapeSig:
    cells: set[Pos] = {(0, 0)}
    while len(cells) < size:
        r, c = rr.choice(tuple(cells))
        nr, nc = rr.choice([(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)])
        cells.add((nr, nc))
    return _normalize(cells)


def _biased_poly_sizes(rr: random.Random, n: int, *, bare: bool) -> list[int]:
    """Prefer triomino/tetromino-sized lumps (authored tasks rarely look like domino salads)."""
    choices = [2, 3, 4, 5]
    weights = [0.06, 0.38, 0.44, 0.12] if bare else [0.05, 0.33, 0.42, 0.20]
    return [rr.choices(choices, weights=weights, k=1)[0] for _ in range(n)]


def _on_perimeter(r: int, c: int, h: int, w: int) -> bool:
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _enumerate_diagonal_singleton_chains_on_perimeter(h: int, w: int, length: int) -> list[set[Pos]]:
    """Straight lattice-diagonal chains (train-style ridges / corner pairs), all cells on boundary."""
    assert length >= 2
    dirs = ((1, 1), (1, -1), (-1, 1), (-1, -1))
    raw: list[set[Pos]] = []
    for r in range(h):
        for c in range(w):
            if not _on_perimeter(r, c, h, w):
                continue
            for dr, dc in dirs:
                cells: list[Pos] = []
                ok = True
                for k in range(length):
                    rr_, cc_ = r + k * dr, c + k * dc
                    if not _in_bounds(h, w, rr_, cc_) or not _on_perimeter(rr_, cc_, h, w):
                        ok = False
                        break
                    cells.append((rr_, cc_))
                if ok and len(set(cells)) == length:
                    raw.append(set(cells))
    seen: set[tuple[Pos, ...]] = set()
    uniq: list[set[Pos]] = []
    for s in raw:
        sig = tuple(sorted(s))
        if sig not in seen:
            seen.add(sig)
            uniq.append(s)
    return uniq


def _try_place_diagonal_singleton_chain(
    rr: random.Random,
    occupied: set[Pos],
    h: int,
    w: int,
    *,
    length: int,
) -> set[Pos] | None:
    cand = _enumerate_diagonal_singleton_chains_on_perimeter(h, w, length)
    rr.shuffle(cand)
    for chain in cand:
        acc = set(occupied)
        ok = True
        for cell in sorted(chain):
            solo = {cell}
            if not _can_place(solo, acc, h, w, forbid_adj4=True):
                ok = False
                break
            acc |= solo
        if ok:
            return chain
    return None


def _raw_has_king_diagonal_bridge(frags: Sequence[set[Pos]]) -> bool:
    """Some orth-islands touch king-only (diagonal cue), matching authored motifs."""
    for i in range(len(frags)):
        for j in range(i + 1, len(frags)):
            if _comps_king_touch(frags[i], frags[j]) and not _orthogonal_adjacent_cells(frags[i], frags[j]):
                return True
    return False


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
    forbid_king_near: set[Pos] | None = None,
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
    if forbid_king_near:
        ban = forbid_king_near
        for r, c in cells:
            for nr, nc in _neighbors8(r, c):
                if _in_bounds(h, w, nr, nc) and (nr, nc) in ban:
                    return False
    return True


def _place_piece_on_edge(
    rr: random.Random,
    sig: ShapeSig,
    edge: str,
    occupied: set[Pos],
    h: int,
    w: int,
    *,
    forbid_king_near: set[Pos] | None = None,
) -> set[Pos] | None:
    offsets = _edge_offsets_for_piece(sig, h, w, edge)
    rr.shuffle(offsets)
    for dr, dc in offsets:
        cells = _translate(sig, dr, dc)
        if _can_place(cells, occupied, h, w, forbid_adj4=True, forbid_king_near=forbid_king_near):
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

    sketches = ["bare", "diag_pair", "diag_triple", "mixed"]
    # Bias toward perimeter diagonal ridges/pairs like authored train examples.
    sketch_w = [0.12, 0.31, 0.31, 0.26]

    for _ in range(22000):
        h = rr.randint(min_h, max_h)
        w = rr.randint(min_w, max_w)
        inp = _empty(h, w)

        color = rr.choice([1, 2, 3, 4, 6, 7, 8, 9])
        sketch = rr.choices(sketches, weights=sketch_w, k=1)[0]
        if sketch == "mixed":
            sketch = "diag_triple" if rr.random() < 0.55 else "diag_pair"

        occupied: set[Pos] = set()
        forbid_king_near: set[Pos] | None = None

        if sketch == "diag_triple":
            ridge = _try_place_diagonal_singleton_chain(rr, occupied, h, w, length=3)
            if ridge is None:
                continue
            occupied |= ridge
            forbid_king_near = set(ridge)
            n_poly_extra = rr.randint(1, 2)
            sizes = _biased_poly_sizes(rr, n_poly_extra, bare=False)
        elif sketch == "diag_pair":
            ridge = _try_place_diagonal_singleton_chain(rr, occupied, h, w, length=2)
            if ridge is None:
                continue
            occupied |= ridge
            forbid_king_near = set(ridge)
            n_poly_extra = rr.randint(1, 2)
            sizes = _biased_poly_sizes(rr, n_poly_extra, bare=False)
        else:
            # Orth-edge lumps only (still forbid orthogonal overlap); mostly tri/tet pentomino mixes.
            n_poly_extra = rr.randint(2, 3)
            sizes = _biased_poly_sizes(rr, n_poly_extra, bare=True)

        piece_sigs = [_random_polyomino(rr, sz) for sz in sizes]

        edges = ["top", "bottom", "left", "right"]
        rr.shuffle(edges)
        piece_edges = [edges[i % len(edges)] for i in range(len(piece_sigs))]
        rr.shuffle(piece_edges)

        ok = True
        for sig, edge in zip(piece_sigs, piece_edges):
            placed = _place_piece_on_edge(
                rr,
                sig,
                edge,
                occupied,
                h,
                w,
                forbid_king_near=forbid_king_near,
            )
            if placed is None:
                ok = False
                break
            occupied |= placed
        if not ok:
            continue

        for r in range(h):
            for c in range(w):
                if (r, c) in occupied:
                    inp[r][c] = color

        raw_frags = _extract_components(inp, color)
        if sketch in ("diag_pair", "diag_triple"):
            if not _raw_has_king_diagonal_bridge(raw_frags):
                continue
        elif sketch == "bare":
            if _raw_has_king_diagonal_bridge(raw_frags):
                continue

        prototypes = _merge_colored_fragments_king_linked(raw_frags)
        if len(prototypes) < 2:
            continue
        merged_sigs = [_normalize(p) for p in prototypes]
        try:
            assemblies_set = _all_assemblies(merged_sigs)
        except ValueError:
            continue
        assemblies = sorted(assemblies_set)
        if len(assemblies) < 2 or len(assemblies) > 420:
            continue
        if any(len(sig) < 4 or len(sig) > 12 for sig in assemblies):
            continue

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
                if _normalize(cand) in assemblies_set:
                    continue
                placed = _place_interior_shape(rr, cand, occupied, h, w)
                if placed is None:
                    continue
                for r, c in placed:
                    inp[r][c] = GREY
                occupied |= placed
                break

        if _grey_has_king_only_bridge(inp) or _grey_king_adjacent_to_task_color(inp, color):
            continue

        out = transform_t16(inp)
        if out == inp:
            continue
        if not any(out[r][c] == color and inp[r][c] == GREY for r in range(h) for c in range(w)):
            continue
        return {"input": inp, "output": out}

    raise RuntimeError("failed to sample valid t16 instance")
