"""Generator for Test2/t20: space-invader style ship destruction.

Rule:
- Alien ships are foreground components (non-zero, non-gray, non-pink).
- Pink cells (6) are shots fired vertically upward.
- Shots are processed by shooter row from top to bottom.
- Within the same row, shot order is chosen to maximize total ships destroyed
  eventually (looking ahead to lower rows). Deterministic tie-break: smallest
  lexicographic column-order permutation.
- One shot destroys one whole ship (its entire connected component).
- Gray cannon (5) at the bottom remains unchanged.
"""

from __future__ import annotations

import itertools
import random
from collections import defaultdict
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Tuple

Grid = List[List[int]]
Pos = Tuple[int, int]

BG = 0
SHOT = 6
CANNON = 5

def _empty(h: int, w: int) -> Grid:
    return [[BG for _ in range(w)] for _ in range(h)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _extract_scene(inp: Grid) -> tuple[list[set[Pos]], dict[int, list[tuple[int, int]]], set[Pos], dict[int, int]]:
    """Return ships, shots grouped by row, cannon cells, and ship color.

    A "ship" is a maximal horizontal run of alien color in a single row.
    (Rows are independent ship lanes; vertical touching does not merge ships.)
    """
    h = len(inp)
    w = len(inp[0]) if h else 0
    ships: list[set[Pos]] = []
    ship_color: dict[int, int] = {}
    for r in range(h):
        c = 0
        while c < w:
            v = inp[r][c]
            if v in (BG, SHOT, CANNON):
                c += 1
                continue
            c0 = c
            while c + 1 < w and inp[r][c + 1] == v:
                c += 1
            comp = {(r, cc) for cc in range(c0, c + 1)}
            sid = len(ships)
            ships.append(comp)
            ship_color[sid] = v
            c += 1

    shots_by_row: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r in range(h):
        for c in range(w):
            if inp[r][c] == SHOT:
                shots_by_row[r].append((r, c))

    cannon_cells = {(r, c) for r in range(h) for c in range(w) if inp[r][c] == CANNON}
    if not cannon_cells:
        raise ValueError("missing cannon cells")

    return ships, dict(shots_by_row), cannon_cells, ship_color


def _build_col_targets(ships: Sequence[set[Pos]], w: int) -> dict[int, list[tuple[int, int]]]:
    """For each column, sorted (bottommost_row_of_ship_in_col, ship_id) desc.

    A vertical shot hits the nearest surviving ship above it, so we compare
    against each ship's deepest (largest) row in that column.
    """
    col_to_ship_row: list[dict[int, int]] = [dict() for _ in range(w)]
    for sid, comp in enumerate(ships):
        max_row_in_col: dict[int, int] = {}
        for r, c in comp:
            prev = max_row_in_col.get(c)
            if prev is None or r > prev:
                max_row_in_col[c] = r
        for c, r in max_row_in_col.items():
            col_to_ship_row[c][sid] = r

    out: dict[int, list[tuple[int, int]]] = {}
    for c in range(w):
        pairs = sorted(((r, sid) for sid, r in col_to_ship_row[c].items()), reverse=True)
        out[c] = pairs
    return out


def _shot_hit_ship(col_targets: dict[int, list[tuple[int, int]]], alive_mask: int, rb: int, c: int) -> int | None:
    for r_ship, sid in col_targets.get(c, []):
        if r_ship >= rb:
            continue
        if (alive_mask >> sid) & 1:
            return sid
    return None


def _simulate_row_perm(
    col_targets: dict[int, list[tuple[int, int]]],
    alive_mask: int,
    rb: int,
    cols_perm: Sequence[int],
) -> tuple[int, int]:
    """Return (new_alive_mask, destroyed_count_in_this_row)."""
    mask = alive_mask
    destroyed = 0
    for c in cols_perm:
        sid = _shot_hit_ship(col_targets, mask, rb, c)
        if sid is None:
            continue
        if (mask >> sid) & 1:
            mask &= ~(1 << sid)
            destroyed += 1
    return mask, destroyed


def _resolve_shots_optimal(
    ships: Sequence[set[Pos]],
    shots_by_row: dict[int, list[tuple[int, int]]],
    w: int,
) -> tuple[int, int]:
    """Return (final_alive_mask, number_of_optimal_global_orders).

    number_of_optimal_global_orders is capped at 2 (1 means unique optimum).
    """
    n = len(ships)
    alive0 = (1 << n) - 1
    col_targets = _build_col_targets(ships, w)
    rows = sorted(shots_by_row.keys())
    row_cols = [tuple(c for _, c in shots_by_row[r]) for r in rows]

    # Unique permutations per row (handle duplicate columns defensively).
    row_perms: list[list[tuple[int, ...]]] = []
    for cols in row_cols:
        perms = sorted(set(itertools.permutations(cols)))
        row_perms.append(perms)

    @lru_cache(maxsize=None)
    def dp(i: int, alive_mask: int) -> tuple[int, int]:
        if i == len(rows):
            return 0, 1
        rb = rows[i]
        best = -1
        ways = 0
        for perm in row_perms[i]:
            nxt_mask, drow = _simulate_row_perm(col_targets, alive_mask, rb, perm)
            drest, wrest = dp(i + 1, nxt_mask)
            total = drow + drest
            if total > best:
                best = total
                ways = wrest
            elif total == best:
                ways = min(2, ways + wrest)
        return best, ways

    def choose(i: int, alive_mask: int) -> int:
        if i == len(rows):
            return alive_mask
        rb = rows[i]
        best, _ = dp(i, alive_mask)
        # Deterministic tie-break: lexicographically smallest permutation.
        for perm in row_perms[i]:
            nxt_mask, drow = _simulate_row_perm(col_targets, alive_mask, rb, perm)
            drest, _ = dp(i + 1, nxt_mask)
            if drow + drest == best:
                return choose(i + 1, nxt_mask)
        raise AssertionError("unreachable")

    _, ways = dp(0, alive0)
    final_alive = choose(0, alive0)
    return final_alive, ways


def _count_tie_rows_along_optimal(
    ships: Sequence[set[Pos]],
    shots_by_row: dict[int, list[tuple[int, int]]],
    w: int,
) -> int:
    """Count row-level ties along the chosen optimal trajectory.

    For each shooter row on the lexicographically chosen optimal path, count a
    tie if more than one column-order permutation can still achieve the global
    optimum from that state.
    """
    n = len(ships)
    alive0 = (1 << n) - 1
    col_targets = _build_col_targets(ships, w)
    rows = sorted(shots_by_row.keys())
    row_cols = [tuple(c for _, c in shots_by_row[r]) for r in rows]
    row_perms: list[list[tuple[int, ...]]] = []
    for cols in row_cols:
        perms = sorted(set(itertools.permutations(cols)))
        row_perms.append(perms)

    @lru_cache(maxsize=None)
    def dp(i: int, alive_mask: int) -> tuple[int, int]:
        if i == len(rows):
            return 0, 1
        rb = rows[i]
        best = -1
        ways = 0
        for perm in row_perms[i]:
            nxt_mask, drow = _simulate_row_perm(col_targets, alive_mask, rb, perm)
            drest, wrest = dp(i + 1, nxt_mask)
            total = drow + drest
            if total > best:
                best = total
                ways = wrest
            elif total == best:
                ways = min(2, ways + wrest)
        return best, ways

    tie_rows = 0
    i = 0
    alive = alive0
    while i < len(rows):
        rb = rows[i]
        best, _ = dp(i, alive)
        optimal_perms: list[tuple[int, ...]] = []
        for perm in row_perms[i]:
            nxt_mask, drow = _simulate_row_perm(col_targets, alive, rb, perm)
            drest, _ = dp(i + 1, nxt_mask)
            if drow + drest == best:
                optimal_perms.append(perm)
        if len(optimal_perms) > 1:
            tie_rows += 1
        # Deterministic continuation: lexicographically smallest optimal perm.
        chosen = optimal_perms[0]
        alive, _ = _simulate_row_perm(col_targets, alive, rb, chosen)
        i += 1
    return tie_rows


def transform_t20(inp: Grid) -> Grid:
    if not inp or not inp[0]:
        raise ValueError("input grid must be non-empty")
    h = len(inp)
    w = len(inp[0])
    if any(len(row) != w for row in inp):
        raise ValueError("input grid must be rectangular")

    ships, shots_by_row, cannon_cells, ship_color = _extract_scene(inp)
    if not ships:
        raise ValueError("no ships found")
    if not shots_by_row:
        raise ValueError("no shots found")

    alive_mask, _ways = _resolve_shots_optimal(ships, shots_by_row, w)

    out = _empty(h, w)
    for sid, comp in enumerate(ships):
        if ((alive_mask >> sid) & 1) == 0:
            continue
        col = ship_color[sid]
        for r, c in comp:
            out[r][c] = col
    for r, c in cannon_cells:
        out[r][c] = CANNON
    return out


def _cannon_shape(h: int, w: int, rr: random.Random) -> set[Pos]:
    c = rr.randint(2, w - 3)
    r_base = h - 1
    cells = {(r_base, c - 1), (r_base, c), (r_base, c + 1), (r_base - 1, c)}
    return cells


def _place_ships(rr: random.Random, h: int, w: int, color: int) -> list[set[Pos]]:
    ships: list[set[Pos]] = []
    used: set[Pos] = set()
    n_rows = rr.randint(3, 4)
    row_candidates = list(range(0, max(1, h // 2)))
    rr.shuffle(row_candidates)
    row_bases = sorted(row_candidates[:n_rows])
    for rb in row_bases:
        nships = rr.randint(2, 4)
        starts = list(range(0, w - 1))
        rr.shuffle(starts)
        placed = 0
        for c0 in starts:
            if placed >= nships:
                break
            width = rr.randint(2, min(4, w - c0))
            comp = {(rb, c) for c in range(c0, c0 + width)}
            # occasionally use a 2-row "chunky" ship
            if rr.random() < 0.3 and rb + 1 < h // 2:
                comp |= {(rb + 1, c) for c in range(c0, c0 + width) if rr.random() < 0.6}
            if any((r, c) in used for r, c in comp):
                continue
            # keep some spacing to avoid merges
            if any(abs(r - rr0) <= 0 and abs(c - cc0) <= 0 for r, c in comp for rr0, cc0 in used):
                continue
            ships.append(comp)
            used |= comp
            placed += 1
    # Ensure enough ships.
    if len(ships) < 6:
        raise ValueError("too few ships")
    return ships


def _ship_cells(ships: Iterable[set[Pos]]) -> set[Pos]:
    out: set[Pos] = set()
    for s in ships:
        out |= s
    return out


def _place_shots(
    rr: random.Random,
    h: int,
    w: int,
    ship_cells: set[Pos],
    cannon: set[Pos],
) -> set[Pos]:
    shot_rows = list(range(h // 2, h - 2))
    if not shot_rows:
        raise ValueError("grid too short")
    rr.shuffle(shot_rows)
    n_rows = rr.randint(3, min(5, len(shot_rows)))
    rows = sorted(shot_rows[:n_rows])
    shots: set[Pos] = set()
    for rb in rows:
        # Prefer one-shot rows; two-shot rows are rarer (ties become rarer).
        k = 1 if rr.random() < 0.85 else 2
        cols = list(range(w))
        rr.shuffle(cols)
        chosen = 0
        for c in cols:
            p = (rb, c)
            if p in cannon or p in shots:
                continue
            # Prefer columns that can hit at least one ship above.
            if not any((r, c) in ship_cells for r in range(rb)):
                if rr.random() < 0.85:
                    continue
            shots.add(p)
            chosen += 1
            if chosen >= k:
                break
    if len(shots) < 4:
        raise ValueError("too few shots")
    return shots


def generate(
    rng: random.Random | None = None,
    *,
    min_h: int = 10,
    max_h: int = 14,
    min_w: int = 10,
    max_w: int = 14,
) -> dict[str, Grid]:
    rr = rng or random.Random()

    for _ in range(2500):
        h = rr.randint(min_h, max_h)
        w = rr.randint(min_w, max_w)
        ship_color = rr.choice([1, 2, 3, 4, 7, 8, 9])
        inp = _empty(h, w)

        try:
            ships = _place_ships(rr, h, w, ship_color)
        except ValueError:
            continue
        for comp in ships:
            for r, c in comp:
                inp[r][c] = ship_color

        cannon = _cannon_shape(h, w, rr)
        if any(inp[r][c] != BG for r, c in cannon):
            continue
        for r, c in cannon:
            inp[r][c] = CANNON

        try:
            shots = _place_shots(rr, h, w, _ship_cells(ships), cannon)
        except ValueError:
            continue
        if any(inp[r][c] != BG for r, c in shots):
            continue
        for r, c in shots:
            inp[r][c] = SHOT

        # Enforce unique optimal resolution to avoid ambiguous transformations.
        scene_ships, shots_by_row, _cannon_cells, _ship_color = _extract_scene(inp)
        alive_mask, ways = _resolve_shots_optimal(scene_ships, shots_by_row, w)
        if ways != 1:
            continue
        tie_rows = _count_tie_rows_along_optimal(scene_ships, shots_by_row, w)
        # Keep some tie cases, but make them uncommon.
        if tie_rows >= 2:
            continue
        if tie_rows == 1 and rr.random() < 0.8:
            continue
        destroyed = len(scene_ships) - (alive_mask.bit_count())
        if destroyed < max(3, len(shots) // 2):
            continue

        try:
            out = transform_t20(inp)
        except ValueError:
            continue
        if out == inp:
            continue
        return {"input": inp, "output": out}

    raise RuntimeError("failed to sample valid t20 instance")
