"""Generator for t31: mirror 2x2 blocks across a green divider unless blocked.

Rule:
- The grid is split into equal left/right halves by a full-height green vertical
  line.
- Each non-green object is a solid 2x2 block.
- Each block has a corresponding 2x2 target position on the opposite half at the
  same row and same within-half column.
- A block moves to its target unless that target would cause a collision.
- Collision is determined from target cells: if a target overlaps any original
  block cell, or if multiple blocks have overlapping targets, the involved block
  stays in its original position. All other blocks move.
- The divider is preserved.

Generation constraints:
- all non-green objects are 2x2 blocks;
- at least one block moves and at least one block stays in most examples;
- both halves have the same dimensions;
- generated outputs contain no overlapping moved blocks.
"""

from __future__ import annotations

import json
import random
from collections import deque
from typing import Dict, Iterable, List, Sequence, Set, Tuple

Grid = List[List[int]]
Cell = Tuple[int, int]
GREEN = 3
BG = 0
COLORS = [1, 2, 4, 5, 6, 7, 8, 9]


class Block:
    def __init__(self, idx: int, color: int, cells: Set[Cell], top: int, left: int, side: str, rel_col: int):
        self.idx = idx
        self.color = color
        self.cells = set(cells)
        self.top = top
        self.left = left
        self.side = side
        self.rel_col = rel_col


def _empty(h: int, w: int, fill: int = BG) -> Grid:
    return [[fill for _ in range(w)] for _ in range(h)]


def _rectangular(grid: Grid) -> bool:
    return bool(grid) and bool(grid[0]) and all(len(row) == len(grid[0]) for row in grid)


def _find_divider(grid: Grid) -> int:
    h, w = len(grid), len(grid[0])
    divs = [c for c in range(w) if all(grid[r][c] == GREEN for r in range(h))]
    if len(divs) != 1:
        raise ValueError('expected exactly one full-height green divider')
    div = divs[0]
    if div * 2 + 1 != w:
        raise ValueError('halves must have equal width')
    return div


def _component_cells(grid: Grid) -> list[tuple[int, Set[Cell]]]:
    h, w = len(grid), len(grid[0])
    seen: Set[Cell] = set()
    comps: list[tuple[int, Set[Cell]]] = []
    for r in range(h):
        for c in range(w):
            color = grid[r][c]
            if color in (BG, GREEN) or (r, c) in seen:
                continue
            q = deque([(r, c)])
            seen.add((r, c))
            cells: Set[Cell] = set()
            while q:
                x, y = q.popleft()
                cells.add((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and (nx, ny) not in seen and grid[nx][ny] == color:
                        seen.add((nx, ny))
                        q.append((nx, ny))
            comps.append((color, cells))
    return comps


def _parse_blocks(grid: Grid) -> tuple[int, list[Block]]:
    if not _rectangular(grid):
        raise ValueError('input grid must be non-empty and rectangular')
    div = _find_divider(grid)
    blocks: list[Block] = []
    for idx, (color, cells) in enumerate(_component_cells(grid)):
        if len(cells) != 4:
            raise ValueError('each object must be a solid 2x2 block')
        rs = sorted({r for r, _ in cells})
        cs = sorted({c for _, c in cells})
        if len(rs) != 2 or len(cs) != 2 or rs[1] != rs[0] + 1 or cs[1] != cs[0] + 1:
            raise ValueError('each object must be a solid 2x2 block')
        if cells != {(rs[0], cs[0]), (rs[0], cs[1]), (rs[1], cs[0]), (rs[1], cs[1])}:
            raise ValueError('malformed block')
        if div in cs:
            raise ValueError('block overlaps divider')
        if cs[1] < div:
            side = 'L'
            rel_col = cs[0]
        elif cs[0] > div:
            side = 'R'
            rel_col = cs[0] - div - 1
        else:
            raise ValueError('block crosses divider')
        blocks.append(Block(idx, color, cells, rs[0], cs[0], side, rel_col))
    return div, blocks


def _target_cells(block: Block, div: int) -> Set[Cell]:
    if block.side == 'L':
        left = div + 1 + block.rel_col
    else:
        left = block.rel_col
    r = block.top
    return {(r, left), (r, left + 1), (r + 1, left), (r + 1, left + 1)}


def transform_t31(grid: Grid) -> Grid:
    div, blocks = _parse_blocks(grid)
    h, w = len(grid), len(grid[0])
    original_occ: Dict[Cell, int] = {}
    for b in blocks:
        for cell in b.cells:
            original_occ[cell] = b.idx
    targets = {b.idx: _target_cells(b, div) for b in blocks}

    blocked: Set[int] = set()
    # Any target overlapping an original block causes that moving block to stay.
    for b in blocks:
        if any(cell in original_occ for cell in targets[b.idx]):
            blocked.add(b.idx)
    # If multiple targets overlap, all involved blocks stay.
    cell_to_targeters: Dict[Cell, list[int]] = {}
    for b in blocks:
        for cell in targets[b.idx]:
            cell_to_targeters.setdefault(cell, []).append(b.idx)
    for ids in cell_to_targeters.values():
        if len(ids) > 1:
            blocked.update(ids)

    out = _empty(h, w)
    for r in range(h):
        out[r][div] = GREEN
    placed: Dict[Cell, int] = {}
    for b in blocks:
        dest = b.cells if b.idx in blocked else targets[b.idx]
        for cell in dest:
            if cell in placed:
                raise ValueError('output collision in malformed input')
            placed[cell] = b.color
    for (r, c), color in placed.items():
        out[r][c] = color
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _cells_for_block(top: int, left: int) -> Set[Cell]:
    return {(top, left), (top, left + 1), (top + 1, left), (top + 1, left + 1)}


def _can_place(cells: Set[Cell], occupied: Set[Cell], h: int, w: int, gap: bool = True) -> bool:
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in occupied:
            return False
        if gap:
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    if (rr, cc) in occupied:
                        return False
    return True


def _place_block(grid: Grid, top: int, left: int, color: int) -> None:
    for r, c in _cells_for_block(top, left):
        grid[r][c] = color


def _add_pair_collision(grid: Grid, occupied: Set[Cell], h: int, half: int, rng: random.Random, colors: list[int]) -> bool:
    """Add one left and one right block at matching coordinates, so both stay."""
    candidates = [(r, c) for r in range(h - 1) for c in range(half - 1)]
    rng.shuffle(candidates)
    div = half
    for top, rel in candidates:
        lcells = _cells_for_block(top, rel)
        rcells = _cells_for_block(top, div + 1 + rel)
        if _can_place(lcells, occupied, h, 2 * half + 1) and _can_place(rcells, occupied, h, 2 * half + 1):
            c1 = colors.pop() if colors else rng.choice(COLORS)
            c2 = colors.pop() if colors else rng.choice(COLORS)
            _place_block(grid, top, rel, c1)
            _place_block(grid, top, div + 1 + rel, c2)
            occupied.update(lcells | rcells)
            return True
    return False


def _add_moving_block(grid: Grid, occupied: Set[Cell], h: int, half: int, rng: random.Random, colors: list[int]) -> bool:
    div = half
    candidates: list[tuple[str, int, int]] = []
    for side in ('L', 'R'):
        for top in range(h - 1):
            for rel in range(half - 1):
                left = rel if side == 'L' else div + 1 + rel
                target_left = div + 1 + rel if side == 'L' else rel
                candidates.append((side, top, rel))
    rng.shuffle(candidates)
    for side, top, rel in candidates:
        left = rel if side == 'L' else div + 1 + rel
        target_left = div + 1 + rel if side == 'L' else rel
        cells = _cells_for_block(top, left)
        target = _cells_for_block(top, target_left)
        if _can_place(cells, occupied, h, 2 * half + 1) and not (target & occupied):
            color = colors.pop() if colors else rng.choice(COLORS)
            _place_block(grid, top, left, color)
            occupied.update(cells)
            return True
    return False


def generate(rng: random.Random | None = None) -> dict[str, Grid]:
    rng = rng or random.Random()
    for _ in range(5000):
        h = rng.randint(6, 14)
        half = rng.randint(5, 9)
        w = 2 * half + 1
        grid = _empty(h, w)
        for r in range(h):
            grid[r][half] = GREEN
        occupied: Set[Cell] = {(r, half) for r in range(h)}
        colors = COLORS[:]
        rng.shuffle(colors)

        # Ensure at least one collision pair and at least one moving block.
        if not _add_pair_collision(grid, occupied, h, half, rng, colors):
            continue
        if not _add_moving_block(grid, occupied, h, half, rng, colors):
            continue
        # Add a few more blocks, with a mix biased toward movable blocks.
        for _k in range(rng.randint(1, 6)):
            if rng.random() < 0.35:
                _add_pair_collision(grid, occupied, h, half, rng, colors)
            else:
                _add_moving_block(grid, occupied, h, half, rng, colors)

        try:
            out = transform_t31(grid)
        except Exception:
            continue
        if out == grid:
            continue
        # Require both moved and stayed blocks in generated examples.
        _, blocks = _parse_blocks(grid)
        moved = stayed = 0
        div = half
        for b in blocks:
            if any(out[r][c] == b.color for r, c in _target_cells(b, div)) and not all(out[r][c] == b.color for r, c in b.cells):
                moved += 1
            else:
                stayed += 1
        if moved == 0 or stayed == 0:
            continue
        return {'input': grid, 'output': out}
    raise RuntimeError('Failed to generate valid t31 example')


if __name__ == '__main__':
    print(json.dumps(generate(rng=random.Random(0)), indent=2))
