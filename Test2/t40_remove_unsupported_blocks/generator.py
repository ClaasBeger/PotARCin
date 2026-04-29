"""Generator for Test2/t40: remove red-marked hollow blocks and unsupported blocks.

Rule:
- The input contains several hollow rectangular blocks.
- Some blocks are marked red by filling the entire interior of the block with
  red cells (color 2).
- In the output, every red-marked block is removed.
- Then any remaining block that is no longer supported is also removed,
  recursively.
- A block is supported if either:
    * its bottom edge lies on the bottom row of the grid, or
    * there exists at least one remaining block whose top edge is exactly one
      row below its bottom edge, with overlapping horizontal span.

Generation constraints used to keep examples well-formed and unambiguous:
- every object is a single hollow axis-aligned rectangle with size at least 3x3;
- blocks do not overlap; blocks on the same level are separated by gaps;
- colors are unique across blocks in generated samples to simplify perception;
- the full interior of each marked block is filled red;
- support relations form a layered DAG, so the cascade is unambiguous.
"""

from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

Grid = List[List[int]]

BG = 0
RED = 2
USABLE_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]


@dataclass(frozen=True)
class Block:
    color: int
    r0: int
    r1: int
    c0: int
    c1: int
    marked: bool

    @property
    def width(self) -> int:
        return self.c1 - self.c0 + 1

    @property
    def height(self) -> int:
        return self.r1 - self.r0 + 1


def _empty(h: int, w: int) -> Grid:
    return [[BG for _ in range(w)] for _ in range(h)]


def _rectangular(grid: Grid) -> bool:
    return bool(grid) and bool(grid[0]) and all(len(row) == len(grid[0]) for row in grid)


def _outline_cells(r0: int, r1: int, c0: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _connected_components_of_color(grid: Grid, color: int) -> list[set[tuple[int, int]]]:
    h, w = len(grid), len(grid[0])
    seen: set[tuple[int, int]] = set()
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] != color or (r, c) in seen:
                continue
            q = deque([(r, c)])
            seen.add((r, c))
            comp: set[tuple[int, int]] = set()
            while q:
                rr, cc = q.popleft()
                comp.add((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and grid[nr][nc] == color:
                        seen.add((nr, nc))
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_blocks(grid: Grid) -> list[Block]:
    if not _rectangular(grid):
        raise ValueError("grid must be non-empty and rectangular")
    h, w = len(grid), len(grid[0])
    blocks: list[Block] = []

    colors = sorted({v for row in grid for v in row if v not in (BG, RED)})
    for color in colors:
        for comp in _connected_components_of_color(grid, color):
            rs = [r for r, _ in comp]
            cs = [c for _, c in comp]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
                raise ValueError("every block must be at least 3x3")
            outline = _outline_cells(r0, r1, c0, c1)
            if comp != outline:
                raise ValueError("non-red component is not a hollow rectangle outline")
            marked = any(grid[r][c] == RED for r in range(r0 + 1, r1) for c in range(c0 + 1, c1))
            blocks.append(Block(color=color, r0=r0, r1=r1, c0=c0, c1=c1, marked=marked))

    # Validate that every red cell lies inside exactly one block interior.
    red_positions = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == RED]
    for r, c in red_positions:
        owners = [b for b in blocks if b.r0 < r < b.r1 and b.c0 < c < b.c1]
        if len(owners) != 1:
            raise ValueError("every red cell must lie in the interior of exactly one block")
    return blocks


def _supports(lower: Block, upper: Block) -> bool:
    if lower.r0 != upper.r1 + 1:
        return False
    return max(lower.c0, upper.c0) <= min(lower.c1, upper.c1)


def transform_t40(grid: Grid) -> Grid:
    blocks = _parse_blocks(grid)
    h, w = len(grid), len(grid[0])

    removed: set[int] = {i for i, b in enumerate(blocks) if b.marked}
    changed = True
    while changed:
        changed = False
        for i, blk in enumerate(blocks):
            if i in removed:
                continue
            if blk.r1 == h - 1:
                continue  # rests on ground
            supported = any(j not in removed and _supports(blocks[j], blk) for j in range(len(blocks)) if j != i)
            if not supported:
                removed.add(i)
                changed = True

    out = _empty(h, w)
    for i, blk in enumerate(blocks):
        if i in removed:
            continue
        for r, c in _outline_cells(blk.r0, blk.r1, blk.c0, blk.c1):
            out[r][c] = blk.color
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _intervals_nonoverlapping(intervals: list[tuple[int, int]]) -> bool:
    intervals = sorted(intervals)
    return all(intervals[i][1] < intervals[i + 1][0] for i in range(len(intervals) - 1))


def _choose_red_cells(blk: Block, rng: random.Random) -> list[tuple[int, int]]:
    del rng  # marking is deterministic: mark the whole interior
    return [(r, c) for r in range(blk.r0 + 1, blk.r1) for c in range(blk.c0 + 1, blk.c1)]


def _render_blocks(h: int, w: int, blocks: list[Block], red_cells_by_idx: dict[int, list[tuple[int, int]]]) -> Grid:
    grid = _empty(h, w)
    for idx, blk in enumerate(blocks):
        for r, c in _outline_cells(blk.r0, blk.r1, blk.c0, blk.c1):
            if grid[r][c] != BG:
                raise ValueError("block overlap in render")
            grid[r][c] = blk.color
        for r, c in red_cells_by_idx.get(idx, []):
            if grid[r][c] != BG:
                raise ValueError("red-cell overlap in render")
            grid[r][c] = RED
    return grid


def generate(
    rng: random.Random | None = None,
    h: int | None = None,
    w: int | None = None,
    num_levels: int | None = None,
) -> dict[str, Grid]:
    rng = rng or random.Random()

    for _attempt in range(4000):
        hh = h if h is not None else rng.randint(12, 22)
        ww = w if w is not None else rng.randint(12, 24)
        levels = num_levels if num_levels is not None else rng.randint(2, 4)
        if not (6 <= hh <= 30 and 6 <= ww <= 30):
            continue

        blocks: list[Block] = []
        colors = USABLE_COLORS[:]
        rng.shuffle(colors)
        color_idx = 0

        # Level 0: base blocks on the ground.
        level_blocks: list[list[int]] = []
        current_level_indices: list[int] = []
        num_base = rng.randint(1, 3)
        x = rng.randint(0, 2)
        intervals: list[tuple[int, int]] = []
        base_top_rows: list[int] = []
        for _ in range(num_base):
            max_base_width = min(8, ww - x - 1)
            if max_base_width < 4:
                break
            width = rng.randint(4, max_base_width)
            height = rng.randint(3, 6)
            c0, c1 = x, x + width - 1
            r1 = hh - 1
            r0 = r1 - height + 1
            if r0 < 0:
                break
            intervals.append((c0, c1))
            if not _intervals_nonoverlapping(intervals):
                intervals.pop()
                break
            blk = Block(color=colors[color_idx], r0=r0, r1=r1, c0=c0, c1=c1, marked=False)
            color_idx += 1
            blocks.append(blk)
            current_level_indices.append(len(blocks) - 1)
            base_top_rows.append(r0)
            x = c1 + rng.randint(1, 3) + 1
            if x + 3 >= ww or color_idx >= len(colors):
                break
        if not current_level_indices:
            continue
        level_blocks.append(current_level_indices)

        # Higher levels.
        prev_level = current_level_indices
        while len(level_blocks) < levels and color_idx < len(colors):
            parent_top = min(blocks[i].r0 for i in prev_level)
            if parent_top < 4:
                break
            count_here = rng.randint(1, min(2, len(prev_level)))
            new_level: list[int] = []
            used_intervals: list[tuple[int, int]] = []
            support_groups = []
            shuffled = prev_level[:]
            rng.shuffle(shuffled)
            # partition some supports into 1- or 2-block groups
            while shuffled and len(support_groups) < count_here:
                take = 1 if len(shuffled) == 1 or rng.random() < 0.65 else 2
                grp = sorted(shuffled[:take], key=lambda i: blocks[i].c0)
                del shuffled[:take]
                support_groups.append(grp)
            for grp in support_groups:
                sup_r0 = min(blocks[i].r0 for i in grp)
                if len({blocks[i].r0 for i in grp}) != 1:
                    continue
                upper_height = rng.randint(3, min(5, sup_r0 - 1))
                r1 = sup_r0 - 1
                r0 = r1 - upper_height + 1
                if r0 < 0:
                    continue
                union_c0 = min(blocks[i].c0 for i in grp)
                union_c1 = max(blocks[i].c1 for i in grp)
                max_width = min(8, union_c1 - union_c0 + 1)
                if max_width < 3:
                    continue
                width = rng.randint(3, max_width)
                start_min = union_c0
                start_max = union_c1 - width + 1
                if start_min > start_max:
                    continue
                c0 = rng.randint(start_min, start_max)
                c1 = c0 + width - 1
                used_intervals.append((c0, c1))
                if not _intervals_nonoverlapping(used_intervals):
                    used_intervals.pop()
                    continue
                blk = Block(color=colors[color_idx], r0=r0, r1=r1, c0=c0, c1=c1, marked=False)
                color_idx += 1
                blocks.append(blk)
                new_level.append(len(blocks) - 1)
                if color_idx >= len(colors):
                    break
            if not new_level:
                break
            level_blocks.append(new_level)
            prev_level = new_level

        if len(blocks) < 2:
            continue

        # Mark some blocks, preferring non-base blocks but allowing any.
        candidate_ids = list(range(len(blocks)))
        non_ground = [i for i, b in enumerate(blocks) if b.r1 < hh - 1]
        rng.shuffle(candidate_ids)
        rng.shuffle(non_ground)
        marked_ids: set[int] = set()
        if non_ground:
            marked_ids.add(non_ground[0])
        marked_target = rng.randint(1, min(3, max(1, len(blocks) // 2)))
        for i in candidate_ids:
            if len(marked_ids) >= marked_target:
                break
            if rng.random() < 0.35:
                marked_ids.add(i)
        # Create marked versions.
        rendered_blocks: list[Block] = []
        red_cells_by_idx: dict[int, list[tuple[int, int]]] = {}
        for i, b in enumerate(blocks):
            marked = i in marked_ids
            rendered_blocks.append(Block(color=b.color, r0=b.r0, r1=b.r1, c0=b.c0, c1=b.c1, marked=marked))
            if marked:
                red_cells_by_idx[i] = _choose_red_cells(rendered_blocks[-1], rng)

        try:
            inp = _render_blocks(hh, ww, rendered_blocks, red_cells_by_idx)
            out = transform_t40(inp)
        except Exception:
            continue

        if out == inp:
            continue

        # Ensure at least one marked block is removed and, ideally, some unmarked cascade occurs.
        parsed = _parse_blocks(inp)
        initially_marked = {i for i, b in enumerate(parsed) if b.marked}
        remaining_out_colors = {(r, c) for r in range(hh) for c in range(ww) if out[r][c] != BG}
        # Count removed blocks by comparing whether any outline cell remains.
        removed_ids = set()
        for i, b in enumerate(parsed):
            if not any((r, c) in remaining_out_colors for r, c in _outline_cells(b.r0, b.r1, b.c0, b.c1)):
                removed_ids.add(i)
        if not initially_marked <= removed_ids:
            continue
        if len(removed_ids) < len(initially_marked):
            continue
        # Prefer examples with at least one cascade block beyond the marked set.
        if len(removed_ids - initially_marked) == 0 and rng.random() < 0.8:
            continue

        return {"input": inp, "output": out}

    raise RuntimeError("failed to generate a valid t40 sample")


if __name__ == "__main__":
    print(json.dumps(generate(rng=random.Random(0)), indent=2))
