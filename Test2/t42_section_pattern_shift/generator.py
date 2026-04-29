"""Generator for Test2/t42: correct shifted color-scheme pattern copies across sections.

Rule:
- The grid is split into equal left/right halves.
- It is also split into several horizontal sections.
- In each section, the left half contains a two-color pattern.
- The right half should contain the *same pattern* but recolored using the two
  colors from the section below (wrapping from bottom to top), except that a
  small number of cells are incorrect.
- The output is the full grid with those right-half mistakes corrected.

Generation constraints used to keep examples well-formed and unambiguous:
- each section's left half uses exactly two colors, and every row in the
  section contains both colors, so the section boundaries are recoverable;
- adjacent sections use different unordered color pairs;
- the right half of each section uses exactly the next section's color pair;
- mistakes are sparse and are only flips between the two colors of that right
  section;
- section sizes and half-widths vary, but the total grid stays within ARC size
  limits.
"""

from __future__ import annotations

import json
import random
from typing import List, Sequence

Grid = List[List[int]]

BG = 0
COLORS = list(range(1, 10))


class Section:
    def __init__(self, start: int, end: int, colors: tuple[int, int]):
        self.start = start
        self.end = end
        self.colors = colors

    @property
    def height(self) -> int:
        return self.end - self.start


def _rectangular(grid: Grid) -> bool:
    return bool(grid) and bool(grid[0]) and all(len(row) == len(grid[0]) for row in grid)


def _group_sections_from_left(grid: Grid) -> list[Section]:
    if not _rectangular(grid):
        raise ValueError("input grid must be non-empty and rectangular")
    h, w = len(grid), len(grid[0])
    if w % 2 != 0:
        raise ValueError("grid width must be even")
    half = w // 2

    row_sets: list[set[int]] = []
    for r in range(h):
        colors = set(grid[r][:half])
        if not (1 <= len(colors) <= 2):
            raise ValueError("each left-half row must use one or two colors")
        row_sets.append(colors)

    sections: list[Section] = []
    start = 0
    while start < h:
        union = set(row_sets[start])
        end = start + 1
        while end < h and len(union | row_sets[end]) <= 2:
            union |= row_sets[end]
            end += 1
        if len(union) != 2:
            raise ValueError("each section must use exactly two colors in total")
        sections.append(Section(start, end, tuple(sorted(union))))
        start = end

    if len(sections) < 2:
        raise ValueError("expected at least two sections")
    return sections


def _render_pattern(bits: list[list[int]], colors: tuple[int, int]) -> Grid:
    a, b = colors
    return [[a if bit == 0 else b for bit in row] for row in bits]


def _section_bits_from_left(grid: Grid, section: Section) -> list[list[int]]:
    half = len(grid[0]) // 2
    a, b = section.colors
    bits: list[list[int]] = []
    for r in range(section.start, section.end):
        row_bits: list[int] = []
        for v in grid[r][:half]:
            if v == a:
                row_bits.append(0)
            elif v == b:
                row_bits.append(1)
            else:
                raise ValueError("left-half row uses a color outside its section pair")
        bits.append(row_bits)
    return bits


def _hamming(a: Grid, b: Grid) -> int:
    return sum(1 for ra, rb in zip(a, b) for va, vb in zip(ra, rb) if va != vb)


def transform_t42(grid: Grid) -> Grid:
    sections = _group_sections_from_left(grid)
    w = len(grid[0])
    half = w // 2
    out = [row[:] for row in grid]

    for i, section in enumerate(sections):
        bits = _section_bits_from_left(grid, section)
        target_colors = sections[(i + 1) % len(sections)].colors
        cand1 = _render_pattern(bits, target_colors)
        cand2 = _render_pattern(bits, (target_colors[1], target_colors[0]))
        actual = [grid[r][half:] for r in range(section.start, section.end)]
        if _hamming(cand1, actual) <= _hamming(cand2, actual):
            corrected = cand1
        else:
            corrected = cand2
        for rr, r in enumerate(range(section.start, section.end)):
            out[r][half:] = corrected[rr]
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _random_binary_pattern(rng: random.Random, h: int, w: int) -> list[list[int]]:
    for _ in range(2000):
        bits = [[1 if rng.random() < 0.5 else 0 for _ in range(w)] for _ in range(h)]
        # Every row uses both colors so row-wise section detection is easy.
        if any(len(set(row)) != 2 for row in bits):
            continue
        # Avoid trivially repetitive patterns.
        if len({tuple(row) for row in bits}) == 1:
            continue
        if len({bits[r][c] for r in range(h) for c in range(w)}) != 2:
            continue
        return bits
    raise RuntimeError("failed to generate a nontrivial binary pattern")


def _flip_cells(grid: Grid, colors: tuple[int, int], num_flips: int, rng: random.Random) -> Grid:
    a, b = colors
    out = [row[:] for row in grid]
    cells = [(r, c) for r in range(len(grid)) for c in range(len(grid[0]))]
    rng.shuffle(cells)
    for r, c in cells[:num_flips]:
        out[r][c] = b if out[r][c] == a else a
    return out


def generate(
    rng: random.Random | None = None,
    num_sections: int | None = None,
    half_width: int | None = None,
    section_heights: Sequence[int] | None = None,
) -> dict[str, Grid]:
    rng = rng or random.Random()

    for _attempt in range(2000):
        n = num_sections if num_sections is not None else rng.randint(2, 4)
        if not (2 <= n <= 4):
            continue
        hw = half_width if half_width is not None else rng.randint(4, 8)
        if hw < 2:
            continue

        if section_heights is not None:
            heights = list(section_heights)
            if len(heights) != n:
                continue
        else:
            heights = [rng.randint(3, 6) for _ in range(n)]
        total_h = sum(heights)
        if total_h > 30 or 2 * hw > 30:
            continue

        pool = COLORS[:]
        rng.shuffle(pool)
        if 2 * n > len(pool):
            continue
        section_pairs: list[tuple[int, int]] = []
        for i in range(n):
            section_pairs.append((pool[2 * i], pool[2 * i + 1]))

        left_sections: list[Grid] = []
        for sec_h in heights:
            left_sections.append(_random_binary_pattern(rng, sec_h, hw))

        inp: Grid = []
        for i in range(n):
            left = _render_pattern(left_sections[i], section_pairs[i])
            target_pair = section_pairs[(i + 1) % n]
            right_clean = _render_pattern(left_sections[i], target_pair)
            area = heights[i] * hw
            max_flips = max(1, min(3, area // 5))
            flips = rng.randint(1, max_flips)
            right = _flip_cells(right_clean, target_pair, flips, rng)
            for lrow, rrow in zip(left, right):
                inp.append(lrow + rrow)

        out = transform_t42(inp)
        if out == inp:
            continue

        sections = _group_sections_from_left(inp)
        half = len(inp[0]) // 2
        if not all(
            any(inp[r][half + c] != out[r][half + c] for r in range(sec.start, sec.end) for c in range(hw))
            for sec in sections
        ):
            continue

        return {"input": inp, "output": out}

    raise RuntimeError("failed to generate a valid t42 sample")


if __name__ == "__main__":
    print(json.dumps(generate(rng=random.Random(0)), indent=2))
