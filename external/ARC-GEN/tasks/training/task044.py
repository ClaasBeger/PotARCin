# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generator."""

import common


def _readings(grid):
  """Counts how many ways the loose shapes can be dropped into the holes.

  A slab is a blob of the commonest colour; the empty cells inside its outline
  are its hole.  Every other colour is a loose shape, matched to a hole by its
  outline alone.  The speckles are a colour too, and once in a long while the
  speckles that happen to land next to each other spell out exactly the outline
  of a hole that a real shape also fits -- then the picture has two answers and
  nothing in it says which is meant.
  """
  height = len(grid)
  width = len(grid[0])
  tally = {}
  for row in grid:
    for cell in row:
      if cell: tally[cell] = tally.get(cell, 0) + 1
  if not tally: return 0
  slab = max(tally, key=lambda cell: tally[cell])

  def outline(cells):
    top, left = min(r for r, _ in cells), min(c for _, c in cells)
    return frozenset((r - top, c - left) for r, c in cells)

  seen = [[False] * width for _ in range(height)]
  holes = []
  for row in range(height):
    for col in range(width):
      if seen[row][col] or grid[row][col] != slab: continue
      stack, cells = [(row, col)], []
      seen[row][col] = True
      while stack:
        r, c = stack.pop()
        cells.append((r, c))
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
          if not (0 <= r + dr < height and 0 <= c + dc < width): continue
          if seen[r + dr][c + dc] or grid[r + dr][c + dc] != slab: continue
          seen[r + dr][c + dc] = True
          stack.append((r + dr, c + dc))
      gaps = [(r, c)
              for r in range(min(r for r, _ in cells), max(r for r, _ in cells) + 1)
              for c in range(min(c for _, c in cells), max(c for _, c in cells) + 1)
              if not grid[r][c]]
      if gaps: holes.append(outline(gaps))

  loose = {}
  for row in range(height):
    for col in range(width):
      cell = grid[row][col]
      if cell and cell != slab: loose.setdefault(cell, []).append((row, col))
  shapes = {color: outline(cells) for color, cells in loose.items()}

  def drop(idx, taken):
    if idx == len(holes): return 1
    found = 0
    for color, shape in shapes.items():
      if color in taken or shape != holes[idx]: continue
      found += drop(idx + 1, taken | {color})
      if found > 1: return found
    return found

  return drop(0, frozenset())


def generate(brows=None, bcols=None, wides=None, talls=None, rows=None,
             cols=None, idxs=None, srows=None, scols=None, colors=None,
             drows=None, dcols=None, dcolor=None, size=10, num_boxes=2):
  """Returns input and output grids according to the given parameters.

  Args:
    brows: a list of vertical coordinates where boxes should be placed
    bcols: a list of horizontal coordinates where boxes should be placed
    wides: a list of widths of boxes
    talls: a list of heights of boxes
    rows: a list of vertical coordinates where pixels should be placed
    cols: a list of horizontal coordinates where pixels should be placed
    idxs: a list of indices of into the boxes
    srows: a list of vertical coordinates where sprites should be placed
    scols: a list of horizontal coordinates where sprites should be placed
    colors: a list of colors of sprites
    drows: a list of vertical coordinates where dust should be placed
    dcols: a list of horizontal coordinates where dust should be placed
    dcolor: the color of pixels
    size: the width and height of the (square) grid
    num_boxes: the number of boxes to place
  """

  def draw(grid, output):
    for brow, bcol, wide, tall in zip(brows, bcols, wides, talls):
      for r in range(brow, brow + tall):
        for c in range(bcol, bcol + wide):
          output[r][c] = grid[r][c] = common.gray()
    for row, col, idx in zip(rows, cols, idxs):
      grid[row + srows[idx]][col + scols[idx]] = colors[idx]
      grid[row + brows[idx] + 1][col + bcols[idx] + 1] = common.black()
      output[row + brows[idx] + 1][col + bcols[idx] + 1] = colors[idx]
    for drow, dcol in zip(drows, dcols):
      output[drow][dcol] = grid[drow][dcol] = dcolor

  if brows is None:
    while True:
      while True:  # Pick a few box heights that fill the space
        talls = [common.randint(3, 5) for _ in range(num_boxes)]
        if sum(talls) in [8, 9]: break
      brows = [0 if sum(talls) == 9 else 1, size - talls[1]]
      while True:  # Pick box widths & sprite locations that don't overlap.
        wides = [common.randint(5, 7) for _ in range(num_boxes)]
        bcols = [common.randint(0, size - wide) for wide in wides]
        swides = [wide - 2 for wide in wides]
        stalls = [tall - 2 for tall in talls]
        while True:  # Pick a unique number of pixels for each sprite.
          areas = [swide * stall for swide, stall in zip(swides, stalls)]
          counts = [common.randint(2, area) for area in areas]
          if len(set(counts)) == num_boxes: break
        # Pick the sprite contents before their places: a creature always starts
        # at its own top left corner, so the only way to leave that corner empty
        # -- as official train example 2 does -- is to slide the creature down and
        # across inside the box afterwards.
        creatures = []
        for idx in range(num_boxes):
          pixels = common.continuous_creature(counts[idx], swides[idx], stalls[idx])
          down = common.randint(0, stalls[idx] - 1 - max(p[0] for p in pixels))
          across = common.randint(0, swides[idx] - 1 - max(p[1] for p in pixels))
          creatures.append([(p[0] + down, p[1] + across) for p in pixels])
        # A sprite may start above or left of the grid by however much of it is
        # empty there, which keeps every one of its cells on the grid.
        srows = [common.randint(-min(p[0] for p in creature), size - stall)
                 for creature, stall in zip(creatures, stalls)]
        scols = [common.randint(-min(p[1] for p in creature), size - swide)
                 for creature, swide in zip(creatures, swides)]
        if not common.overlaps(brows + srows, bcols + scols, wides + swides,
                               talls + stalls): break
      rows, cols, idxs = [], [], []
      for idx in range(num_boxes):
        rows.extend([p[0] for p in creatures[idx]])
        cols.extend([p[1] for p in creatures[idx]])
        idxs.extend([idx] * len(creatures[idx]))
      drows, dcols = [], []
      dcolor = common.random_color(exclude=[common.gray()])
      colors = common.random_colors(2, exclude=[common.gray(), dcolor])
      # Finally, draw the grid, so we can figure our where to place the dust.
      grid, output = common.grids(size, size)
      draw(grid, output)
      for r in range(size):
        for c in range(size):
          if grid[r][c] or output[r][c] or common.randint(0, 9): continue
          drows.append(r)
          dcols.append(c)

      # Speckles can spell out a hole that a real shape also fits; that
      # picture has two answers, so draw another.  All four official
      # examples read exactly one way and are kept.
      grid, output = common.grids(size, size)
      draw(grid, output)
      if _readings(grid) == 1: break

  grid, output = common.grids(size, size)
  draw(grid, output)
  return {"input": grid, "output": output}


def validate():
  """Validates the generator."""
  train = [
      generate(brows=[1, 7], bcols=[1, 5], wides=[5, 5], talls=[5, 3],
               rows=[0, 0, 1, 1, 0, 0], cols=[0, 1, 0, 1, 1, 2],
               idxs=[0, 0, 0, 0, 1, 1], srows=[8, 2], scols=[1, 6],
               colors=[8, 6], drows=[0, 0, 0, 5, 7], dcols=[0, 8, 9, 8, 4],
               dcolor=7),
      generate(brows=[0, 6], bcols=[0, 3], wides=[5, 6], talls=[4, 4],
               rows=[0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1],
               cols=[0, 1, 2, 2, 0, 1, 2, 3, 0, 1, 2, 3],
               idxs=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], srows=[5, 1],
               scols=[0, 6], colors=[3, 9],
               drows=[4, 4, 5, 5, 8, 8, 9, 9],
               dcols=[7, 9, 6, 7, 0, 1, 0, 1],
               dcolor=6),
      generate(brows=[0, 5], bcols=[4, 0], wides=[6, 5], talls=[4, 5],
               rows=[0, 0, 0, 1, 1, 1, 1, 2, 2, 2],
               cols=[0, 1, 2, 2, 3, 0, 1, 0, 1, 2],
               idxs=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1], srows=[8, -1], scols=[6, 0],
               colors=[8, 2], drows=[3, 3, 4, 4, 5, 5, 7],
               dcols=[1, 2, 2, 5, 7, 8, 9], dcolor=4),
  ]
  test = [
      generate(brows=[0, 7], bcols=[2, 3], wides=[5, 7], talls=[5, 3],
               rows=[0, 0, 0, 1, 0, 0, 0], cols=[0, 1, 2, 1, 0, 1, 2],
               idxs=[0, 0, 0, 0, 1, 1, 1], srows=[5, 2], scols=[0, 7],
               colors=[7, 4], drows=[0, 1, 4, 5, 5, 8, 9],
               dcols=[9, 0, 9, 5, 7, 1, 0], dcolor=2),
  ]
  return {"train": train, "test": test}
