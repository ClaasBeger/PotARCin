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


def _readings(grid, cap=2):
  """Counts (up to cap) ways the grey shapes can fill the bites in the bar.

  Every shape slides up into the bar, and together they have to fill its bites
  exactly -- that is the whole puzzle.  Telling the shapes apart by the part of
  each that shows in the bar is not enough: two shapes with different footprints
  can still swap places along the bar, leaving the bar looking the same while the
  parts hanging below it land somewhere else.  A picture like that has two right
  answers, and only pictures with exactly one are worth drawing.
  """
  height = len(grid)
  width = len(grid[0])
  deep = [r for r in range(height) if common.red() in grid[r]]
  bar = {(r, c) for r in deep for c in range(width)}
  empty = {(r, c) for r in range(height) for c in range(width) if not grid[r][c]}
  bites = bar & empty
  deepest = deep[-1] if deep else -1

  seen = [[False] * width for _ in range(height)]
  shapes = []
  for row in range(height):
    for col in range(width):
      if seen[row][col] or grid[row][col] != common.gray(): continue
      stack, cells = [(row, col)], []
      seen[row][col] = True
      while stack:
        r, c = stack.pop()
        cells.append((r, c))
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
          if not (0 <= r + dr < height and 0 <= c + dc < width): continue
          if seen[r + dr][c + dc] or grid[r + dr][c + dc] != common.gray(): continue
          seen[r + dr][c + dc] = True
          stack.append((r + dr, c + dc))
      top, left = min(r for r, _ in cells), min(c for _, c in cells)
      shapes.append(frozenset((r - top, c - left) for r, c in cells))

  # Where each shape could sit: on empty cells only, and biting into the bar.
  places = []
  for shape in shapes:
    tall = max(r for r, _ in shape) + 1
    wide = max(c for _, c in shape) + 1
    spots = []
    for dr in range(-tall + 1, deepest + 1):
      for dc in range(-wide + 1, width):
        put = frozenset((r + dr, c + dc) for r, c in shape)
        if any(not (0 <= r < height and 0 <= c < width) for r, c in put): continue
        if put - empty or not put & bar or (put & bar) - bites: continue
        spots.append(put)
    places.append(spots)

  order = sorted(range(len(shapes)), key=lambda i: len(places[i]))
  answers = set()

  def deal(idx, used):
    if len(answers) >= cap: return
    if idx == len(order):
      if used & bar == bites: answers.add(used)
      return
    for put in places[order[idx]]:
      if put & used: continue
      deal(idx + 1, used | put)

  deal(0, frozenset())
  return len(answers)


def generate(rows=None, cols=None, idxs=None, bluerows=None, bluecols=None,
             grayrows=None, graycols=None, width=15, height=10):
  """Returns input and output grids according to the given parameters.

  Args:
    rows: a list of vertical coordinates where pixels should be placed
    cols: a list of horizontal coordinates where pixels should be placed
    idxs: a list of indices into the sprite list
    bluerows: a list of vertical coordinates where blue boxes should be placed
    bluecols: a list of horizontal coordinates where blue boxes should be placed
    grayrows: a list of vertical coordinates where gray boxes should be placed
    graycols: a list of horizontal coordinates where gray boxes should be placed
    width: the width of the grid
    height: the height of the grid
  """

  if rows is None:
    while True:
      # The number of creatures has to be resampled along with everything else:
      # four creatures have to be packed into a single 15-wide strip twice over
      # (once in blue, once in gray), and a count drawn once up front left the
      # loop spinning for tens of thousands of iterations.
      num_shapes = max(3, common.randint(0, 4))
      # Choose the dimensions and contents of the creatures.
      wides = [common.randint(1, 4) for _ in range(num_shapes)]
      talls = [common.randint(2, 4) for _ in range(num_shapes)]
      sizes = [common.randint(max(2, wide * tall // 2), wide * tall)
               for wide, tall in zip(wides, talls)]
      creatures = [common.continuous_creature(size, wide, tall)
                   for size, wide, tall in zip(sizes, wides, talls)]
      # Choose the blue locations of the creatures.  The creatures always
      # overlap vertically, so this is a one-dimensional packing that is nearly
      # always satisfiable yet rarely hit on the first try -- retry the columns
      # on their own instead of rebuilding the creatures every time.
      bluerows = [common.randint(1, 2) for _ in range(num_shapes)]
      for _ in range(200):
        bluecols = [common.randint(0, width - wide) for wide in wides]
        # Make sure the creatures don't overlap.
        if not common.overlaps(bluerows, bluecols, wides, talls, 1): break
      else:
        continue
      # Extract the rows and columns, along with the creatures actual heights.
      rows, cols, idxs, max_rows = [], [], [], []
      for idx, creature in enumerate(creatures):
        rows.extend(p[0] for p in creature)
        cols.extend(p[1] for p in creature)
        idxs.extend([idx] * len(creature))
        max_rows.append(max(p[0] for p in creature))
      grayrows = [height - max_row - 1 for max_row in max_rows]
      for _ in range(200):
        graycols = [common.randint(0, width - wide) for wide in wides]
        if not common.overlaps(grayrows, graycols, wides, talls, 1): break
      else:
        continue
      # Make sure the footprints are unique (so we can match them exactly).
      footprints = []
      for creature in creatures:
        one_deep = [p for p in creature if p[0] < 1]
        two_deep = [p for p in creature if p[0] < 2]
        one_deep.sort()
        two_deep.sort()
        footprints.append([one_deep, two_deep])
      # Both the convexity of the red bar and the uniqueness of the footprints
      # depend on the creatures and on how deeply they sit in the bar, but not
      # on where they sit along it, so reroll the depths on their own rather
      # than rebuilding the creatures for every collision.
      for _ in range(50):
        bluerows = [common.randint(1, 2) for _ in range(num_shapes)]
        if common.overlaps(bluerows, bluecols, wides, talls, 1): continue
        # Draw the shapes, and make sure the red bar remains convex.
        illegal = False
        grid = common.grid(width, height)
        for r in range(3):
          for c in range(width):
            grid[r][c] = common.red()
        for r, c, idx in zip(rows, cols, idxs):
          grid[r + bluerows[idx]][c + bluecols[idx]] = common.black()
        for r in range(1, height):
          for c in range(width):
            if grid[r][c] == common.red() and grid[r - 1][c] == common.black():
              illegal = True
        for i, creature in enumerate(creatures):
          footprint = [p for p in creature if p[0] < 3 - bluerows[i]]
          footprint.sort()
          for j in range(len(creatures)):
            if i != j and footprint in footprints[j]: illegal = True
        if not illegal: break
      else:
        continue
      # Different footprints still leave the shapes free to swap places along
      # the bar, which fills the bar the same way but hangs the tails below it
      # somewhere else; such a picture has two right answers.  Keep only the
      # pictures that can be read one way -- all three official examples can.
      grid = common.grid(width, height)
      for r in range(3):
        for c in range(width):
          grid[r][c] = common.red()
      for r, c, idx in zip(rows, cols, idxs):
        grid[r + grayrows[idx]][c + graycols[idx]] = common.gray()
        grid[r + bluerows[idx]][c + bluecols[idx]] = common.black()
      if _readings(grid) != 1: continue
      break

  grid, output = common.grids(width, height)
  for r in range(3):
    for c in range(width):
      output[r][c] = grid[r][c] = common.red()
  for r, c, idx in zip(rows, cols, idxs):
    grid[r + grayrows[idx]][c + graycols[idx]] = common.gray()
    grid[r + bluerows[idx]][c + bluecols[idx]] = common.black()
    output[r + bluerows[idx]][c + bluecols[idx]] = common.blue()
  return {"input": grid, "output": output}


def validate():
  """Validates the generator."""
  train = [
      generate(rows=[0, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 0, 1, 2, 3],
               cols=[0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0, 0, 0],
               idxs=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
               bluerows=[1, 2, 1],
               bluecols=[1, 6, 14],
               grayrows=[7, 8, 6],
               graycols=[6, 1, 13]),
      generate(rows=[0, 0, 1, 1, 1, 0, 1, 2, 0, 1, 1, 1, 0, 0, 1, 1, 2, 2, 2, 2,
                     3, 3],
               cols=[1, 2, 0, 1, 2, 0, 0, 0, 1, 0, 1, 2, 1, 2, 1, 2, 0, 1, 2, 3,
                     1, 2],
               idxs=[0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
                     3, 3],
               bluerows=[2, 1, 1, 1],
               bluecols=[0, 4, 7, 10],
               grayrows=[8, 7, 8, 6],
               graycols=[12, 10, 6, 0]),
  ]
  test = [
      generate(rows=[0, 1, 1, 1, 2, 2, 2, 2, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0,
                     1, 1, 2, 3, 3],
               cols=[0, 0, 1, 2, 0, 1, 2, 3, 0, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
                     0, 1, 0, 0, 1],
               idxs=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
                     2, 2, 2, 2, 2],
               bluerows=[1, 1, 1],
               bluecols=[11, 6, 1],
               grayrows=[7, 6, 6],
               graycols=[1, 11, 7]),
  ]
  return {"train": train, "test": test}
