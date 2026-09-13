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


def generate(rows=None, cols=None, idxs=None, brows=None, bcols=None,
             bidxs=None, size=10):
  """Returns input and output grids according to the given parameters.

  Args:
    rows: a list of vertical coordinates where pixels should be placed
    cols: a list of horizontal coordinates where pixels should be placed
    idxs: a list of indices into the sprite list
    brows: a list of vertical coordinates where boxes should be placed
    bcols: a list of horizontal coordinates where boxes should be placed
    bidxs: a list of indices into the sprite list
    size: the width and height of the (square) grid
  """
  if rows is None:
    width = common.randint(3, 5)
    # Two of the widest sprites can't sit side by side on the grid, so they have
    # to stack -- which only leaves room for them if they aren't too tall.
    height = common.randint(2, 4 if 2 * (width + 1) <= size else 3)
    wides, talls, bidxs = [width, width, width - 1], [height] * 3, [0, 0, 1]
    while True:
      brows = [common.randint(0, size - tall) for tall in talls]
      bcols = [common.randint(0, size - wide) for wide in wides]
      if not common.overlaps(brows, bcols, wides, talls, 1): break
    while True:  # Keep trying until the smaller sprite stays in one piece.
      rows, cols, idxs = [], [], []
      # Sometimes make a box.
      if common.randint(0, 1):
        for r in range(height):
          for c in range(width):
            if r not in [0, height - 1] and c not in [0, width - 1]: continue
            rows.append(r)
            cols.append(c)
            idxs.append(0)
      # Otherwise, make a creature (and use up each of the columns!)
      else:
        num_pixels = common.randint(width * height // 2, width * height)
        while True:
          pixels = common.continuous_creature(num_pixels, width, height)
          if max([p[1] for p in pixels]) + 1 == width: break
        rows.extend([p[0] for p in pixels])
        cols.extend([p[1] for p in pixels])
        idxs.extend([0] * len(pixels))
      # Pick some "bad" column to remove from the sprite to create a smaller
      # one; the smaller sprite has to stay contiguous, just like the big one.
      smaller = None
      for badcol in common.shuffle(list(range(width))):
        pixels = [(r, c if c < badcol else c - 1)
                  for r, c in zip(rows, cols) if c != badcol]
        if common.connected(pixels):
          smaller = pixels
          break
      if smaller is None: continue
      rows.extend([p[0] for p in smaller])
      cols.extend([p[1] for p in smaller])
      idxs.extend([1] * len(smaller))
      break

  grid, output = common.grids(size, size)
  for brow, bcol, bidx in zip(brows, bcols, bidxs):
    for row, col, idx in zip(rows, cols, idxs):
      if bidx != idx: continue
      grid[brow + row][bcol + col] = common.cyan()
      output[brow + row][bcol + col] = common.red() if idx else common.blue()
  return {"input": grid, "output": output}


def validate():
  """Validates the generator."""
  train = [
      generate(rows=[0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 0, 0, 0, 1, 1, 2, 2, 2],
               cols=[0, 1, 2, 3, 0, 3, 0, 1, 2, 3, 0, 1, 2, 0, 2, 0, 1, 2],
               idxs=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
               brows=[1, 2, 7],
               bcols=[7, 1, 5],
               bidxs=[1, 0, 0]),
      generate(rows=[0, 0, 0, 0, 1, 1, 2, 2, 2, 0, 0, 0, 0, 1, 1, 2, 2],
               cols=[0, 1, 2, 3, 2, 3, 2, 3, 4, 0, 1, 2, 3, 2, 3, 2, 3],
               idxs=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
               brows=[1, 2, 6], bcols=[6, 1, 3], bidxs=[1, 0, 0]),
      generate(rows=[0, 0, 0, 1, 1, 0, 0, 1, 1],
               cols=[0, 1, 2, 0, 2, 0, 1, 0, 1],
               idxs=[0, 0, 0, 0, 0, 1, 1, 1, 1],
               brows=[1, 5, 6], bcols=[1, 0, 3], bidxs=[0, 1, 0]),
  ]
  test = [
      generate(rows=[0, 0, 0, 1, 2, 3, 3, 3, 3, 0, 0, 1, 2, 3, 3, 3],
               cols=[0, 1, 2, 2, 1, 0, 1, 2, 3, 0, 1, 1, 0, 0, 1, 2],
               idxs=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
               brows=[1, 1, 6], bcols=[1, 6, 3], bidxs=[1, 0, 0]),
  ]
  return {"train": train, "test": test}
