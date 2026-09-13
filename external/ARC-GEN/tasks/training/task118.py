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


def _crosses_that_fit(reds, nonzero, height, width, length):
  """Every cross of this arm length that the picture could be showing.

  A cross whose center sits near an edge shows only the part that fits, so the
  cells looked at are the ones inside the grid; a cross has to lie on drawn
  cells, and has to explain at least one red cell to be worth reading.
  """
  crosses = []
  for row in range(height):
    for col in range(width):
      cells = [(row, col)]
      for delta in range(1, length + 1):
        cells += [(row - delta, col), (row + delta, col),
                  (row, col - delta), (row, col + delta)]
      seen = frozenset(cell for cell in cells
                       if 0 <= cell[0] < height and 0 <= cell[1] < width)
      if not seen & reds: continue
      if seen - nonzero: continue
      crosses.append(seen)
  return crosses


def _readings(grid, height, width, budget=200000):
  """The repairs that the picture admits, as sets of cells to turn cyan.

  Every cross is drawn in the same two colors: red where it crossed empty space
  and cyan where it crossed static, and the input shows that cyan as static
  again. So the red cells are the whole of the evidence, and a reading is a set
  of crosses of one common arm length, each lying on drawn cells, together
  covering the red cells, none of them redundant.
  """
  reds = frozenset((row, col) for row in range(height) for col in range(width)
                   if grid[row][col] == 2)
  nonzero = frozenset((row, col) for row in range(height)
                      for col in range(width) if grid[row][col])
  readings, calls = set(), [0]
  for length in (2, 3):
    crosses = _crosses_that_fit(reds, nonzero, height, width, length)
    def cover(left, drawn):
      calls[0] += 1
      if calls[0] > budget: raise RuntimeError("too many readings")
      if not left:
        for idx, cross in enumerate(drawn):
          rest = frozenset().union(frozenset(),
                                   *[d for j, d in enumerate(drawn) if j != idx])
          if not (cross & reds) - rest: return  # a cross nothing needs
        cells = frozenset().union(frozenset(), *drawn)
        readings.add(frozenset(cell for cell in cells
                               if grid[cell[0]][cell[1]] != 2))
        return
      first = min(left)
      for cross in crosses:
        if first in cross: cover(left - cross, drawn + [cross])
    cover(reds, [])
  return readings


def _readable(grid, height, width):
  """Whether the picture names exactly one repair, the one that was drawn.

  A cross that happened to fall entirely on static left no red cell behind and
  so cannot be seen at all, and a set of red cells that a different arm length
  or different centers would explain just as well leaves the repair undecided.
  Either way the picture has two answers, so it is not worth asking about.
  """
  drawn = frozenset((row, col) for row in range(height) for col in range(width)
                    if grid[row][col] == 8)
  try:
    readings = _readings(grid, height, width)
  except RuntimeError:
    return False
  return readings == {drawn}


def generate(width=None, height=None, colors=None):
  """Returns input and output grids according to the given parameters.

  Args:
    width: the width of the input grid
    height: the height of the input grid
    colors: a list of colors to be used in the input grid
  """
  if width is None:
    # The input hides every cyan cell as static again, so only the red cells
    # say where the crosses are. Keep drawing until they say it just once.
    while True:
      height = common.randint(10, 25)
      width = height + common.randint(-3, 3)
      # First, create random static.
      pixels = common.random_pixels(width, height)
      rows, cols = zip(*pixels)
      grid = common.grid(width, height)
      for r, c in zip(rows, cols):
        grid[r][c] = common.gray()
      # Second, create a few crosses underneath.
      length = common.randint(2, 3)
      rows, cols, lens = [], [], []
      for _ in range(4):
        r = common.randint(length, height - length - 1)
        c = common.randint(length, width - length - 1)
        if common.overlaps(rows + [r], cols + [c], lens + [2 * length],
                           lens + [2 * length], 2): continue
        rows.append(r)
        cols.append(c)
        lens.append(2 * length)
        for idx in range(-length, length + 1):
          if grid[r][c + idx]:
            grid[r][c + idx] = common.cyan()
          else:
            grid[r][c + idx] = common.red()
          if grid[r + idx][c]:
            grid[r + idx][c] = common.cyan()
          else:
            grid[r + idx][c] = common.red()
      if _readable(grid, height, width): break
    # Finally, flatten the colors into a list.
    colors = []
    for r in range(height):
      for c in range(width):
        colors.append(grid[r][c])

  grid, output = common.grids(width, height)
  for r in range(height):
    for c in range(width):
      output[r][c] = colors[r * width + c]
      if colors[r * width + c] != common.cyan():
        grid[r][c] = colors[r * width + c]
      else:
        grid[r][c] = common.gray()

  return {"input": grid, "output": output}


def validate():
  """Validates the generator."""
  train = [
      generate(width=22, height=20,
               colors=[0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 5,
                       0, 0, 0, 0, 5, 0, 0, 5, 5, 5, 5, 0, 0, 5, 0, 5, 0, 5, 0,
                       0, 5, 0, 0, 5, 5, 5, 0, 5, 5, 0, 5, 5, 5, 0, 0, 5, 5, 2,
                       0, 0, 0, 0, 0, 0, 0, 5, 0, 5, 0, 0, 5, 5, 0, 0, 0, 5, 0,
                       0, 0, 2, 5, 5, 5, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 5, 0,
                       0, 0, 5, 0, 0, 2, 5, 5, 0, 0, 5, 0, 5, 5, 0, 0, 5, 0, 0,
                       5, 0, 0, 0, 5, 2, 8, 2, 8, 8, 8, 2, 5, 0, 5, 0, 0, 0, 0,
                       5, 5, 0, 5, 0, 0, 0, 0, 0, 5, 0, 2, 5, 0, 0, 5, 0, 0, 5,
                       5, 5, 0, 0, 0, 0, 0, 5, 0, 0, 0, 5, 5, 0, 2, 5, 0, 5, 5,
                       0, 5, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 5, 0, 0, 0, 2, 0,
                       0, 0, 0, 0, 0, 5, 0, 5, 5, 0, 0, 5, 0, 0, 0, 0, 0, 5, 5,
                       0, 5, 5, 0, 0, 0, 0, 0, 5, 0, 0, 0, 5, 0, 5, 0, 5, 5, 5,
                       5, 5, 0, 0, 0, 0, 5, 0, 5, 5, 5, 0, 5, 5, 0, 5, 5, 0, 0,
                       8, 0, 0, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 0, 5, 5, 0, 0,
                       5, 5, 0, 2, 5, 5, 5, 0, 0, 5, 0, 0, 0, 0, 0, 5, 5, 0, 0,
                       0, 5, 0, 5, 0, 0, 8, 5, 5, 0, 0, 0, 0, 0, 5, 0, 0, 5, 5,
                       0, 0, 0, 5, 0, 0, 2, 8, 8, 2, 2, 2, 2, 0, 0, 0, 5, 5, 0,
                       5, 0, 0, 5, 0, 5, 0, 0, 5, 5, 0, 0, 8, 5, 0, 5, 0, 0, 5,
                       0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 5, 0, 0, 0, 5, 2, 0, 5, 5,
                       0, 5, 0, 0, 5, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 5, 5, 0, 2,
                       5, 0, 0, 0, 5, 0, 0, 0, 5, 5, 0, 0, 0, 5, 5, 5, 0, 5, 5,
                       5, 0, 0, 0, 5, 5, 5, 5, 0, 0, 5, 5, 0, 5, 0, 0, 0, 5, 5,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 5, 0, 5, 0,
                       0, 0, 5]),
      generate(width=20, height=20,
               colors=[0, 5, 0, 5, 0, 0, 0, 5, 5, 0, 5, 5, 0, 0, 0, 5, 5, 0, 5,
                       5, 5, 5, 0, 5, 5, 5, 0, 5, 5, 0, 5, 0, 0, 5, 0, 0, 0, 5,
                       5, 0, 0, 5, 0, 5, 5, 0, 8, 5, 0, 5, 0, 0, 5, 0, 0, 5, 0,
                       0, 5, 5, 5, 0, 0, 5, 5, 0, 2, 5, 0, 5, 0, 5, 0, 0, 0, 5,
                       5, 5, 5, 5, 0, 5, 0, 5, 2, 8, 2, 2, 2, 0, 5, 5, 0, 5, 0,
                       5, 5, 0, 0, 0, 5, 5, 0, 0, 5, 5, 2, 5, 5, 5, 0, 5, 0, 0,
                       5, 5, 0, 0, 0, 0, 0, 0, 5, 5, 0, 0, 8, 5, 0, 0, 5, 5, 0,
                       0, 5, 0, 0, 5, 0, 5, 0, 0, 0, 5, 0, 5, 0, 5, 5, 5, 0, 5,
                       5, 5, 0, 0, 5, 5, 0, 5, 5, 0, 0, 0, 5, 0, 0, 5, 5, 5, 5,
                       0, 5, 5, 8, 0, 0, 5, 0, 5, 5, 0, 0, 5, 0, 5, 5, 5, 0, 5,
                       5, 0, 5, 0, 8, 5, 5, 5, 5, 5, 5, 0, 5, 5, 0, 5, 5, 5, 5,
                       5, 0, 5, 2, 8, 2, 2, 2, 0, 0, 5, 0, 0, 5, 0, 0, 0, 0, 0,
                       0, 5, 5, 5, 0, 0, 8, 0, 0, 5, 0, 5, 0, 0, 5, 0, 0, 5, 0,
                       5, 5, 0, 5, 5, 5, 5, 8, 5, 5, 5, 5, 0, 5, 5, 0, 0, 5, 5,
                       0, 8, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 5, 5, 0, 0, 0,
                       5, 5, 8, 0, 5, 5, 0, 5, 5, 5, 5, 0, 0, 5, 5, 0, 0, 5, 0,
                       5, 8, 8, 2, 2, 8, 5, 0, 0, 5, 0, 0, 5, 5, 0, 0, 0, 5, 5,
                       0, 0, 5, 5, 2, 5, 0, 5, 5, 0, 0, 5, 0, 5, 5, 0, 0, 0, 0,
                       5, 0, 5, 0, 5, 8, 0, 5, 5, 5, 0, 0, 5, 0, 0, 0, 5, 0, 0,
                       0, 5, 0, 5, 5, 0, 5, 5, 5, 0, 5, 5, 5, 0, 5, 0, 0, 5, 5,
                       5, 5, 5, 0, 5, 0, 5, 0, 5, 5, 0, 0, 5, 5, 0, 0, 0, 0, 0,
                       5]),
      generate(width=19, height=18,
               colors=[0, 0, 5, 0, 5, 0, 5, 5, 5, 5, 0, 5, 5, 0, 0, 0, 5, 5, 0,
                       0, 0, 5, 5, 5, 0, 5, 5, 5, 5, 0, 0, 5, 5, 5, 5, 5, 0, 5,
                       0, 5, 5, 5, 0, 5, 0, 5, 5, 0, 0, 0, 5, 5, 5, 0, 5, 0, 0,
                       5, 5, 5, 5, 5, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 5, 0, 0,
                       5, 5, 0, 0, 0, 5, 5, 5, 0, 5, 5, 8, 5, 0, 0, 0, 5, 0, 0,
                       5, 0, 0, 0, 0, 0, 5, 0, 5, 0, 5, 2, 5, 0, 0, 5, 0, 5, 5,
                       5, 0, 5, 0, 0, 5, 5, 0, 5, 2, 2, 8, 2, 2, 5, 5, 0, 5, 0,
                       0, 8, 0, 5, 5, 5, 5, 5, 0, 5, 0, 8, 5, 5, 5, 0, 5, 5, 5,
                       5, 8, 5, 0, 5, 5, 5, 5, 0, 0, 5, 2, 5, 5, 5, 0, 0, 0, 0,
                       8, 2, 2, 8, 0, 0, 5, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0,
                       5, 2, 5, 5, 5, 0, 0, 5, 5, 5, 5, 0, 0, 5, 5, 5, 5, 0, 5,
                       0, 2, 5, 0, 5, 5, 0, 0, 5, 5, 5, 5, 5, 0, 5, 5, 5, 5, 0,
                       0, 5, 0, 0, 5, 5, 0, 0, 5, 5, 0, 0, 5, 0, 0, 5, 0, 0, 0,
                       5, 0, 0, 5, 5, 5, 5, 5, 0, 0, 5, 5, 5, 0, 5, 5, 5, 0, 5,
                       0, 5, 5, 5, 0, 0, 5, 0, 0, 0, 5, 0, 5, 5, 5, 5, 0, 0, 0,
                       5, 5, 0, 0, 5, 5, 5, 5, 0, 5, 5, 0, 5, 0, 5, 0, 0, 0, 0,
                       5, 0, 5, 0, 5, 0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0, 5, 0, 5,
                       0, 5, 5, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0, 5, 5, 5,
                       5]),
      generate(width=12, height=11,
               colors=[0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 5, 0, 5, 0, 0, 0, 0,
                       0, 5, 0, 0, 5, 5, 0, 5, 0, 0, 5, 5, 0, 2, 0, 5, 0, 5, 5,
                       0, 0, 5, 0, 5, 0, 2, 5, 0, 5, 5, 0, 0, 5, 5, 5, 2, 8, 2,
                       2, 2, 0, 5, 5, 5, 0, 5, 5, 0, 5, 2, 0, 0, 5, 5, 5, 5, 0,
                       5, 0, 0, 5, 8, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0,
                       0, 0, 5, 5, 0, 5, 0, 0, 0, 0, 5, 0, 0, 5, 0, 0, 0, 5, 5,
                       5, 5, 5, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 5, 5, 5, 5]),
  ]
  test = [
      generate(width=22, height=19,
               colors=[0, 5, 0, 5, 0, 0, 5, 5, 0, 5, 0, 0, 0, 5, 0, 5, 0, 0, 0,
                       5, 5, 0, 0, 5, 0, 5, 5, 0, 0, 0, 5, 5, 0, 0, 5, 5, 0, 0,
                       0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 5, 5, 8, 0, 0, 0, 0, 5, 5,
                       0, 0, 5, 5, 0, 0, 5, 5, 5, 0, 0, 5, 5, 0, 5, 8, 5, 0, 5,
                       0, 5, 0, 5, 0, 5, 5, 0, 5, 5, 5, 0, 0, 5, 0, 5, 2, 2, 8,
                       2, 2, 5, 0, 0, 5, 0, 5, 5, 5, 0, 0, 5, 5, 0, 0, 0, 0, 5,
                       0, 5, 2, 5, 5, 5, 0, 5, 0, 0, 0, 0, 5, 5, 5, 5, 0, 0, 5,
                       5, 0, 0, 5, 5, 2, 0, 5, 5, 0, 0, 0, 8, 0, 0, 0, 5, 5, 5,
                       5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 8, 0, 5, 0,
                       0, 5, 0, 5, 0, 5, 5, 5, 5, 5, 0, 0, 5, 5, 0, 5, 2, 8, 2,
                       8, 8, 0, 0, 5, 5, 5, 0, 0, 0, 0, 5, 5, 5, 0, 0, 5, 0, 0,
                       0, 5, 8, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 5,
                       0, 5, 0, 5, 5, 2, 5, 0, 5, 0, 0, 5, 5, 0, 0, 5, 5, 5, 0,
                       0, 0, 5, 5, 5, 5, 0, 0, 5, 0, 5, 5, 0, 0, 0, 5, 5, 5, 5,
                       0, 0, 5, 8, 5, 0, 0, 5, 5, 0, 5, 0, 5, 5, 0, 0, 5, 5, 0,
                       5, 0, 0, 5, 5, 5, 8, 5, 5, 5, 5, 0, 0, 5, 5, 5, 5, 5, 0,
                       0, 5, 0, 5, 5, 5, 0, 8, 8, 2, 2, 2, 5, 5, 5, 0, 5, 8, 5,
                       0, 5, 0, 0, 5, 5, 0, 5, 0, 0, 0, 5, 2, 5, 0, 5, 0, 5, 0,
                       5, 8, 5, 5, 0, 0, 0, 0, 5, 5, 5, 5, 5, 0, 0, 2, 0, 5, 5,
                       0, 0, 2, 2, 2, 2, 2, 5, 0, 5, 0, 5, 5, 5, 0, 5, 0, 0, 5,
                       0, 5, 0, 0, 0, 0, 0, 8, 0, 5, 5, 5, 0, 5, 5, 0, 5, 5, 5,
                       5, 5, 0, 5, 0, 5, 5, 5, 5, 0, 8, 0, 0, 5, 5, 0, 5, 0,
                       5]),
  ]
  return {"train": train, "test": test}
