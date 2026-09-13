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


def _picture(size, rows, cols, colors):
  """The input grid as it will be drawn, before any flip or transpose."""
  grid = common.grid(size, size)
  for r, c, color in zip(rows, cols, colors):
    grid[r][c] = common.black() if color == common.blue() else color
  return grid


def _route_count(grid):
  """How many ways the picture can be read.

  The trail leaves the green end, runs until the static stops it, turns across,
  runs until that is stopped too, and drops into the red end. Both turns have to
  be forced -- that is what picks one crossing line out of the several that are
  merely clear, and it is the reading the official examples use. More than one
  such trail and the picture has no single answer.
  """
  size = len(grid)
  clear = (common.black(), common.red(), common.green())
  def cells_of(colour):
    return [(r, c) for r in range(len(grid)) for c in range(len(grid[0]))
            if grid[r][c] == colour]
  green, red = cells_of(common.green()), cells_of(common.red())
  if len(green) != 2 or len(red) != 2: return -1
  if green[0][0] == green[1][0]:  # stubs lie along rows; read the picture sideways
    grid = common.transpose(grid)
    green, red = cells_of(common.green()), cells_of(common.red())
  if green[0][1] != green[1][1] or red[0][1] != red[1][1]: return -1
  gtop, gbot, gcol = min(r for r, _ in green), max(r for r, _ in green), green[0][1]
  rtop, rbot, rcol = min(r for r, _ in red), max(r for r, _ in red), red[0][1]
  if gcol == rcol: return -1
  step = 1 if rcol > gcol else -1

  def blocked(r, c):
    return not (0 <= r < len(grid) and 0 <= c < len(grid[0])) or grid[r][c] not in clear

  found = 0
  for row in range(len(grid)):
    if gtop <= row <= gbot or rtop <= row <= rbot: continue
    down = row > gbot
    leg_g = [(r, gcol) for r in (range(gbot + 1, row + 1) if down else range(row, gtop))]
    leg_r = [(r, rcol) for r in (range(rbot + 1, row + 1) if row > rbot else range(row, rtop))]
    across = [(row, c) for c in range(gcol, rcol + step, step)]
    if not all(grid[r][c] in clear for r, c in leg_g + leg_r + across): continue
    if not blocked(row + (1 if down else -1), gcol): continue
    if not blocked(row, rcol + step): continue
    found += 1
  return found

def generate(size=None, rows=None, cols=None, colors=None, flip=None,
             xpose=None):
  """Returns input and output grids according to the given parameters.

  Args:
    size: the width and height of the (square) grid
    rows: a list of vertical coordinates where pixels should be placed
    cols: a list of horizontal coordinates where pixels should be placed
    colors: a list of digits representing the color to be used
    flip: whether to flip the grids
    xpose: whether to transpose the grids
  """
  if size is None:
    while True:
      size, rows, cols = common.randint(10, 20), [], []
      # Option 'S'
      if common.randint(0, 1):
        height = common.randint(3 * size // 4, 7 * size // 8)
        width = common.randint(size // 2, 3 * size // 4)
        row = common.randint(1, size - height - 1)
        col = common.randint(1, size - width - 1)
        mid = common.randint(row + 3, row + height - 3)
        for r in range(row, mid):
          rows.append(r)
          cols.append(col + width - 1)
        for c in range(col, col + width):
          rows.append(mid)
          cols.append(c)
        for r in range(mid + 1, row + height):
          rows.append(r)
          cols.append(col)
        colors = [common.blue()] * len(rows)
        colors[0] = colors[1] = common.red()
        colors[-1] = colors[-2] = common.green()
        rows.append(mid - 1)
        cols.append(col)
        colors.append(common.cyan())
        rows.append(mid)
        cols.append(col + width)
        colors.append(common.cyan())
      # Option 'U'
      else:
        height = common.randint(size // 2, 3 * size // 4)
        width = common.randint(size // 2, 3 * size // 4)
        row = common.randint(1, size - height - 1)
        col = common.randint(1, size - width - 1)
        mid1 = common.randint(row, row + height - 4)  # No hard turns @ red dot.
        mid2 = common.randint(row, row + height - 3)
        for r in range(mid1, row + height):
          rows.append(r)
          cols.append(col + width - 1)
        for c in range(col, col + width):
          rows.append(row + height - 1)
          cols.append(c)
        for r in range(row + height - 1, mid2 - 1, -1):
          rows.append(r)
          cols.append(col)
        colors = [common.blue()] * len(rows)
        colors[0] = colors[1] = common.red()
        colors[-1] = colors[-2] = common.green()
        rows.append(row + height)
        cols.append(col)
        colors.append(common.cyan())
        rows.append(row + height - 1)
        cols.append(col + width)
        colors.append(common.cyan())
        # Extra sky blue to prevent sideways entry.
        rows.append(mid1)
        cols.append(col + width - 2)
        colors.append(common.cyan())
      # Prepend some static (will get overridden by the path). The official
      # examples run from ~11% static up to ~31%; a fixed one-in-five could reach
      # neither end, so draw the density once per grid.
      density = common.randint(3, 9)
      xrows, xcols, xcolors = [], [], []
      for r in range(size):
        for c in range(size):
          if common.randint(0, density - 1): continue
          xrows.append(r)
          xcols.append(c)
          xcolors.append(common.cyan())
      rows, cols, colors = xrows + rows, xcols + cols, xcolors + colors
      # Finally, flip and/or transpose.
      flip, xpose = common.randint(0, 1), common.randint(0, 1)

      # The static can leave a second forced trail between the two ends, and
      # then the picture has two answers; draw again when it does.
      if _route_count(_picture(size, rows, cols, colors)) == 1: break

  grid, output = common.grids(size, size)
  for r, c, color in zip(rows, cols, colors):
    output[r][c] = grid[r][c] = color
    if color == common.blue():
      grid[r][c] = common.black()
      output[r][c] = common.green()
  if flip: grid, output = grid[::-1], output[::-1]
  if xpose: grid, output = common.transpose(grid), common.transpose(output)
  return {"input": grid, "output": output}


def validate():
  """Validates the generator."""
  train = [
      generate(size=20,
               rows=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                     2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4,
                     4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                     6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10,
                     10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11,
                     11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13,
                     13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15,
                     15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17,
                     17, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19,
                     19],
               cols=[4, 5, 6, 7, 8, 11, 13, 14, 15, 17, 19, 1, 7, 13, 15, 0, 1,
                     2, 3, 4, 6, 8, 12, 13, 14, 17, 0, 2, 3, 10, 11, 12, 13, 14,
                     15, 17, 0, 3, 4, 11, 13, 17, 3, 13, 17, 0, 3, 4, 5, 6, 7,
                     8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 2, 3, 18, 19, 0,
                     3, 12, 18, 19, 3, 9, 12, 18, 19, 1, 3, 6, 7, 8, 10, 13, 15,
                     16, 0, 3, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 3, 6,
                     7, 8, 11, 12, 13, 15, 18, 19, 3, 6, 7, 12, 16, 19, 3, 7, 9,
                     11, 14, 17, 19, 3, 6, 7, 8, 12, 13, 14, 15, 1, 7, 9, 10,
                     12, 14, 16, 7, 11, 17, 18, 3, 7, 9, 12, 13, 14, 19, 4, 5,
                     6, 7, 10, 15, 16, 17],
               colors=[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                       8, 8, 8, 8, 8, 8, 2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 2, 8, 8,
                       8, 8, 8, 1, 8, 8, 1, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 8, 8, 1, 8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 8, 8,
                       8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8,
                       8, 8, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8,
                       3, 8, 8, 8, 8, 8, 8, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                       8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                       8, 8, 8],
               flip=0, xpose=0),
      generate(size=10,
               rows=[0, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6,
                     7, 7, 8, 9],
               cols=[9, 1, 2, 1, 7, 1, 2, 3, 4, 5, 6, 9, 1, 3, 5, 3, 5, 1, 2, 5,
                     2, 5, 2, 0],
               colors=[8, 3, 8, 3, 8, 1, 1, 1, 1, 1, 8, 8, 8, 8, 1, 8, 1, 8, 8,
                       2, 8, 2, 8, 8],
               flip=0, xpose=0),
      generate(size=15,
               rows=[0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4,
                     5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8,
                     8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11,
                     11, 11, 11, 11, 12, 12, 13, 13, 13, 13, 13, 14, 14, 14],
               cols=[5, 7, 10, 13, 3, 6, 11, 13, 14, 0, 4, 5, 6, 11, 12, 13, 5,
                     7, 9, 13, 1, 2, 3, 4, 5, 6, 7, 8, 12, 7, 11, 13, 1, 2, 5,
                     7, 8, 10, 11, 1, 7, 0, 1, 2, 3, 4, 5, 6, 7, 13, 0, 7, 8, 9,
                     1, 4, 6, 10, 11, 12, 13, 5, 7, 5, 6, 8, 11, 14, 1, 4, 5],
               colors=[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                       8, 3, 3, 1, 1, 1, 1, 1, 8, 8, 1, 8, 8, 8, 8, 8, 1, 8, 8,
                       8, 8, 1, 8, 2, 2, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8, 8,
                       8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
               flip=0, xpose=0),
  ]
  test = [
      generate(size=13,
               rows=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                     4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                     8, 8, 8, 8, 9, 9, 10, 10, 10, 11, 12],
               cols=[0, 1, 2, 3, 9, 10, 0, 5, 6, 7, 8, 9, 1, 4, 5, 9, 2, 8, 9,
                     12, 2, 6, 9, 12, 3, 8, 9, 10, 9, 10, 0, 2, 3, 4, 5, 6, 7,
                     8, 9, 10, 1, 2, 5, 10, 6, 7, 1, 2, 6, 3, 11],
               colors=[8, 8, 8, 8, 8, 8, 8, 8, 2, 2, 1, 1, 8, 8, 8, 1, 8, 8, 1,
                       8, 8, 8, 1, 8, 8, 8, 1, 8, 1, 8, 8, 8, 3, 3, 1, 1, 1, 1,
                       1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
               flip=0, xpose=0),
  ]
  return {"train": train, "test": test}
