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


def _reading_unique(grid, ground):
  """True if only one of the small shapes explains the enlarged drawing.

  The enlargement runs off the grid, so part of it is missing, and a different
  small shape can sometimes be slid under the same visible cells and account for
  them just as well -- hiding the same amount, in the same direction. The picture
  then does not say which shape was enlarged, and neither reading can be called
  the answer, so the instance is rejected rather than emitted.

  Read off the finished picture rather than the parameters that drew it: double
  each small shape, slide it everywhere, clip at the edge, and see which ones land
  exactly on the enlarged drawing. If more than one shape can, the instance is
  dropped, whatever the amounts hidden.
  """
  height = len(grid)
  width = len(grid[0]) if height else 0
  marks = {}
  for r in range(height):
    for c in range(width):
      if grid[r][c] != ground:
        marks.setdefault(grid[r][c], set()).add((r, c))
  owners = set()
  for big_colour, big_cells in marks.items():
    for small_colour, small_cells in marks.items():
      if small_colour == big_colour:
        continue
      top = min(r for r, _c in small_cells)
      left = min(c for _r, c in small_cells)
      tall = max(r for r, _c in small_cells) - top + 1
      wide = max(c for _r, c in small_cells) - left + 1
      shape = [(r - top, c - left) for r, c in small_cells]
      doubled = [(2 * r + dr, 2 * c + dc)
                 for r, c in shape for dr in (0, 1) for dc in (0, 1)]
      for mrow in range(-2 * tall, height):
        for mcol in range(-2 * wide, width):
          seen, hidden = set(), 0
          for r, c in doubled:
            if 0 <= mrow + r < height and 0 <= mcol + c < width:
              seen.add((mrow + r, mcol + c))
            else:
              hidden += 1
          if not hidden or seen != big_cells:
            continue
          owners.add(small_colour)
  return len(owners) == 1


def generate(width=None, height=None, rows=None, cols=None, idxs=None,
             brows=None, bcols=None, magrow=None, magcol=None, colors=None,
             magcolor=None, bgcolor=None):
  """Returns input and output grids according to the given parameters.

  Args:
    width: the width of the (square) grid
    height: the height of the (square) grid
    rows: a list of vertical coordinates where pixels should be placed
    cols: a list of horizontal coordinates where pixels should be placed
    idxs: a list of indices of the sprites
    brows: a list of vertical coordinates where sprites should be placed
    bcols: a list of horizontal coordinates where sprites should be placed
    magrow: the vertical coordinate where the magnified sprite should be
    magcol: the horizontal coordinate where the magnified sprite should be
    colors: a list of colors to be used
    magcolor: a digit representing a magnified color to be used
    bgcolor: a digit representing a background color to be used
  """

  def draw(b, mrow, mcol, mcolor):
    legal = True
    wide = max([c for c, i in zip(cols, idxs) if not i]) + 1
    tall = max([r for r, i in zip(rows, idxs) if not i]) + 1
    grid = common.grid(width, height, b)
    output = common.grid(wide, tall, b)
    some_hidden = False
    some_shown = False
    for row, col, idx in zip(rows, cols, idxs):
      r, c = brows[idx] + row, bcols[idx] + col
      if grid[r][c] != b: legal = False
      grid[r][c] = colors[idx]
      if idx: continue  # Not the magnified sprite.
      output[row][col] = colors[idx]
      for dr, dc in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        r, c = mrow + 2 * row + dr, mcol + 2 * col + dc
        if common.get_pixel(grid, r, c) == -1:
          some_hidden = True
        else:
          some_shown = True
        if common.get_pixel(grid, r, c) not in [b, -1]: legal = False
        common.draw(grid, r, c, mcolor)
    # Part of the enlargement has to run off the grid, and part of it has to stay
    # on: an enlargement placed entirely outside leaves nothing to read, and one
    # placed entirely inside is not this task.
    if not (some_hidden and some_shown): legal = False
    return grid, output, legal

  if rows is None:
    width, height = common.randint(15, 19), common.randint(15, 19)
    bgcolor = common.random_color()
    magcolor = common.random_color(exclude=[bgcolor])
    colors = common.random_colors(2, exclude=[bgcolor, magcolor])
    while True:
      rows, cols, idxs, brows, bcols = [], [], [], [], []
      for idx in range(2):
        wide, tall = common.randint(3, 5), common.randint(3, 5)
        while True:
          # conway_sprite removes pixels on wide + tall attempts and almost all
          # of them land, so a 5x5 sprite came out with 15 pixels 95% of the
          # time. The official examples are denser than that -- the first one
          # keeps 17 of the 25 cells in both of its 5x5 sprites -- so vary how
          # many pixels are taken away instead of always taking the most.
          tries = common.randint(max(wide, tall), wide + tall)
          xrows, xcols = common.conway_sprite(wide, tall, tries)
          if common.diagonally_connected(list(zip(xrows, xcols))): break
        rows.extend(xrows)
        cols.extend(xcols)
        idxs.extend([idx] * len(xrows))
        brows.append(common.randint(0, height - tall))
        bcols.append(common.randint(0, width - wide))
        if idx: continue
        # Anywhere that leaves some of the enlargement on the grid. The earlier
        # version always pushed it off by exactly two pixels at exactly one edge;
        # the task's own examples are cut by one and by three pixels, and two of
        # them are cut at two edges at once, so none of them was reachable.
        magrow = common.randint(-(2 * tall - 1), height - 1)
        magcol = common.randint(-(2 * wide - 1), width - 1)
      drawn, _, legal = draw(bgcolor, magrow, magcol, magcolor)
      # Also require that the picture says which shape was enlarged.
      if legal and _reading_unique(drawn, bgcolor): break

  grid, output, _ = draw(bgcolor, magrow, magcol, magcolor)
  return {"input": grid, "output": output}


def validate():
  """Validates the generator."""
  train = [
      generate(width=17, height=17,
               rows=[0, 0, 0, 0, 0, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4,
                     0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 3, 4, 4, 4, 4, 4],
               cols=[0, 1, 2, 3, 4, 0, 2, 4, 4, 0, 2, 4, 0, 1, 2, 3, 4,
                     0, 1, 3, 4, 0, 4, 0, 1, 2, 3, 4, 2, 0, 1, 2, 3, 4],
               idxs=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               brows=[3, 2], bcols=[3, 11], magrow=9, magcol=3,
               colors=[2, 3], magcolor=8, bgcolor=1),
      generate(width=18, height=18,
               rows=[0, 1, 1, 1, 2, 3, 3, 3, 4, 0, 0, 1, 1, 1, 1, 1, 2, 2],
               cols=[1, 0, 1, 2, 1, 0, 1, 2, 1, 1, 3, 0, 1, 2, 3, 4, 1, 3],
               idxs=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               brows=[8, 4], bcols=[11, 3], magrow=10, magcol=-1,
               colors=[4, 3], magcolor=6, bgcolor=8),
      generate(width=17, height=19,
               rows=[0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 4,
                     0, 0, 0, 1, 1, 1, 1, 1, 2],
               cols=[0, 1, 2, 0, 0, 1, 2, 2, 0, 1, 2,
                     0, 2, 4, 0, 1, 2, 3, 4, 2],
               idxs=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     1, 1, 1, 1, 1, 1, 1, 1, 1],
               brows=[2, 10], bcols=[4, 8], magrow=13, magcol=-2,
               colors=[8, 3], magcolor=1, bgcolor=2),
      generate(width=17, height=15,
               rows=[0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                     0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3],
               cols=[1, 2, 3, 0, 1, 3, 4, 1, 2, 3,
                     0, 3, 0, 1, 2, 3, 0, 3, 0, 1, 2, 3],
               idxs=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               brows=[9, 2], bcols=[8, 5], magrow=9, magcol=-3,
               colors=[3, 8], magcolor=2, bgcolor=1),
  ]
  test = [
      generate(width=18, height=18,
               rows=[0, 1, 1, 1, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2],
               cols=[1, 0, 1, 2, 1, 0, 1, 2, 0, 2, 3, 0, 1, 3, 0, 2, 3],
               idxs=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               brows=[5, 2], bcols=[2, 9], magrow=12, magcol=6,
               colors=[6, 1], magcolor=8, bgcolor=3),
  ]
  return {"train": train, "test": test}
