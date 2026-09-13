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


def generate(width=None, height=None, rows=None, cols=None, bgcolor=None,
             color=None, rdiff=None, cdiff=None, brow=None, bcol=None,
             prow=None, pcol=None, flip=None, xpose=None):
  """Returns input and output grids according to the given parameters.

  Args:
    width: the width of the input grid
    height: the height of the input grid
    rows: a list of vertical coordinates where pixels should be placed
    cols: a list of horizontal coordinates where pixels should be placed
    bgcolor: a digit representing the background color
    color: a digit representing the foreground color
    rdiff: the row delta to add for each iteration of the sprite
    cdiff: the column delta to add for each iteration of the sprite
    brow: the vertical coordinate of the origin sprite
    bcol: the horizontal coordinate of the origin sprite
    prow: the vertical coordinate of the pixel hint
    pcol: the horizontal coordinate of the pixel hint
    flip: whether to flip the sprite
    xpose: whether to transpose the sprite
  """
  if width is None:
    while True:
      width, height = common.randint(10, 20), common.randint(10, 20)
      # The official train[2] uses a 5x5 sprite, which randint(3, 4) excluded.
      length = common.randint(3, 5)
      # All four official sprites are symmetric under both mirrors, so only a
      # quarter of the box is drawn at random and then mirrored into place, once
      # every row and column of the quarter is occupied (which keeps the sprite
      # exactly length by length).  Carving pixels out of the whole box instead,
      # as this used to, could not reach the sparse official sprites: it always
      # appended the (0, 0) pixel, which forced all four corners in -- the
      # official diamond and plus have none -- and it listed each cell up to
      # four times over, so the shortest list it could build had sixteen entries
      # against the official four.
      half = (length + 1) // 2
      while True:
        quarter = common.random_pixels(half, half)
        if len({pixel[0] for pixel in quarter}) < half: continue
        if len({pixel[1] for pixel in quarter}) < half: continue
        break
      cells = []
      for row, col in quarter:
        for r in (row, length - row - 1):
          for c in (col, length - col - 1):
            if (r, c) not in cells: cells.append((r, c))
      pixels = set(cells)
      rows = [cell[0] for cell in cells]
      cols = [cell[1] for cell in cells]
      # How far the sprite walks each time is a property of its shape rather
      # than of its size: the copies step by the smallest diagonal offset at
      # which the sprite no longer lands on itself.  That is length only when
      # the sprite has its corners filled in, which is why the official diamond
      # and plus (3x3 sprites) step by two and not by three.
      step = 1
      for k in range(1, length + 1):
        if any((r + k, c + k) in pixels for r, c in cells): step = k + 1
      # The repeat step may point in any diagonal direction: the official test
      # uses rdiff=cdiff=-3 and train[2] uses rdiff=-5 with cdiff=+5, while the
      # old code only ever stepped down and to the right.
      rdiff = step if common.randint(0, 1) else -step
      cdiff = step if common.randint(0, 1) else -step
      brow = common.randint(length, height // 2)
      bcol = common.randint(length, width // 2)
      # The pixel hint is one of the cells of the first copy (it is not always
      # that copy's corner: the official hints sit wherever the sprite happens
      # to have a pixel).  Keep only the cells that stay on the grid, that do
      # not eat into the black sprite, and that no other diagonal direction
      # could explain, so the direction the hint points in stays unambiguous.
      spots = []
      for row, col in cells:
        prow, pcol = brow + rdiff + row, bcol + cdiff + col
        if not 0 <= prow < height or not 0 <= pcol < width: continue
        if (prow - brow, pcol - bcol) in pixels: continue
        hits = 0
        for dr in (-step, step):
          for dc in (-step, step):
            if (prow - brow - dr, pcol - bcol - dc) in pixels: hits += 1
        if hits == 1: spots.append((prow, pcol))
      if not spots: continue
      prow, pcol = spots[common.randint(0, len(spots) - 1)]
      color = common.random_color()
      bgcolor = common.random_color(exclude=[color])
      flip, xpose = common.randint(0, 1), common.randint(0, 1)
      break

  grid, output = common.grids(width, height, bgcolor)
  for row, col in zip(rows, cols):
    grid[brow + row][bcol + col] = common.black()
    output[brow + row][bcol + col] = common.black()
  grid[prow][pcol] = color
  for repeat in range(1, width + height + 1):
    for row, col in zip(rows, cols):
      common.draw(output, brow + repeat * rdiff + row,
                  bcol + repeat * cdiff + col, color)
  if flip: grid, output = grid[::-1], output[::-1]
  if xpose: grid, output = common.transpose(grid), common.transpose(output)
  return {"input": grid, "output": output}


def validate():
  """Validates the generator."""
  train = [
      generate(width=14, height=12, rows=[0, 1, 1, 2], cols=[1, 0, 2, 1],
               bgcolor=8, color=2, rdiff=2, cdiff=2, brow=3, bcol=3, prow=5,
               pcol=6, flip=0, xpose=0),
      generate(width=15, height=13, rows=[0, 1, 1, 1, 2], cols=[1, 0, 1, 2, 1],
               bgcolor=1, color=3, rdiff=2, cdiff=-2, brow=5, bcol=5, prow=7,
               pcol=4, flip=0, xpose=0),
      generate(width=16, height=12,
               rows=[0, 0, 0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 3, 4, 4, 4, 4],
               cols=[0, 1, 3, 4, 0, 1, 3, 4, 2, 0, 1, 3, 4, 0, 1, 3, 4],
               bgcolor=4, color=8, rdiff=-5, cdiff=5, brow=5, bcol=6, prow=4,
               pcol=11, flip=0, xpose=0),
  ]
  test = [
      generate(width=16, height=18, rows=[0, 0, 1, 2, 2], cols=[0, 2, 1, 0, 2],
               bgcolor=3, color=6, rdiff=-3, cdiff=-3, brow=6, bcol=4, prow=5,
               pcol=3, flip=0, xpose=0),
  ]
  return {"train": train, "test": test}
