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


def _placement_unambiguous(rows, cols, colors, shows, wide, tall, irow, icol, mag):
  """True if the magnified copies pin down the magnification and the offset.

  Only some of the sprite's cells are copied into the box, magnified; the rest of
  the answer has to be read off the small sprite outside it. That only works if
  the copies say *how much* the sprite was magnified and *where* it sits. When a
  different magnification or offset would put exactly the same copies in exactly
  the same places, the picture has two readings and neither can be called the
  answer, so the instance is rejected rather than emitted.

  Stated over the picture rather than the parameters that drew it: the sprite's
  extent is its drawn bounding box, which is what a reader can see.
  """
  srow0, srow1 = min(rows), max(rows)
  scol0, scol1 = min(cols), max(cols)
  sh, sw = srow1 - srow0 + 1, scol1 - scol0 + 1
  sprite = [[0] * sw for _ in range(sh)]
  for idx, color in enumerate(colors):
    sprite[rows[idx] - srow0][cols[idx] - scol0] = color

  shown = {}
  for idx in shows:
    r0 = irow + (rows[idx] - srow0) * mag
    c0 = icol + (cols[idx] - scol0) * mag
    for dr in range(mag):
      for dc in range(mag):
        shown[(r0 + dr, c0 + dc)] = colors[idx]

  found = 0
  for mg in range(2, 9):
    if mg * sh > tall - 2 or mg * sw > wide - 2:
      continue
    for ir in range(1, tall - mg * sh + 1):
      for ic in range(1, wide - mg * sw + 1):
        ok = True
        # every copied cell must land on a sprite cell of its own colour
        for (rr, cc), color in shown.items():
          dr, dc = rr - ir, cc - ic
          if dr < 0 or dc < 0 or dr >= mg * sh or dc >= mg * sw:
            ok = False
            break
          if sprite[dr // mg][dc // mg] != color:
            ok = False
            break
        if not ok:
          continue
        # and every sprite cell must be copied whole or not at all
        for sr in range(sh):
          for sc in range(sw):
            color = sprite[sr][sc]
            block = set()
            for dr in range(mg):
              for dc in range(mg):
                block.add(shown.get((ir + sr * mg + dr, ic + sc * mg + dc), 0))
            if block != {0} and block != {color}:
              ok = False
              break
          if not ok:
            break
        if ok:
          found += 1
          if found > 1:
            return False
  return found == 1


def generate(width=None, height=None, rows=None, cols=None, colors=None,
             shows=None, wide=None, tall=None, brow=None, bcol=None, irow=None,
             icol=None, srow=None, scol=None, mag=None):
  """Returns input and output grids according to the given parameters.

  Args:
    width: the width of the input grid
    height: the height of the input grid
    rows: a list of vertical coordinates where pixels should be placed
    cols: a list of horizontal coordinates where pixels should be placed
    colors: a list of colors to be used for the pixels
    shows: a subset of pixel indices to show magnified
    wide: the width of the yellow box
    tall: the height of the yellow box
    brow: a vertical coordinate where the box should be placed
    bcol: a horizontal coordinate where the box should be placed
    irow: a vertical coordinate inside the box where the sprite should be
    icol: a horizontal coordinate inside the box where the sprite should be
    srow: a vertical coordinate where the full sprite should be
    scol: a horizontal coordinate where the full sprite should be
    mag: the magnification factor
  """
  if width is None:
   while True:  # re-roll until the copies determine magnification and offset
    width, height = common.randint(15, 20), common.randint(15, 20)
    mag = common.randint(2, min(4, (height + 1) // 5))
    w, t = common.randint(2, 7 - mag), common.randint(2, 3)
    num_pixels = common.randint(max(4, w * t // 2), w * t)
    pixels = common.continuous_creature(num_pixels, w, t)
    rows, cols = zip(*pixels)
    colors = [common.choice([1, 2, 3, 8]) for _ in pixels]
    num_show = common.randint(2, num_pixels // 2)
    shows = common.sample(range(len(pixels)), num_show)
    wide = common.randint(mag * w + 2, width)
    tall = common.randint(mag * t + 2, height - 4)
    brow = common.randint(0, height - 4 - tall)
    bcol = common.randint(0, width - wide)
    # Upper bound is tall/wide - mag*t, not one less: the magnified sprite may sit
    # against the far border, as it does in the task's own test example, which the
    # tighter bound made unreachable.
    irow = common.randint(1, tall - mag * t)
    icol = common.randint(1, wide - mag * w)
    srow = height - t - 1
    scol = common.randint(1, width - 1 - w)
    # Reject instances whose magnified copies admit a second reading.
    if not _placement_unambiguous(
        rows, cols, colors, shows, wide, tall, irow, icol, mag):
      continue
    break

  grid, output = common.grid(width, height), common.grid(wide, tall)
  for r, c in [(0, 0), (0, wide - 1), (tall - 1, 0), (tall - 1, wide - 1)]:
    grid[brow + r][bcol + c] = common.yellow()
    output[r][c] = common.yellow()
  for idx, color in enumerate(colors):
    row, col = rows[idx], cols[idx]
    grid[srow + row][scol + col] = color
    for dr in range(mag):
      for dc in range(mag):
        r, c = irow + row * mag + dr, icol + col * mag + dc
        if idx in shows: grid[brow + r][bcol + c] = color
        output[r][c] = color
  return {"input": grid, "output": output}


def validate():
  """Validates the generator."""
  train = [
      generate(width=17, height=17,
               rows=[0, 0, 1, 1, 1, 1, 2, 2, 2],
               cols=[0, 4, 0, 1, 3, 4, 1, 2, 3],
               colors=[3, 2, 1, 1, 1, 1, 1, 1, 1],
               shows=[0, 1, 7], wide=14, tall=9,
               brow=0, bcol=2, irow=2, icol=1, srow=12, scol=2, mag=2),
      generate(width=18, height=17,
               rows=[0, 0, 1, 1],
               cols=[0, 1, 0, 1],
               colors=[2, 3, 3, 8],
               shows=[0, 3], wide=7, tall=7,
               brow=1, bcol=2, irow=1, icol=1, srow=13, scol=10, mag=2),
      generate(width=18, height=16,
               rows=[0, 0, 0, 1, 1],
               cols=[0, 1, 2, 1, 2],
               colors=[2, 1, 1, 3, 1],
               shows=[0, 2, 3], wide=11, tall=11,
               brow=0, bcol=3, irow=3, icol=1, srow=13, scol=6, mag=3),
  ]
  test = [
      generate(width=18, height=19,
               rows=[0, 0, 1, 1, 1, 2],
               cols=[0, 2, 0, 1, 2, 1],
               colors=[8, 3, 1, 1, 1, 1],
               shows=[0, 1], wide=18, tall=14,
               brow=0, bcol=0, irow=2, icol=4, srow=15, scol=8, mag=4),
  ]
  return {"train": train, "test": test}
