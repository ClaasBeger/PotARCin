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


def _turns(pixels):
  """The eight ways a sprite can be turned over and around."""
  out = []
  for flip in (False, True):
    pts = [(c, r) for r, c in pixels] if flip else list(pixels)
    for _ in range(4):
      out.append(pts)
      pts = [(c, -r) for r, c in pts]
  return out


def _readable(sprites):
  """Whether every arrangement of key cells names one sprite and one turn.

  A diagonal triple is left where it was by a transpose, and with two sprites
  sharing a palette a triple can fit the other sprite as well -- either way the
  clone could be filled in two ways, so the picture has no single answer.
  """
  seen = {}
  for pixels, pos in sprites:
    for turned in _turns(pixels):
      anchor = turned[pos[0]]
      keys = tuple(sorted((turned[p][0] - anchor[0], turned[p][1] - anchor[1], j)
                          for j, p in enumerate(pos)))
      shape = frozenset((r - anchor[0], c - anchor[1]) for r, c in turned)
      if keys in seen and seen[keys] != shape: return False
      seen[keys] = shape
  return True


def generate(width=None, height=None, rows=None, cols=None, idxs=None,
             colors=None, brows=None, bcols=None, rotates=None):
  """Returns input and output grids according to the given parameters.

  Args:
    width: the width of the (square) grid
    height: the height of the (square) grid
    rows: a list of vertical coordinates where pixels should be placed
    cols: a list of horizontal coordinates where pixels should be placed
    idxs: a list of indices into the sprite list
    colors: a list of digits representing the colors to be used
    brows: a list of vertical coordinates where the sprites should be placed
    bcols: a list of horizontal coordinates where the sprites should be placed
    rotates: a list of rotations for the sprites
  """
  if width is None:
    num_sprites = common.randint(1, 2)
    # Choose locations for sprites and their clones until they don't overlap.
    while True:
      width, height = common.randint(12, 24), common.randint(12, 24)
      wides, talls, lengths, rotates, brows, bcols = [], [], [], [], [], []
      for _ in range(num_sprites):
        wides.append(common.randint(3, 6))
        talls.append(9 - wides[-1])
        lengths.append(max(wides[-1], talls[-1]))
        rotates.append(common.randint(1, 4))
      brows.extend([common.randint(0, height - length) for length in lengths])
      brows.extend([common.randint(0, height - length) for length in lengths])
      bcols.extend([common.randint(0, width - length) for length in lengths])
      bcols.extend([common.randint(0, width - length) for length in lengths])
      o = common.overlaps(brows, bcols, lengths + lengths, lengths + lengths, 3)
      if not o: break
    # Choose the contents of the sprites.
    # TODO: Use something less bulky than a continuous creature.
    color_list = common.random_colors(4)
    # A clone shows only its three key cells, so those cells have to say both
    # which sprite it is and which way that sprite was turned. Keep drawing
    # sprites until they do.
    while True:
      sprites = []
      for idx in range(num_sprites):
        wide, tall = wides[idx], talls[idx]
        pixels = common.continuous_creature(common.randint(6, 12), wide, tall)
        pos = None
        for _ in range(200):  # Pick positions that aren't all in a line
          choice = common.sample(range(len(pixels)), 3)
          if len(set(pixels[p][0] for p in choice)) == 1: continue
          if len(set(pixels[p][1] for p in choice)) == 1: continue
          pos = choice
          break
        if pos is None: break
        sprites.append((pixels, pos))
      if len(sprites) == num_sprites and _readable(sprites): break
    rows, cols, idxs, colors = [], [], [], []
    for idx, (pixels, pos) in enumerate(sprites):
      sprite_colors = [color_list[3]] * len(pixels)
      sprite_colors[pos[0]] = color_list[0]
      sprite_colors[pos[1]] = color_list[1]
      sprite_colors[pos[2]] = color_list[2]
      rows.extend(p[0] for p in pixels)
      cols.extend(p[1] for p in pixels)
      idxs.extend([idx] * len(pixels))
      colors.extend(sprite_colors)

  num_sprites, mode = max(idxs) + 1, max(set(colors), key=colors.count)
  grid, output = common.grids(width, height)
  for row, col, i, color in zip(rows, cols, idxs, colors):
    r, c = row, col
    brow, bcol = brows[i], bcols[i]
    grid[brow + r][bcol + c] = color
    brow, bcol, rot = brows[num_sprites + i], bcols[num_sprites + i], rotates[i]
    wide = max([c for c, idx in zip(cols, idxs) if idx == i]) + 1
    tall = max([r for r, idx in zip(rows, idxs) if idx == i]) + 1
    if rot == 1: r, c = wide - c - 1, r
    if rot == 2: r, c = c, tall - r - 1
    if rot == 3: r = tall - r - 1
    if rot == 4: r, c = c, r
    output[brow + r][bcol + c] = color
    if color == mode: continue  # For the clone, we hide the common color.
    grid[brow + r][bcol + c] = color
  return {"input": grid, "output": output}


def validate():
  """Validates the generator."""
  train = [
      generate(width=18, height=14,
               rows=[0, 1, 1, 1, 2, 2, 2, 0, 1, 1, 2, 2, 2, 3, 3, 4],
               cols=[1, 0, 1, 2, 0, 1, 2, 0, 0, 2, 0, 1, 2, 0, 2, 0],
               idxs=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               colors=[8, 3, 8, 1, 8, 4, 8, 3, 8, 8, 8, 8, 4, 8, 8, 1],
               brows=[1, 6, 9, 2], bcols=[2, 7, 1, 13], rotates=[1, 1]),
      generate(width=15, height=14,
               rows=[0, 1, 1, 1, 2, 3, 4, 5, 5, 5],
               cols=[1, 0, 1, 2, 1, 1, 1, 0, 1, 2],
               idxs=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               colors=[2, 4, 3, 3, 3, 3, 3, 3, 1, 3],
               brows=[3, 10], bcols=[3, 9], rotates=[2]),
      generate(width=14, height=16,
               rows=[0, 1, 1, 2, 2, 2, 2, 2, 2, 3],
               cols=[4, 0, 4, 0, 1, 2, 3, 4, 5, 4],
               idxs=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               colors=[4, 8, 8, 1, 8, 8, 8, 2, 8, 8],
               brows=[2, 10], bcols=[5, 1], rotates=[3]),
  ]
  test = [
      generate(width=19, height=24,
               rows=[0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 0, 0, 1, 1, 2, 2, 2],
               cols=[2, 1, 2, 3, 4, 5, 2, 5, 0, 1, 2, 3, 0, 1, 1, 3, 1, 2, 3],
               idxs=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
               colors=[5, 5, 1, 5, 5, 4, 5, 5, 2, 5, 5, 5, 5, 2, 5, 5, 4, 5, 1],
               brows=[3, 9, 16, 12], bcols=[5, 10, 9, 2], rotates=[1, 4]),
  ]
  return {"train": train, "test": test}
