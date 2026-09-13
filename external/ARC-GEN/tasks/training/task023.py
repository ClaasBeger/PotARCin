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


def _answers(cells, cap=2):
  """Counts (up to cap) different answers a set of grey cells can be given.

  The toys are the only three shapes that exist here: a 2x2 box, a tall stick
  and a flat stick.  Two toys that touch can melt into one grey blob that can be
  cut up more than one way, and then the picture has more than one right answer.
  Cutting always starts at the first cell left over in reading order, which only
  three toys can cover -- each of them has that cell as its own top left corner,
  because everything above it and to its left is already spoken for.  What is
  counted is the answers, not the cuts: the answer only says which cells are
  boxes, so two cuts that disagree about where one stick stops but paint the
  picture the same way are one answer and no trouble at all.
  """
  answers = set()

  def cut(left, boxes):
    if len(answers) >= cap: return
    if not left:
      answers.add(frozenset(boxes))
      return
    r, c = min(left)
    for toy in (((0, 0), (0, 1), (1, 0), (1, 1)),
                ((0, 0), (0, 1), (0, 2)),
                ((0, 0), (1, 0), (2, 0))):
      piece = {(r + dr, c + dc) for dr, dc in toy}
      if piece <= left:
        cut(left - piece, boxes | piece if len(piece) == 4 else boxes)

  cut(set(cells), frozenset())
  return len(answers)


def _cells(rows, cols, idxs):
  """The grey cells covered by a set of toys."""
  shapes = {0: ((0, 0), (0, 1), (1, 0), (1, 1)),
            1: ((0, 0), (1, 0), (2, 0)),
            2: ((0, 0), (0, 1), (0, 2))}
  out = set()
  for row, col, idx in zip(rows, cols, idxs):
    out.update((row + dr, col + dc) for dr, dc in shapes[idx])
  return out


def generate(width=None, height=None, rows=None, cols=None, idxs=None):
  """Returns input and output grids according to the given parameters.

  Args:
    width: the width of the grid
    height: the height of the grid
    rows: a list of vertical coordinates where toys should be placed
    cols: a list of horizontal coordinates where toys should be placed
    idxs: a list of toy indices (0=box, 1=tall stick, 2=flat stick)
  """
  if width is None:
    width, height = common.randint(9, 11), common.randint(8, 9)
    while True:
      num_boxes = 2 if width < 10 else 3
      while True:
        # The official test example puts toys on row 0, which the old lower bound
        # of 1 excluded.  A 2-row box still fits: row 5 is the last legal start.
        brows = [common.randint(0, 5) for _ in range(num_boxes)]
        bcols = [common.randint(1, 6) for _ in range(num_boxes)]
        bidxs = [0] * num_boxes
        # Check that no two boxes are alongside + parallel to each other.
        parallel = False
        for j in range(num_boxes):
          for i in range(j):
            if brows[i] == brows[j] and abs(bcols[i] - bcols[j]) <= 2:
              parallel = True
            if bcols[i] == bcols[j] and abs(brows[i] - brows[j]) <= 2:
              parallel = True
        if parallel: continue
        bwides, btalls = [2] * num_boxes, [2] * num_boxes
        if not common.overlaps(brows, bcols, bwides, btalls): break
      num_sticks = num_boxes + common.randint(0, 1)
      while True:
        sidxs = [common.randint(1, 2) for _ in range(num_sticks)]
        # Draw the kinds first so the row can depend on them: a flat stick is a
        # single row tall and the first official example has one on row 6, while
        # a tall stick needs three rows and so still stops at row 5.  Row 0 is
        # allowed for the same reason as the boxes above.
        srows = [common.randint(0, 5 if idx == 1 else 6) for idx in sidxs]
        scols = [common.randint(1, 6) for _ in range(num_sticks)]
        # Check that no two sticks are alongside + parallel to each other.
        parallel = False
        for j in range(num_sticks):
          for i in range(j):
            if sidxs[i] != sidxs[j]: continue
            if srows[i] == srows[j] and abs(scols[i] - scols[j]) <= 1:
              parallel = True
            if scols[i] == scols[j] and abs(srows[i] - srows[j]) <= 1:
              parallel = True
        if parallel: continue
        swides = [1 if idx == 1 else 3 for idx in sidxs]
        stalls = [3 if idx == 1 else 1 for idx in sidxs]
        if not common.overlaps(brows + srows, bcols + scols, bwides + swides,
                               btalls + stalls): break
      rows = brows + srows
      cols = bcols + scols
      idxs = bidxs + sidxs
      # Toys that touch can merge into a blob that comes apart more than one
      # way, and the other way round is as good an answer as the one drawn
      # here.  Keep only pictures with a single answer; all four official
      # examples have one already.
      if _answers(_cells(rows, cols, idxs)) == 1: break

  grid, output = common.grids(width, height)
  for row, col, idx in zip(rows, cols, idxs):
    if idx == 0:
      for dr, dc in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        grid[row + dr][col + dc] = common.gray()
        output[row + dr][col + dc] = common.cyan()
    if idx == 1:
      for dr, dc in [(0, 0), (1, 0), (2, 0)]:
        grid[row + dr][col + dc] = common.gray()
        output[row + dr][col + dc] = common.red()
    if idx == 2:
      for dr, dc in [(0, 0), (0, 1), (0, 2)]:
        grid[row + dr][col + dc] = common.gray()
        output[row + dr][col + dc] = common.red()
  return {"input": grid, "output": output}


def validate():
  """Validates the generator."""
  train = [
      generate(width=11, height=9, rows=[1, 3, 4, 3, 2, 6],
               cols=[2, 4, 6, 3, 4, 5], idxs=[0, 0, 0, 1, 2, 2]),
      generate(width=10, height=8, rows=[1, 1, 4, 1, 1, 4],
               cols=[1, 4, 5, 3, 6, 4], idxs=[0, 0, 0, 1, 1, 1]),
      generate(width=9, height=8, rows=[1, 4, 1, 3], cols=[4, 4, 1, 3],
               idxs=[0, 0, 2, 1]),
  ]
  test = [
      generate(width=11, height=8, rows=[0, 2, 5, 0, 2, 4, 1],
               cols=[2, 4, 5, 5, 1, 3, 6], idxs=[0, 0, 0, 2, 2, 2, 1]),
  ]
  return {"train": train, "test": test}
