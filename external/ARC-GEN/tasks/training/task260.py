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


def generate(diag=None, rows=None, cols=None, color=None, size=10):
  """Returns input and output grids according to the given parameters.

  Args:
    diag: a diagonal to be drawn
    rows: a list of rows for the gray corners
    cols: a list of columns for the gray corners
    color: a digit representing a color to be used
    size: the width and height of the (square) grid
  """
  if diag is None:
    diag = common.randint(-5, 5)
    locol, lorow, hicol, hirow = None, None, None, None
    while not (locol or lorow or hicol or hirow):
      # The math below required a lot of trial and error.
      row, col = common.randint(1, size - 2), common.randint(1, size - 2)
      if row - col < diag - 1 and col < (size - diag) and row >= diag:
        lorow, locol = row, col
      if row - col > diag + 1 and col >= -diag and row < (size + diag):
        hirow, hicol = row, col
    # The third official example and the test example both have a corner on
    # BOTH sides of the diagonal.  A single (row, col) can never satisfy both
    # conditions above, and the loop stops as soon as either side is filled in,
    # so two corners was unreachable: half the time, keep drawing until the
    # other side is filled in too (bounded, since the first side already has a
    # corner and a failure to find the second just yields the one-corner case).
    if common.randint(0, 1):
      for _ in range(100):
        if lorow and hirow: break
        row, col = common.randint(1, size - 2), common.randint(1, size - 2)
        if row - col < diag - 1 and col < (size - diag) and row >= diag:
          lorow, locol = row, col
        if row - col > diag + 1 and col >= -diag and row < (size + diag):
          hirow, hicol = row, col
    rows, cols = [], []
    if lorow and locol:
      rows.append(lorow)
      cols.append(locol)
    if hicol and hirow:
      rows.append(hirow)
      cols.append(hicol)
    color = common.random_color(exclude=[common.gray()])

  grid, output = common.grids(size, size)
  for r in range(size):
    for c in range(size):
      mydiag = r - c
      if mydiag == diag: output[r][c] = grid[r][c] = color
      for row, col in zip(rows, cols):
        linediag = row - col
        if diag < mydiag and mydiag <= linediag and r <= row and c >= col:
          grid[r][c] = common.gray()
        if diag > mydiag and mydiag >= linediag and r >= row and c <= col:
          grid[r][c] = common.gray()
        if linediag < diag and linediag == mydiag + 2: output[r][c] = color
        if linediag > diag and linediag == mydiag - 2: output[r][c] = color
  return {"input": grid, "output": output}


def validate():
  """Validates the generator."""
  train = [
      generate(diag=0, rows=[3], cols=[5], color=7),
      generate(diag=-5, rows=[4], cols=[5], color=9),
      generate(diag=1, rows=[3, 7], cols=[4, 3], color=2),
  ]
  test = [
      generate(diag=-1, rows=[1, 8], cols=[4, 4], color=1),
  ]
  return {"train": train, "test": test}
