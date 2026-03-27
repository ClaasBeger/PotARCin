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


def _edgefill_output(output, GREEN):
  """Post-process output with edge-fill branch rules (PotARCin adaptation)."""
  h = len(output)
  w = len(output[0]) if h else 0
  if h == 0 or w == 0:
    return output

  def _is_other(v):
    return v != 0 and v != GREEN

  src = output
  out = [list(row) for row in src]
  visited = [[False for _ in range(w)] for _ in range(h)]

  def _can_place_at(rr, cc):
    if _is_other(src[rr][cc]):
      return False
    for dr in (-1, 0, 1):
      for dc in (-1, 0, 1):
        if dr == 0 and dc == 0:
          continue
        nr, nc = rr + dr, cc + dc
        if 0 <= nr < h and 0 <= nc < w and _is_other(src[nr][nc]):
          return False
    return True

  from collections import deque, Counter

  def _contiguous_runs(rows):
    if not rows:
      return []
    rows = sorted(set(rows))
    runs = []
    start = rows[0]
    prev = rows[0]
    for r in rows[1:]:
      if r == prev + 1:
        prev = r
        continue
      runs.append((start, prev))
      start = r
      prev = r
    runs.append((start, prev))
    return runs

  def _prune_non_edge_additions(src_grid, out_grid):
    visited2 = [[False for _ in range(w)] for _ in range(h)]
    for sr2 in range(h):
      for sc2 in range(w):
        if visited2[sr2][sc2] or out_grid[sr2][sc2] != GREEN:
          continue
        q2 = deque([(sr2, sc2)])
        visited2[sr2][sc2] = True
        comp2 = []
        while q2:
          r2, c2 = q2.popleft()
          comp2.append((r2, c2))
          for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr2, nc2 = r2 + dr, c2 + dc
            if 0 <= nr2 < h and 0 <= nc2 < w and not visited2[nr2][nc2] and out_grid[nr2][nc2] == GREEN:
              visited2[nr2][nc2] = True
              q2.append((nr2, nc2))

        rows = sorted({r2 for (r2, _c2) in comp2})
        for a2, b2 in _contiguous_runs(rows):
          run_rows = list(range(a2, b2 + 1))
          touches_side = False
          for rr2 in run_rows:
            cols = [c2 for (r2, c2) in comp2 if r2 == rr2]
            if not cols:
              continue
            if min(cols) == 0 or max(cols) == w - 1:
              touches_side = True
              break
          if touches_side:
            continue
          for rr2 in run_rows:
            for cc2 in range(w):
              if out_grid[rr2][cc2] == GREEN and src_grid[rr2][cc2] != GREEN:
                out_grid[rr2][cc2] = src_grid[rr2][cc2]

  def _prune_non_edge_branches_from_mainbar(src_grid, out_grid):
    def _num_cc(cells):
      if not cells:
        return 0
      cells = set(cells)
      seen = set()
      ncc = 0
      for cell in list(cells):
        if cell in seen:
          continue
        ncc += 1
        stack = [cell]
        seen.add(cell)
        while stack:
          rr, cc = stack.pop()
          for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nn = (rr + dr, cc + dc)
            if nn in cells and nn not in seen:
              seen.add(nn)
              stack.append(nn)
      return ncc

    while True:
      changed = False
      visited3 = [[False for _ in range(w)] for _ in range(h)]
      for sr3 in range(h):
        for sc3 in range(w):
          if visited3[sr3][sc3] or out_grid[sr3][sc3] != GREEN:
            continue
          q3 = deque([(sr3, sc3)])
          visited3[sr3][sc3] = True
          comp3 = []
          while q3:
            r3, c3 = q3.popleft()
            comp3.append((r3, c3))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
              nr3, nc3 = r3 + dr, c3 + dc
              if 0 <= nr3 < h and 0 <= nc3 < w and not visited3[nr3][nc3] and out_grid[nr3][nc3] == GREEN:
                visited3[nr3][nc3] = True
                q3.append((nr3, nc3))

          rows = sorted({r3 for (r3, _c3) in comp3})
          if not rows:
            continue
          row_min = {}
          row_max = {}
          row_w = {}
          for rr3 in rows:
            cs = [cc3 for (r3, cc3) in comp3 if r3 == rr3]
            row_min[rr3] = min(cs)
            row_max[rr3] = max(cs)
            row_w[rr3] = row_max[rr3] - row_min[rr3] + 1

          main_w = Counter(row_w.values()).most_common(1)[0][0]
          main_rows = [r for r in rows if row_w[r] == main_w]
          if not main_rows:
            continue
          core_l = Counter([row_min[r] for r in main_rows]).most_common(1)[0][0]
          core_r = core_l + main_w - 1
          comp_set = set(comp3)

          for rr3 in rows:
            if row_w[rr3] == main_w:
              continue
            l = row_min[rr3]
            r = row_max[rr3]
            left_nonedge = l < core_l and l > 0
            right_nonedge = r > core_r and r < w - 1
            reaches_left = l == 0
            reaches_right = r == w - 1

            if left_nonedge and reaches_right:
              cand = {
                  (rr3, cc3)
                  for cc3 in range(l, core_l)
                  if out_grid[rr3][cc3] == GREEN and src_grid[rr3][cc3] != GREEN
              }
              remain = comp_set - cand
              if _num_cc(remain) <= 1:
                for rrn, ccn in cand:
                  out_grid[rrn][ccn] = 0
                  changed = True
                comp_set = remain
            if right_nonedge and reaches_left:
              cand = {
                  (rr3, cc3)
                  for cc3 in range(core_r + 1, r + 1)
                  if out_grid[rr3][cc3] == GREEN and src_grid[rr3][cc3] != GREEN
              }
              remain = comp_set - cand
              if _num_cc(remain) <= 1:
                for rrn, ccn in cand:
                  out_grid[rrn][ccn] = 0
                  changed = True
                comp_set = remain

      if not changed:
        break

  for sr in range(h):
    for sc in range(w):
      if visited[sr][sc] or src[sr][sc] != GREEN:
        continue
      q = deque([(sr, sc)])
      visited[sr][sc] = True
      comp = []
      touches_left = sc == 0
      touches_right = sc == w - 1
      while q:
        r, c = q.popleft()
        comp.append((r, c))
        if c == 0:
          touches_left = True
        if c == w - 1:
          touches_right = True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
          nr, nc = r + dr, c + dc
          if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and src[nr][nc] == GREEN:
            visited[nr][nc] = True
            q.append((nr, nc))
      if not (touches_left or touches_right):
        continue
      if touches_left:
        left_rows = [r for (r, c) in comp if c == 0]
        for a, b in _contiguous_runs(left_rows):
          row_span = list(range(a, b + 1))
          eligible_rows = [rr for rr in row_span if _can_place_at(rr, 0)]
          for ea, eb in _contiguous_runs(eligible_rows):
            run_rows = list(range(ea, eb + 1))
            end = -1
            for cc in range(w):
              if all(_can_place_at(rr, cc) for rr in run_rows):
                end = cc
              else:
                break
            if end >= 0:
              for rr in run_rows:
                for cc in range(0, end + 1):
                  out[rr][cc] = GREEN
      if touches_right:
        right_rows = [r for (r, c) in comp if c == w - 1]
        for a, b in _contiguous_runs(right_rows):
          row_span = list(range(a, b + 1))
          eligible_rows = [rr for rr in row_span if _can_place_at(rr, w - 1)]
          for ea, eb in _contiguous_runs(eligible_rows):
            run_rows = list(range(ea, eb + 1))
            start = w
            for cc in range(w - 1, -1, -1):
              if all(_can_place_at(rr, cc) for rr in run_rows):
                start = cc
              else:
                break
            if start < w:
              for rr in run_rows:
                for cc in range(start, w):
                  out[rr][cc] = GREEN
        # Conservative right-edge closure for rows one column short.
        comp_rows = sorted({r for (r, _c) in comp})
        row_to_max = {}
        for rr in comp_rows:
          cols = [cc for (r2, cc) in comp if r2 == rr]
          if cols:
            row_to_max[rr] = max(cols)
        right_row_set = set(right_rows)
        for a, b in _contiguous_runs(comp_rows):
          run = list(range(a, b + 1))
          if not any(r in right_row_set for r in run):
            continue
          for rr in run:
            if row_to_max.get(rr, -1) == w - 2 and _can_place_at(rr, w - 1):
              out[rr][w - 1] = GREEN

  _prune_non_edge_additions(src, out)
  _prune_non_edge_branches_from_mainbar(src, out)
  return out


def generate(colors=None, size=30):
  """Returns input and output grids according to the given parameters.

  Args:
    colors: a list of digits representing the colors to be used
    size: the width and height of the (square) grid
  """
  if colors is None:
    color = common.random_color(exclude=[common.green()])
    bitmap = common.grid(size, size)
    for r, c in common.random_pixels(size, size, 0.5):
      bitmap[r][c] = color
    rows, cols, wides, talls = [], [], [], []
    # Add the artery
    rows.append(-1 if common.randint(0, 3) else 4)
    cols.append(common.randint(5, 10))
    wides.append(common.randint(6, 12))
    talls.append(size + 2)
    # Maybe add a big vein on the left.
    if common.randint(0, 1):
      rows.append(rows[0] + common.randint(6, 12))
      cols.append(-1)
      wides.append(cols[0] + wides[0])
      talls.append(common.randint(10, 14))
    # Or, maybe add a couple small veins on the left.
    elif common.randint(0, 1):
      rows.append(rows[0] + common.randint(6, 9))
      cols.append(-1)
      wides.append(cols[0] + wides[0])
      talls.append(common.randint(3, 4))
      rows.append(rows[1] + common.randint(6, 9))
      cols.append(-1)
      wides.append(cols[0] + wides[0])
      talls.append(common.randint(3, 4))
    # On the right, maybe add a vein up high.
    if common.randint(0, 1):
      rows.append(rows[0] + common.randint(6, 9))
      cols.append(cols[0] + wides[0] - 2)
      wides.append(size)
      talls.append(common.randint(3, 4))
    # And maybe another much lower?
    if common.randint(0, 1):
      rows.append(rows[0] + common.randint(18, 24))
      cols.append(cols[0] + wides[0] - 2)
      wides.append(size)
      talls.append(common.randint(3, 4))
    # For each box, remove its outline.
    for row, col, wide, tall in zip(rows, cols, wides, talls):
      for r in range(row, row + tall):
        for c in range(col, col + wide):
          common.draw(bitmap, r, c, common.black())
    for row, col, wide, tall in zip(rows, cols, wides, talls):
      for r in range(row + 1, row + tall - 1):
        for c in range(col + 1, col + wide - 1):
          common.draw(bitmap, r, c, common.green())
    # Sometimes we transpose:
    if common.randint(0, 1): bitmap = common.transpose(bitmap)
    # Sometimes we flip:
    if common.randint(0, 1): bitmap = bitmap[::-1]
    colors = []
    for row in bitmap:
      colors.extend(row)

  grid, output = common.grids(size, size)
  for r in range(size):
    for c in range(size):
      color = colors[r * size + c]
      grid[r][c] = color if color != common.green() else common.black()
      output[r][c] = color

  # PotARCin adaptation: run edge-fill post-process directly in task255.py so
  # a64e4611 uses the custom generator behavior without external rewiring.
  output = _edgefill_output(output, common.green())
  for r in range(size):
    for c in range(size):
      grid[r][c] = common.black() if output[r][c] == common.green() else output[r][c]
  return {"input": grid, "output": output}


def validate():
  """Validates the generator."""
  train = [
      generate(
          colors=[8, 8, 0, 8, 0, 8, 0, 8, 8, 8, 8, 8, 0, 8, 8, 8, 0, 8, 0, 0, 8,
                  0, 8, 0, 0, 0, 8, 8, 0, 8, 0, 0, 0, 8, 8, 8, 8, 0, 0, 8, 0, 8,
                  0, 0, 8, 8, 0, 0, 8, 0, 0, 0, 0, 0, 8, 8, 8, 8, 0, 8, 8, 0, 0,
                  0, 8, 8, 0, 0, 8, 0, 8, 8, 0, 8, 8, 0, 8, 0, 8, 0, 8, 8, 8, 8,
                  0, 0, 8, 0, 0, 0, 0, 8, 8, 0, 0, 0, 0, 8, 8, 0, 0, 0, 0, 8, 8,
                  0, 8, 8, 0, 0, 0, 8, 8, 0, 8, 0, 0, 0, 0, 0, 8, 8, 8, 0, 8, 0,
                  0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 8, 0, 8, 8,
                  0, 0, 8, 0, 8, 0, 0, 0, 8, 8, 8, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0,
                  8, 8, 8, 8, 0, 8, 0, 8, 0, 0, 0, 8, 0, 8, 8, 8, 8, 0, 0, 8, 0,
                  3, 3, 3, 3, 3, 3, 3, 0, 8, 8, 8, 0, 0, 0, 0, 0, 8, 0, 8, 8, 8,
                  0, 8, 8, 8, 8, 0, 0, 8, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 8, 0,
                  8, 0, 8, 8, 8, 0, 0, 8, 8, 8, 0, 8, 8, 0, 8, 8, 8, 0, 3, 3, 3,
                  3, 3, 3, 3, 0, 8, 0, 0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 8, 8, 8, 8,
                  0, 8, 8, 0, 8, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 8, 0, 8, 8, 0, 0, 8, 8, 0, 3, 3, 3, 3, 3, 3,
                  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 8, 8, 0, 0, 0,
                  0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                  3, 3, 3, 0, 8, 0, 0, 8, 0, 0, 8, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 0, 0, 0, 0,
                  3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 8, 0, 8, 8, 0, 8, 8, 0, 8,
                  8, 0, 8, 8, 0, 0, 8, 8, 0, 3, 3, 3, 3, 3, 3, 3, 0, 8, 8, 0, 8,
                  0, 0, 0, 8, 0, 0, 0, 8, 8, 8, 0, 8, 0, 0, 8, 8, 0, 0, 3, 3, 3,
                  3, 3, 3, 3, 0, 0, 0, 8, 0, 8, 8, 0, 0, 0, 8, 0, 8, 8, 0, 0, 8,
                  8, 8, 8, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 8, 8, 0, 0, 0, 8, 8,
                  0, 8, 8, 0, 0, 8, 8, 0, 8, 0, 0, 8, 8, 8, 0, 3, 3, 3, 3, 3, 3,
                  3, 0, 8, 8, 8, 8, 0, 8, 8, 0, 0, 0, 8, 8, 0, 8, 0, 8, 8, 0, 8,
                  0, 8, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 8, 8, 8, 0, 8, 0, 8, 0, 8,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0,
                  8, 0, 0, 0, 0, 8, 8, 8, 0, 8, 8, 8, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                  3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 8, 0, 8, 0,
                  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 8, 0, 0, 0,
                  0, 8, 0, 8, 8, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3,
                  3, 3, 3, 3, 0, 0, 8, 0, 0, 0, 0, 8, 0, 8, 0, 8, 8, 8, 8, 8, 0,
                  0, 8, 8, 0, 8, 0, 3, 3, 3, 3, 3, 3, 3, 0, 8, 8, 8, 0, 8, 0, 0,
                  0, 0, 8, 8, 8, 8, 0, 8, 8, 8, 8, 0, 0, 8, 0, 3, 3, 3, 3, 3, 3,
                  3, 0, 8, 0, 8, 0, 8, 0, 8, 8, 0, 0, 0, 8, 8, 0, 8, 8, 0, 8, 8,
                  8, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 8, 0, 8, 0,
                  8, 0, 8, 8, 0, 8, 8, 8, 0, 8, 8, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0,
                  8, 8, 8, 0, 8, 0, 8, 8, 0, 0, 0, 8, 8, 0, 8, 0, 8, 0, 8, 0, 0,
                  3, 3, 3, 3, 3, 3, 3, 0, 8, 0, 8, 8, 0, 8, 8, 0, 8, 0, 0, 8, 0,
                  0, 8, 8, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 8, 0, 0,
                  8, 8, 0, 8, 8, 8, 0, 0, 0, 8, 8, 8, 0, 8, 0, 0, 8, 0, 3, 3, 3,
                  3, 3, 3, 3, 0, 8, 8, 0, 8, 0, 8, 8, 0, 8, 0, 8, 8, 0]),
      generate(
          colors=[1, 1, 1, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 1, 0, 1, 0, 0, 0, 1,
                  1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 3, 3, 3, 3, 3, 3,
                  3, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1,
                  1, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1,
                  0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 3, 3, 3, 3, 3, 3, 3, 0, 1,
                  1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0,
                  3, 3, 3, 3, 3, 3, 3, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1,
                  0, 0, 1, 0, 0, 1, 1, 1, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 1, 0,
                  0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 3, 3, 3,
                  3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  1, 1, 1, 0, 1, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                  3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 1, 0, 1, 0, 3, 3, 3, 3, 3, 3,
                  3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
                  1, 1, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0,
                  0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0,
                  0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,
                  3, 3, 3, 3, 3, 3, 3, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                  0, 0, 1, 1, 1, 1, 0, 1, 0, 3, 3, 3, 3, 3, 3, 3, 0, 1, 0, 1, 0,
                  0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 3, 3, 3,
                  3, 3, 3, 3, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1,
                  0, 1, 1, 0, 1, 0, 3, 3, 3, 3, 3, 3, 3, 0, 1, 1, 0, 0, 1, 0, 1,
                  1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 3, 3, 3, 3, 3, 3,
                  3, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1,
                  0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 0, 1, 0, 0, 0,
                  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                  3, 3, 3, 1, 1, 0, 1, 1, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 3, 3, 3,
                  3, 3, 3, 3, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1,
                  0, 0, 1, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 1, 1, 0, 0, 0, 0, 0,
                  1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3,
                  3, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0,
                  1, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
                  0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0,
                  0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0,
                  3, 3, 3, 3, 3, 3, 3, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1,
                  1, 0, 0, 0, 0, 1, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 1, 1, 0, 0,
                  0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 3, 3, 3,
                  3, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1,
                  1, 0, 1, 0, 1, 0, 3, 3, 3, 3, 3, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0,
                  1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 3, 3, 3, 3, 3, 3,
                  3, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0]),
      generate(
          colors=[0, 2, 0, 2, 2, 2, 2, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 2, 0, 0, 0,
                  0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 3, 3, 3, 3,
                  3, 3, 3, 3, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 2, 2, 0, 0, 2, 0,
                  0, 0, 2, 2, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 2, 0, 2, 0, 0, 0, 0,
                  2, 0, 2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 0, 3, 3, 3, 3, 3, 3, 3,
                  3, 0, 2, 2, 0, 0, 2, 2, 0, 2, 2, 0, 0, 0, 0, 0, 2, 0, 0, 2, 2,
                  0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 2, 0, 0, 0, 0, 2, 2, 0, 0,
                  2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 2,
                  2, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0, 0, 3,
                  3, 3, 3, 3, 3, 3, 3, 0, 2, 2, 0, 2, 0, 0, 2, 2, 2, 0, 0, 0, 0,
                  2, 2, 2, 2, 0, 0, 2, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 2, 0, 0,
                  0, 2, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3,
                  3, 3, 3, 3, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 2, 2, 0, 2, 3, 3, 3,
                  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 2, 0, 0, 0, 0, 0, 0,
                  0, 0, 2, 2, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                  3, 0, 0, 0, 2, 2, 0, 2, 2, 2, 0, 0, 0, 0, 2, 3, 3, 3, 3, 3, 3,
                  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 2, 2, 2, 2, 0, 2, 0, 0,
                  0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0,
                  0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                  3, 3, 3, 3, 3, 3, 3, 0, 2, 0, 0, 0, 0, 0, 2, 2, 0, 2, 0, 0, 2,
                  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 2, 0, 0, 0,
                  0, 2, 0, 0, 2, 0, 0, 2, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                  3, 3, 3, 3, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 0, 2, 0, 3, 3, 3,
                  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 2, 0, 2, 2, 0, 0,
                  2, 2, 0, 0, 0, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                  3, 0, 0, 2, 2, 2, 2, 0, 2, 2, 2, 0, 2, 0, 2, 3, 3, 3, 3, 3, 3,
                  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0,
                  0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 3,
                  3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 2, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0,
                  0, 0, 2, 0, 0, 2, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0,
                  2, 2, 0, 0, 0, 2, 2, 0, 2, 0, 0, 2, 2, 0, 2, 0, 0, 3, 3, 3, 3,
                  3, 3, 3, 3, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 2, 0,
                  0, 0, 2, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 2, 0, 0, 2, 0, 0,
                  0, 0, 0, 2, 0, 2, 2, 0, 2, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3,
                  3, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 2, 0,
                  0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 2, 2, 0, 0, 2, 0, 0, 2, 2,
                  0, 0, 0, 2, 0, 0, 0, 2, 2, 2, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 2,
                  0, 0, 2, 0, 2, 0, 0, 0, 2, 2, 2, 2, 0, 2, 0, 2, 0, 0, 2, 0, 3,
                  3, 3, 3, 3, 3, 3, 3, 0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 2, 0, 2, 2,
                  2, 0, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 2, 0, 0, 0,
                  2, 2, 0, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2, 2, 0, 0, 0, 3, 3, 3, 3,
                  3, 3, 3, 3, 0, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 2, 2]),
  ]
  test = [
      generate(
          colors=[0, 4, 4, 0, 4, 0, 4, 4, 0, 0, 0, 3, 3, 0, 4, 4, 4, 4, 4, 0, 3,
                  0, 4, 0, 4, 4, 4, 0, 0, 0, 4, 4, 4, 0, 0, 4, 4, 0, 0, 0, 0, 3,
                  3, 0, 0, 4, 4, 0, 4, 0, 3, 0, 0, 0, 0, 4, 4, 0, 4, 4, 0, 0, 0,
                  4, 0, 0, 0, 0, 4, 4, 0, 3, 3, 0, 0, 4, 0, 4, 4, 0, 3, 0, 4, 4,
                  0, 0, 4, 0, 0, 4, 4, 0, 0, 0, 4, 4, 4, 0, 4, 0, 0, 3, 3, 0, 4,
                  0, 0, 0, 4, 0, 3, 0, 0, 0, 0, 0, 4, 0, 0, 4, 4, 0, 4, 4, 4, 0,
                  4, 0, 0, 4, 0, 3, 3, 0, 0, 0, 0, 4, 0, 0, 3, 0, 0, 4, 0, 4, 4,
                  0, 4, 0, 0, 0, 4, 0, 4, 0, 0, 0, 4, 4, 0, 3, 3, 0, 4, 0, 0, 4,
                  0, 0, 3, 0, 0, 4, 0, 0, 0, 0, 0, 0, 4, 0, 4, 4, 0, 0, 4, 0, 0,
                  4, 0, 3, 3, 0, 4, 0, 4, 4, 0, 0, 3, 0, 4, 0, 0, 4, 4, 0, 4, 4,
                  0, 4, 0, 4, 4, 4, 0, 4, 0, 0, 0, 3, 3, 0, 0, 0, 0, 4, 0, 0, 3,
                  0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 4, 4, 4, 0, 0, 3,
                  3, 0, 4, 4, 0, 0, 4, 0, 3, 0, 0, 0, 0, 0, 4, 0, 4, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                  3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3,
                  3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0,
                  0, 0, 0, 0, 4, 0, 4, 0, 3, 3, 0, 0, 4, 0, 0, 4, 0, 0, 0, 4, 0,
                  4, 4, 0, 0, 4, 4, 4, 4, 0, 4, 4, 0, 0, 4, 0, 0, 0, 3, 3, 0, 0,
                  4, 0, 4, 0, 0, 4, 0, 0, 4, 4, 4, 0, 4, 0, 0, 0, 0, 4, 0, 4, 4,
                  4, 0, 4, 0, 0, 3, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 4, 4, 4, 4,
                  4, 4, 4, 0, 4, 4, 0, 4, 0, 0, 0, 0, 0, 0, 3, 3, 0, 4, 4, 4, 0,
                  4, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0,
                  4, 0, 3, 3, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 4, 0, 0,
                  4, 0, 4, 0, 0, 0, 4, 0, 0, 0, 0, 3, 3, 0, 0, 0, 4, 4, 0, 0, 0,
                  0, 0, 0, 4, 0, 4, 4, 0, 4, 4, 0, 4, 4, 0, 0, 4, 4, 0, 0, 0, 3,
                  3, 0, 0, 0, 4, 4, 4, 0, 0, 4, 0, 4, 0, 4, 0, 0, 4, 4, 0, 4, 4,
                  4, 4, 0, 4, 0, 0, 0, 0, 3, 3, 0, 4, 0, 4, 4, 0, 4, 0, 0, 0, 4,
                  0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 4, 4, 0, 4, 0, 3, 3, 0, 4,
                  4, 0, 4, 0, 0, 4, 0, 4, 0, 4, 0, 4, 4, 4, 0, 4, 0, 4, 0, 0, 0,
                  4, 0, 0, 4, 0, 3, 3, 0, 4, 0, 4, 4, 0, 0, 0, 0, 4, 0, 4, 4, 0,
                  4, 0, 4, 0, 4, 0, 4, 0, 0, 0, 0, 0, 4, 0, 3, 3, 0, 4, 0, 4, 0,
                  4, 4, 4, 0, 0, 4, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 4, 0, 4, 4, 0,
                  0, 0, 3, 3, 0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 4, 0, 0, 4, 4,
                  4, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 4,
                  0, 4, 0, 4, 4, 4, 0, 4, 4, 0, 0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 3,
                  3, 0, 0, 4, 0, 0, 4, 0, 0, 0, 0, 4, 0, 4, 4, 0, 0, 0]),
  ]
  return {"train": train, "test": test}
