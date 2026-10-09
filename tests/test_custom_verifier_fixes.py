"""Custom verifiers for 29ec7d0e, 3345333e and 3631a71a.

Each must agree with every example the task ships (train, test, ARC-GEN stable,
RE-ARC) and with seeded generator draws, be the canonical slot, and handle the
inputs that broke the verifiers it replaces.
"""

import copy
import importlib
import random

import pytest

from framework.tasks.arc_dataset import load_task
from framework.verifier_selection import select_verifier_for_task

TASKS = ("29ec7d0e", "3345333e", "3631a71a")
GENERATOR_DRAWS = 500


def _solve(task_id):
    return importlib.import_module(f"framework.custom_verifiers.fixes.{task_id}").solve


@pytest.fixture(scope="module", params=TASKS)
def task(request):
    return load_task(request.param)


def _misses(solve, pairs):
    return [k for k, (inp, out) in enumerate(pairs) if solve(copy.deepcopy(inp)) != out]


def test_is_canonical(task):
    slot, _ = select_verifier_for_task(task)
    assert slot == "custom"


def test_matches_train_and_test(task):
    pairs = [(p.input, p.output) for p in task.train_pairs] + list(zip(task.test_inputs, task.test_outputs))
    assert _misses(_solve(task.task_id), pairs) == []


def test_matches_arc_gen_stable(task):
    if not task.arc_gen_synthetic_pairs:
        pytest.skip("ARC-GEN stable set not available")
    pairs = [(p.input, p.output) for p in task.arc_gen_synthetic_pairs]
    assert _misses(_solve(task.task_id), pairs) == []


def test_matches_re_arc(task):
    if not task.re_arc_synthetic_pairs:
        pytest.skip("RE-ARC samples not available")
    pairs = [(p.input, p.output) for p in task.re_arc_synthetic_pairs]
    assert _misses(_solve(task.task_id), pairs) == []


def test_matches_generator_draws(task):
    if task.arc_gen_generator is None:
        pytest.skip("ARC-GEN generator not available")
    rng = random.Random(20261008)
    pairs = []
    while len(pairs) < GENERATOR_DRAWS:
        try:
            pair = task.arc_gen_generator(1, rng=rng)[0]
        except Exception:
            continue
        pairs.append((pair.input, pair.output))
    assert _misses(_solve(task.task_id), pairs) == []


# --- 29ec7d0e -------------------------------------------------------------


def _table(size, modulus):
    return [[(r * c) % modulus + 1 for c in range(size)] for r in range(size)]


def test_29ec7d0e_fills_holes_at_any_size():
    # The golf verifiers always answer with an 18x18 table.
    out = _table(10, 5)
    inp = copy.deepcopy(out)
    for r, c in [(1, 2), (1, 3), (2, 2), (2, 3), (6, 6), (6, 7), (7, 6), (7, 7)]:
        inp[r][c] = 0
    assert _solve("29ec7d0e")(inp) == out


def test_29ec7d0e_rebuilds_tile_cells_hidden_in_every_copy():
    # Period 7 on an 8x8 grid: cell (3, 3) has no other copy, so repetition
    # alone cannot recover it.
    out = _table(8, 7)
    inp = copy.deepcopy(out)
    for r in (2, 3):
        for c in (2, 3, 4):
            inp[r][c] = 0
    assert _solve("29ec7d0e")(inp) == out


def test_29ec7d0e_other_noise_colour():
    tile = [[0, 3, 3], [4, 0, 4]]
    out = [[tile[r % 2][c % 3] for c in range(12)] for r in range(9)]
    inp = copy.deepcopy(out)
    for r in range(2, 5):
        for c in range(5, 8):
            inp[r][c] = 6
    assert _solve("29ec7d0e")(inp) == out


# --- 3345333e -------------------------------------------------------------


def _blank(size=16):
    return [[0] * size for _ in range(size)]


def test_3345333e_restores_hidden_half():
    out = _blank()
    shape = [(3, 5), (3, 8), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (5, 5), (5, 8), (6, 6), (6, 7)]
    for r, c in shape:
        out[r][c] = 6
    inp = copy.deepcopy(out)
    for r in range(3, 7):
        for c in range(8, 12):
            inp[r][c] = 1
    assert _solve("3345333e")(inp) == out


def test_3345333e_rectangular_shape_uses_axis_away_from_block():
    # A rectangle is symmetric about both axes; only the vertical one, which
    # the block does not straddle, restores the whole hidden part.
    out = _blank()
    for r in range(5, 8):
        for c in range(4, 10):
            out[r][c] = 9
    inp = copy.deepcopy(out)
    for r in range(6, 9):
        for c in range(8, 12):
            inp[r][c] = 4
    assert _solve("3345333e")(inp) == out


def test_3345333e_two_rectangles_block_is_the_lower_one():
    out = _blank()
    for r in range(4, 6):
        for c in range(6, 14):
            out[r][c] = 7
    inp = copy.deepcopy(out)
    for r in range(6, 11):
        for c in range(6, 10):
            inp[r][c] = 2
    assert _solve("3345333e")(inp) == out


# --- 3631a71a -------------------------------------------------------------


def test_3631a71a_mask_over_black_cells_only():
    # re_arc treats black as the mask here and copies the mask across the grid.
    task = load_task("3631a71a")
    out = task.train_pairs[0].output
    size = len(out)
    black = [(r, c) for r in range(2, size // 2) for c in range(2, size // 2) if out[r][c] == 0]
    r, c = black[0]
    inp = copy.deepcopy(out)
    inp[r][c] = 9
    assert _solve("3631a71a")(inp) == out


def test_3631a71a_keeps_the_offset_two_layout():
    # A picture symmetric about the centre instead of the offset axis is not
    # this task's structure, so the masked cell is not filled from the centre
    # mirror.
    size = 8
    picture = [[(min(r, size - 1 - r) * 3 + min(c, size - 1 - c)) % 5 + 1 for c in range(size)] for r in range(size)]
    picture = [[picture[min(r, c)][max(r, c)] for c in range(size)] for r in range(size)]
    inp = copy.deepcopy(picture)
    inp[1][6] = 9
    inp[6][1] = 9
    assert _solve("3631a71a")(inp) != picture
