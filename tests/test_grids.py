from framework.grids import (
    Grid,
    batch_equal,
    clone_grid,
    is_equal_grid,
    validate_grid,
)


def test_is_equal_grid_basic():
    a: Grid = [[1, 2], [3, 4]]
    b: Grid = [[1, 2], [3, 4]]
    c: Grid = [[1, 2], [4, 3]]
    assert is_equal_grid(a, b)
    assert not is_equal_grid(a, c)


def test_clone_grid_creates_independent_copy():
    a: Grid = [[1, 2], [3, 4]]
    b = clone_grid(a)
    assert is_equal_grid(a, b)
    b[0][0] = 9
    assert not is_equal_grid(a, b)


def test_batch_equal():
    seq1 = [[[1]], [[2]]]
    seq2 = [[[1]], [[2]]]
    seq3 = [[[1]], [[3]]]
    assert batch_equal(seq1, seq2)
    assert not batch_equal(seq1, seq3)


def test_validate_grid_rejects_non_rectangular():
    bad = [[1], [2, 3]]
    try:
        validate_grid(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for non-rectangular grid")

