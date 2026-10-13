"""The offline verifier validity check: sample count, seeding and per-call time limit."""

import random
import time

from framework.dimensions.classification_distribution import verifier_matches_train_test_stable_dynamic50
from framework.grids import Grid, GridPair
from framework.tasks.base import ArcTask


def _plus_one(grid: Grid) -> Grid:
    return [[cell + 1 for cell in row] for row in grid]


def _task(calls: list, *, hang_on: int | None = None) -> ArcTask:
    def generator(n: int, *, rng=None) -> list[GridPair]:
        rng = rng or random.Random()
        calls.append(n)
        pairs = []
        for _ in range(n):
            inp = [[rng.randint(0, 5)]]
            if inp[0][0] == hang_on:
                time.sleep(30)
            pairs.append(GridPair(input=inp, output=_plus_one(inp)))
        return pairs

    return ArcTask(
        task_id="dummy",
        train_pairs=[GridPair(input=[[1]], output=[[2]])],
        test_inputs=[[[3]]],
        test_outputs=[[[4]]],
        arc_gen_generator=generator,
    )


def test_draws_the_requested_number_of_generator_pairs():
    calls: list = []
    assert verifier_matches_train_test_stable_dynamic50(_task(calls), _plus_one, dynamic_samples=7, seed=0)
    assert calls == [1] * 7


def test_seed_gives_every_verifier_the_same_pairs():
    seen = []

    def recording(grid: Grid) -> Grid:
        seen.append(grid[0][0])
        return _plus_one(grid)

    task = _task([])
    for _ in range(2):
        verifier_matches_train_test_stable_dynamic50(task, recording, dynamic_samples=20, seed=3)
    first, second = seen[: len(seen) // 2], seen[len(seen) // 2 :]
    assert first == second


def test_hanging_verifier_counts_as_a_mismatch():
    def hangs_on_five(grid: Grid) -> Grid:
        if grid[0][0] == 5:
            time.sleep(30)
        return _plus_one(grid)

    start = time.monotonic()
    ok = verifier_matches_train_test_stable_dynamic50(
        _task([]), hangs_on_five, dynamic_samples=40, seed=0, call_timeout_s=0.2
    )
    assert not ok
    assert time.monotonic() - start < 5


def test_hanging_generator_draw_is_skipped_not_held_against_the_verifier():
    calls: list = []
    start = time.monotonic()
    ok = verifier_matches_train_test_stable_dynamic50(
        _task(calls, hang_on=0), _plus_one, dynamic_samples=20, seed=0, call_timeout_s=0.2
    )
    assert ok
    assert len(calls) > 20  # the hanging draws were replaced
    assert time.monotonic() - start < 10
