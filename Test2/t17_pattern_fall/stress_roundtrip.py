"""Stress round-trip: official t17 pairs + N ``generate()`` samples.

``generate()`` only returns after a successful ``transform_t17`` check inside the
sampler, so this script does not re-run the verifier on every sample (avoids doubling
the cost of prototype resolution per draw).

With ``--jobs`` > 1, workers use independent RNG seeds (not a single shared
``Random(42)`` stream) but still run the same total number of successful generates.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generator import generate  # noqa: E402


def _verify_t17_json() -> None:
    from verifier import verify_t17

    data = json.loads((ROOT.parent / "t17.json").read_text(encoding="utf-8"))
    for split in ("train", "test"):
        for i, ex in enumerate(data[split]):
            if verify_t17(ex["input"]) != ex["output"]:
                raise SystemExit(f"t17.json mismatch: {split}[{i}]")


def _worker_generate_many(seed: int, count: int) -> tuple[int, float]:
    """Return (count, elapsed_seconds)."""
    rng = random.Random(seed)
    t0 = time.time()
    for _ in range(count):
        generate(rng=rng)
    return count, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--random-samples", type=int, default=600, help="Total successful generate() calls")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel workers (each uses its own RNG seed)")
    args = ap.parse_args()

    _verify_t17_json()
    print("t17.json train+test: OK")

    t0 = time.time()
    if args.jobs <= 1:
        rng = random.Random(42)
        for k in range(args.random_samples):
            generate(rng=rng)
            if (k + 1) % 50 == 0:
                print(f"  {k + 1} generates OK  ({time.time() - t0:.1f}s elapsed)")
        print(f"{args.random_samples} sequential generates OK in {time.time() - t0:.1f}s")
        return

    n = args.random_samples
    j = max(1, args.jobs)
    base = [n // j + (1 if i < n % j else 0) for i in range(j)]
    seeds = [42 + k * 1_000_003 for k in range(j)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(j) as pool:
        parts = pool.starmap(_worker_generate_many, [(seeds[i], base[i]) for i in range(j)])
    total = sum(p[0] for p in parts)
    wall = time.time() - t0
    print(f"{total} generates across {j} workers in {wall:.1f}s wall time")


if __name__ == "__main__":
    main()
