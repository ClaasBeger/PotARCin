from __future__ import annotations

from pathlib import Path
import re

from framework.tasks.arc_dataset import load_task
from framework.grids import is_equal_grid


def main() -> None:
    # Use the latest smaller run log for reverification.
    log_path = Path("caller_run_2026-03-17_14-02-46.log")
    if not log_path.exists():
        print("Log file not found:", log_path)
        return

    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    tasks: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    in_block = False

    for line in lines:
        line = line.rstrip()
        if line.startswith("Task: "):
            current = {"id": line.split("Task: ", 1)[1].strip(), "lines": []}
            tasks.append(current)
            in_block = True
            continue
        if in_block and line.startswith("=== Verifier check on ARC-GEN examples"):
            continue
        if in_block and line.startswith("========"):
            in_block = False
            current = None
            continue
        if in_block and current is not None:
            current["lines"].append(line)

    # We want tasks where the log shows v1, v2, v4 but NO v3 line at all,
    # i.e. v3 was absent in that run while the task was logged because all
    # present verifiers failed.
    cands: list[str] = []
    for b in tasks:
        tid = b["id"]
        assert isinstance(tid, str)
        blk = "\n".join(b["lines"])  # type: ignore[arg-type]
        if (
            "v1 (re_arc):" in blk
            and "v2 (google-code-golf-2025):" in blk
            and "v4 (NeurIPS-Code-Golf-2025):" in blk
            and "v3 (golf):" not in blk
        ):
            cands.append(tid)

    cands = sorted(set(cands))
    print("Found", len(cands), "candidate tasks where v3 was absent in that run.")
    if not cands:
        return
    print("\nTasks:", ", ".join(cands))
    print("\nFull v3 verification for these tasks (train/test/stable/dynamic):\n")

    for tid in cands:
        print("=" * 80)
        print("Task:", tid)
        t = load_task(tid)
        v3 = getattr(t, "tertiary_verifier", None)
        if v3 is None:
            print("  v3 missing; skipping.")
            continue

        # 1) Train
        if t.train_pairs:
            ok = 0
            fail = 0
            for i, pair in enumerate(t.train_pairs):
                try:
                    out = v3(pair.input)
                    if is_equal_grid(out, pair.output):
                        ok += 1
                    else:
                        fail += 1
                        print(f"  train mismatch at pair {i}: expected vs got differ")
                except Exception as e:
                    fail += 1
                    print(f"  train error at pair {i}: {e}")
            print(f"  train: ok={ok}, fail={fail}, total={len(t.train_pairs)}")
        else:
            print("  train: no train pairs")

        # 2) Test (if primary labels available)
        if t.test_inputs and t.verifier is not None:
            primary = t.verifier
            try:
                primary_outs = [primary(inp) for inp in t.test_inputs]
            except Exception as e:
                print("  test: primary labeler failed:", e)
                primary_outs = None

            if primary_outs is not None:
                ok = 0
                fail = 0
                for i, inp in enumerate(t.test_inputs):
                    try:
                        out = v3(inp)
                        if is_equal_grid(out, primary_outs[i]):
                            ok += 1
                        else:
                            fail += 1
                            print(f"  test mismatch at input {i}: expected vs got differ")
                    except Exception as e:
                        fail += 1
                        print(f"  test error at input {i}: {e}")
                print(f"  test: ok={ok}, fail={fail}, total={len(t.test_inputs)}")
        else:
            print("  test: n/a (no test inputs or no primary verifier)")

        # 3) Stable ARC-GEN
        if t.arc_gen_synthetic_pairs:
            pairs = t.arc_gen_synthetic_pairs
            ok = 0
            fail = 0
            for i, pair in enumerate(pairs):
                try:
                    out = v3(pair.input)
                    if is_equal_grid(out, pair.output):
                        ok += 1
                    else:
                        fail += 1
                        print(f"  stable mismatch at pair {i}: expected vs got differ")
                except Exception as e:
                    fail += 1
                    print(f"  stable error at pair {i}: {e}")
            print(f"  stable ARC-GEN: ok={ok}, fail={fail}, total={len(pairs)}")
        else:
            print("  stable ARC-GEN: n/a (no pairs)")

        # 4) Dynamic ARC-GEN (25 examples, like caller)
        if t.arc_gen_generator is not None:
            dyn_pairs = t.arc_gen_generator(25)
            ok = 0
            fail = 0
            for i, pair in enumerate(dyn_pairs):
                try:
                    out = v3(pair.input)
                    if is_equal_grid(out, pair.output):
                        ok += 1
                    else:
                        fail += 1
                        print(f"  dynamic mismatch at pair {i}: expected vs got differ")
                except Exception as e:
                    fail += 1
                    print(f"  dynamic error at pair {i}: {e}")
            print(f"  dynamic ARC-GEN (25): ok={ok}, fail={fail}, total={len(dyn_pairs)}")
        else:
            print("  dynamic ARC-GEN: n/a (no generator)")


if __name__ == "__main__":
    main()

