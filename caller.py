from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from framework.tasks.arc_dataset import load_task, iter_tasks
from framework.tasks.base import ArcTask, TaskSource
from framework.grids import Grid, pretty_grid, is_equal_grid


# Simple ANSI color helpers for nicer CLI output.
GREEN = "\x1b[32m"
RED = "\x1b[31m"
YELLOW = "\x1b[33m"
RESET = "\x1b[0m"

# ARC color palette (same as Viewer).
_SYMBOL_TO_COLOR = {
    0: "#000000",
    1: "#0074D9",
    2: "#FF4136",
    3: "#2ECC40",
    4: "#FFDC00",
    5: "#AAAAAA",
    6: "#F012BE",
    7: "#FF851B",
    8: "#7FDBFF",
    9: "#870C25",
}


def _grid_to_html(grid: Grid, title: str, cell_px: int = 20) -> str:
    """Return HTML for one grid with colored cells (viewer-style)."""
    from html import escape
    h = len(grid)
    w = len(grid[0]) if h else 0
    rows = []
    for r in range(h):
        cells = []
        for c in range(w):
            sym = int(grid[r][c])
            color = _SYMBOL_TO_COLOR.get(sym, "#000000")
            cells.append(
                f'<div style="width:{cell_px}px;height:{cell_px}px;background:{color};'
                'border:1px solid #555;box-sizing:border-box;display:inline-block;'
                'vertical-align:top;"></div>'
            )
        rows.append("<div>" + "".join(cells) + "</div>")
    return f'<div style="margin:8px;"><div style="font-weight:600;margin-bottom:4px;">{escape(title)}</div>' + "".join(rows) + "</div>"


def _write_mismatch_html(
    out_dir: Path,
    task_id: str,
    index: int,
    input_grid: Grid,
    expected: Grid,
    got: Grid,
) -> Path:
    """Write a single HTML file showing input, expected, and got side by side."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{task_id}_dynamic_{index}.html"
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'/><title>Mismatch "
        + task_id
        + " dynamic "
        + str(index)
        + "</title></head><body style='font-family:sans-serif;'><h2>"
        + task_id
        + " — dynamic example "
        + str(index)
        + "</h2><div style='display:flex;flex-wrap:wrap;gap:16px;'>"
        + _grid_to_html(input_grid, "Input")
        + _grid_to_html(expected, "Expected (ARC-GEN)")
        + _grid_to_html(got, "Got (verifier)")
        + "</div></body></html>"
    )
    path.write_text(html, encoding="utf-8")
    return path


def _progress_bar(ok: int, total: int, width: int = 30) -> str:
    """Return a simple ASCII progress bar string."""
    if total <= 0:
        return "[" + " " * width + "]"
    ratio = ok / total
    filled = int(width * ratio)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def inspect_task(task_id: str) -> ArcTask:
    """Print a quick summary of a single task and return it."""
    task: ArcTask = load_task(task_id)

    print("Task ID:", task.task_id)
    print("Num train pairs:", len(task.train_pairs))
    print("Num test inputs:", len(task.test_inputs))

    if task.train_pairs:
        print("\nFirst train input grid:")
        print(pretty_grid(task.train_pairs[0].input))
        print("\nFirst train output grid:")
        print(pretty_grid(task.train_pairs[0].output))

    print("\nHas re_arc synthetic pairs?", task.re_arc_synthetic_pairs is not None)
    if task.re_arc_synthetic_pairs is not None:
        print("  Num re_arc synthetic pairs:", len(task.re_arc_synthetic_pairs))
        print("  First re_arc pair:", task.re_arc_synthetic_pairs[0])
    print("Has arc_gen synthetic pairs?", task.arc_gen_synthetic_pairs is not None)
    if task.arc_gen_synthetic_pairs is not None:
        print("  Num arc_gen synthetic pairs:", len(task.arc_gen_synthetic_pairs))
        print("  First arc_gen pair:", task.arc_gen_synthetic_pairs[0])
    print("Has re_arc generator?", task.re_arc_generator is not None)
    if task.re_arc_generator is not None:
        print("  Sample from re_arc generator (1 example):", task.re_arc_generator(1)[0])
    print("Has arc_gen generator?", task.arc_gen_generator is not None)
    if task.arc_gen_generator is not None:
        print("  Sample from arc_gen generator (1 example):", task.arc_gen_generator(1)[0])
    print("Has primary verifier?", task.verifier is not None)
    if task.verifier is not None and task.train_pairs:
        print("  Verifier:", task.verifier)
        print(
            "  Verifier output on first train input:",
            task.verifier(task.train_pairs[0].input),
        )
    print("Has secondary verifier?", task.secondary_verifier is not None)
    if task.secondary_verifier is not None and task.train_pairs:
        print("  Secondary verifier:", task.secondary_verifier)
        print(
            "  Secondary output on first train input:",
            task.secondary_verifier(task.train_pairs[0].input),
        )

    return task


def check_verifiers_on_arc_gen(task: ArcTask, max_examples: int = 5) -> dict:
    """Check verifiers on original ARC data and ARC-GEN examples.

    For a given task, this function runs available verifiers on:
    - original ARC training pairs
    - original ARC test inputs (if labels are available via primary verifier)
    - all stable ARC-GEN synthetic pairs
    - a handful of dynamically generated ARC-GEN examples (if the generator exists)
    and prints simple match/mismatch statistics, returning a small stats dict.
    """
    print("\n=== Verifier check on ARC-GEN examples ===")

    if task.verifier is None and task.secondary_verifier is None:
        print("No verifiers available; skipping check.")
        return {
            "primary": {"train_ok": None, "test_ok": None, "stable_ok": None, "dynamic_ok": None},
            "secondary": {"train_ok": None, "test_ok": None, "stable_ok": None, "dynamic_ok": None},
        }

    stats = {
        "primary": {
            "train_ok": None,
            "test_ok": None,
            "stable_ok": None,
            "dynamic_ok": None,
            "dynamic_mismatches": [],
        },
        "secondary": {
            "train_ok": None,
            "test_ok": None,
            "stable_ok": None,
            "dynamic_ok": None,
            "dynamic_mismatches": [],
        },
    }

    # 0) Original ARC training pairs.
    if task.train_pairs:
        print(f"\nOriginal ARC train pairs: {len(task.train_pairs)}")
        for name, key, verifier in (
            ("primary (re_arc)", "primary", task.verifier),
            ("secondary (NeurIPS)", "secondary", task.secondary_verifier),
        ):
            if verifier is None:
                continue
            ok = 0
            fail = 0
            total = len(task.train_pairs)
            for i, pair in enumerate(task.train_pairs, start=1):
                try:
                    out = verifier(pair.input)
                    if is_equal_grid(out, pair.output):
                        ok += 1
                    else:
                        fail += 1
                        print(
                            f"  [{name}] train mismatch at pair {i-1}: expected vs got differ"
                        )
                except Exception as e:
                    fail += 1
                    print(f"  [{name}] train error at pair {i-1}: {e}")
            all_ok = fail == 0 and ok > 0
            stats[key]["train_ok"] = all_ok
            bar = _progress_bar(ok, total)
            color = GREEN if all_ok else RED
            print(
                f"  [{name}] train: {color}{bar}{RESET} (ok={ok}, fail={fail})"
            )
    else:
        print("\nNo original ARC train pairs available.")

    # 0b) Original ARC test inputs, if we can get labels via primary verifier.
    if task.test_inputs and task.verifier is not None:
        print(f"\nOriginal ARC test inputs: {len(task.test_inputs)}")
        # Use primary as a "label generator" for test; others should match it.
        primary = task.verifier
        primary_outputs = [primary(inp) for inp in task.test_inputs]

        for name, key, verifier in (
            ("primary (re_arc)", "primary", task.verifier),
            ("secondary (NeurIPS)", "secondary", task.secondary_verifier),
        ):
            if verifier is None:
                continue
            ok = 0
            fail = 0
            total = len(task.test_inputs)
            for i, inp in enumerate(task.test_inputs, start=1):
                try:
                    out = verifier(inp)
                    if is_equal_grid(out, primary_outputs[i - 1]):
                        ok += 1
                    else:
                        fail += 1
                        print(
                            f"  [{name}] test mismatch at input {i-1}: expected vs got differ"
                        )
                except Exception as e:
                    fail += 1
                    print(f"  [{name}] test error at input {i-1}: {e}")
            all_ok = fail == 0 and ok > 0
            stats[key]["test_ok"] = all_ok
            bar = _progress_bar(ok, total)
            color = GREEN if all_ok else RED
            print(
                f"  [{name}] test: {color}{bar}{RESET} (ok={ok}, fail={fail})"
            )

    # 1) Stable ARC-GEN synthetic pairs: use ALL available.
    if task.arc_gen_synthetic_pairs:
        pairs = task.arc_gen_synthetic_pairs
        print(f"\nStable ARC-GEN pairs: {len(pairs)} total")

        for name, key, verifier in (
            ("primary (re_arc)", "primary", task.verifier),
            ("secondary (NeurIPS)", "secondary", task.secondary_verifier),
        ):
            if verifier is None:
                continue
            ok = 0
            fail = 0
            total = len(pairs)
            for i, pair in enumerate(pairs, start=1):
                try:
                    out = verifier(pair.input)
                    if is_equal_grid(out, pair.output):
                        ok += 1
                    else:
                        fail += 1
                        print(
                            f"  [{name}] stable mismatch at pair {i-1}: expected vs got differ"
                        )
                except Exception as e:
                    fail += 1
                    print(
                        f"  [{name}] stable error at pair {i-1}: {e}"
                    )
            all_ok = fail == 0 and ok > 0
            stats[key]["stable_ok"] = all_ok
            bar = _progress_bar(ok, total)
            color = GREEN if all_ok else RED
            print(
                f"  [{name}] stable ARC-GEN: {color}{bar}{RESET} (ok={ok}, fail={fail})"
            )
    else:
        print("\nNo stable ARC-GEN synthetic pairs available.")

    # 2) Dynamically generated ARC-GEN examples: limit to max_examples.
    if task.arc_gen_generator is not None:
        print(f"\nDynamic ARC-GEN examples (up to {max_examples})")
        dyn_pairs = task.arc_gen_generator(max_examples)
        for name, key, verifier in (
            ("primary (re_arc)", "primary", task.verifier),
            ("secondary (NeurIPS)", "secondary", task.secondary_verifier),
        ):
            if verifier is None:
                continue
            ok = 0
            fail = 0
            total = len(dyn_pairs)
            for i, pair in enumerate(dyn_pairs, start=1):
                try:
                    out = verifier(pair.input)
                    if is_equal_grid(out, pair.output):
                        ok += 1
                    else:
                        fail += 1
                        print(
                            f"  [{name}] dynamic mismatch at pair {i-1}: expected vs got differ"
                        )
                        # Record mismatch details for later inspection.
                        stats[key]["dynamic_mismatches"].append(
                            {
                                "index": i - 1,
                                "input": pair.input,
                                "expected": pair.output,
                                "got": out,
                            }
                        )
                except Exception as e:
                    fail += 1
                    print(
                        f"  [{name}] dynamic error at pair {i-1}: {e}"
                    )
            all_ok = fail == 0 and ok > 0
            stats[key]["dynamic_ok"] = all_ok
            bar = _progress_bar(ok, total)
            color = GREEN if all_ok else RED
            print(
                f"  [{name}] dynamic ARC-GEN: {color}{bar}{RESET} (ok={ok}, fail={fail})"
            )
    else:
        print("\nNo ARC-GEN generator available.")
    return stats


if __name__ == "__main__":
    # How many tasks (from the original ARC set) to check.
    num_tasks_to_check = 400

    # Tee all output to a log file so the full run is never lost to scrollback.
    log_path = Path("caller_run_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log")
    _log_file = open(log_path, "w", encoding="utf-8")

    class _Tee:
        def write(self, s: str) -> None:
            sys.__stdout__.write(s)
            _log_file.write(s)
        def flush(self) -> None:
            sys.__stdout__.flush()
            _log_file.flush()

    _orig_stdout = sys.stdout
    sys.stdout = _Tee()
    try:
        print(f"Full log: {log_path}")
        print(f"Running verifier checks on first {num_tasks_to_check} ARC tasks...\n")
        task_stats = []
        for idx, task in enumerate(
            iter_tasks(split="train", source=TaskSource.ORIGINAL_ARC)
        ):
            if idx >= num_tasks_to_check:
                break
            print("=" * 80)
            print(f"[Task {idx+1}] ID: {task.task_id}")
            stats = check_verifiers_on_arc_gen(task, max_examples=5)
            task_stats.append((task.task_id, stats))

        # Summary focusing on tasks where the primary verifier fails anywhere.
        print("\n" + "=" * 80)
        print("SUMMARY: Tasks where the primary verifier failed on any ARC-GEN check\n")

        failing_primary = []
        for task_id, s in task_stats:
            p_train = s["primary"]["train_ok"]
            p_test = s["primary"]["test_ok"]
            p_stable = s["primary"]["stable_ok"]
            p_dyn = s["primary"]["dynamic_ok"]
            # Treat None as "no data" (does not cause failure).
            primary_all_ok = (
                (p_train in (True, None))
                and (p_test in (True, None))
                and (p_stable in (True, None))
                and (p_dyn in (True, None))
            )
            if not primary_all_ok:
                failing_primary.append((task_id, s))

        if not failing_primary:
            print("Primary verifier passed on all checked ARC-GEN examples for all tasks.")
        else:
            for task_id, s in failing_primary:
                p_train = s["primary"]["train_ok"]
                p_test = s["primary"]["test_ok"]
                p_stable = s["primary"]["stable_ok"]
                p_dyn = s["primary"]["dynamic_ok"]
                sec_train = s["secondary"]["train_ok"]
                sec_test = s["secondary"]["test_ok"]
                sec_stable = s["secondary"]["stable_ok"]
                sec_dyn = s["secondary"]["dynamic_ok"]

                def _fmt(flag):
                    if flag is True:
                        return f"{GREEN}ok{RESET}"
                    if flag is False:
                        return f"{RED}fail{RESET}"
                    return "n/a"

                secondary_all_ok = all(
                    flag in (True, None)
                    for flag in (sec_train, sec_test, sec_stable, sec_dyn)
                )

                print(f"- Task {task_id}:")
                print(
                    f"    primary:   train={_fmt(p_train)}, test={_fmt(p_test)}, "
                    f"stable={_fmt(p_stable)}, dynamic={_fmt(p_dyn)}"
                )
                print(
                    f"    secondary: train={_fmt(sec_train)}, test={_fmt(sec_test)}, "
                    f"stable={_fmt(sec_stable)}, dynamic={_fmt(sec_dyn)}"
                )
                if secondary_all_ok:
                    print("    => Secondary verifier is fully consistent on ARC-GEN.")
                else:
                    print("    => No verifier is fully consistent on ARC-GEN.")

                # Write dynamic mismatches to HTML files (viewer-style) for inspection.
                dyn_mismatches = s["primary"]["dynamic_mismatches"] or []
                mismatch_dir = Path("caller_mismatches")
                if dyn_mismatches:
                    print("    Dynamic ARC-GEN mismatches (primary) — saved to HTML:")
                    for m in dyn_mismatches:
                        idx = m["index"]
                        p = _write_mismatch_html(
                            mismatch_dir,
                            task_id,
                            idx,
                            m["input"],
                            m["expected"],
                            m["got"],
                        )
                        print(f"      {p}")
    finally:
        sys.stdout = _orig_stdout
        _log_file.close()