from __future__ import annotations

import sys
from datetime import datetime
import copy
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


def _normalize_grid(grid: Grid) -> Grid:
    """Convert tuple-based grids to list-of-lists so equality is semantic."""
    # Many golfed solutions return tuples of tuples; we want structural
    # equality up to container type, not strict type equality.
    if not isinstance(grid, (list, tuple)):
        return grid
    rows = []
    for row in grid:
        if isinstance(row, (list, tuple)):
            rows.append(list(row))
        else:
            rows.append(row)  # type: ignore[list-item]
    return rows  # type: ignore[return-value]


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
    verifier_key: str,
    verifier_name: str,
    failed_check: str,
    index: int,
    input_grid: Grid,
    expected: Grid,
    got: Grid,
) -> Path:
    """Write a single HTML file showing input, expected, and got side by side."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{task_id}_{verifier_key}_{failed_check}_{index}.html"
    if failed_check == "train":
        expected_label = "Expected (train label)"
    elif failed_check == "test":
        expected_label = "Expected (test label)"
    elif failed_check == "stable":
        expected_label = "Expected (ARC-GEN stable)"
    elif failed_check == "dynamic":
        expected_label = "Expected (ARC-GEN dynamic)"
    else:
        expected_label = "Expected"
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'/><title>Mismatch "
        + task_id
        + " "
        + verifier_key
        + " "
        + failed_check
        + " "
        + str(index)
        + "</title></head><body style='font-family:sans-serif;'>"
        + "<h2>"
        + task_id
        + " — "
        + verifier_name
        + " — failed "
        + failed_check
        + " ["
        + str(index)
        + "]"
        + "</h2>"
        + "<div style='opacity:0.8;margin-bottom:10px;'>File: "
        + str(path)
        + "</div>"
        + "<div style='display:flex;flex-wrap:wrap;gap:16px;'>"
        + _grid_to_html(input_grid, "Input")
        + _grid_to_html(expected, expected_label)
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
    print("Has verifier #2 (google-code-golf-2025)?", task.secondary_verifier is not None)
    if task.secondary_verifier is not None and task.train_pairs:
        print("  Verifier #2:", task.secondary_verifier)
        print(
            "  Verifier #2 output on first train input:",
            task.secondary_verifier(task.train_pairs[0].input),
        )
    print("Has verifier #3 (golf)?", getattr(task, "tertiary_verifier", None) is not None)
    if getattr(task, "tertiary_verifier", None) is not None and task.train_pairs:
        v = task.tertiary_verifier  # type: ignore[assignment]
        print("  Verifier #3:", v)
        print("  Verifier #3 output on first train input:", v(task.train_pairs[0].input))
    print("Has verifier #4 (NeurIPS-Code-Golf-2025)?", getattr(task, "quaternary_verifier", None) is not None)
    if getattr(task, "quaternary_verifier", None) is not None and task.train_pairs:
        v = task.quaternary_verifier  # type: ignore[assignment]
        print("  Verifier #4:", v)
        print("  Verifier #4 output on first train input:", v(task.train_pairs[0].input))
    print("Has verifier #5 (custom)?", getattr(task, "quinary_verifier", None) is not None)
    if getattr(task, "quinary_verifier", None) is not None and task.train_pairs:
        v = task.quinary_verifier  # type: ignore[assignment]
        print("  Verifier #5:", v)
        print("  Verifier #5 output on first train input:", v(task.train_pairs[0].input))

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
    verifiers = [
        ("v1 (re_arc)", "v1", task.verifier),
        ("v2 (google-code-golf-2025)", "v2", task.secondary_verifier),
        ("v3 (golf)", "v3", getattr(task, "tertiary_verifier", None)),
        ("v4 (NeurIPS-Code-Golf-2025)", "v4", getattr(task, "quaternary_verifier", None)),
        ("v5 (custom)", "v5", getattr(task, "quinary_verifier", None)),
    ]

    present = [(n, k, v) for (n, k, v) in verifiers if v is not None]
    if not present:
        return {}

    # Ground-truth test outputs from arc_original.zip, when available.
    test_labels = None
    if task.test_inputs and getattr(task, "test_outputs", None):
        # Normalize here so all verifiers can compare against the same form.
        test_labels = [_normalize_grid(out) for out in task.test_outputs]

    stats: dict[str, dict] = {}
    for _name, key, _verifier in present:
        stats[key] = {
            "train_ok": None,
            "test_ok": None,
            "stable_ok": None,
            "dynamic_ok": None,
            "failed_fast": False,
            "first_failure": None,  # dict with kind/index/error/input/expected/got
            "counts": {
                "train": {"ok": 0, "total": 0},
                "test": {"ok": 0, "total": 0},
                "stable": {"ok": 0, "total": 0},
                "dynamic": {"ok": 0, "total": 0},
            },
        }

    def _mark_failed_fast(key: str, failure: dict) -> None:
        stats[key]["failed_fast"] = True
        stats[key]["first_failure"] = failure
        # Ensure this verifier cannot be considered "all ok".
        for f in ("train_ok", "test_ok", "stable_ok", "dynamic_ok"):
            if stats[key][f] is None:
                stats[key][f] = False

    # Run checks per verifier; for each verifier, stop as soon as ANY grid fails.
    for name, key, verifier in present:
        # 0) Train
        if task.train_pairs and not stats[key]["failed_fast"]:
            ok = 0
            stats[key]["counts"]["train"]["total"] = len(task.train_pairs)
            for i, pair in enumerate(task.train_pairs):
                try:
                    orig_input = pair.input
                    work_input = copy.deepcopy(orig_input)
                    out = _normalize_grid(verifier(work_input))
                    if is_equal_grid(out, _normalize_grid(pair.output)):
                        ok += 1
                        stats[key]["counts"]["train"]["ok"] = ok
                    else:
                        _mark_failed_fast(
                            key,
                            {
                                "kind": "train",
                                "index": i,
                                "input": copy.deepcopy(orig_input),
                                "expected": copy.deepcopy(pair.output),
                                "got": out,
                            },
                        )
                        break
                except Exception as e:
                    _mark_failed_fast(key, {"kind": "train", "index": i, "error": str(e)})
                    break
            stats[key]["train_ok"] = (not stats[key]["failed_fast"]) and ok == len(task.train_pairs) and ok > 0

        # 0b) Test against ground-truth outputs from arc_original.zip.
        if task.test_inputs and test_labels is not None and not stats[key]["failed_fast"]:
            ok = 0
            stats[key]["counts"]["test"]["total"] = len(task.test_inputs)
            for i, inp in enumerate(task.test_inputs):
                try:
                    orig_input = inp
                    work_input = copy.deepcopy(inp)
                    out = _normalize_grid(verifier(work_input))
                    if is_equal_grid(out, test_labels[i]):
                        ok += 1
                        stats[key]["counts"]["test"]["ok"] = ok
                    else:
                        _mark_failed_fast(
                            key,
                            {
                                "kind": "test",
                                "index": i,
                                "input": copy.deepcopy(orig_input),
                                "expected": copy.deepcopy(test_labels[i]),
                                "got": out,
                            },
                        )
                        break
                except Exception as e:
                    _mark_failed_fast(key, {"kind": "test", "index": i, "error": str(e)})
                    break
            stats[key]["test_ok"] = (not stats[key]["failed_fast"]) and ok == len(task.test_inputs) and ok > 0

        # 1) Stable ARC-GEN
        if task.arc_gen_synthetic_pairs and not stats[key]["failed_fast"]:
            pairs = task.arc_gen_synthetic_pairs
            ok = 0
            stats[key]["counts"]["stable"]["total"] = len(pairs)
            for i, pair in enumerate(pairs):
                try:
                    orig_input = pair.input
                    work_input = copy.deepcopy(orig_input)
                    out = _normalize_grid(verifier(work_input))
                    if is_equal_grid(out, _normalize_grid(pair.output)):
                        ok += 1
                        stats[key]["counts"]["stable"]["ok"] = ok
                    else:
                        _mark_failed_fast(
                            key,
                            {
                                "kind": "stable",
                                "index": i,
                                "input": copy.deepcopy(orig_input),
                                "expected": copy.deepcopy(pair.output),
                                "got": out,
                            },
                        )
                        break
                except Exception as e:
                    _mark_failed_fast(key, {"kind": "stable", "index": i, "error": str(e)})
                    break
            stats[key]["stable_ok"] = (not stats[key]["failed_fast"]) and ok == len(pairs) and ok > 0

        # 2) Dynamic ARC-GEN
        if task.arc_gen_generator is not None and not stats[key]["failed_fast"]:
            dyn_pairs = task.arc_gen_generator(max_examples)
            ok = 0
            stats[key]["counts"]["dynamic"]["total"] = len(dyn_pairs)
            for i, pair in enumerate(dyn_pairs):
                try:
                    orig_input = pair.input
                    work_input = copy.deepcopy(orig_input)
                    out = _normalize_grid(verifier(work_input))
                    if is_equal_grid(out, _normalize_grid(pair.output)):
                        ok += 1
                        stats[key]["counts"]["dynamic"]["ok"] = ok
                    else:
                        _mark_failed_fast(
                            key,
                            {
                                "kind": "dynamic",
                                "index": i,
                                "input": copy.deepcopy(orig_input),
                                "expected": copy.deepcopy(pair.output),
                                "got": out,
                            },
                        )
                        break
                except Exception as e:
                    _mark_failed_fast(key, {"kind": "dynamic", "index": i, "error": str(e)})
                    break
            stats[key]["dynamic_ok"] = (not stats[key]["failed_fast"]) and ok == len(dyn_pairs) and ok > 0

        # If a verifier never ran any checks (no data), keep fields as None.
        if not task.train_pairs:
            stats[key]["train_ok"] = None
        if not (task.test_inputs and test_labels is not None):
            stats[key]["test_ok"] = None
        if not task.arc_gen_synthetic_pairs:
            stats[key]["stable_ok"] = None
        if task.arc_gen_generator is None:
            stats[key]["dynamic_ok"] = None

    # Only log tasks where NONE of the verifiers pass all available checks.
    def _verifier_all_ok(vs: dict) -> bool:
        return all(vs.get(f) in (True, None) for f in ("train_ok", "test_ok", "stable_ok", "dynamic_ok"))

    any_all_ok = any(_verifier_all_ok(vs) for vs in stats.values())
    if any_all_ok:
        return {}

    # Logging/printing only for "all verifiers failed" tasks.
    print("\n=== Verifier check on ARC-GEN examples (logged: no verifier fully consistent) ===")
    print(f"Task: {task.task_id}")

    def _fmt(flag):
        if flag is True:
            return f"{GREEN}ok{RESET}"
        if flag is False:
            return f"{RED}fail{RESET}"
        return "n/a"

    for name, key, verifier in present:
        vs = stats[key]
        c_train = vs["counts"]["train"]
        c_test = vs["counts"]["test"]
        c_stable = vs["counts"]["stable"]
        c_dynamic = vs["counts"]["dynamic"]
        print(f"- {name}:")
        if c_train["total"]:
            print(f"    train   {_fmt(vs['train_ok'])} {_progress_bar(c_train['ok'], c_train['total'])} (ok={c_train['ok']}, total={c_train['total']})")
        else:
            print(f"    train   {_fmt(vs['train_ok'])}")
        if c_test["total"]:
            print(f"    test    {_fmt(vs['test_ok'])} {_progress_bar(c_test['ok'], c_test['total'])} (ok={c_test['ok']}, total={c_test['total']})")
        else:
            print(f"    test    {_fmt(vs['test_ok'])}")
        if c_stable["total"]:
            print(f"    stable  {_fmt(vs['stable_ok'])} {_progress_bar(c_stable['ok'], c_stable['total'])} (ok={c_stable['ok']}, total={c_stable['total']})")
        else:
            print(f"    stable  {_fmt(vs['stable_ok'])}")
        if c_dynamic["total"]:
            print(f"    dynamic {_fmt(vs['dynamic_ok'])} {_progress_bar(c_dynamic['ok'], c_dynamic['total'])} (ok={c_dynamic['ok']}, total={c_dynamic['total']})")
        else:
            print(f"    dynamic {_fmt(vs['dynamic_ok'])}")
        if vs["first_failure"] is not None:
            ff = vs["first_failure"]
            if "error" in ff:
                print(f"    first failure: {ff['kind']}[{ff['index']}] error={ff['error']}")
            else:
                print(f"    first failure: {ff['kind']}[{ff['index']}] mismatch")

    # Only write mismatch HTMLs for these logged tasks, and only when we have grids.
    mismatch_dir = Path("caller_mismatches")
    for name, key, _verifier in present:
        ff = stats[key]["first_failure"]
        if not ff or "input" not in ff or "expected" not in ff or "got" not in ff:
            continue
        out_dir = mismatch_dir / task.task_id
        p = _write_mismatch_html(
            out_dir,
            task.task_id,
            key,
            name,
            str(ff.get("kind", "unknown")),
            int(ff["index"]),
            ff["input"],
            ff["expected"],
            ff["got"],
        )
        print(f"    mismatch HTML ({name}): {p}")

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
            # Dynamic ARC-GEN samples per task (only tasks where no verifier is fully
            # consistent are printed/logged).
            _stats = check_verifiers_on_arc_gen(task, max_examples=50)
            task_stats.append((task.task_id, _stats))

        # Small final note: by design, only "all-verifiers-failed" tasks were logged above.
        print("\n" + "=" * 80)
        print("DONE. Only tasks where no verifier was fully consistent were logged.\n")
    finally:
        sys.stdout = _orig_stdout
        _log_file.close()