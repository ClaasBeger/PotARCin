"""Build unified evaluation HTML pages for every task in a multi-dimension run summary.

Reads ``task_id`` values from ``evaluation/multi_dimension_eval_runs/run_*.json`` and
runs ``build_unified_evaluation_visualization`` for each task so the latest per-task
eval JSONs under ``evaluation/`` are shown (same logic as
``Demonstrations/unified_evaluations``).

By default the builder module is loaded **once** and ``main()`` is called in-process
for each task (fast). Use ``--subprocess-per-task`` only when you need process isolation.

Usage::

    python scripts/build_test_run_unified_visualizations.py \\
        --run-json evaluation/multi_dimension_eval_runs/Test_Run_2026-04-08/run_2026-04-08_13-57-30.json \\
        --out-dir Demonstrations/Test_Run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from html import escape
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT_DIR / "scripts" / "build_unified_evaluation_visualization.py"


def _ensure_matplotlib_config_dir() -> None:
    """Avoid per-process Matplotlib font-cache rebuilds when ~/.matplotlib is not writable."""
    d = ROOT_DIR / ".mplconfig"
    d.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(d))
    os.environ.setdefault("MPLBACKEND", "Agg")


def _load_builder_module():
    _ensure_matplotlib_config_dir()
    spec = importlib.util.spec_from_file_location(
        "build_unified_evaluation_visualization",
        BUILD_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load builder module from {BUILD_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_entries(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    runs = data.get("runs")
    if not isinstance(runs, list):
        raise ValueError(f"Expected 'runs' list in {path}")
    rows: list[dict] = []
    for r in runs:
        if isinstance(r, dict) and r.get("task_id"):
            rows.append(r)
    return rows


def _inject_prev_next_nav(html: str, *, prev_href: str | None, next_href: str | None) -> str:
    """Inject Prev/Index/Next nav under the subtitle line.

    Idempotent via a marker comment.
    """
    marker = "<!-- task-nav-links -->"
    if marker in html:
        return html
    # We insert after the subtitle paragraph, which is stable across builders.
    needle = "</h1>\n<p class=\"sub\">"
    k = html.find(needle)
    if k < 0:
        return html
    # Find end of the subtitle paragraph.
    p_end = html.find("</p>", k)
    if p_end < 0:
        return html
    p_end = html.find("\n", p_end)
    if p_end < 0:
        return html

    def _link(href: str, label: str) -> str:
        return f'<a class="tasknav-link" href="{escape(href)}">{escape(label)}</a>'

    parts: list[str] = []
    parts.append(marker)
    parts.append('<nav class="tasknav">')
    parts.append(_link(prev_href, "← Prev") if prev_href else '<span class="tasknav-disabled">← Prev</span>')
    parts.append(_link("index.html", "Index"))
    parts.append(_link(next_href, "Next →") if next_href else '<span class="tasknav-disabled">Next →</span>')
    parts.append("</nav>")
    parts.append(
        """<style>
.tasknav { display:flex; gap: 12px; align-items:center; margin: 0.75rem 0 1.5rem; }
.tasknav-link, .tasknav-disabled {
  display:inline-block;
  padding: 6px 10px;
  border-radius: 10px;
  border: 1px solid #2a3140;
  background: #12151c;
  color: #5c9ded;
  text-decoration: none;
  font-size: 0.9rem;
}
.tasknav-link:hover { filter: brightness(1.15); }
.tasknav-disabled { color: #9aa0a6; opacity: 0.65; cursor: not-allowed; }
</style>
"""
    )
    injection = "\n".join(parts) + "\n"
    return html[: p_end + 1] + injection + html[p_end + 1 :]


def _inject_nav_for_out_dir(out_dir: Path, task_ids: list[str]) -> None:
    for i, tid in enumerate(task_ids):
        prev_href = f"{task_ids[i-1]}_unified_evaluation.html" if i > 0 else None
        next_href = f"{task_ids[i+1]}_unified_evaluation.html" if i + 1 < len(task_ids) else None
        p = out_dir / f"{tid}_unified_evaluation.html"
        if not p.is_file():
            continue
        html = p.read_text(encoding="utf-8")
        new_html = _inject_prev_next_nav(html, prev_href=prev_href, next_href=next_href)
        if new_html != html:
            p.write_text(new_html, encoding="utf-8")


def _write_index(out_dir: Path, task_ids: list[str], run_json: Path) -> None:
    rows = "".join(
        f'<tr><td><a href="{escape(tid)}_unified_evaluation.html">{escape(tid)}</a></td></tr>\n'
        for tid in task_ids
    )
    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Test run — unified evaluations</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; background: #0f1115; color: #e8eaed; }}
a {{ color: #5c9ded; }}
table {{ border-collapse: collapse; margin-top: 16px; }}
td {{ padding: 6px 12px; border-bottom: 1px solid #2a3140; }}
code {{ background: #171a21; padding: 2px 6px; }}
</style>
</head>
<body>
<h1>Test run — unified evaluations</h1>
<p>Source run: <code>{escape(str(run_json))}</code></p>
<p>{len(task_ids)} tasks — each link opens the same five-dimension layout as <code>Demonstrations/unified_evaluations/</code>.</p>
<table>
<thead><tr><th>Task ID</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
</body>
</html>
"""
    (out_dir / "index.html").write_text(body, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build unified evaluation HTML for each task in a multi_dimension_eval_runs JSON."
    )
    p.add_argument(
        "--run-json",
        type=Path,
        required=True,
        help="Path to run_*.json under evaluation/multi_dimension_eval_runs/",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT_DIR / "Demonstrations" / "Test_Run",
        help="Output directory (default: Demonstrations/Test_Run)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print task ids and paths; do not invoke the builder",
    )
    p.add_argument(
        "--inject-nav-only",
        action="store_true",
        help="Do not rebuild pages; only inject Prev/Next nav into existing HTML files in --out-dir.",
    )
    p.add_argument(
        "--subprocess-per-task",
        action="store_true",
        help=(
            "Spawn a new Python interpreter for each task (slow; isolates failures). "
            "Default is in-process: load the builder once and reuse imports (much faster)."
        ),
    )
    args = p.parse_args()

    run_path = args.run_json if args.run_json.is_absolute() else ROOT_DIR / args.run_json
    if not run_path.is_file():
        raise SystemExit(f"Run JSON not found: {run_path}")

    entries = _run_entries(run_path)
    task_ids = [str(e["task_id"]) for e in entries]
    if not task_ids:
        raise SystemExit("No task_id entries found in run JSON")

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for tid in task_ids:
            print(tid, "->", out_dir / f"{tid}_unified_evaluation.html")
        return

    if args.inject_nav_only:
        _inject_nav_for_out_dir(out_dir, task_ids)
        _write_index(out_dir, task_ids, run_path)
        print(f"Injected Prev/Next nav into pages under {out_dir.resolve()}")
        return

    if not BUILD_SCRIPT.is_file():
        raise SystemExit(f"Missing builder script: {BUILD_SCRIPT}")

    builder = None if args.subprocess_per_task else _load_builder_module()

    failed: list[str] = []
    for i, entry in enumerate(entries, 1):
        tid = str(entry["task_id"])
        out_html = out_dir / f"{tid}_unified_evaluation.html"
        dims = entry.get("dimensions", {}) if isinstance(entry, dict) else {}
        def_path = ((dims.get("definition") or {}).get("result_path")) if isinstance(dims, dict) else None
        cls_path = (
            ((dims.get("classification_distribution") or {}).get("result_path"))
            if isinstance(dims, dict)
            else None
        )
        cg_path = (
            ((dims.get("constrained_generation") or {}).get("result_path"))
            if isinstance(dims, dict)
            else None
        )
        ed_path = ((dims.get("editing") or {}).get("result_path")) if isinstance(dims, dict) else None
        inv_path = ((dims.get("inversion") or {}).get("result_path")) if isinstance(dims, dict) else None
        print(f"[{i}/{len(task_ids)}] {tid}")
        cmd = [
            sys.executable,
            str(BUILD_SCRIPT),
            "--task-id",
            tid,
            "--out",
            str(out_html),
        ]
        if def_path:
            cmd += ["--definition-json", str(def_path)]
        if cls_path:
            cmd += ["--classification-json", str(cls_path)]
        if cg_path:
            cmd += ["--constrained-generation-json", str(cg_path)]
        if ed_path:
            cmd += ["--editing-json", str(ed_path)]
        if inv_path:
            cmd += ["--inversion-json", str(inv_path)]

        if args.subprocess_per_task:
            _ensure_matplotlib_config_dir()
            proc = subprocess.run(cmd, cwd=str(ROOT_DIR))
            if proc.returncode != 0:
                failed.append(tid)
                print(f"  WARNING: builder failed for {tid} (exit {proc.returncode})")
        else:
            assert builder is not None
            argv_bak = sys.argv[:]
            try:
                sys.argv = cmd[1:]
                builder.main()
            except SystemExit as e:
                c = e.code
                if c is None or c is False:
                    code = 0
                elif isinstance(c, int):
                    code = c
                else:
                    code = 1
                if code != 0:
                    failed.append(tid)
                    print(f"  WARNING: builder failed for {tid} (exit {code})")
            except Exception as e:
                failed.append(tid)
                print(f"  WARNING: builder failed for {tid}: {e}")
            finally:
                sys.argv = argv_bak

    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}")

    _write_index(out_dir, task_ids, run_path)
    ok_n = len(task_ids) - len(failed)
    print(f"Wrote {ok_n}/{len(task_ids)} pages + index.html under {out_dir.resolve()}")
    if failed:
        raise SystemExit(1)

    _inject_nav_for_out_dir(out_dir, task_ids)


if __name__ == "__main__":
    main()
