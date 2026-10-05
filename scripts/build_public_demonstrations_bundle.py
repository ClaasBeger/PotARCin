"""Sample unified-evaluation HTML demos into PotARCin-Public/Demonstrations/.

Picks 10 files from PotARCin/Demonstrations (recursive ``*_unified_evaluation.html``),
ensuring:
  - distinct ``task_id`` (filename stem before ``_unified_evaluation``)
  - at least one demo each for GPT54, Gemini31Pro, MiniMaxM25, ClaudeO46

Writes:
  ``<dest>/demo_01.html`` … ``demo_10.html``, ``index.html``, ``manifest.json``.

Navigation matches ``build_test_run_unified_visualizations._inject_prev_next_nav``
(Prev / Index / Next), with ``index.html`` as the hub.

Usage (from PotARCin repo root)::

    python scripts/build_public_demonstrations_bundle.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from html import escape
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

REQUIRED_MODELS = ("GPT54", "Gemini31Pro", "MiniMaxM25", "ClaudeO46")


def infer_model(path: Path) -> str | None:
    s = str(path).lower()
    if "kimik25" in s or "/kimi" in s:
        return "KimiK25"
    if "claudeo46" in s or "claude_" in s or "opus46" in s:
        return "ClaudeO46"
    if "minimax" in s:
        return "MiniMaxM25"
    if "gemini" in s:
        return "Gemini31Pro"
    if "gpt54" in s or "gpt-5" in s:
        return "GPT54"
    return None


def task_id_from_demo_path(path: Path) -> str:
    name = path.name
    suf = "_unified_evaluation.html"
    if not name.endswith(suf):
        raise ValueError(f"Unexpected demo filename: {path}")
    return name[: -len(suf)]


def logical_task_key(stem: str) -> str:
    """Treat ``abcd0123_seed1006`` as the same task id as ``abcd0123`` for diversity."""

    if stem.startswith("test2_"):
        return stem
    m = re.fullmatch(r"([0-9a-f]{8})_seed(\d+)", stem)
    if m:
        return m.group(1)
    return stem


def strip_legacy_run_navigation(html: str) -> str:
    """Remove prev/next injected by ``build_test_run_unified_visualizations`` (broken in bundle dir)."""

    old_nav = re.compile(
        r"\n<!-- task-nav-links -->\n<nav class=\"tasknav\">.*?</nav>\n<style>.*?</style>",
        re.DOTALL,
    )
    return old_nav.sub("", html)


def collect_buckets(src: Path) -> dict[str, list[tuple[str, Path]]]:
    buckets: dict[str, list[tuple[str, Path]]] = {m: [] for m in REQUIRED_MODELS}
    for p in src.rglob("*_unified_evaluation.html"):
        if p.name == "index.html":
            continue
        model = infer_model(p)
        if model is None or model not in buckets:
            continue
        tid = task_id_from_demo_path(p)
        buckets[model].append((tid, p))
    return buckets


def pick_ten(
    buckets: dict[str, list[tuple[str, Path]]], rng: random.Random
) -> list[tuple[str, str, Path]]:
    for m in REQUIRED_MODELS:
        if not buckets[m]:
            raise RuntimeError(f"No *_unified_evaluation.html matched model {m} under source tree.")

    picked: list[tuple[str, str, Path]] = []
    used_logical: set[str] = set()

    def candidates_for(model: str) -> list[tuple[str, Path]]:
        out: list[tuple[str, Path]] = []
        for tid, p in buckets[model]:
            if logical_task_key(tid) in used_logical:
                continue
            out.append((tid, p))
        return out

    for m in REQUIRED_MODELS:
        cand = candidates_for(m)
        if not cand:
            raise RuntimeError(
                f"Cannot assign a logically distinct task for required model {m} "
                "(all compatible tasks already used)."
            )
        tid, pth = rng.choice(cand)
        picked.append((tid, m, pth))
        used_logical.add(logical_task_key(tid))

    remaining: list[tuple[str, str, Path]] = []
    for m in REQUIRED_MODELS:
        for tid, pth in buckets[m]:
            if logical_task_key(tid) not in used_logical:
                remaining.append((tid, m, pth))
    rng.shuffle(remaining)

    counts = Counter(model for _, model, _ in picked)
    while len(picked) < 10 and remaining:
        min_c = min(counts[m] for m in REQUIRED_MODELS)
        tier = [it for it in remaining if counts[it[1]] == min_c]
        pool_pick = tier if tier else remaining[:]
        i = rng.randrange(len(pool_pick))
        tid, m, pth = pool_pick[i]
        remaining.remove((tid, m, pth))
        picked.append((tid, m, pth))
        used_logical.add(logical_task_key(tid))
        counts[m] += 1

    if len(picked) < 10:
        raise RuntimeError(
            f"Only {len(picked)} demos with logically distinct tasks available "
            f"(need 10 across {REQUIRED_MODELS})."
        )

    rng.shuffle(picked)
    return picked[:10]


def inject_prev_next_nav(html: str, *, prev_href: str | None, next_href: str | None) -> str:
    marker = "<!-- public-demo-nav -->"
    if marker in html:
        return html
    needle = "</h1>\n<p class=\"sub\">"
    k = html.find(needle)
    if k < 0:
        needle2 = "</h1>\r\n<p class=\"sub\">"
        k = html.find(needle2)
        if k < 0:
            wrap = '<div class="wrap">'
            wk = html.find(wrap)
            if wk < 0:
                return html
            ins = wk + len(wrap)
            nav_block = (
                f"\n{marker}\n{_nav_inner(prev_href, next_href)}\n"
            )
            return html[:ins] + nav_block + html[ins:]
        needle = needle2
    p_end = html.find("</p>", k)
    if p_end < 0:
        return html
    p_end = html.find("\n", p_end)
    if p_end < 0:
        return html
    injection = "\n" + marker + "\n" + _nav_inner(prev_href, next_href) + "\n"
    return html[: p_end + 1] + injection + html[p_end + 1 :]


def _nav_inner(prev_href: str | None, next_href: str | None) -> str:
    def _link(href: str, label: str) -> str:
        return f'<a class="tasknav-link" href="{escape(href)}">{escape(label)}</a>'

    parts = [
        '<nav class="tasknav">',
        _link(prev_href, "← Prev") if prev_href else '<span class="tasknav-disabled">← Prev</span>',
        _link("index.html", "Index"),
        _link(next_href, "Next →") if next_href else '<span class="tasknav-disabled">Next →</span>',
        "</nav>",
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
</style>""",
    ]
    return "\n".join(parts)


def write_index(dest: Path, items: list[tuple[str, str, str]]) -> None:
    """items: (demo_filename, task_id, model)"""
    rows = []
    for demo_fn, tid, model in items:
        rows.append(
            f"<tr><td><code>{escape(tid)}</code></td>"
            f"<td>{escape(model)}</td>"
            f"<td><a href=\"{escape(demo_fn)}\">{escape(demo_fn)}</a></td></tr>"
        )
    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>PotARCin — demonstration sample</title>
<style>
:root {{
  --bg: #0f1115;
  --surface: #171a21;
  --border: #2a3140;
  --text: #e8eaed;
  --muted: #9aa0a6;
  --accent: #5c9ded;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: "Segoe UI", system-ui, -apple-system, Roboto, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.5;
}}
.wrap {{ max-width: 960px; margin: 0 auto; padding: 2rem 1.25rem 3rem; }}
h1 {{ font-size: 1.6rem; margin: 0 0 0.5rem; }}
p.sub {{ color: var(--muted); margin-bottom: 1.5rem; }}
table {{
  width: 100%;
  border-collapse: collapse;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
}}
th, td {{ padding: 0.65rem 1rem; text-align: left; border-bottom: 1px solid var(--border); }}
th {{ background: #12151c; color: var(--accent); font-weight: 600; }}
tr:last-child td {{ border-bottom: none; }}
code {{ font-size: 0.9rem; }}
a {{ color: var(--accent); }}
</style>
</head>
<body>
<div class="wrap">
<h1>Demonstration sample</h1>
<p class="sub">Ten unified multi-dimension evaluation pages (random seed), covering GPT-5.4, Gemini 3.1 Pro,
MiniMax M2.5, and Claude Opus 4.6, each on a distinct task. Use Prev / Next inside each page to browse.</p>
<table>
<thead><tr><th>Task</th><th>Model</th><th>Page</th></tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody>
</table>
<p class="sub" style="margin-top:1.5rem">Open <a href="demo_01.html"><code>demo_01.html</code></a> to start the chain.</p>
</div>
</body>
</html>
"""
    (dest / "index.html").write_text(body, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Bundle random demos into PotARCin-Public/Demonstrations.")
    ap.add_argument(
        "--source",
        type=Path,
        default=ROOT_DIR / "Demonstrations",
        help="Root containing unified_evaluation HTML (default: PotARCin/Demonstrations)",
    )
    ap.add_argument(
        "--dest",
        type=Path,
        default=ROOT_DIR.parent / "PotARCin-Public" / "Demonstrations",
        help="Output directory (default: sibling PotARCin-Public/Demonstrations)",
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    args = ap.parse_args()

    src: Path = args.source
    dest: Path = args.dest
    if not src.is_dir():
        raise SystemExit(f"Missing source directory: {src}")

    rng = random.Random(args.seed)
    buckets = collect_buckets(src)
    picked = pick_ten(buckets, rng)

    dest.mkdir(parents=True, exist_ok=True)

    manifest_items: list[dict[str, str]] = []
    index_rows: list[tuple[str, str, str]] = []

    for i, (tid, model, src_path) in enumerate(picked, start=1):
        demo_name = f"demo_{i:02d}.html"
        prev_name = f"demo_{i-1:02d}.html" if i > 1 else None
        next_name = f"demo_{i+1:02d}.html" if i < len(picked) else None

        html = src_path.read_text(encoding="utf-8")
        html = strip_legacy_run_navigation(html)
        html = inject_prev_next_nav(html, prev_href=prev_name, next_href=next_name)
        out_path = dest / demo_name
        out_path.write_text(html, encoding="utf-8")

        manifest_items.append(
            {
                "demo_file": demo_name,
                "task_id": tid,
                "logical_task_key": logical_task_key(tid),
                "model_alias": model,
                "source_relative": str(src_path.relative_to(src)),
            }
        )
        index_rows.append((demo_name, tid, model))

    manifest = {
        "seed": args.seed,
        "source_root": str(src.resolve()),
        "dest_root": str(dest.resolve()),
        "demos": manifest_items,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    write_index(dest, index_rows)

    print(f"Wrote {len(picked)} demos + index.html + manifest.json under {dest.resolve()}")


if __name__ == "__main__":
    main()
