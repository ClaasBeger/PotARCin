from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from framework.dimensions.classification import (
    run_distribution_classification_api_evaluation_once,
)
from framework.dimensions.constrained_generation import (
    run_constrained_generation_api_evaluation_once,
)
from framework.dimensions.definition import evaluate_definition_script_on_task
from framework.dimensions.editing import run_editing_api_evaluation_once
from framework.dimensions.inversion import run_inversion_api_evaluation_once
from framework.eval_paths import dimension_out_dirs, model_alias, multi_run_summary_dir
from framework.grids import GridPair
from framework.tasks.arc_dataset import load_task


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _grid_sig(g: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in g)


def _pair_sig(inp: list[list[int]], out: list[list[int]]) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    return _grid_sig(inp), _grid_sig(out)


def _load_latest_definition_script_from_dir(dir_path: Path, task_id: str) -> str | None:
    if not dir_path.is_dir():
        return None
    matches = sorted(dir_path.glob(f"{task_id}_definition_eval_*.json"))
    if not matches:
        return None
    for p in reversed(matches):
        try:
            payload = _read_json(p)
        except Exception:
            continue
        script = payload.get("script")
        if isinstance(script, str) and script.strip():
            return script
    return None


def _load_reference_definition_script(
    *,
    task_id: str,
    preferred_dirs: Sequence[Path],
    fallback_root: Path,
) -> str:
    for d in preferred_dirs:
        script = _load_latest_definition_script_from_dir(d, task_id)
        if script is not None:
            return script

    # Fallback: search any definition eval result directory (legacy / custom layouts).
    globbed = sorted(fallback_root.glob(f"**/{task_id}_definition_eval_*.json"))
    for p in reversed(globbed):
        try:
            payload = _read_json(p)
        except Exception:
            continue
        script = payload.get("script")
        if isinstance(script, str) and script.strip():
            return script
    raise FileNotFoundError(f"Could not locate a reference definition script for task {task_id}.")


def _iter_successful_cg_pairs_from_dir(dir_path: Path, task_id: str) -> Iterable[GridPair]:
    if not dir_path.is_dir():
        return
    for p in sorted(dir_path.glob(f"{task_id}_constrained_generation_eval_*.json")):
        try:
            payload = _read_json(p)
        except Exception:
            continue
        if payload.get("evaluation_ok") is not True:
            continue
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        if float(metrics.get("success_rate") or 0.0) < 1.0:
            continue
        pred = payload.get("prediction")
        if not isinstance(pred, dict):
            continue
        inp = pred.get("input")
        out = pred.get("output")
        if not isinstance(inp, list) or not isinstance(out, list):
            continue
        try:
            yield GridPair(inp, out)
        except Exception:
            continue


def _dedupe_pairs(pairs: Sequence[GridPair]) -> list[GridPair]:
    seen: set[tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]] = set()
    out: list[GridPair] = []
    for p in pairs:
        try:
            sig = _pair_sig(p.input, p.output)
        except Exception:
            continue
        if sig in seen:
            continue
        seen.add(sig)
        out.append(p)
    return out


@dataclass(frozen=True)
class StressConfig:
    model: str
    provider: str | None
    reasoning_effort: str
    tools_enabled: bool
    run_tag: str
    seeds: list[int]
    definition_arc_gen_dynamic_n: int
    history_run_tags: list[str]


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _seed_from_definition_filename(path: Path) -> int | None:
    name = path.name
    marker = "_seed"
    i = name.find(marker)
    if i < 0:
        return None
    j = i + len(marker)
    digits: list[str] = []
    while j < len(name) and name[j].isdigit():
        digits.append(name[j])
        j += 1
    if not digits:
        return None
    try:
        return int("".join(digits))
    except Exception:
        return None


def _build_skip_report_entry(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    ok = payload.get("evaluation_ok")
    if ok is False:
        return {
            "ok": False,
            "error": str(payload.get("error") or "evaluation_ok is false"),
            "result_path": str(path),
            "metrics": payload.get("metrics"),
            "attempts": 0,
            "skipped_rerun": True,
        }
    return {
        "ok": True,
        "result_path": str(path),
        "metrics": payload.get("metrics"),
        "attempts": 0,
        "skipped_rerun": True,
    }


def _existing_report_for_task_seed(
    *,
    task_id: str,
    seed: int,
    cfg: StressConfig,
    out_dirs: dict[str, Path],
) -> dict[str, Any] | None:
    task_int = int(task_id, 16)

    def _mix(tag: int) -> int:
        return int((seed * 1_000_003 + task_int + tag) % (1 << 31))

    dim_seeds = {
        "classification_distribution": _mix(11),
        "editing": _mix(13),
        "inversion": _mix(17),
        "definition_arcgen": _mix(19),
        "constrained_generation": _mix(23),
    }

    # Definition is keyed by base seed in filename.
    def_paths = sorted(
        out_dirs["definition"].glob(
            f"{task_id}_definition_eval_fixedscript_arcgendyn{cfg.definition_arc_gen_dynamic_n}_seed{seed}_*.json"
        )
    )
    if not def_paths:
        return None
    def_path = def_paths[-1]
    try:
        def_payload = _read_json(def_path)
    except Exception:
        return None

    # Classification / Editing / Inversion are keyed by derived per-dimension seed in payload.
    def _find_by_payload_seed(dir_path: Path, pattern: str, target_seed: int) -> tuple[Path, dict[str, Any]] | None:
        matches = sorted(dir_path.glob(pattern))
        for p in reversed(matches):
            try:
                payload = _read_json(p)
            except Exception:
                continue
            if payload.get("seed") == target_seed:
                return p, payload
        return None

    cls = _find_by_payload_seed(
        out_dirs["classification_distribution"],
        f"{task_id}_distribution_classification_eval_*.json",
        dim_seeds["classification_distribution"],
    )
    edt = _find_by_payload_seed(
        out_dirs["editing"],
        f"{task_id}_editing_eval_*.json",
        dim_seeds["editing"],
    )
    inv = _find_by_payload_seed(
        out_dirs["inversion"],
        f"{task_id}_inversion_eval_*.json",
        dim_seeds["inversion"],
    )
    if cls is None or edt is None or inv is None:
        return None
    cls_path, cls_payload = cls
    edt_path, edt_payload = edt
    inv_path, inv_payload = inv

    # Constrained generation has no seed in payload/filename. We align by completed-seed order
    # for this task: definition seed order <-> constrained-generation timestamp order.
    cg_paths = sorted(
        out_dirs["constrained_generation"].glob(f"{task_id}_constrained_generation_eval_*.json")
    )
    if not cg_paths:
        return None
    all_def_paths = sorted(
        out_dirs["definition"].glob(
            f"{task_id}_definition_eval_fixedscript_arcgendyn{cfg.definition_arc_gen_dynamic_n}_seed*_*.json"
        )
    )
    ordered_def_seeds = [s for s in (_seed_from_definition_filename(p) for p in all_def_paths) if s is not None]
    try:
        idx = ordered_def_seeds.index(seed)
    except ValueError:
        return None
    if idx >= len(cg_paths):
        return None
    cg_path = cg_paths[idx]
    try:
        cg_payload = _read_json(cg_path)
    except Exception:
        return None

    return {
        "task_id": task_id,
        "model": cfg.model,
        "provider": cfg.provider,
        "reasoning_effort": cfg.reasoning_effort,
        "tools_enabled": cfg.tools_enabled,
        "seed": seed,
        "dimension_seeds": dim_seeds,
        "dimensions": {
            "definition": _build_skip_report_entry(def_payload, def_path),
            "classification_distribution": _build_skip_report_entry(cls_payload, cls_path),
            "constrained_generation": _build_skip_report_entry(cg_payload, cg_path),
            "editing": _build_skip_report_entry(edt_payload, edt_path),
            "inversion": _build_skip_report_entry(inv_payload, inv_path),
        },
    }


def _resolve_history_dirs(history_run_tags: Sequence[str]) -> list[Path]:
    dirs: list[Path] = []
    base = ROOT_DIR / "evaluation" / "constrained_generation_eval_results"
    for tag in history_run_tags:
        # We only know the directory layout when the run tag is within model alias buckets.
        # This helper assumes the default GPT54 alias; users can override tags if desired.
        for run_bucket in ("Run1", "Run2"):
            p = base / "GPT54" / run_bucket / tag
            if p.is_dir():
                dirs.append(p)
    return dirs


def run_for_task_and_seed(
    *,
    task_id: str,
    seed: int,
    cfg: StressConfig,
    out_dirs: dict[str, Path],
    reference_definition_dirs: Sequence[Path],
) -> dict[str, Any]:
    task = load_task(task_id)

    # Dimension seed derivations: deterministic but independent-ish.
    # We keep them in 31-bit range for consistency with other runners.
    def _mix(tag: int) -> int:
        return int((seed * 1_000_003 + int(task_id, 16) + tag) % (1 << 31))

    dim_seeds = {
        "classification_distribution": _mix(11),
        "editing": _mix(13),
        "inversion": _mix(17),
        "definition_arcgen": _mix(19),
        "constrained_generation": _mix(23),
    }

    report: dict[str, Any] = {
        "task_id": task_id,
        "model": cfg.model,
        "provider": cfg.provider,
        "reasoning_effort": cfg.reasoning_effort,
        "tools_enabled": cfg.tools_enabled,
        "seed": seed,
        "dimension_seeds": dim_seeds,
        "dimensions": {},
    }

    # 1) Definition: fixed script, scaled ARC-GEN dynamic sample size.
    script = _load_reference_definition_script(
        task_id=task_id,
        preferred_dirs=reference_definition_dirs,
        fallback_root=ROOT_DIR / "evaluation" / "definition_eval_results",
    )
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dirs["definition"].mkdir(parents=True, exist_ok=True)
    def_out_path = out_dirs["definition"] / f"{task_id}_definition_eval_fixedscript_arcgendyn{cfg.definition_arc_gen_dynamic_n}_seed{seed}_{ts}.json"
    rng = random.Random(dim_seeds["definition_arcgen"])
    def_payload: dict[str, Any] = {
        "task_id": task_id,
        "model": cfg.model,
        "provider": cfg.provider,
        "reasoning_effort": cfg.reasoning_effort,
        "tools_enabled": cfg.tools_enabled,
        "seed": seed,
        "definition_mode": "fixed_script_rescore",
        "arc_gen_dynamic_n": cfg.definition_arc_gen_dynamic_n,
        "arc_gen_dynamic_seed": dim_seeds["definition_arcgen"],
        "script": script,
        "evaluation_ok": False,
        "metrics": {},
        "first_failures": {},
    }
    try:
        eval_result = evaluate_definition_script_on_task(
            task=task,
            script=script,
            test_idx=0,
            arc_gen_dynamic_n=cfg.definition_arc_gen_dynamic_n,
            arc_gen_dynamic_rng=rng,
        )
        def_payload["script_loaded"] = eval_result["script_loaded"]
        def_payload["script_load_error"] = eval_result["script_load_error"]
        def_payload["metrics"] = eval_result["metrics"]
        def_payload["first_failures"] = eval_result["first_failures"]
        def_payload["evaluation_ok"] = True
    except Exception as e:
        def_payload["error"] = str(e)
        def_payload["script_loaded"] = False
        def_payload["script_load_error"] = str(e)
    def_out_path.write_text(json.dumps(def_payload, indent=2, default=str), encoding="utf-8")
    report["dimensions"]["definition"] = {
        "ok": bool(def_payload.get("evaluation_ok")),
        "result_path": str(def_out_path),
        "metrics": def_payload.get("metrics"),
        "attempts": 0,
        "skipped_rerun": True,
    }

    # 2) Classification (distribution): normal re-run per seed.
    report["dimensions"]["classification_distribution"] = _wrap_dimension_call(
        lambda: run_distribution_classification_api_evaluation_once(
            task_id=task_id,
            model=cfg.model,
            provider=cfg.provider,
            reasoning_effort=cfg.reasoning_effort,
            tools_enabled=cfg.tools_enabled,
            seed=dim_seeds["classification_distribution"],
            out_dir=out_dirs["classification_distribution"],
        )
    )

    # 3) Constrained generation: augment demonstrations with prior successful generations.
    prior_pairs: list[GridPair] = []
    for history_dir in _resolve_history_dirs(cfg.history_run_tags):
        prior_pairs.extend(list(_iter_successful_cg_pairs_from_dir(history_dir, task_id)))
    # Add successes from this run-tag so far (earlier seeds).
    prior_pairs.extend(list(_iter_successful_cg_pairs_from_dir(out_dirs["constrained_generation"], task_id)))
    prior_pairs = _dedupe_pairs(prior_pairs)

    report["dimensions"]["constrained_generation"] = _wrap_dimension_call(
        lambda: run_constrained_generation_api_evaluation_once(
            task_id=task_id,
            model=cfg.model,
            provider=cfg.provider,
            reasoning_effort=cfg.reasoning_effort,
            tools_enabled=cfg.tools_enabled,
            extra_demonstrations=prior_pairs,
            out_dir=out_dirs["constrained_generation"],
        )
    )

    # 4) Editing: normal re-run per seed.
    report["dimensions"]["editing"] = _wrap_dimension_call(
        lambda: run_editing_api_evaluation_once(
            task=task,
            model=cfg.model,
            provider=cfg.provider,
            reasoning_effort=cfg.reasoning_effort,
            tools_enabled=cfg.tools_enabled,
            seed=dim_seeds["editing"],
            out_dir=out_dirs["editing"],
        )
    )

    # 5) Inversion: normal re-run per seed.
    report["dimensions"]["inversion"] = _wrap_dimension_call(
        lambda: run_inversion_api_evaluation_once(
            task_id=task_id,
            model=cfg.model,
            provider=cfg.provider,
            reasoning_effort=cfg.reasoning_effort,
            tools_enabled=cfg.tools_enabled,
            seed=dim_seeds["inversion"],
            out_dir=out_dirs["inversion"],
        )
    )

    return report


def _wrap_dimension_call(fn) -> dict[str, Any]:
    try:
        res = fn()
        if isinstance(res, dict) and res.get("evaluation_ok") is False:
            return {
                "ok": False,
                "error": str(res.get("error") or "evaluation_ok is false"),
                "result_path": res.get("result_path"),
                "metrics": res.get("metrics"),
                "attempts": 1,
            }
        return {
            "ok": True,
            "result_path": res.get("result_path"),
            "metrics": res.get("metrics"),
            "attempts": 1,
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "attempts": 1}


def main() -> None:
    p = argparse.ArgumentParser(description="Stress-test subset with multiple seeds and special-case definition/constrained-generation.")
    p.add_argument(
        "--subset-json",
        default="Test2/gpt54_nontrivial_success_subset_20_manual.json",
        help="Path to subset JSON containing selected_task_ids.",
    )
    p.add_argument("--model", default="gpt-5.4", help="Model name (not hard-coded).")
    p.add_argument(
        "--provider",
        default=None,
        choices=["openai", "anthropic", "google", "together", "openrouter"],
        help="LLM provider (default: infer from --model).",
    )
    p.add_argument("--reasoning-effort", default="low")
    p.add_argument("--tools-enabled", action="store_true")
    p.add_argument("--run-tag", required=True, help="Run tag for output directories.")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip task/seed pairs that already have all five dimension artifacts (resume mode).",
    )
    p.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore existing artifacts and re-run every task/seed pair.",
    )
    p.add_argument("--seed", type=int, default=77, help="Base seed for the 10-seed sweep (default: 77).")
    p.add_argument("--num-seeds", type=int, default=10, help="Number of seeds per task (default: 10).")
    p.add_argument(
        "--definition-arcgen-dynamic-n",
        type=int,
        default=1000,
        help="ARC-GEN dynamic sample size for fixed-script Definition (default: 1000).",
    )
    p.add_argument(
        "--history-run-tag",
        action="append",
        default=[
            "Evaluation_Run_2026-04-14_gpt54_low",
            "Evaluation_Run_2026-04-14_gpt54_low_run2",
        ],
        help="Run-tag(s) to mine prior successful constrained-generation examples from (can be repeated).",
    )
    args = p.parse_args()
    if args.skip_existing and args.force_rerun:
        raise SystemExit("Use either --skip-existing or --force-rerun, not both.")

    subset_path = (ROOT_DIR / args.subset_json).resolve() if not str(args.subset_json).startswith("/") else Path(args.subset_json)
    subset = _read_json(subset_path)
    task_ids = subset.get("selected_task_ids")
    if not isinstance(task_ids, list) or not all(isinstance(t, str) for t in task_ids):
        raise SystemExit(f"subset-json must contain selected_task_ids: {subset_path}")

    num_seeds = int(args.num_seeds)
    if num_seeds <= 0:
        raise SystemExit("--num-seeds must be >= 1")
    base_seed = int(args.seed)
    seeds = [base_seed + i for i in range(num_seeds)]

    cfg = StressConfig(
        model=str(args.model),
        provider=args.provider,
        reasoning_effort=str(args.reasoning_effort),
        tools_enabled=bool(args.tools_enabled),
        run_tag=str(args.run_tag),
        seeds=seeds,
        definition_arc_gen_dynamic_n=int(args.definition_arcgen_dynamic_n),
        history_run_tags=[str(x) for x in (args.history_run_tag or [])],
    )

    out_dirs = dimension_out_dirs(root_dir=ROOT_DIR, model=cfg.model, run_tag=cfg.run_tag)

    # Prefer definition scripts from the previously recorded GPT-5.4 runs, but fall back if missing.
    reference_definition_dirs = [
        ROOT_DIR / "evaluation" / "definition_eval_results" / "GPT54" / "Run1" / "Evaluation_Run_2026-04-14_gpt54_low",
        ROOT_DIR / "evaluation" / "definition_eval_results" / "GPT54" / "Run2" / "Evaluation_Run_2026-04-14_gpt54_low_run2",
    ]

    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_path = multi_run_summary_dir(root_dir=ROOT_DIR, model=cfg.model, run_tag=cfg.run_tag) / f"sampling_stress_{ts}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "mode": "subset_sampling_stress",
        "created_at": ts,
        "subset_json": str(subset_path),
        "task_count": len(task_ids),
        "seeds": cfg.seeds,
        "run_tag": cfg.run_tag,
        "model": cfg.model,
        "provider": cfg.provider,
        "reasoning_effort": cfg.reasoning_effort,
        "tools_enabled": cfg.tools_enabled,
        "definition_arc_gen_dynamic_n": cfg.definition_arc_gen_dynamic_n,
        "history_run_tags": cfg.history_run_tags,
        "runs": [],
    }
    _write_summary(summary_path, payload)

    skip_existing = bool(args.skip_existing) and not bool(args.force_rerun)
    skipped = 0
    executed = 0
    try:
        for ti, task_id in enumerate(task_ids):
            for si, seed in enumerate(cfg.seeds):
                print(f"[task {ti+1}/{len(task_ids)}] {task_id}  [seed {si+1}/{len(cfg.seeds)}] {seed}")
                rep: dict[str, Any] | None = None
                if skip_existing:
                    rep = _existing_report_for_task_seed(
                        task_id=task_id,
                        seed=seed,
                        cfg=cfg,
                        out_dirs=out_dirs,
                    )
                    if rep is not None:
                        skipped += 1
                        print("  -> skipped (reused existing artifacts)")
                if rep is None:
                    rep = run_for_task_and_seed(
                        task_id=task_id,
                        seed=seed,
                        cfg=cfg,
                        out_dirs=out_dirs,
                        reference_definition_dirs=reference_definition_dirs,
                    )
                    executed += 1
                payload["runs"].append(rep)
                _write_summary(summary_path, payload)
    except KeyboardInterrupt:
        _write_summary(summary_path, payload)
        print(f"Interrupted. Partial summary saved: {summary_path}")
        print(f"Progress: runs_recorded={len(payload['runs'])}, executed={executed}, skipped={skipped}")
        raise SystemExit(130)

    print(f"Summary saved: {summary_path}")
    print(f"Progress: runs_recorded={len(payload['runs'])}, executed={executed}, skipped={skipped}")


if __name__ == "__main__":
    main()

