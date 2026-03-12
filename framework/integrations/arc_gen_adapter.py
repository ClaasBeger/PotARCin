from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
from types import ModuleType
from typing import Any, Dict, List, Optional

from framework.grids import Grid, GridPair


_ARC_GEN_MODULE: ModuleType | None = None


def _candidate_arc_gen_dirs() -> list[Path]:
    """Return possible locations of the vendored ARC-GEN repo."""
    root = Path(__file__).resolve().parents[2]
    return [
        root / "external" / "ARC-GEN",
        root / "PotARCin" / "external" / "ARC-GEN",
    ]


def _load_arc_gen_module() -> ModuleType | None:
    """Best-effort loader for the `arc_gen.py` module in external/ARC-GEN."""
    global _ARC_GEN_MODULE
    if _ARC_GEN_MODULE is not None:
        return _ARC_GEN_MODULE

    for base_dir in _candidate_arc_gen_dirs():
        arc_gen_path = base_dir / "arc_gen.py"
        if not arc_gen_path.exists():
            continue

        spec = importlib.util.spec_from_file_location(
            "arc_gen_external", arc_gen_path
        )
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        sys.modules["arc_gen_external"] = module
        spec.loader.exec_module(module)
        _ARC_GEN_MODULE = module
        return module

    return None


def is_available() -> bool:
    """Return True if the ARC-GEN module can be loaded from `external/ARC-GEN`."""
    return _load_arc_gen_module() is not None


def generate_synthetic_pairs(
    task_id: str,
    num_examples: int = 1,
) -> List[GridPair]:
    """Generate additional synthetic (input, output) examples for a task.

    This is a thin wrapper that delegates to ARC-GEN if it is present.
    The exact function name and signature in ARC-GEN may differ; adapt
    this once you decide on a concrete integration strategy.
    """
    module = _load_arc_gen_module()
    if module is None:
        raise RuntimeError("ARC-GEN is not available under external/ARC-GEN")

    # The concrete API here is intentionally loose; adjust as needed.
    generate_fn = getattr(module, "generate_variations_for_task", None)
    if not callable(generate_fn):
        raise RuntimeError(
            "ARC-GEN module does not expose `generate_variations_for_task`"
        )

    raw_examples: List[Dict[str, Any]] = generate_fn(task_id, num_examples)  # type: ignore[misc]
    pairs: List[GridPair] = []
    for ex in raw_examples:
        inp: Grid = ex["input"]
        out: Grid = ex["output"]
        pairs.append(GridPair(inp, out))
    return pairs

