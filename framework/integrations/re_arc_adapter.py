from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
from types import ModuleType
from typing import Callable, Optional

from framework.grids import Grid


_VERIFIERS_MODULE: ModuleType | None = None


def _candidate_re_arc_dirs() -> list[Path]:
    """Return possible locations of the vendored re_arc repo."""
    root = Path(__file__).resolve().parents[2]
    return [
        root / "external" / "re_arc",
        root / "PotARCin" / "external" / "re_arc",
    ]


def _load_verifiers_module() -> ModuleType | None:
    """Best-effort loader for the re_arc `verifiers.py` module.

    This keeps all re_arc-specific import logic in one place so that
    the rest of the framework can depend on a simple callable interface.
    """
    global _VERIFIERS_MODULE
    if _VERIFIERS_MODULE is not None:
        return _VERIFIERS_MODULE

    for base_dir in _candidate_re_arc_dirs():
        verifiers_path = base_dir / "verifiers.py"
        if not verifiers_path.exists():
            continue

        spec = importlib.util.spec_from_file_location(
            "re_arc_verifiers", verifiers_path
        )
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        sys.modules["re_arc_verifiers"] = module
        spec.loader.exec_module(module)
        _VERIFIERS_MODULE = module
        return module

    return None


def is_available() -> bool:
    """Return True if the re_arc verifiers module can be loaded."""
    return _load_verifiers_module() is not None


def get_re_arc_verifier(task_id: str) -> Optional[Callable[[Grid], Grid]]:
    """Return a callable verifier for the given task_id, if available.

    The exact API of re_arc may evolve; this adapter tries a few common
    patterns (e.g. a dictionary `VERIFIERS` or a `get_verifier` helper)
    and degrades gracefully to `None` if nothing matches.
    """
    module = _load_verifiers_module()
    if module is None:
        return None

    # Pattern 1: module exposes `get_verifier(task_id)`.
    get_verifier = getattr(module, "get_verifier", None)
    if callable(get_verifier):
        try:
            verifier = get_verifier(task_id)  # type: ignore[misc]
        except Exception:
            verifier = None
        if callable(verifier):
            return verifier

    # Pattern 2: module exposes a mapping `VERIFIERS` keyed by task_id.
    verifiers_map = getattr(module, "VERIFIERS", None)
    if isinstance(verifiers_map, dict):
        candidate = verifiers_map.get(task_id)
        if callable(candidate):
            return candidate

    return None

