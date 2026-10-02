"""Verifier selection using optional ``task_valid_verifiers.csv`` (offline audit).

Runtimes should prefer CSV slot order and avoid scanning every verifier with full
train/test/stable and generator-sampled (``task.arc_gen_generator``) checks. When a
dimension has instance-specific generated pairs, pass them as ``dynamic_pairs``; the first
CSV-listed verifier that both exists on :class:`~framework.tasks.base.ArcTask` and
matches those pairs is returned. If none match, falls back to the legacy full probe
(:func:`~framework.dimensions.classification_distribution.verifier_matches_train_test_stable_dynamic50`).
"""

from __future__ import annotations

import copy
import csv
import os
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple

from framework.grids import GridPair, is_equal_grid
from framework.tasks.base import ArcTask, Verifier
from framework.tasks.test2_dataset import parse_test2_task_num

VerifierSlot = Literal["re_arc", "google", "keymoon", "neurips", "custom"]

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CSV = _ROOT / "task_valid_verifiers.csv"
_CSV_SLOTS: dict[str, list[VerifierSlot]] | None = None
# The one verifier a task is meant to be judged by. Several slots can be listed
# as valid and still disagree with the generator on draws the audit did not
# see, so the canonical choice is recorded per task and tried first.
_CSV_SELECTED: dict[str, VerifierSlot] | None = None
_CSV_SELECTED_FROM: Path | None = None
_CSV_LOADED_FROM: Path | None = None


def default_verifiers_csv_path() -> Path:
    """CSV path from ``POTARCIN_VALID_VERIFIERS_CSV`` or repo-root default."""
    env = os.environ.get("POTARCIN_VALID_VERIFIERS_CSV")
    return Path(env).expanduser() if env else _DEFAULT_CSV


def clear_verifier_csv_cache() -> None:
    """Drop cached CSV mapping (e.g. after swapping files in tests)."""
    global _CSV_SLOTS, _CSV_SELECTED, _CSV_SELECTED_FROM, _CSV_LOADED_FROM
    _CSV_SLOTS = None
    _CSV_SELECTED = None
    _CSV_SELECTED_FROM = None
    _CSV_LOADED_FROM = None


def _load_selected_csv(path: Path) -> dict[str, VerifierSlot]:
    """task_id -> the slot named in ``selected_verifier_slot``, when it is a real slot."""
    if not path.is_file():
        return {}
    out: dict[str, VerifierSlot] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            tid = (row.get("task_id") or "").strip()
            sel = (row.get("selected_verifier_slot") or "").strip()
            if tid and sel in ("re_arc", "google", "keymoon", "neurips", "custom"):
                out[tid] = sel  # type: ignore[assignment]
    return out


def csv_selected_slot_for_task(task_id: str, csv_path: Path | None = None) -> VerifierSlot | None:
    """The canonical slot recorded for *task_id*, or ``None``."""
    global _CSV_SELECTED, _CSV_SELECTED_FROM
    path = csv_path if csv_path is not None else default_verifiers_csv_path()
    if _CSV_SELECTED is None or _CSV_SELECTED_FROM != path:
        _CSV_SELECTED = _load_selected_csv(path)
        _CSV_SELECTED_FROM = path
    return _CSV_SELECTED.get(task_id)


def _canonical_first(task_id: str, slots: Sequence[VerifierSlot], csv_path: Path | None) -> list[VerifierSlot]:
    """*slots* with the task's canonical slot moved to the front, when it is listed."""
    canon = csv_selected_slot_for_task(task_id, csv_path)
    ordered = list(slots)
    if canon in ordered:
        ordered.remove(canon)
        ordered.insert(0, canon)
    return ordered


def _load_csv(path: Path) -> dict[str, list[VerifierSlot]]:
    if not path.is_file():
        return {}
    out: dict[str, list[VerifierSlot]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = (row.get("task_id") or "").strip()
            if not tid:
                continue
            raw = (row.get("valid_verifier_slots") or "").strip()
            slots: list[VerifierSlot] = []
            for part in raw.split(";"):
                p = part.strip()
                if p in ("re_arc", "google", "keymoon", "neurips", "custom"):
                    slots.append(p)  # type: ignore[arg-type]
            out[tid] = slots
    return out


def _slots_for_task_id(task_id: str, csv_path: Path | None) -> list[VerifierSlot]:
    global _CSV_SLOTS, _CSV_LOADED_FROM
    path = csv_path if csv_path is not None else default_verifiers_csv_path()
    if _CSV_SLOTS is None or _CSV_LOADED_FROM != path:
        _CSV_SLOTS = _load_csv(path)
        _CSV_LOADED_FROM = path
    return _CSV_SLOTS.get(task_id, [])


def verifier_matches_pairs(v: Verifier, pairs: Sequence[GridPair]) -> bool:
    """True if *v* agrees with every (input, output) pair (safe w.r.t. verifier errors)."""
    for p in pairs:
        try:
            out = v(copy.deepcopy(p.input))
            if not is_equal_grid(out, p.output):
                return False
        except Exception:
            return False
    return True


def _callable_for_slot(task: ArcTask, slot: VerifierSlot) -> Verifier | None:
    if slot == "re_arc":
        return task.verifier
    if slot == "google":
        return task.secondary_verifier
    if slot == "keymoon":
        return task.tertiary_verifier
    if slot == "neurips":
        return task.quaternary_verifier
    if slot == "custom":
        return task.quinary_verifier
    return None


# custom first: hand-written verifiers correct bugs in the slot they replace.
_PRIORITY: tuple[VerifierSlot, ...] = ("custom", "re_arc", "google", "keymoon", "neurips")


def _legacy_select_verifier(
    task: ArcTask,
    dynamic_pairs: Sequence[GridPair] | None,
) -> Tuple[VerifierSlot, Verifier] | None:
    from framework.dimensions.classification_distribution import (
        verifier_matches_train_test_stable_dynamic50,
    )

    for slot in _PRIORITY:
        fn = _callable_for_slot(task, slot)
        if fn is None:
            continue
        if not verifier_matches_train_test_stable_dynamic50(task, fn):
            continue
        if dynamic_pairs and not verifier_matches_pairs(fn, dynamic_pairs):
            continue
        return slot, fn
    return None


def select_verifier_for_task(
    task: ArcTask,
    *,
    dynamic_pairs: Sequence[GridPair] | None = None,
    csv_path: Path | None = None,
) -> Tuple[VerifierSlot, Verifier] | None:
    """Choose a verifier: the CSV's canonical slot, then CSV order, optional instance dynamic check, then legacy probe."""
    try:
        parse_test2_task_num(task.task_id)
        if task.verifier is None:
            return None
        if dynamic_pairs and not verifier_matches_pairs(task.verifier, dynamic_pairs):
            return None
        return "re_arc", task.verifier
    except ValueError:
        pass

    csv_order = _canonical_first(task.task_id, _slots_for_task_id(task.task_id, csv_path), csv_path)
    for slot in csv_order:
        fn = _callable_for_slot(task, slot)
        if fn is None:
            continue
        if dynamic_pairs and not verifier_matches_pairs(fn, dynamic_pairs):
            continue
        return slot, fn
    return _legacy_select_verifier(task, dynamic_pairs)


def valid_verifiers_for_task(
    task: ArcTask,
    *,
    csv_path: Path | None = None,
) -> list[Tuple[VerifierSlot, Verifier]]:
    """Return all valid verifiers (train/test/stable + 50 ``arc_gen`` samples), CSV order first."""
    try:
        parse_test2_task_num(task.task_id)
        return [("re_arc", task.verifier)] if task.verifier is not None else []
    except ValueError:
        pass

    from framework.dimensions.classification_distribution import (
        verifier_matches_train_test_stable_dynamic50,
    )

    out: list[Tuple[VerifierSlot, Verifier]] = []
    seen: set[VerifierSlot] = set()

    ordered_slots = _canonical_first(task.task_id, _slots_for_task_id(task.task_id, csv_path), csv_path) + list(_PRIORITY)
    for slot in ordered_slots:
        if slot in seen:
            continue
        seen.add(slot)
        fn = _callable_for_slot(task, slot)
        if fn is None:
            continue
        if not verifier_matches_train_test_stable_dynamic50(task, fn):
            continue
        out.append((slot, fn))
    return out
