from __future__ import annotations

import importlib.util
import inspect
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterator, Optional

from framework.grids import Grid, GridPair, is_int_grid
from framework.tasks.base import ArcTask

ROOT_DIR = Path(__file__).resolve().parents[2]
TEST2_DIR = ROOT_DIR / "Test2"

_TASK_DIR_CACHE: dict[int, Path] | None = None
_VERIFIER_CACHE: dict[int, Callable[[Grid], Grid]] = {}
_GENERATOR_CACHE: dict[int, Callable[..., list[GridPair]]] = {}


def parse_test2_task_num(task_id: str) -> int:
    s = task_id.strip().lower()
    for pat in (r"test2_t(\d+)", r"t(\d+)"):
        m = re.fullmatch(pat, s)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 50:
                return n
    raise ValueError(f"Invalid Test2 task id: {task_id!r}")


def canonical_test2_task_id(task_id: str) -> str:
    return f"test2_t{parse_test2_task_num(task_id)}"


def _discover_task_dirs() -> dict[int, Path]:
    global _TASK_DIR_CACHE
    if _TASK_DIR_CACHE is not None:
        return _TASK_DIR_CACHE
    mapping: dict[int, Path] = {}
    if not TEST2_DIR.is_dir():
        _TASK_DIR_CACHE = mapping
        return mapping
    for p in sorted(TEST2_DIR.iterdir()):
        if not p.is_dir():
            continue
        m = re.match(r"t(\d+)_", p.name)
        if not m:
            continue
        n = int(m.group(1))
        if 1 <= n <= 50 and n not in mapping:
            mapping[n] = p
    _TASK_DIR_CACHE = mapping
    return mapping


def _task_dir(task_num: int) -> Path:
    mapping = _discover_task_dirs()
    if task_num not in mapping:
        raise FileNotFoundError(f"Missing Test2 subdirectory for t{task_num} under {TEST2_DIR}")
    return mapping[task_num]


def _load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    old_generator_module = sys.modules.pop("generator", None)
    old_verifier_module = sys.modules.pop("verifier", None)
    try:
        task_dir = str(path.parent)
        if task_dir not in sys.path:
            sys.path.insert(0, task_dir)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path = old_path
        if old_generator_module is not None:
            sys.modules["generator"] = old_generator_module
        else:
            sys.modules.pop("generator", None)
        if old_verifier_module is not None:
            sys.modules["verifier"] = old_verifier_module
        else:
            sys.modules.pop("verifier", None)


def _wrap_verifier_fn(fn: Callable[..., object], task_num: int) -> Callable[[Grid], Grid]:
    sig = inspect.signature(fn)
    required = [
        p
        for p in sig.parameters.values()
        if p.default is inspect._empty
        and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    arity = len(required)
    if arity == 1:
        def _one_arg(grid: Grid) -> Grid:
            out = fn(grid)  # type: ignore[misc]
            if not is_int_grid(out):
                raise ValueError(f"Test2 verifier verify_t{task_num} returned non-grid output")
            return out

        return _one_arg
    raise ValueError(
        f"Unsupported verifier signature for t{task_num}: expected one required arg, got {arity}"
    )


def _load_test2_verifier(task_num: int) -> Optional[Callable[[Grid], Grid]]:
    if task_num in _VERIFIER_CACHE:
        return _VERIFIER_CACHE[task_num]
    path = _task_dir(task_num) / "verifier.py"
    if not path.is_file():
        return None
    module = _load_module(path, f"test2_verifier_t{task_num}")
    candidates = (
        f"verify_t{task_num}",
        f"transform_t{task_num}",
        "verify",
        "transform",
    )
    for name in candidates:
        raw = getattr(module, name, None)
        if callable(raw):
            try:
                wrapped = _wrap_verifier_fn(raw, task_num)
            except Exception:
                continue
            _VERIFIER_CACHE[task_num] = wrapped
            return wrapped
    # Some tasks expose a boolean verifier(inp, out) plus transform_tN in the same module.
    verify_bool = getattr(module, f"verify_t{task_num}", None)
    transform = getattr(module, f"transform_t{task_num}", None)
    if callable(verify_bool) and callable(transform):
        try:
            verify_sig = inspect.signature(verify_bool)
            if len(
                [
                    p
                    for p in verify_sig.parameters.values()
                    if p.default is inspect._empty
                    and p.kind
                    in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                ]
            ) == 2:
                wrapped = _wrap_verifier_fn(transform, task_num)
                _VERIFIER_CACHE[task_num] = wrapped
                return wrapped
        except Exception:
            pass
    return None


def _load_test2_generator(task_num: int) -> Optional[Callable[..., list[GridPair]]]:
    if task_num in _GENERATOR_CACHE:
        return _GENERATOR_CACHE[task_num]
    path = _task_dir(task_num) / "generator.py"
    if not path.is_file():
        return None
    module = _load_module(path, f"test2_generator_t{task_num}")
    raw = getattr(module, "generate", None)
    if not callable(raw):
        return None
    task_dir = _task_dir(task_num)
    verifier_path = task_dir / "verifier.py"

    def _gen(num_examples: int, *, rng=None) -> list[GridPair]:
        out: list[GridPair] = []
        for _ in range(num_examples):
            old_path = list(sys.path)
            old_verifier_module = sys.modules.pop("verifier", None)
            try:
                p = str(task_dir)
                if p not in sys.path:
                    sys.path.insert(0, p)
                if verifier_path.is_file():
                    sys.modules["verifier"] = _load_module(verifier_path, f"test2_runtime_verifier_t{task_num}")
                try:
                    sample = raw(rng=rng)
                except TypeError:
                    sample = raw()
            finally:
                sys.path = old_path
                if old_verifier_module is not None:
                    sys.modules["verifier"] = old_verifier_module
                else:
                    sys.modules.pop("verifier", None)
            if not isinstance(sample, dict):
                raise ValueError(f"Test2 generator t{task_num} returned non-dict sample")
            inp = sample.get("input")
            output = sample.get("output")
            if not is_int_grid(inp) or not is_int_grid(output):
                raise ValueError(f"Test2 generator t{task_num} returned invalid grids")
            out.append(GridPair(inp, output))
        return out

    _GENERATOR_CACHE[task_num] = _gen
    return _gen


def load_test2_task(task_id: str) -> ArcTask:
    task_num = parse_test2_task_num(task_id)
    task_json = TEST2_DIR / f"t{task_num}.json"
    if not task_json.is_file():
        raise FileNotFoundError(f"Missing Test2 task file: {task_json}")
    payload = json.loads(task_json.read_text(encoding="utf-8"))
    train = payload.get("train")
    test = payload.get("test")
    if not isinstance(train, list) or not isinstance(test, list):
        raise ValueError(f"Malformed Test2 task JSON: {task_json}")

    train_pairs: list[GridPair] = []
    for i, ex in enumerate(train):
        if not isinstance(ex, dict):
            raise ValueError(f"{task_json}: train[{i}] must be an object")
        inp, out = ex.get("input"), ex.get("output")
        if not is_int_grid(inp) or not is_int_grid(out):
            raise ValueError(f"{task_json}: invalid train[{i}] grid(s)")
        train_pairs.append(GridPair(inp, out))

    test_inputs: list[Grid] = []
    test_outputs: list[Grid] = []
    for i, ex in enumerate(test):
        if not isinstance(ex, dict):
            raise ValueError(f"{task_json}: test[{i}] must be an object")
        inp, out = ex.get("input"), ex.get("output")
        if not is_int_grid(inp) or not is_int_grid(out):
            raise ValueError(f"{task_json}: invalid test[{i}] grid(s)")
        test_inputs.append(inp)
        test_outputs.append(out)

    verifier = _load_test2_verifier(task_num)
    generator = _load_test2_generator(task_num)

    return ArcTask(
        task_id=canonical_test2_task_id(task_id),
        train_pairs=train_pairs,
        test_inputs=test_inputs,
        test_outputs=test_outputs,
        verifier=verifier,
        arc_gen_generator=generator,
    )


def iter_test2_tasks() -> Iterator[ArcTask]:
    for n in range(1, 51):
        yield load_test2_task(f"test2_t{n}")
