from __future__ import annotations

import importlib.util
import itertools
import json
import random
from collections import defaultdict
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple

from framework.custom_verifiers import get_custom_verifier
from framework.grids import Grid, GridPair
from framework.tasks.base import ArcTask, TaskSource
from framework.integrations.re_arc_adapter import get_re_arc_verifier


ROOT_DIR = Path(__file__).resolve().parents[2]

ARC_ORIGINAL_DIR = ROOT_DIR / "external" / "arc_original_train"
RE_ARC_ZIP = ROOT_DIR / "external" / "re_arc" / "re_arc.zip"
ARC_GEN_STABLE_ZIP = ROOT_DIR / "external" / "arc_gen_stable.zip"

_ARC_GEN_TASK_LIST_MODULE: ModuleType | None = None


def _load_arc_original(task_id: str) -> Tuple[List[GridPair], List[Grid], List[Grid]]:
    """Load canonical ARC train/test examples for a task from arc_original_train/.

    Returns (train_pairs, test_inputs, test_outputs), where test_outputs
    come from the same JSON file (ground truth for the original ARC test
    inputs).
    """
    if not ARC_ORIGINAL_DIR.exists():
        raise FileNotFoundError(
            f"Missing arc_original_train directory at {ARC_ORIGINAL_DIR}"
        )

    path = ARC_ORIGINAL_DIR / f"{task_id}.json"
    if not path.exists():
        raise KeyError(f"No original ARC task {task_id!r} at {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    train_pairs = [
        GridPair(example["input"], example["output"]) for example in data["train"]
    ]
    test_inputs: List[Grid] = [example["input"] for example in data["test"]]
    test_outputs: List[Grid] = [example["output"] for example in data["test"]]
    return train_pairs, test_inputs, test_outputs


def _load_re_arc_synthetic_pairs(task_id: str) -> Optional[List[GridPair]]:
    """Load stable re_arc synthetic input/output pairs from re_arc.zip, if present."""
    import zipfile

    if not RE_ARC_ZIP.exists():
        return None

    with zipfile.ZipFile(RE_ARC_ZIP) as zf:
        path = f"re_arc/tasks/{task_id}.json"
        try:
            raw = zf.read(path)
        except KeyError:
            return None
    data = json.loads(raw)
    return [GridPair(entry["input"], entry["output"]) for entry in data]


def _load_arc_gen_stable_pairs(task_id: str) -> Optional[List[GridPair]]:
    """Load stable ARC-GEN synthetic pairs from arc_gen_stable.zip, if present."""
    import zipfile

    if not ARC_GEN_STABLE_ZIP.exists():
        return None

    with zipfile.ZipFile(ARC_GEN_STABLE_ZIP) as zf:
        path = f"{task_id}.json"
        try:
            raw = zf.read(path)
        except KeyError:
            return None
    data = json.loads(raw)
    return [GridPair(entry["input"], entry["output"]) for entry in data]


def _load_re_arc_generators_module() -> Optional[ModuleType]:
    """Load external/re_arc/generators.py as a module."""
    path = ROOT_DIR / "external" / "re_arc" / "generators.py"
    if not path.exists():
        return None

    # Ensure relative imports like `from dsl import *` and `from utils import *`
    # inside generators.py can be resolved by temporarily adding the re_arc
    # directory to sys.path.
    re_arc_dir = path.parent
    original_sys_path = list(sys.path)
    try:
        if str(re_arc_dir) not in sys.path:
            sys.path.insert(0, str(re_arc_dir))

        spec = importlib.util.spec_from_file_location("re_arc_generators", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ModuleNotFoundError:
            # If re_arc's own dependencies are not importable, just skip
            # exposing generators rather than failing task loading entirely.
            return None
        return module
    finally:
        sys.path = original_sys_path


def _make_re_arc_generator(task_id: str) -> Optional[Callable[[int], List[GridPair]]]:
    """Return a callable that generates re_arc synthetic pairs for a task."""
    module = _load_re_arc_generators_module()
    if module is None:
        return None
    gen_fn = getattr(module, f"generate_{task_id}", None)
    if not callable(gen_fn):
        return None

    def generator(num_examples: int, *, diff_lb: float = 0.0, diff_ub: float = 1.0) -> List[GridPair]:
        pairs: List[GridPair] = []
        for _ in range(num_examples):
            example = gen_fn(diff_lb, diff_ub)  # type: ignore[misc]
            pairs.append(GridPair(example["input"], example["output"]))
        return pairs

    # Expose as a simple Callable[[int], List[GridPair]] by fixing difficulty.
    return lambda n: generator(n)


def _load_arc_gen_task_list_module() -> Optional[ModuleType]:
    """Load external/ARC-GEN/task_list.py as a module."""
    global _ARC_GEN_TASK_LIST_MODULE
    if _ARC_GEN_TASK_LIST_MODULE is not None:
        return _ARC_GEN_TASK_LIST_MODULE

    path = ROOT_DIR / "external" / "ARC-GEN" / "task_list.py"
    if not path.exists():
        return None

    arc_gen_dir = path.parent
    original_sys_path = list(sys.path)
    try:
        # Ensure `from tasks.training import ...` can resolve by adding the
        # ARC-GEN root (which contains the `tasks` package) to sys.path.
        if str(arc_gen_dir) not in sys.path:
            sys.path.insert(0, str(arc_gen_dir))

        spec = importlib.util.spec_from_file_location("arc_gen_task_list", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ModuleNotFoundError:
            # If ARC-GEN's internal imports fail (e.g. missing dependencies),
            # degrade gracefully and skip exposing its generators.
            return None

        _ARC_GEN_TASK_LIST_MODULE = module
        return module
    finally:
        sys.path = original_sys_path


def _arc_gen_id_to_task_num_and_generator(
    task_id: str,
) -> Optional[Tuple[int, Callable[[], Dict[str, Grid]]]]:
    """Map ARC task_id to (task_num, generator) using ARC-GEN's task_list."""
    module = _load_arc_gen_task_list_module()
    if module is None:
        return None
    task_list_fn = getattr(module, "task_list", None)
    if not callable(task_list_fn):
        return None
    mapping: Dict[int, Tuple[str, Callable[[], Dict[str, Grid]], Callable]] = task_list_fn()  # type: ignore[misc]
    for num, (arc_id, generator, _validator) in mapping.items():
        if arc_id == task_id:
            return num, generator
    return None


def _arc_gen_90f3ed37_cols_match_global_awide(cols: List[int]) -> bool:
    """task219 uses one pre-sample ``awide`` in {1,2}; each band has ``col = awide * k``, k∈{1,2}.

    So either every col is in {1, 2} (original awide=1) or every col is in {2, 4} (original awide=2).
    Mixing 1 and 4 across bands is impossible for the real generator.
    """
    if any(c == 1 for c in cols):
        return all(c in (1, 2) for c in cols)
    if any(c == 4 for c in cols):
        return all(c in (2, 4) for c in cols)
    return True


def _arc_gen_90f3ed37_analyze_latent_fits(
    example: Dict[str, Grid],
    *,
    apply_col_lexmax: bool = True,
) -> Tuple[Set[Tuple[Tuple[int, ...], ...]], bool]:
    """Enumerate task219-consistent outputs for ``example["input"]`` (no verifier calls).

    Returns ``(output_keys, expected_matches)`` where ``output_keys`` is the set of
    distinct output grids (as row-tuple keys) consistent with the input, and
    ``expected_matches`` is True iff ``example["output"]`` is among them.

    If ``apply_col_lexmax`` is True, within each fixed band geometry, only column
    assignments in the ``{1,2}`` regime that are lexicographically maximal are kept
    (removes degenerate ``1`` vs ``2`` ambiguities). Set False for the raw latent set.
    """
    inp = example["input"]
    exp = example["output"]
    h = len(inp)
    w = len(inp[0]) if h else 0
    if h == 0 or w == 0:
        return False

    BLACK, BLUE, CYAN = 0, 1, 8

    cyan_rows = sorted(r for r in range(h) if any(inp[r][c] == CYAN for c in range(w)))
    if not cyan_rows:
        return set(), False

    def _cluster_cyan_rows(rows_list: List[int], tall: int) -> List[List[int]]:
        clusters: List[List[int]] = []
        cluster = [rows_list[0]]
        for r in rows_list[1:]:
            lo = min(cluster + [r])
            hi = max(cluster + [r])
            if hi - lo <= tall - 1:
                cluster.append(r)
            else:
                clusters.append(cluster)
                cluster = [r]
        clusters.append(cluster)
        return clusters

    def _bases_for_cluster(cluster: List[int], tall: int) -> List[int]:
        lo, hi = min(cluster), max(cluster)
        lo_b, hi_b = hi - tall + 1, lo
        return [b for b in range(lo_b, hi_b + 1) if b >= 0 and b + tall <= h]

    def _pack(vals: List[List[int]], wide: int, tall_local: int) -> int:
        m = 0
        for rr in range(tall_local):
            for cc in range(wide):
                if vals[rr][cc]:
                    m |= 1 << (rr * wide + cc)
        return m

    exp_key = tuple(tuple(row) for row in exp)
    # Per fixed (tall, band placement), keep only lexicographically maximal ``cols``
    # among assignments that match the input — removes spurious (1,*,*,...) vs (2,*,*,...)
    # ties when both are valid under global awide=1.
    fits_by_geom: Dict[
        Tuple[int, Tuple[Tuple[int, int], ...]],
        List[Tuple[Tuple[int, ...], Tuple[Tuple[int, ...], ...]]],
    ] = defaultdict(list)

    for tall in (1, 2, 3):
        clusters = _cluster_cyan_rows(cyan_rows, tall)
        base_opts: List[List[int]] = []
        skip_tall = False
        for cl in clusters:
            bs = _bases_for_cluster(cl, tall)
            if not bs:
                skip_tall = True
                break
            base_opts.append(bs)
        if skip_tall:
            continue

        for base_choice in itertools.product(*base_opts):
            bc = list(base_choice)
            ok_gap = True
            for i in range(len(bc) - 1):
                if bc[i + 1] < bc[i] + tall + 1:
                    ok_gap = False
                    break
            if not ok_gap:
                continue
            bands = [(b, b + tall - 1) for b in bc]
            n = len(bands)

            def _render(
                aw: int,
                bw: int,
                cw: int,
                cols: List[int],
                ma: int,
                mb: int,
                mc: int,
                output_mode: bool,
            ) -> Grid:
                out = [[BLACK for _ in range(w)] for _ in range(h)]
                for bi, (s, _e) in enumerate(bands):
                    col = cols[bi]
                    for rr in range(tall):
                        row = s + rr
                        for a0 in range(0, col, aw):
                            for lc in range(aw):
                                if ((ma >> (rr * aw + lc)) & 1):
                                    c = a0 + lc
                                    if 0 <= c < w:
                                        out[row][c] = CYAN
                        for lc in range(bw):
                            if ((mb >> (rr * bw + lc)) & 1):
                                c = col + lc
                                if 0 <= c < w:
                                    out[row][c] = CYAN
                        if output_mode:
                            c_color = CYAN if bi == 0 else BLUE
                            for c0 in range(col + bw, w, cw):
                                for lc in range(cw):
                                    if ((mc >> (rr * cw + lc)) & 1):
                                        c = c0 + lc
                                        if 0 <= c < w:
                                            out[row][c] = c_color
                        elif bi == 0:
                            for c0 in range(col + bw, w, cw):
                                for lc in range(cw):
                                    if ((mc >> (rr * cw + lc)) & 1):
                                        c = c0 + lc
                                        if 0 <= c < w:
                                            out[row][c] = CYAN
                return out

            for aw in (1, 2):
                for bw in (1, 2):
                    for cw in (1, 2):
                        col_options = [aw, 2 * aw] + ([4 * aw] if aw == 1 else [])
                        for col_mask in range(len(col_options) ** n):
                            tmp = col_mask
                            cols = []
                            for _ in range(n):
                                cols.append(col_options[tmp % len(col_options)])
                                tmp //= len(col_options)

                            if not _arc_gen_90f3ed37_cols_match_global_awide(cols):
                                continue

                            a_vals = [[-1 for _ in range(aw)] for _ in range(tall)]
                            b_vals = [[-1 for _ in range(bw)] for _ in range(tall)]
                            c_vals = [[-1 for _ in range(cw)] for _ in range(tall)]
                            valid = True

                            def _set(arr: List[List[int]], rr: int, cc: int, val: int) -> None:
                                nonlocal valid
                                old = arr[rr][cc]
                                if old == -1:
                                    arr[rr][cc] = val
                                elif old != val:
                                    valid = False

                            for bi, (s, _e) in enumerate(bands):
                                col = cols[bi]
                                for rr in range(tall):
                                    row = s + rr
                                    for a0 in range(0, col, aw):
                                        for lc in range(aw):
                                            cc = a0 + lc
                                            if not (0 <= cc < w):
                                                valid = False
                                                break
                                            _set(a_vals, rr, lc, 1 if inp[row][cc] == CYAN else 0)
                                        if not valid:
                                            break
                                    if not valid:
                                        break
                                    for lc in range(bw):
                                        cc = col + lc
                                        if not (0 <= cc < w):
                                            valid = False
                                            break
                                        _set(b_vals, rr, lc, 1 if inp[row][cc] == CYAN else 0)
                                    if not valid:
                                        break
                                    if bi == 0:
                                        for c0 in range(col + bw, w, cw):
                                            for lc in range(cw):
                                                cc = c0 + lc
                                                if 0 <= cc < w:
                                                    _set(c_vals, rr, lc, 1 if inp[row][cc] == CYAN else 0)
                                            if not valid:
                                                break
                                    if not valid:
                                        break
                                if not valid:
                                    break

                            if not valid:
                                continue

                            for rr in range(tall):
                                for cc in range(aw):
                                    if a_vals[rr][cc] == -1:
                                        a_vals[rr][cc] = 0
                                for cc in range(bw):
                                    if b_vals[rr][cc] == -1:
                                        b_vals[rr][cc] = 0
                                for cc in range(cw):
                                    if c_vals[rr][cc] == -1:
                                        c_vals[rr][cc] = 0

                            ma = _pack(a_vals, aw, tall)
                            mb = _pack(b_vals, bw, tall)
                            mc = _pack(c_vals, cw, tall)
                            if ma == 0 or mb == 0 or mc == 0:
                                continue

                            if _render(aw, bw, cw, cols, ma, mb, mc, output_mode=False) != inp:
                                continue

                            out = _render(aw, bw, cw, cols, ma, mb, mc, output_mode=True)
                            key = tuple(tuple(row) for row in out)
                            geom = (tall, tuple(bands))
                            fits_by_geom[geom].append((tuple(cols), key))

    output_keys: Set[Tuple[Tuple[int, ...], ...]] = set()
    expected_matches = False
    for _geom, lst in fits_by_geom.items():
        if not lst:
            continue
        feasible_cols = {t[0] for t in lst}
        col_filter: Optional[Tuple[int, ...]] = None
        if apply_col_lexmax:
            # Only apply tie-break when ambiguity is within orig_awide=1 ({1,2} per band).
            strict12 = {c for c in feasible_cols if all(x in (1, 2) for x in c)}
            if len(strict12) >= 2:
                col_filter = max(strict12)
        for cols_tuple, key in lst:
            if col_filter is not None and cols_tuple != col_filter:
                continue
            output_keys.add(key)
            if key == exp_key:
                expected_matches = True

    return output_keys, expected_matches


def _arc_gen_90f3ed37_example_is_unambiguous(example: Dict[str, Grid]) -> bool:
    """Dynamic filter: accept if raw latent outputs are unique, or if canonical {1,2}-cols tie-break yields a unique label-consistent output."""
    keys_raw, _ = _arc_gen_90f3ed37_analyze_latent_fits(example, apply_col_lexmax=False)
    if len(keys_raw) == 1:
        return True
    keys_canon, ok = _arc_gen_90f3ed37_analyze_latent_fits(example, apply_col_lexmax=True)
    return len(keys_canon) == 1 and ok


def _arc_gen_a64e4611_example_is_unambiguous(example: Dict[str, Grid]) -> bool:
    """Task255/a64e4611 dynamic filter: exclude edge-adjacent empty-column ambiguity.

    Reject only when green touches a side border and the adjacent inner column is
    entirely empty of green, which makes edge-fill behavior ambiguous.
    """
    out = example["output"]
    h = len(out)
    w = len(out[0]) if h else 0
    if h == 0 or w < 2:
        return False
    GREEN = 3
    left_touches = any(out[r][0] == GREEN for r in range(h))
    right_touches = any(out[r][w - 1] == GREEN for r in range(h))
    left_adjacent_empty = all(out[r][1] != GREEN for r in range(h))
    right_adjacent_empty = all(out[r][w - 2] != GREEN for r in range(h))

    if left_touches and left_adjacent_empty:
        return False
    if right_touches and right_adjacent_empty:
        return False

    # Keep all stable pairs even if the structural rule is conservative.
    inp_sig = tuple(tuple(int(v) for v in row) for row in example["input"])
    out_sig = tuple(tuple(int(v) for v in row) for row in example["output"])
    if (inp_sig, out_sig) in _arc_gen_a64e4611_stable_signatures():
        return True

    for r in range(h):
        if out[r][0] == GREEN and out[r][1] == GREEN:
            return True
        if out[r][w - 1] == GREEN and out[r][w - 2] == GREEN:
            return True
    return True


def _arc_gen_a64e4611_stable_signatures() -> Set[Tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]]]:
    """Cached stable (input, output) signatures for task a64e4611."""
    cached = getattr(_arc_gen_a64e4611_stable_signatures, "_cache", None)
    if cached is not None:
        return cached
    pairs = _load_arc_gen_stable_pairs("a64e4611")
    sigs: Set[Tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]]] = set()
    for p in pairs:
        inp_sig = tuple(tuple(int(v) for v in row) for row in p.input)
        out_sig = tuple(tuple(int(v) for v in row) for row in p.output)
        sigs.add((inp_sig, out_sig))
    setattr(_arc_gen_a64e4611_stable_signatures, "_cache", sigs)
    return sigs


def verify_arc_gen_90f3ed37_stable_coverage() -> Tuple[int, int, List[int]]:
    """Each stable pair has at least one latent fit that reproduces its labeled output.

    This is weaker than :func:`_arc_gen_90f3ed37_example_is_unambiguous` (which also
    requires uniqueness). Use this to confirm the stable dataset lies in the
    generator's *support*.

    Returns ``(accepted_count, total, failed_indices)``. If ``arc_gen_stable.zip`` is
    missing, returns ``(0, 0, [])``.
    """
    pairs = _load_arc_gen_stable_pairs("90f3ed37")
    if not pairs:
        return 0, 0, []
    failed: List[int] = []
    for i, p in enumerate(pairs):
        _keys, ok = _arc_gen_90f3ed37_analyze_latent_fits(
            {"input": p.input, "output": p.output}
        )
        if not ok:
            failed.append(i)
    return len(pairs) - len(failed), len(pairs), failed


def verify_arc_gen_90f3ed37_stable_unambiguous_coverage() -> Tuple[int, int, List[int]]:
    """Same as stable coverage but requires the strict dynamic-generator predicate."""
    pairs = _load_arc_gen_stable_pairs("90f3ed37")
    if not pairs:
        return 0, 0, []
    failed: List[int] = []
    for i, p in enumerate(pairs):
        if not _arc_gen_90f3ed37_example_is_unambiguous({"input": p.input, "output": p.output}):
            failed.append(i)
    return len(pairs) - len(failed), len(pairs), failed


def _make_arc_gen_generator(task_id: str) -> Optional[Callable[[int], List[GridPair]]]:
    """Return a callable that generates ARC-GEN synthetic pairs for a task."""
    lookup = _arc_gen_id_to_task_num_and_generator(task_id)
    if lookup is None:
        return None
    task_num, generator = lookup

    # Adaptation for a64e4611:
    # task255.py now includes the edge-fill post-process directly. The loader-side
    # replacement is kept as a defensive fallback for environments that still carry an
    # older task255 implementation.
    if task_id == "a64e4611":
        try:
            import importlib.util
            from pathlib import Path

            edgefill_path = (
                ROOT_DIR
                / "external"
                / "ARC-GEN"
                / "tasks"
                / "training"
                / "task255_edgefill.py"
            )
            spec = importlib.util.spec_from_file_location(
                "arcgen_task255_edgefill", edgefill_path
            )
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                edge_gen = getattr(module, "generate", None)
                if callable(edge_gen):
                    generator = edge_gen  # type: ignore[assignment]
        except Exception:
            # Fall back to the original generator.
            pass

    def gen(num_examples: int) -> List[GridPair]:
        pairs: List[GridPair] = []
        if task_id == "a64e4611":
            max_attempts = max(2000, num_examples * 120)
            example_id = 0
            attempts = 0
            while len(pairs) < num_examples and attempts < max_attempts:
                random.seed(task_num + example_id)
                example = generator()
                attempts += 1
                example_id += 1
                if not _arc_gen_a64e4611_example_is_unambiguous(example):
                    continue
                pairs.append(GridPair(example["input"], example["output"]))
            if len(pairs) < num_examples:
                raise RuntimeError(
                    f"Custom ARC-GEN generator acceptance too low for {task_id!r}: "
                    f"got {len(pairs)}/{num_examples} after {attempts} attempts"
                )
            return pairs

        for example_id in range(num_examples):
            random.seed(task_num + example_id)
            example = generator()
            pairs.append(GridPair(example["input"], example["output"]))
        return pairs

    return gen


def _load_golf_verifier_from_neurips(task_id: str) -> Optional[Callable[[Grid], Grid]]:
    """Best-effort loader from `external/NeurIPS-Code-Golf-2025/solutions`."""
    solutions_root = ROOT_DIR / "external" / "NeurIPS-Code-Golf-2025" / "solutions"
    if not solutions_root.exists():
        return None

    lookup = _arc_gen_id_to_task_num_and_generator(task_id)
    if lookup is None:
        return None
    task_num, _generator = lookup
    path = solutions_root / f"task{task_num:03d}.py"
    if not path.exists():
        return None

    spec = importlib.util.spec_from_file_location(f"cg_solutions_{task_id}", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    # Some golfed solutions trigger SyntaxWarning on import under certain
    # Python versions. We don't want these warnings to spam the UI.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        spec.loader.exec_module(module)

    solve_fn = getattr(module, "solve", None)
    if callable(solve_fn):
        return solve_fn  # type: ignore[return-value]

    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        attr = getattr(module, attr_name)
        if callable(attr):
            return attr  # type: ignore[return-value]

    return None


def _load_golf_verifier_from_google_code_golf_2025(
    task_id: str,
) -> Optional[Callable[[Grid], Grid]]:
    """Best-effort loader from `external/google-code-golf-2025/submission`."""
    root = ROOT_DIR / "external" / "google-code-golf-2025" / "submission"
    if not root.exists():
        return None

    lookup = _arc_gen_id_to_task_num_and_generator(task_id)
    if lookup is None:
        return None
    task_num, _generator = lookup
    path = root / f"task{task_num:03d}.py"
    if not path.exists():
        return None

    spec = importlib.util.spec_from_file_location(f"google_code_golf_{task_id}", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        spec.loader.exec_module(module)

    solve_fn = getattr(module, "solve", None)
    if callable(solve_fn):
        return solve_fn  # type: ignore[return-value]

    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        attr = getattr(module, attr_name)
        if callable(attr):
            return attr  # type: ignore[return-value]

    return None


def _load_golf_verifier_from_keymoon(task_id: str) -> Optional[Callable[[Grid], Grid]]:
    """Best-effort loader from `external/golf` (key-moon/golf)."""
    root = ROOT_DIR / "external" / "golf"
    if not root.exists():
        return None

    lookup = _arc_gen_id_to_task_num_and_generator(task_id)
    if lookup is None:
        return None
    task_num, _generator = lookup

    # key-moon/golf stores solutions under `sols/taskNNN.py`.
    sols_dir = root / "sols"
    if not sols_dir.exists():
        return None
    path = sols_dir / f"task{task_num:03d}.py"
    if not path.exists():
        return None

    spec = importlib.util.spec_from_file_location(f"golf_keymoon_{task_id}", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        spec.loader.exec_module(module)

    solve_fn = getattr(module, "solve", None)
    if callable(solve_fn):
        return solve_fn  # type: ignore[return-value]

    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        attr = getattr(module, attr_name)
        if callable(attr):
            return attr  # type: ignore[return-value]

    return None


def _load_alternative_verifiers(
    task_id: str,
) -> tuple[
    Optional[Callable[[Grid], Grid]],
    Optional[Callable[[Grid], Grid]],
    Optional[Callable[[Grid], Grid]],
    Optional[Callable[[Grid], Grid]],
]:
    """Load up to 4 alternative verifiers from external/custom sources.

    Order:
    - google-code-golf-2025 (`external/google-code-golf-2025/submission`)
    - key-moon/golf (`external/golf/sols`)
    - NeurIPS-Code-Golf-2025 (`external/NeurIPS-Code-Golf-2025/solutions`)
    - custom local verifiers (`framework/custom_verifiers`)
    """
    v2 = _load_golf_verifier_from_google_code_golf_2025(task_id)
    v3 = _load_golf_verifier_from_keymoon(task_id)
    v4 = _load_golf_verifier_from_neurips(task_id)
    v5 = get_custom_verifier(task_id)
    return v2, v3, v4, v5


def load_task(task_id: str) -> ArcTask:
    """Load a single ARC task by ID, with attached synthetic data and verifiers."""
    # Canonical ARC examples (including ground-truth test outputs).
    train_pairs, test_inputs, test_outputs = _load_arc_original(task_id)

    # Optional synthetic data.
    re_arc_pairs = _load_re_arc_synthetic_pairs(task_id)
    arc_gen_pairs = _load_arc_gen_stable_pairs(task_id)

    # Optional generators.
    re_arc_gen = _make_re_arc_generator(task_id)
    arc_gen_gen = _make_arc_gen_generator(task_id)

    # Primary and alternative verifiers.
    verifier = get_re_arc_verifier(task_id)
    secondary_verifier, tertiary_verifier, quaternary_verifier, quinary_verifier = (
        _load_alternative_verifiers(task_id)
    )

    return ArcTask(
        task_id=task_id,
        train_pairs=train_pairs,
        test_inputs=test_inputs,
        test_outputs=test_outputs,
        verifier=verifier,
        secondary_verifier=secondary_verifier,
        tertiary_verifier=tertiary_verifier,
        quaternary_verifier=quaternary_verifier,
        quinary_verifier=quinary_verifier,
        re_arc_synthetic_pairs=re_arc_pairs,
        arc_gen_synthetic_pairs=arc_gen_pairs,
        re_arc_generator=re_arc_gen,
        arc_gen_generator=arc_gen_gen,
    )


def iter_tasks(
    split: Optional[str] = None,
    *,
    source: Optional[TaskSource] = None,
) -> Iterator[ArcTask]:
    """Iterate over tasks from the original ARC JSONs."""
    if source not in (None, TaskSource.ORIGINAL_ARC):
        raise ValueError(
            f"iter_tasks currently only supports source={TaskSource.ORIGINAL_ARC!r}"
        )

    # For now we only expose the training split from the original ARC set.
    if split not in (None, "train"):
        raise ValueError("Only split='train' (or None) is supported at the moment.")

    if not ARC_ORIGINAL_DIR.exists():
        raise FileNotFoundError(
            f"Missing arc_original_train directory at {ARC_ORIGINAL_DIR}"
        )

    for path in sorted(ARC_ORIGINAL_DIR.glob("*.json")):
        task_id = path.stem
        yield load_task(task_id)


def to_grid_pairs(
    inputs: List[Grid],
    outputs: List[Grid],
) -> List[GridPair]:
    """Utility to create `GridPair` objects from separate input/output lists."""
    if len(inputs) != len(outputs):
        raise ValueError("inputs and outputs must have the same length")
    return [GridPair(inp, out) for inp, out in zip(inputs, outputs)]

