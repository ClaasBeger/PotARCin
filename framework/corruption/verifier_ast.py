"""AST-based corruption of re_arc-style verifier functions.

Primary strategy (**drop_assign_rewire**): delete *one* intermediate assignment
``x_k = expr`` from a straight-line verifier and rewrite all *loads* of ``x_k``
in later statements to use ``x_{k-1}`` instead. That removes one computation
step from the pipeline while keeping the rest of the graph wired (a real
"pruned strain", not identity input).
"""

from __future__ import annotations

import ast
import copy
import random
import textwrap
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from framework.grids import Grid, is_equal_grid, normalized_cell_edit_between_outputs


ROOT_DIR = Path(__file__).resolve().parents[2]
RE_ARC_VERIFIERS_PATH = ROOT_DIR / "external" / "re_arc" / "verifiers.py"


def _load_verifier_function_ast(task_id: str) -> ast.FunctionDef:
    if not RE_ARC_VERIFIERS_PATH.exists():
        raise FileNotFoundError(f"Missing {RE_ARC_VERIFIERS_PATH}")
    src = RE_ARC_VERIFIERS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    name = f"verify_{task_id}"
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise KeyError(f"No {name} in {RE_ARC_VERIFIERS_PATH}")


class _RewireLoad(ast.NodeTransformer):
    """Replace loads of ``old_id`` with ``new_id`` (leave stores unchanged)."""

    def __init__(self, old_id: str, new_id: str) -> None:
        self._old = old_id
        self._new = new_id

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id == self._old:
            return ast.copy_location(ast.Name(id=self._new, ctx=ast.Load()), node)
        return node


def corrupt_verifier_drop_assign_rewire(
    fn: ast.FunctionDef,
    *,
    drop_index: int,
) -> ast.FunctionDef:
    """Remove assignment at *drop_index* in the assign-only prelude; rewire loads.

    *drop_index* is 0-based in the list of assignment statements (not including
    ``return``). Must satisfy ``1 <= drop_index < len(assigns) - 1`` so we drop
    a middle step (not the first binding, not the last before return).
    """
    if not fn.body or not isinstance(fn.body[-1], ast.Return):
        raise ValueError(f"Function {fn.name}: expected trailing return.")
    prelude = fn.body[:-1]
    assigns = [s for s in prelude if isinstance(s, ast.Assign)]
    if len(assigns) != len(prelude):
        raise ValueError(
            f"Function {fn.name}: expected only Assign statements before return."
        )
    n = len(assigns)
    if n < 3:
        raise ValueError(f"Function {fn.name}: need at least 3 assignments to drop one middle.")
    if not (1 <= drop_index < n - 1):
        raise ValueError(f"drop_index must be in [1, {n - 2}], got {drop_index}")

    rem = assigns[drop_index]
    prev = assigns[drop_index - 1]
    if len(rem.targets) != 1 or len(prev.targets) != 1:
        raise ValueError("Expected simple name targets.")
    if not isinstance(rem.targets[0], ast.Name) or not isinstance(prev.targets[0], ast.Name):
        raise ValueError("Expected Name targets.")
    removed_name = rem.targets[0].id
    prev_name = prev.targets[0].id

    rewriter = _RewireLoad(removed_name, prev_name)
    new_pre: List[ast.stmt] = []
    for i, stmt in enumerate(assigns):
        if i == drop_index:
            continue
        if i > drop_index:
            stmt = rewriter.visit(copy.deepcopy(stmt))
            ast.fix_missing_locations(stmt)
        new_pre.append(stmt)

    new_body = new_pre + [fn.body[-1]]
    return ast.FunctionDef(
        name=fn.name,
        args=fn.args,
        body=new_body,
        decorator_list=list(fn.decorator_list),
        returns=fn.returns,
        type_comment=getattr(fn, "type_comment", None),
        lineno=fn.lineno,
        col_offset=fn.col_offset,
    )


def corrupt_verifier_truncate_last_assign(
    fn: ast.FunctionDef,
    *,
    rng: random.Random,
    num_drops: int = 1,
) -> ast.FunctionDef:
    """Drop the last *num_drops* assignments before ``return``; return the new last value.

    Requires a linear body of ``Assign`` statements followed by a single
    ``return <Name>`` referencing the final assigned name.
    """
    del rng  # reserved for future randomized drop counts / targets
    body = fn.body
    if not body or not isinstance(body[-1], ast.Return):
        raise ValueError(f"Function {fn.name}: expected trailing return.")

    assigns = [s for s in body[:-1] if isinstance(s, ast.Assign)]
    if len(assigns) < num_drops + 1:
        raise ValueError(
            f"Function {fn.name}: need at least {num_drops + 1} assignments to truncate."
        )

    ret = body[-1]
    assert isinstance(ret, ast.Return) and isinstance(ret.value, ast.Name)
    final_name = ret.value.id
    last_t = assigns[-1].targets[0]
    if not isinstance(last_t, ast.Name) or last_t.id != final_name:
        raise ValueError(
            f"Function {fn.name}: last assign target should match return name."
        )

    kept_assigns = assigns[: -num_drops]
    prev_t = kept_assigns[-1].targets[0]
    if not isinstance(prev_t, ast.Name):
        raise ValueError(f"Function {fn.name}: expected simple name targets.")
    new_ret_name = prev_t.id
    new_body: List[ast.stmt] = list(kept_assigns) + [
        ast.Return(value=ast.Name(id=new_ret_name, ctx=ast.Load()))
    ]

    new_fn = ast.FunctionDef(
        name=fn.name,
        args=fn.args,
        body=new_body,
        decorator_list=list(fn.decorator_list),
        returns=fn.returns,
        type_comment=getattr(fn, "type_comment", None),
        lineno=fn.lineno,
        col_offset=fn.col_offset,
    )
    return new_fn


def compile_re_arc_verifier_function(
    fn: ast.FunctionDef,
    *,
    task_id: str,
) -> Callable[[Grid], Grid]:
    """Compile a single ``verify_*`` function with ``from dsl import *`` in scope."""
    import sys

    re_arc_dir = str(RE_ARC_VERIFIERS_PATH.parent)
    old_path = list(sys.path)
    try:
        if re_arc_dir not in sys.path:
            sys.path.insert(0, re_arc_dir)
        g: dict = {"__builtins__": __builtins__}
        exec("from dsl import *", g)
        code = ast.unparse(fn)
        exec(code, g)
        raw = g.get(f"verify_{task_id}")
        if not callable(raw):
            raise RuntimeError(f"Expected callable verify_{task_id} after exec")
    finally:
        sys.path = old_path

    def wrapped(grid: Grid) -> Grid:
        tuple_grid = tuple(tuple(row) for row in grid)
        out = raw(tuple_grid)  # type: ignore[misc]
        return [list(row) for row in out]

    return wrapped


def uncorrupted_re_arc_verifier(task_id: str) -> Callable[[Grid], Grid]:
    """Compile the original ``verify_*`` from ``verifiers.py`` (no corruption)."""
    return compile_re_arc_verifier_function(
        _load_verifier_function_ast(task_id), task_id=task_id
    )


def load_and_corrupt_re_arc_verifier(
    task_id: str,
    *,
    rng: random.Random,
    sample_input: Optional[Grid] = None,
    max_attempts: int = 40,
    max_normalized_cell_edit_distance: Optional[float] = 0.70,
) -> Tuple[Callable[[Grid], Grid], str, int]:
    """Load ``verify_{task_id}``, apply drop-and-rewire corruption, return (fn, src, drop_index).

    Tries several random ``drop_index`` values (middle assignments only). If
    *sample_input* is given, prefers a corruption that runs without error and
    produces an output different from the uncorrupted verifier on that input.

    If *max_normalized_cell_edit_distance* is set (default ``0.70``), also require
    the corrupted output to be close to gold in the sense that the fraction of
    differing cells (same shape) is at most that value. Set to ``None`` to skip.
    """
    fn = _load_verifier_function_ast(task_id)
    prelude = fn.body[:-1]
    assigns = [s for s in prelude if isinstance(s, ast.Assign)]
    n = len(assigns)
    if n < 3:
        raise ValueError(f"verify_{task_id}: too few assignments for middle drop.")

    indices = list(range(1, n - 1))
    rng.shuffle(indices)

    def _ok_corrupt(cfn: Callable[[Grid], Grid], _di: int) -> bool:
        if sample_input is None:
            return True
        try:
            gold = uncorrupted_re_arc_verifier(task_id)(copy.deepcopy(sample_input))
            bad = cfn(copy.deepcopy(sample_input))
            if is_equal_grid(gold, bad):
                return False
            if max_normalized_cell_edit_distance is not None:
                if normalized_cell_edit_between_outputs(gold, bad) > max_normalized_cell_edit_distance:
                    return False
            return True
        except Exception:
            return False

    last_err: Optional[Exception] = None
    for di in indices[:max_attempts]:
        try:
            corrupted = corrupt_verifier_drop_assign_rewire(fn, drop_index=di)
            cfn = compile_re_arc_verifier_function(corrupted, task_id=task_id)
            if _ok_corrupt(cfn, di):
                return cfn, textwrap.dedent(ast.unparse(corrupted)), di
        except Exception as e:
            last_err = e
            continue

    if sample_input is not None:
        raise RuntimeError(
            f"Could not corrupt verify_{task_id} to a distinct output on the sample input. "
            f"Last error: {last_err!r}"
        )

    for di in indices:
        try:
            corrupted = corrupt_verifier_drop_assign_rewire(fn, drop_index=di)
            cfn = compile_re_arc_verifier_function(corrupted, task_id=task_id)
            return cfn, textwrap.dedent(ast.unparse(corrupted)), di
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        f"Could not build a working corrupted verifier for {task_id}. Last error: {last_err!r}"
    )
