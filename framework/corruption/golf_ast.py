"""AST corruption for golf-style solution files (short lambdas / one-liners).

When there is no straight-line ``verify_*`` assign chain (re_arc), golf sources often
expose ``p = lambda g: <expr>`` or ``def solve(g): ...``. We prune **one** binary
operation by replacing a :class:`ast.BinOp` node with its **left** subtree (same idea
as removing one computational step).
"""

from __future__ import annotations

import ast
import copy
import random
import textwrap
import warnings
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple

from framework.grids import Grid, is_equal_grid, normalized_cell_edit_between_outputs
from framework.tasks.arc_dataset import _arc_gen_id_to_task_num_and_generator


ROOT_DIR = Path(__file__).resolve().parents[2]

GolfSource = Literal["google", "keymoon", "neurips"]


def golf_solution_path(task_id: str, source: GolfSource) -> Optional[Path]:
    """Return path to ``taskNNN.py`` for *source*, or None if missing."""
    lookup = _arc_gen_id_to_task_num_and_generator(task_id)
    if lookup is None:
        return None
    task_num, _ = lookup
    if source == "keymoon":
        p = ROOT_DIR / "external" / "golf" / "sols" / f"task{task_num:03d}.py"
    elif source == "google":
        p = (
            ROOT_DIR
            / "external"
            / "google-code-golf-2025"
            / "submission"
            / f"task{task_num:03d}.py"
        )
    else:
        p = (
            ROOT_DIR
            / "external"
            / "NeurIPS-Code-Golf-2025"
            / "solutions"
            / f"task{task_num:03d}.py"
        )
    return p if p.exists() else None


def _collect_binops(node: ast.AST) -> List[ast.BinOp]:
    return [n for n in ast.walk(node) if isinstance(n, ast.BinOp)]


class _PruneBinOpToLeft(ast.NodeTransformer):
    def __init__(self, target: ast.BinOp) -> None:
        self._target = target

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        if node is self._target:
            return node.left
        return self.generic_visit(node)


def _prune_kth_binop(expr: ast.expr, k: int) -> ast.expr:
    tree = copy.deepcopy(expr)
    binops = _collect_binops(tree)
    if not binops or k < 0 or k >= len(binops):
        raise IndexError(f"binop index {k} out of range (have {len(binops)})")
    target = binops[k]
    pruned = _PruneBinOpToLeft(target).visit(tree)
    assert isinstance(pruned, ast.expr)
    return pruned


def _mutate_assign_lambda(
    mod: ast.Module, mutator: Callable[[ast.Lambda], ast.Lambda]
) -> ast.Module:
    new_body: List[ast.stmt] = []
    for stmt in mod.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Lambda):
            new_body.append(
                ast.Assign(
                    targets=stmt.targets,
                    value=mutator(copy.deepcopy(stmt.value)),
                    lineno=stmt.lineno,
                    col_offset=stmt.col_offset,
                )
            )
        else:
            new_body.append(copy.deepcopy(stmt))
    return ast.Module(body=new_body, type_ignores=[])


def _mutate_function_return_expr(
    fn: ast.FunctionDef, mutator: Callable[[ast.expr], ast.expr]
) -> ast.FunctionDef:
    if len(fn.body) != 1 or not isinstance(fn.body[0], ast.Return):
        raise ValueError(f"Expected single-return body for {fn.name}")
    ret = fn.body[0]
    assert ret.value is not None
    new_ret = ast.Return(value=mutator(copy.deepcopy(ret.value)))
    return ast.FunctionDef(
        name=fn.name,
        args=fn.args,
        body=[new_ret],
        decorator_list=list(fn.decorator_list),
        returns=fn.returns,
        type_comment=getattr(fn, "type_comment", None),
        lineno=fn.lineno,
        col_offset=fn.col_offset,
    )


def corrupt_golf_module_ast(
    mod: ast.Module,
    *,
    binop_index: int,
) -> ast.Module:
    """Return a new module AST with one BinOp pruned in the primary callable."""
    for stmt in mod.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Lambda):
            lam = stmt.value

            def _m(old: ast.Lambda, bi=binop_index) -> ast.Lambda:
                new_body = _prune_kth_binop(old.body, bi)
                return ast.Lambda(
                    args=old.args,
                    body=new_body,
                    lineno=old.lineno,
                    col_offset=old.col_offset,
                )

            return _mutate_assign_lambda(mod, _m)

    for stmt in mod.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "solve":
            fn = copy.deepcopy(stmt)
            if len(fn.body) == 1 and isinstance(fn.body[0], ast.Return):
                new_fn = _mutate_function_return_expr(
                    fn, lambda expr: _prune_kth_binop(expr, binop_index)
                )
                return _replace_stmt_by_name(mod, "solve", new_fn)
            raise ValueError("Unsupported multi-statement solve() for golf corruption.")

    raise ValueError("No assign-lambda or single-return solve() found in golf module.")


def _replace_stmt_by_name(
    mod: ast.Module, name: str, new_fn: ast.FunctionDef
) -> ast.Module:
    body: List[ast.stmt] = []
    for stmt in mod.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == name:
            body.append(new_fn)
        else:
            body.append(copy.deepcopy(stmt))
    return ast.Module(body=body, type_ignores=[])


def compile_golf_module_ast(mod: ast.Module) -> Callable[[Grid], Grid]:
    """Compile a golf solution module; return a grid in/out callable."""
    ast.fix_missing_locations(mod)
    src = ast.unparse(mod)
    g: dict = {"__builtins__": __builtins__}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        exec(src, g)

    raw: Optional[Callable[..., object]] = g.get("solve")
    if not callable(raw):
        raw = None
        for name, attr in g.items():
            if name.startswith("_") or not callable(attr):
                continue
            raw = attr  # type: ignore[assignment]
            break
    if raw is None:
        raise RuntimeError("No callable found after compiling golf module.")

    def wrapped(grid: Grid) -> Grid:
        out = raw(grid)  # type: ignore[misc]
        if isinstance(out, tuple):
            return [list(row) for row in out]
        return [list(row) if isinstance(row, tuple) else row for row in out]  # type: ignore[misc]

    return wrapped


def uncorrupted_golf_verifier(task_id: str, source: GolfSource) -> Callable[[Grid], Grid]:
    path = golf_solution_path(task_id, source)
    if path is None:
        raise FileNotFoundError(f"No golf file for {task_id!r} source={source!r}")
    mod = ast.parse(path.read_text(encoding="utf-8"))
    return compile_golf_module_ast(mod)


def load_and_corrupt_golf_verifier(
    task_id: str,
    source: GolfSource,
    *,
    rng: random.Random,
    sample_input: Optional[Grid] = None,
    max_attempts: int = 80,
    max_normalized_cell_edit_distance: Optional[float] = 0.70,
) -> Tuple[Callable[[Grid], Grid], str, int]:
    """Build corrupted golf verifier; *binop_index* (or drop index) recorded as third value.

    If *max_normalized_cell_edit_distance* is set (default ``0.70``), require the
    corrupted output to differ from gold but with at most that fraction of cells
    changed (same-shape Hamming). Set to ``None`` to skip.
    """
    path = golf_solution_path(task_id, source)
    if path is None:
        raise FileNotFoundError(f"No golf file for {task_id!r} source={source!r}")
    mod = ast.parse(path.read_text(encoding="utf-8"))

    # Enumerate candidate binop indices from uncorrupted tree (upper bound).
    try:
        probe = copy.deepcopy(mod)
        # Same shape as corrupt_golf for lambda path — count binops on first assign lambda
        n_candidates = 0
        for stmt in probe.body:
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Lambda):
                n_candidates = max(n_candidates, len(_collect_binops(stmt.value.body)))
                break
        if n_candidates == 0:
            for stmt in probe.body:
                if isinstance(stmt, ast.FunctionDef) and stmt.name == "solve":
                    if len(stmt.body) == 1 and isinstance(stmt.body[0], ast.Return):
                        v = stmt.body[0].value
                        if v is not None:
                            n_candidates = max(n_candidates, len(_collect_binops(v)))
                    break
    except Exception:
        n_candidates = 32

    if n_candidates == 0:
        raise ValueError(f"golf {task_id}: no BinOp nodes to prune.")

    order = list(range(n_candidates))
    rng.shuffle(order)
    gold_fn = uncorrupted_golf_verifier(task_id, source)

    def _differs(cfn: Callable[[Grid], Grid], bi: int) -> bool:
        if sample_input is None:
            return True
        ggrid = gold_fn(copy.deepcopy(sample_input))
        bgrid = cfn(copy.deepcopy(sample_input))
        if is_equal_grid(ggrid, bgrid):
            return False
        if max_normalized_cell_edit_distance is not None:
            if normalized_cell_edit_between_outputs(ggrid, bgrid) > max_normalized_cell_edit_distance:
                return False
        return True

    last_err: Optional[Exception] = None
    for bi in order[:max_attempts]:
        try:
            corrupted = corrupt_golf_module_ast(copy.deepcopy(mod), binop_index=bi)
            cfn = compile_golf_module_ast(corrupted)
            if _differs(cfn, bi):
                return cfn, textwrap.dedent(ast.unparse(corrupted)), bi
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        f"Could not build golf corruption for {task_id} ({source}) with distinct output. "
        f"Last error: {last_err!r}"
    )
