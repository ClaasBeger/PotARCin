"""AST corruption for Test2 tasks (``verifier.py`` and, when present, ``generator.transform_tN``).

Many tasks delegate ``verify_tN`` to ``transform_tN`` in ``generator.py``; corruption targets
the implementation that contains the real logic (generator first when ``transform_tN`` exists).

Strategies (same spirit as ARC-AGI-1 / re_arc):

1. ``drop_assign_rewire`` — drop a middle assignment in a straight-line assign chain.
2. ``truncate_last_assign`` — drop trailing assignments before ``return``.
3. ``binop_prune`` — replace one :class:`ast.BinOp` with its left subtree (golf-style).
"""

from __future__ import annotations

import ast
import copy
import random
import signal
import sys
import textwrap
import types
import warnings
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple

from framework.corruption.verifier_ast import (
    corrupt_verifier_drop_assign_rewire,
    corrupt_verifier_truncate_last_assign,
)
from framework.grids import Grid, is_equal_grid, normalized_cell_edit_between_outputs

SourceKind = Literal["generator", "verifier"]


def _find_function(mod: ast.Module, name: str) -> Optional[ast.FunctionDef]:
    for stmt in mod.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == name:
            return stmt
    return None


def _replace_function(mod: ast.Module, name: str, new_fn: ast.FunctionDef) -> ast.Module:
    body: list[ast.stmt] = []
    for stmt in mod.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == name:
            body.append(new_fn)
        else:
            body.append(stmt)
    return ast.Module(body=body, type_ignores=[])


def _prune_kth_binop_in_function(fn: ast.FunctionDef, k: int) -> ast.FunctionDef:
    fn2 = copy.deepcopy(fn)
    binops = [n for n in ast.walk(fn2) if isinstance(n, ast.BinOp)]
    if k < 0 or k >= len(binops):
        raise IndexError(f"binop {k} / {len(binops)}")

    class T(ast.NodeTransformer):
        def __init__(self, target: ast.BinOp) -> None:
            self._t = target

        def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
            if node is self._t:
                return node.left
            return self.generic_visit(node)

    tgt = binops[k]
    Tpr = T(tgt)
    new_body: list[ast.stmt] = []
    for st in fn2.body:
        st2 = Tpr.visit(st)
        ast.fix_missing_locations(st2)
        new_body.append(st2)
    return ast.FunctionDef(
        name=fn2.name,
        args=fn2.args,
        body=new_body,
        decorator_list=fn2.decorator_list,
        returns=fn2.returns,
        type_comment=getattr(fn2, "type_comment", None),
        lineno=fn2.lineno,
        col_offset=fn2.col_offset,
    )


def _pick_corruption_target(
    task_num: int, g_mod: ast.Module, v_mod: ast.Module
) -> Tuple[SourceKind, str]:
    tname = f"transform_t{task_num}"
    if _find_function(g_mod, tname) is not None:
        return "generator", tname
    for cand in (f"verify_t{task_num}", f"transform_t{task_num}", "verify", "transform"):
        if _find_function(v_mod, cand) is not None:
            return "verifier", cand
    raise ValueError(f"Test2 t{task_num}: no verify/transform entry point found in AST.")


def _iter_corruption_strategies(
    fn: ast.FunctionDef, rng: random.Random
) -> List[Tuple[str, int]]:
    del rng  # order is deterministic (cheap strategies first)
    out: list[Tuple[str, int]] = []
    fn_probe = copy.deepcopy(fn)
    compares = [n for n in ast.walk(fn_probe) if isinstance(n, ast.Compare)]
    # Compare flips first (often yield valid-but-wrong grids cheaply).
    for ci in range(min(len(compares), 12)):
        out.append(("cmp_flip", ci))
    binops = [n for n in ast.walk(fn) if isinstance(n, ast.BinOp)]
    for bi in range(len(binops)):
        out.append(("binop", bi))
    if fn.body and isinstance(fn.body[-1], ast.Return):
        prelude = fn.body[:-1]
        assigns = [s for s in prelude if isinstance(s, ast.Assign)]
        if len(assigns) == len(prelude) and len(assigns) >= 3:
            n = len(assigns)
            for di in range(1, n - 1):
                out.append(("drop_rewire", di))
    if fn.body and isinstance(fn.body[-1], ast.Return):
        assigns = [s for s in fn.body[:-1] if isinstance(s, ast.Assign)]
        if len(assigns) >= 2:
            out.append(("truncate", 1))
        if len(assigns) >= 3:
            out.append(("truncate", 2))
    return out


def _corrupt_flip_kth_compare(fn: ast.FunctionDef, k: int) -> ast.FunctionDef:
    """Flip the operator on the *k*-th :class:`ast.Compare` (e.g. ``!=`` ↔ ``==``)."""
    fn2 = copy.deepcopy(fn)
    compares = [n for n in ast.walk(fn2) if isinstance(n, ast.Compare)]
    if k < 0 or k >= len(compares):
        raise IndexError
    tgt = compares[k]
    if len(tgt.ops) != 1:
        raise ValueError("multi-op Compare not supported")
    op = tgt.ops[0]
    if isinstance(op, ast.Eq):
        flip = ast.NotEq()
    elif isinstance(op, ast.NotEq):
        flip = ast.Eq()
    elif isinstance(op, ast.Lt):
        flip = ast.GtE()
    elif isinstance(op, ast.LtE):
        flip = ast.Gt()
    elif isinstance(op, ast.Gt):
        flip = ast.LtE()
    elif isinstance(op, ast.GtE):
        flip = ast.Lt()
    else:
        raise ValueError("unsupported compare op")
    tgt.ops = [flip]
    ast.fix_missing_locations(tgt)
    return fn2


def _stub_wrong_flat_function(fn: ast.FunctionDef) -> ast.FunctionDef:
    """Replace body with ``return [[0]*width ...]`` using the first parameter name."""
    if not fn.args.args:
        raise ValueError("stub_wrong needs at least one parameter")
    arg = fn.args.args[0].arg
    stub_src = (
        f"def {fn.name}({arg}):\n"
        f"    return [[0] * len({arg}[0]) for _ in {arg}]\n"
    )
    mod = ast.parse(stub_src)
    got = mod.body[0]
    assert isinstance(got, ast.FunctionDef)
    return got


def _apply_strategy(
    fn: ast.FunctionDef, strat: str, param: int, rng: random.Random
) -> ast.FunctionDef:
    if strat == "drop_rewire":
        return corrupt_verifier_drop_assign_rewire(fn, drop_index=param)
    if strat == "truncate":
        return corrupt_verifier_truncate_last_assign(fn, rng=rng, num_drops=param)
    if strat == "binop":
        return _prune_kth_binop_in_function(fn, param)
    if strat == "cmp_flip":
        return _corrupt_flip_kth_compare(fn, param)
    if strat == "stub_wrong":
        del param
        return _stub_wrong_flat_function(fn)
    raise ValueError(strat)


def _exec_test2_modules(task_num: int, *, gen_src: str, ver_src: str) -> Callable[[Grid], Grid]:
    """Full exec of generator + verifier (used for sanity / external callers)."""
    gmod = types.ModuleType("generator")
    if gen_src.strip():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            exec(compile(gen_src, "<test2_generator>", "exec"), gmod.__dict__)
    sys.modules["generator"] = gmod
    ns: dict = {"__builtins__": __builtins__}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        exec(compile(ver_src, "<test2_verifier>", "exec"), ns)
    for name in (f"verify_t{task_num}", f"transform_t{task_num}", "verify", "transform"):
        fn = ns.get(name)
        if callable(fn):
            return fn  # type: ignore[return-value]
    raise RuntimeError(f"Test2 t{task_num}: no verify_* / transform entry in exec namespace.")


def _verify_callable_from_exec_ns(task_num: int, ns: dict) -> Callable[[Grid], Grid]:
    for name in (f"verify_t{task_num}", f"transform_t{task_num}", "verify", "transform"):
        fn = ns.get(name)
        if callable(fn):
            return fn  # type: ignore[return-value]
    raise RuntimeError(f"Test2 t{task_num}: no verify_* in namespace.")


def load_and_corrupt_test2_verifier(
    task_id: str,
    task_num: int,
    task_dir: Path,
    *,
    rng: random.Random,
    gold_fn: Callable[[Grid], Grid],
    sample_input: Optional[Grid] = None,
    max_attempts: int = 160,
    max_normalized_cell_edit_distance: Optional[float] = 0.70,
) -> Tuple[Callable[[Grid], Grid], str, str]:
    """Build a corrupted verifier for Test2; return ``(callable, unparse_or_note, strategy_tag)``.

    Compares against *gold_fn* (the task's ground-truth verifier) on *sample_input* when set.
    """
    del task_id  # reserved for diagnostics
    g_path = task_dir / "generator.py"
    v_path = task_dir / "verifier.py"
    gen_src = g_path.read_text(encoding="utf-8") if g_path.is_file() else ""
    ver_src = v_path.read_text(encoding="utf-8")
    if not v_path.is_file():
        raise FileNotFoundError(task_dir / "verifier.py")

    g_mod = ast.parse(gen_src) if gen_src.strip() else ast.Module(body=[], type_ignores=[])
    v_mod = ast.parse(ver_src)
    where, fname = _pick_corruption_target(task_num, g_mod, v_mod)
    parent = g_mod if where == "generator" else v_mod
    fn_orig = _find_function(parent, fname)
    if fn_orig is None:
        raise RuntimeError(f"Missing function {fname}")

    strategies = _iter_corruption_strategies(fn_orig, rng)
    last_err: Optional[Exception] = None

    class _AlarmWall(Exception):
        pass

    def _ok(cfn: Callable[[Grid], Grid], tag: str) -> bool:
        del tag
        if sample_input is None:
            return True
        try:
            gin = copy.deepcopy(sample_input)
            gold = gold_fn(copy.deepcopy(sample_input))

            def _eval_corrupt_out() -> Grid:
                if sys.platform.startswith("win"):
                    return cfn(gin)
                def _handler(_sig: int, _frame: object) -> None:  # pragma: no cover
                    raise _AlarmWall()
                old = signal.signal(signal.SIGALRM, _handler)
                try:
                    signal.setitimer(signal.ITIMER_REAL, 1.2)
                    return cfn(gin)
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0.0)
                    signal.signal(signal.SIGALRM, old)

            bad = _eval_corrupt_out()
            if is_equal_grid(gold, bad):
                return False
            if max_normalized_cell_edit_distance is not None:
                if normalized_cell_edit_between_outputs(gold, bad) > max_normalized_cell_edit_distance:
                    return False
            return True
        except _AlarmWall:
            return False
        except Exception:
            return False

    # Live modules: patch one function, re-import verifier so ``from generator import X`` rebinds.
    gen_live = types.ModuleType("generator")
    if gen_src.strip():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            exec(compile(gen_src, "<test2_generator>", "exec"), gen_live.__dict__)
    sys.modules["generator"] = gen_live
    orig_impl = getattr(gen_live, fname, None)

    for strat, param in strategies[:max_attempts]:
        try:
            new_fn_ast = _apply_strategy(copy.deepcopy(fn_orig), strat, param, rng)
            fn_py = ast.unparse(new_fn_ast)
            tag = f"{where}:{strat}:{param}"
            if where == "generator":
                assert orig_impl is not None
                exec(compile(fn_py, "<corrupt_fn>", "exec"), gen_live.__dict__)
                sys.modules["generator"] = gen_live
                ver_ns: dict = {"__builtins__": __builtins__}
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    exec(compile(ver_src, "<test2_verifier>", "exec"), ver_ns)
                cfn = _verify_callable_from_exec_ns(task_num, ver_ns)
                if _ok(cfn, tag):
                    setattr(gen_live, fname, orig_impl)
                    snippet = textwrap.dedent(fn_py)[:8000]
                    return cfn, snippet, tag
                setattr(gen_live, fname, orig_impl)
            else:
                ver_ns2: dict = {"__builtins__": __builtins__}
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    exec(compile(ver_src, "<test2_verifier>", "exec"), ver_ns2)
                orig_v = ver_ns2.get(f"verify_t{task_num}") or ver_ns2.get("verify")
                exec(compile(fn_py, "<corrupt_verify>", "exec"), ver_ns2)
                cfn = _verify_callable_from_exec_ns(task_num, ver_ns2)
                if _ok(cfn, tag):
                    snippet = textwrap.dedent(fn_py)[:8000]
                    return cfn, snippet, tag
                if orig_v is not None:
                    ver_ns2[f"verify_t{task_num}"] = orig_v
        except Exception as e:
            last_err = e
            if where == "generator" and orig_impl is not None:
                setattr(gen_live, fname, orig_impl)
            continue

    try:
        new_fn_ast = _apply_strategy(copy.deepcopy(fn_orig), "stub_wrong", 0, rng)
        fn_py = ast.unparse(new_fn_ast)
        tag_fb = f"{where}:stub_wrong:0"
        if where == "generator":
            assert orig_impl is not None
            exec(compile(fn_py, "<stub_fn>", "exec"), gen_live.__dict__)
            sys.modules["generator"] = gen_live
            ver_fb: dict = {"__builtins__": __builtins__}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                exec(compile(ver_src, "<test2_verifier>", "exec"), ver_fb)
            cfn_fb = _verify_callable_from_exec_ns(task_num, ver_fb)
            if _ok(cfn_fb, tag_fb):
                setattr(gen_live, fname, orig_impl)
                return cfn_fb, textwrap.dedent(fn_py)[:8000], tag_fb
            setattr(gen_live, fname, orig_impl)
        else:
            ver_fb2: dict = {"__builtins__": __builtins__}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                exec(compile(ver_src, "<test2_verifier>", "exec"), ver_fb2)
            exec(compile(fn_py, "<stub_verify>", "exec"), ver_fb2)
            cfn_fb2 = _verify_callable_from_exec_ns(task_num, ver_fb2)
            if _ok(cfn_fb2, tag_fb):
                return cfn_fb2, textwrap.dedent(fn_py)[:8000], tag_fb
    except Exception as e:
        last_err = e

    raise RuntimeError(
        f"Could not build corrupted Test2 verifier for t{task_num}. Last error: {last_err!r}"
    )
