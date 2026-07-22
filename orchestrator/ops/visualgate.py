"""Optional visual/runtime gate — assert over a running app's scene-graph state.

A green deterministic gate means "it compiles/tests pass", not "it renders
correctly": a three.js scene can build clean and show a grey screen. This gate
starts the app in the candidate worktree, drives a pluggable MCP inspector to
fetch structured scene-graph facts, and evaluates project-defined assertions
over them. Zero LLM tokens.

Design points:
  * Project-agnostic: the inspector is a `mcp:` entry (threejs now, unity/blender
    later). This op is its OWN MCP CLIENT (the existing plumbing only hands config
    paths to the Claude CLI); the client is injected/best-effort.
  * Safe assertions: a restricted AST evaluator over the returned JSON — NEVER
    eval(). Only comparisons, boolean/arithmetic ops, attribute/subscript access,
    comprehensions, and a tiny allowlist of pure builtins.
  * Process lifecycle: the dev/app process is spawned in its own process group
    and the whole group is killed on teardown (no `npm run dev &` orphans).
"""

from __future__ import annotations

import ast
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("orchestrator.visualgate")

# ---- safe assertion evaluator ----------------------------------------------

_ALLOWED_FUNCS = {
    "len": len, "any": any, "all": all, "min": min, "max": max,
    "abs": abs, "int": int, "float": float, "bool": bool, "sum": sum,
}
_ALLOWED_NODES = (
    ast.Expression, ast.BoolOp, ast.UnaryOp, ast.BinOp, ast.Compare,
    ast.Name, ast.Load, ast.Store, ast.Constant, ast.Attribute, ast.Subscript,
    ast.Index, ast.Call, ast.List, ast.Tuple, ast.ListComp, ast.GeneratorExp,
    ast.comprehension, ast.And, ast.Or, ast.Not, ast.USub,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Slice,
)


class AssertionError_(Exception):
    """A malformed / unsafe assertion expression."""


class _Dot:
    """Attribute + subscript access over JSON, so `scene.visibleMeshCount` and
    `scene['visibleMeshCount']` both work. Refuses dunder access."""

    def __init__(self, data):
        object.__setattr__(self, "_d", data)

    def __getattr__(self, key):
        if key.startswith("__"):
            raise AssertionError_(f"refused attribute: {key}")
        return _wrap(self._d[key]) if isinstance(self._d, dict) and key in self._d \
            else _missing(key)

    def __getitem__(self, key):
        return _wrap(self._d[key])

    def __iter__(self):
        return (_wrap(v) for v in self._d)

    def __len__(self):
        return len(self._d)

    def __eq__(self, other):
        return self._d == (other._d if isinstance(other, _Dot) else other)

    def __bool__(self):
        return bool(self._d)


def _missing(key):
    raise AssertionError_(f"unknown field: {key}")


def _wrap(v):
    return _Dot(v) if isinstance(v, (dict, list)) else v


def _unwrap(v):
    return v._d if isinstance(v, _Dot) else v


def _eval(node, env):
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise AssertionError_(f"unknown name: {node.id}")
    if isinstance(node, ast.Attribute):
        if node.attr.startswith("__"):
            raise AssertionError_("refused dunder access")
        return getattr(_as_dot(_eval(node.value, env)), node.attr)
    if isinstance(node, ast.Subscript):
        target = _eval(node.value, env)
        key = _eval(node.slice, env) if not isinstance(node.slice, ast.Slice) else None
        return _as_dot(target)[_unwrap(key)]
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, env)
        return not _truthy(v) if isinstance(node.op, ast.Not) else -_unwrap(v)
    if isinstance(node, ast.BoolOp):
        vals = [_eval(v, env) for v in node.values]
        if isinstance(node.op, ast.And):
            return all(_truthy(v) for v in vals)
        return any(_truthy(v) for v in vals)
    if isinstance(node, ast.BinOp):
        a, b = _unwrap(_eval(node.left, env)), _unwrap(_eval(node.right, env))
        return _BINOPS[type(node.op)](a, b)
    if isinstance(node, ast.Compare):
        left = _unwrap(_eval(node.left, env))
        for op, comp in zip(node.ops, node.comparators):
            right = _unwrap(_eval(comp, env))
            if not _CMPOPS[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise AssertionError_("only allowlisted functions may be called")
        args = [_unwrap(_eval(a, env)) for a in node.args]
        return _ALLOWED_FUNCS[node.func.id](*args)
    if isinstance(node, (ast.ListComp, ast.GeneratorExp)):
        return list(_eval_comp(node, env))
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_unwrap(_eval(e, env)) for e in node.elts]
    raise AssertionError_(f"unsupported expression: {type(node).__name__}")


def _eval_comp(node, env):
    (gen,) = node.generators
    if gen.ifs or gen.is_async:
        # keep it simple + safe: support `x for x in seq` (no filters)
        raise AssertionError_("comprehension filters not supported")
    iterable = _eval(gen.iter, env)
    for item in iterable:
        local = dict(env)
        local[gen.target.id] = item
        yield _unwrap(_eval(node.elt, local))


def _as_dot(v):
    return v if isinstance(v, _Dot) else _Dot(v)


def _truthy(v):
    return bool(_unwrap(v))


import operator as _op  # noqa: E402
_BINOPS = {ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul,
           ast.Div: _op.truediv, ast.Mod: _op.mod}
_CMPOPS = {ast.Eq: _op.eq, ast.NotEq: _op.ne, ast.Lt: _op.lt, ast.LtE: _op.le,
           ast.Gt: _op.gt, ast.GtE: _op.ge,
           ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}


def evaluate_assertion(expr: str, facts: dict) -> bool:
    """Evaluate one assertion against the scene-graph facts. Raises
    AssertionError_ on any unsafe/unsupported construct."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise AssertionError_(f"bad assertion syntax: {e}") from e
    for sub in ast.walk(tree):
        if not isinstance(sub, _ALLOWED_NODES):
            raise AssertionError_(f"disallowed syntax: {type(sub).__name__}")
    env = {k: _wrap(v) for k, v in facts.items()}
    env.update(_ALLOWED_FUNCS)
    return _truthy(_eval(tree, env))


def run_assertions(assertions: list[str], facts: dict) -> list[str]:
    """Return the list of FAILED assertion strings (empty => all pass). A
    malformed assertion counts as a failure with its error text."""
    failures = []
    for a in assertions:
        try:
            if not evaluate_assertion(a, facts):
                failures.append(a)
        except AssertionError_ as e:
            failures.append(f"{a}  [invalid: {e}]")
    return failures


# ---- runtime supervision ----------------------------------------------------

@dataclass
class VisualResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    # True only when assertions were actually evaluated against real inspector
    # facts. A disabled gate or an enabled-but-no-fact-source pass-through returns
    # enforced=False, so callers can distinguish a genuine visual PASS from a blind
    # pass-through and surface the latter (see visual_gate node's skipped event).
    enforced: bool = False


def _spawn(run_cmd: str, cwd: Path) -> subprocess.Popen:
    """Foreground command supervised by us in its OWN process group (never a
    shell-backgrounded `&`), so teardown can kill the whole group."""
    return subprocess.Popen(run_cmd, shell=True, cwd=str(cwd),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)


def _teardown(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            proc.kill()


def check(cfg, worktree: Path, *, fetch_facts=None, wait_ready=None) -> VisualResult:
    """Run the visual gate for one candidate worktree.

    `fetch_facts()` returns the scene-graph facts dict from the MCP inspector
    (injected so this is testable and so the MCP client stays best-effort).
    `wait_ready()` blocks until the app is serving. When the gate is disabled or
    no fact source is available, returns a PASS (pass-through) — the harness must
    run fine with visual_gate.enabled:false.
    """
    vg = cfg.get("visual_gate") or {}
    get = vg.get if hasattr(vg, "get") else (lambda k, d=None: d)
    if not get("enabled", False):
        return VisualResult(passed=True)
    assertions = list(get("assertions", []) or [])
    run_cmd = get("run_cmd")

    proc = None
    try:
        if run_cmd and fetch_facts is not None:
            proc = _spawn(run_cmd, worktree)
            if wait_ready is not None:
                wait_ready()
        if fetch_facts is None:
            # No inspector wired (best-effort) — don't block the pipeline.
            log.warning("visual_gate enabled but no fact source; passing through")
            return VisualResult(passed=True)
        facts = fetch_facts()
        failures = run_assertions(assertions, facts)
        return VisualResult(passed=not failures, failures=failures, facts=facts,
                            enforced=True)
    finally:
        _teardown(proc)
