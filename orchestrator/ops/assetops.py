"""Deterministic asset operations. Zero tokens, zero LLM discretion.

Some tasks need a real external binary-processing tool run against a file no LLM
role can touch: decimating a .glb with gltf-transform, transcoding a texture,
baking an atlas. Neither role can do it and neither should be able to — workers
are patch-only with no shell by design (that "tiny explicit context, no tools"
contract is what makes a cheap worker trustworthy), and the planner/reviewer tier
is Read/Grep/Glob.

So the command is not something an LLM produces. `asset_ops` in the project
profile is a map of NAMED, HUMAN-AUTHORED, FIXED command strings — exactly the
trust level of `gate.commands`. A spec may reference one BY NAME (`asset_op:
reduce_ae86_poly`); the name is the only thing an LLM ever writes, and an
unknown name fails loudly rather than silently doing nothing.

Design points, all mirroring ops/gate.py:
  * Pure `subprocess` in the candidate's own worktree — same isolation, same
    trust boundary as the gate. Nothing new is exposed to any LLM role.
  * Runs ONCE per attempt, after the worker's patch is applied and before the
    gate, so the gate sees the processed asset and the outputs land in the
    candidate's diff for review/merge.
  * A non-zero exit, a timeout, or an unknown op name FAILS the candidate,
    exactly like a red gate. It never passes through quietly.
  * Disabled by default: an empty `asset_ops` map plus a spec with no `asset_op`
    is a no-op that costs one dict lookup (same pattern as visual_gate with no
    facts_cmd).
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("orchestrator.assetops")

DEFAULT_TIMEOUT_S = 1200


class UnknownAssetOp(Exception):
    """A spec named an `asset_op` the profile does not define.

    Loud on purpose. Silently skipping would hand the reviewer a diff whose
    asset was never processed while every log line says the attempt was clean —
    and the planner would keep re-emitting the bad name because nothing ever
    contradicted it."""


@dataclass
class AssetOpResult:
    passed: bool
    # False => nothing was requested (no `asset_op` on the spec). Distinguishes
    # "no op configured" from "the op ran and passed", so the caller can log a
    # real execution and stay silent otherwise.
    ran: bool = False
    op: str = ""
    cmd: str = ""
    log_tail: str = ""


def _ops(cfg) -> dict:
    ops = cfg.get("asset_ops") or {}
    return ops.as_dict() if hasattr(ops, "as_dict") else dict(ops)


def available_ops(cfg) -> list[str]:
    """Sorted names of every configured op. Used by plan-time validation and by
    the planner payload — the planner cannot reference an op it never saw."""
    return sorted(_ops(cfg))


def resolve_op(cfg, name: str) -> tuple[str, int]:
    """(command, timeout_s) for `name`. Raises UnknownAssetOp.

    An entry is either a bare command string (the common case) or a mapping
    `{cmd, timeout_s}` — asset tooling is not on the gate's timescale in either
    direction, and pinning that per op beats one global knob.
    """
    ops = _ops(cfg)
    if name not in ops:
        raise UnknownAssetOp(
            f"unknown asset_op {name!r}; profile defines: "
            f"{', '.join(available_ops(cfg)) or '(none)'}")
    entry = ops[name]
    default_timeout = int(cfg.gate.get("timeout_s", DEFAULT_TIMEOUT_S)
                          or DEFAULT_TIMEOUT_S)
    if isinstance(entry, str):
        return entry, default_timeout
    entry = entry.as_dict() if hasattr(entry, "as_dict") else dict(entry)
    cmd = entry.get("cmd")
    if not cmd or not isinstance(cmd, str):
        raise UnknownAssetOp(
            f"asset_op {name!r} has no `cmd` string: {entry!r}")
    return cmd, int(entry.get("timeout_s") or default_timeout)


def run_asset_op(worktree: Path, cfg, spec: dict) -> AssetOpResult:
    """Run the spec's asset op in the candidate worktree, if it has one.

    Returns ran=False, passed=True when the spec requests nothing — the default
    for every project that never configures this.
    """
    name = (spec.get("asset_op") or "").strip()
    if not name:
        return AssetOpResult(passed=True, ran=False)

    tail = int(cfg.gate.get("log_tail_chars", 6000) or 6000)
    try:
        cmd, timeout = resolve_op(cfg, name)
    except UnknownAssetOp as e:
        # Not a crash: the candidate fails with the reason, same as a red gate,
        # so the retry/escalation ladder and the event log both see it.
        return AssetOpResult(passed=False, ran=True, op=name, log_tail=str(e))

    log.info("asset_op %s: %s (%s)", name, cmd, worktree.name)
    try:
        proc = subprocess.run(cmd, shell=True, cwd=str(worktree),
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return AssetOpResult(passed=False, ran=True, op=name, cmd=cmd,
                             log_tail=f"TIMEOUT after {timeout}s")
    except OSError as e:
        return AssetOpResult(passed=False, ran=True, op=name, cmd=cmd,
                             log_tail=f"could not run: {e}")
    if proc.returncode != 0:
        return AssetOpResult(
            passed=False, ran=True, op=name, cmd=cmd,
            log_tail=f"exited {proc.returncode}\n"
                     + (proc.stdout + "\n" + proc.stderr)[-tail:])
    return AssetOpResult(passed=True, ran=True, op=name, cmd=cmd,
                         log_tail=(proc.stdout + "\n" + proc.stderr)[-tail:])
