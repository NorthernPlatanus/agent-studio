"""The deterministic self-check gate. Zero tokens.

Runs the project-configured commands (typecheck/lint/test/build or anything
else) inside a candidate worktree. Failure output goes back to the worker as
retry feedback — no LLM is spent on detecting broken code.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..core.errors import OrchestratorError

log = logging.getLogger("orchestrator.gate")


@dataclass
class GateResult:
    passed: bool
    failed_step: str | None = None
    log_tail: str = ""


def ensure_deps(worktree: Path, cfg) -> None:
    install_cmd = cfg.gate.get("install_cmd")
    marker = cfg.gate.get("install_marker")
    if not install_cmd:
        return
    if marker and (worktree / marker).exists():
        return
    log.info("gate: installing deps in %s (%s)", worktree.name, install_cmd)
    timeout = int(cfg.gate.timeout_s)
    try:
        proc = subprocess.run(install_cmd, shell=True, cwd=str(worktree),
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise OrchestratorError(
            f"dependency install timed out after {timeout}s in {worktree}")
    if proc.returncode != 0:
        raise OrchestratorError(
            f"dependency install failed in {worktree}:\n{proc.stderr[-2000:]}")


def run_gate(worktree: Path, cfg) -> GateResult:
    tail = int(cfg.gate.log_tail_chars)
    timeout = int(cfg.gate.timeout_s)
    for cmd in cfg.gate.commands:
        log.info("gate: %s (%s)", cmd, worktree.name)
        try:
            proc = subprocess.run(cmd, shell=True, cwd=str(worktree),
                                  capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return GateResult(False, cmd, f"TIMEOUT after {timeout}s")
        if proc.returncode != 0:
            output = (proc.stdout + "\n" + proc.stderr)[-tail:]
            return GateResult(False, cmd, output)
    return GateResult(True)
