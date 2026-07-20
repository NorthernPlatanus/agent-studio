"""Claude Code subscription provider — `claude -p` subprocess.

Uses the installed Claude Code CLI, which picks up subscription auth on its
own; no API key, no per-token cash cost (reported cost is logged as
subscription usage). Detects weekly/usage limit exhaustion and raises
LimitExhausted so the run can checkpoint and pause.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path

from ..core.errors import LimitExhausted, OrchestratorError
from .base import LLMProvider, LLMResult

log = logging.getLogger("orchestrator.claude_cli")

# Heuristics over CLI error text. The CLI's exact wording changes between
# releases — keep patterns broad and add new ones here as observed.
LIMIT_PATTERNS = re.compile(
    r"(usage limit|weekly limit|rate limit|limit reached|out of (usage|credits)"
    r"|upgrade to continue|limit will reset)",
    re.I,
)


class ClaudeCliProvider(LLMProvider):
    type = "claude_cli"

    async def complete(self, *, model: str, system: str, user: str,
                       cwd: str | None = None) -> LLMResult:
        binary = os.environ.get("CLAUDE_BIN", self.pcfg.binary)
        args = [
            binary, "-p", user,
            "--output-format", "json",
            "--model", model,
        ]
        if system:
            args += ["--append-system-prompt", system]
        allowed = self.pcfg.get("allowed_tools")
        if allowed:
            args += ["--allowedTools", allowed]
        # Optional MCP servers (e.g. a live scene inspector for reviewer).
        for name in (self.cfg.get("mcp") or {}).keys():
            entry = self.cfg.mcp.get(name)
            if entry and entry.get("enabled"):
                mcp_path = Path(entry.mcp_config)
                if not mcp_path.is_absolute():
                    mcp_path = self.cfg.root / mcp_path
                args += ["--mcp-config", str(mcp_path)]

        timeout = int(self.pcfg.get("timeout_s", 600))
        log.debug("claude_cli: %s ... (cwd=%s)", " ".join(args[:2]), cwd)
        proc = await asyncio.create_subprocess_exec(
            *args, cwd=cwd,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()  # reap; avoid a zombie process
            raise OrchestratorError(f"claude CLI timed out after {timeout}s")

        out, err = stdout.decode(), stderr.decode()
        combined = out + "\n" + err

        if proc.returncode != 0:
            if LIMIT_PATTERNS.search(combined):
                raise LimitExhausted(f"claude CLI limit: {combined.strip()[:500]}")
            raise OrchestratorError(
                f"claude CLI exited {proc.returncode}: {combined.strip()[:1000]}")

        try:
            payload = json.loads(out)
        except json.JSONDecodeError:
            raise OrchestratorError(f"claude CLI returned non-JSON: {out[:500]}")

        if payload.get("is_error"):
            msg = str(payload.get("result", ""))
            if LIMIT_PATTERNS.search(msg):
                raise LimitExhausted(f"claude CLI limit: {msg[:500]}")
            raise OrchestratorError(f"claude CLI error: {msg[:1000]}")

        usage = payload.get("usage") or {}
        return LLMResult(
            text=payload.get("result", ""),
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            cost_usd=float(payload.get("total_cost_usd", 0.0)),
            model=model,
            raw=payload,
        )
