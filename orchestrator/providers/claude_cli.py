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
import uuid
from pathlib import Path

from ..core.errors import LimitExhausted, OrchestratorError, SessionLost
from .base import LLMProvider, LLMResult

log = logging.getLogger("orchestrator.claude_cli")

# Heuristics over CLI error text. The CLI's exact wording changes between
# releases — keep patterns broad and add new ones here as observed.
LIMIT_PATTERNS = re.compile(
    r"(usage limit|weekly limit|rate limit|limit reached|out of (usage|credits)"
    r"|upgrade to continue|limit will reset)",
    re.I,
)
# A --resume that couldn't find/load its conversation. Only consulted when we
# actually passed --resume, so a broad pattern can't misclassify other failures.
SESSION_LOST_PATTERNS = re.compile(
    r"(no conversation|session .*(not found|does not exist|expired|invalid)"
    r"|could not (find|resume|load) .*session|invalid session)",
    re.I,
)


# Accepted `--effort` values, from this CLI's own `--help` (and its warning text
# on a bad value). Kept as a literal set because the CLI does NOT reject an
# unknown level — it prints a warning to stderr, ignores it, and runs at the
# default effort. That would silently change the smart tier's behavior mid-study,
# so a typo has to fail at config load instead.
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


class ClaudeCliProvider(LLMProvider):
    type = "claude_cli"

    def __init__(self, name, pcfg, cfg):
        super().__init__(name, pcfg, cfg)
        # continuity key -> CLI session uuid. Instance state, and provider
        # instances are cached per-Config (providers/__init__), so sessions are
        # scoped to one run and never leak across projects or tests.
        self._sessions: dict[str, str] = {}

    def session_active(self, key: str) -> bool:
        return key in self._sessions

    def end_session(self, key: str) -> None:
        self._sessions.pop(key, None)

    async def complete(self, *, model: str, system: str, user: str,
                       cwd: str | None = None,
                       params: dict | None = None,
                       session: str | None = None,
                       effort: str | None = None,
                       allowed_tools: str | None = None,
                       mcp_config: str | None = None) -> LLMResult:
        # `params` (sampling overrides) is accepted for signature parity and
        # ignored: `claude -p` has no convenient temperature flag, and the spec
        # is explicit about not forcing one on the CLI tier.
        binary = os.environ.get("CLAUDE_BIN", self.pcfg.binary)
        args = [
            binary, "-p", user,
            "--output-format", "json",
            "--model", model,
        ]
        # Reasoning effort. The caller passes a per-ROLE level (planner and
        # reviewer are not equally hard); `providers.<name>.effort` is the
        # tier-wide fallback when a role sets none.
        level = effort or self.pcfg.get("effort")
        if level:
            if level not in EFFORT_LEVELS:
                raise OrchestratorError(
                    f"effort={level!r} is not a valid level. Use one of "
                    f"{', '.join(EFFORT_LEVELS)} — the CLI would ignore an "
                    f"unknown value and silently run at its default effort. "
                    f"(Set per role as roles.<role>.effort / run.escalate_effort, "
                    f"or tier-wide as providers.{self.name}.effort.)")
            args += ["--effort", str(level)]
        # Session continuity: the first call with a key pins a uuid via
        # --session-id, later calls continue it with --resume. Verified against
        # this CLI's `--help` (both flags exist and --session-id takes a uuid) —
        # per the project rule about never guessing external CLI surface.
        resuming = False
        if session is not None:
            sid = self._sessions.get(session)
            if sid is None:
                sid = str(uuid.uuid4())
                self._sessions[session] = sid
                args += ["--session-id", sid]
            else:
                args += ["--resume", sid]
                resuming = True
        if system:
            args += ["--append-system-prompt", system]
        # Per-role tool policy wins over the provider-wide allowlist, so one role
        # can hold an inspector's MCP tools without granting them tier-wide.
        allowed = allowed_tools or self.pcfg.get("allowed_tools")
        if allowed:
            # NOTE: `--allowedTools` is VARIADIC in this CLI ("<tools...>"), so it
            # swallows every following argument that does not start with "-".
            # Verified: the prompt is positional at args[2], well before this, and
            # `--mcp-config` below starts with "-" and terminates the list — so
            # this order is safe. Never append a bare positional after this point.
            args += ["--allowedTools", allowed]
        # MCP servers. An explicit per-call config (the verify phase) replaces the
        # global `mcp:` scan — that scan applies to EVERY claude_cli role, which
        # is almost never what you want for an inspector.
        if mcp_config:
            args += ["--mcp-config", str(mcp_config)]
        else:
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
            self._raise_if_session_lost(session, resuming, combined)
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
            self._raise_if_session_lost(session, resuming, msg)
            raise OrchestratorError(f"claude CLI error: {msg[:1000]}")

        usage = payload.get("usage") or {}
        in_tok, hit, miss = cache_tokens(usage)
        return LLMResult(
            text=payload.get("result", ""),
            input_tokens=in_tok,
            output_tokens=int(usage.get("output_tokens", 0)),
            cost_usd=float(payload.get("total_cost_usd", 0.0)),
            model=model,
            raw=payload,
            cache_hit_tokens=hit,
            cache_miss_tokens=miss,
        )


    def _raise_if_session_lost(self, session: str | None, resuming: bool,
                               text: str) -> None:
        """A failed --resume is recoverable, not a task failure: drop the dead id
        so the next call opens a fresh session, and signal the caller to retry
        with the full self-contained payload it abbreviated away."""
        if not (resuming and session and SESSION_LOST_PATTERNS.search(text)):
            return
        self._sessions.pop(session, None)
        log.warning("claude CLI could not resume session %s; falling back to a "
                    "fresh one: %s", session, text.strip()[:200])
        raise SessionLost(f"claude CLI could not resume: {text.strip()[:300]}")


def cache_tokens(usage: dict) -> tuple[int, int, int]:
    """(total_input, cache_hit, cache_miss) from a Claude CLI `usage` object.

    The CLI mirrors the Anthropic API shape, where `input_tokens` counts ONLY the
    uncached remainder — cached prefix bytes are reported separately as
    `cache_read_input_tokens` (billed at the cheap cached rate) and
    `cache_creation_input_tokens` (the write that seeds the cache). Reporting the
    bare `input_tokens` therefore understates the real prompt weight, which is the
    exact number the subscription tier is rationed on.

    So `input_tokens` here is the SUM of all three, and hit + miss == that sum —
    the same invariant openai_compatible._cache_tokens maintains, so the `usage`
    ledger is comparable across tiers. Absent fields degrade to 0 (an older CLI
    just reports no cache activity, never a fabricated hit).
    """
    fresh = int(usage.get("input_tokens", 0) or 0)
    created = int(usage.get("cache_creation_input_tokens", 0) or 0)
    read = int(usage.get("cache_read_input_tokens", 0) or 0)
    return fresh + created + read, read, fresh + created
