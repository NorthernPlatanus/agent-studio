"""Codex CLI provider — `codex exec` subprocess (smart-tier alternative to Claude).

Mirrors claude_cli.py: a non-interactive CLI call that reuses the installed
Codex CLI's saved auth. Verified against the OpenAI Codex non-interactive docs
(codex exec): it defaults to a READ-ONLY sandbox, streams progress to stderr,
and prints the final agent message to stdout; `--json` emits JSONL events and
`--output-last-message <file>` writes the final message to a file.

Invariants honored:
  * Read-only by design. `allowed_tools` maps to `--sandbox <mode>`, defaulting
    to `read-only` so planner/reviewer never edit the repo — same guarantee as
    claude_cli's `allowed_tools: "Read,Grep,Glob"`.
  * MCP is best-effort and OFF by default (enable_mcp: false). `codex exec`
    cannot approve MCP tool calls without a dangerous bypass, so by default we do
    NOT trade away the sandbox to enable it (Fable N6): if MCP can't be granted
    safely we simply degrade (skip MCP for that role). The one opt-in escape is
    Fix 2's `mcp_bypass` — gated behind BOTH enable_mcp and mcp_bypass, OFF by
    default, loudly named — a temporary workaround for openai/codex#24135 that
    swaps --sandbox for --dangerously-bypass-approvals-and-sandbox. Enable ONLY
    inside a disposable worktree/container.

Codex has no `--append-system-prompt` analogue, so the system text is prepended
to the user prompt.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path

from ..core.errors import LimitExhausted, OrchestratorError
from ._process import CliTimeout, stream_cli
from .base import LLMProvider, LLMResult

log = logging.getLogger("orchestrator.codex_cli")

#: Fallbacks when the provider config sets neither. See claude_cli for the
#: reasoning behind an idle clock and a loose absolute backstop.
DEFAULT_IDLE_TIMEOUT_S = 300
DEFAULT_TOTAL_TIMEOUT_S = 3600

# Codex limit/quota wording (broad, like claude_cli — extend as observed).
LIMIT_PATTERNS = re.compile(
    r"(usage limit|weekly limit|rate limit|limit reached|too many requests"
    r"|quota|429|out of (usage|credits)|upgrade to continue|limit will reset)",
    re.I,
)

_SANDBOX_MODES = {"read-only", "workspace-write", "danger-full-access"}


class CodexCliProvider(LLMProvider):
    type = "codex_cli"

    def _sandbox_mode(self) -> str:
        """Map allowed_tools -> a codex --sandbox mode. Anything that isn't an
        explicit codex mode (e.g. a Claude-style 'Read,Grep,Glob') is treated as
        read-only — never widen the sandbox implicitly."""
        v = (self.pcfg.get("allowed_tools") or "read-only").strip()
        return v if v in _SANDBOX_MODES else "read-only"

    def _mcp_config_args(self) -> list[str]:
        """Best-effort MCP translation to `-c mcp_servers.*` overrides. OFF
        unless enable_mcp is set, and NEVER adds a bypass/danger flag."""
        args: list[str] = []
        if not self.pcfg.get("enable_mcp"):
            return args
        for name in (self.cfg.get("mcp") or {}).keys():
            entry = self.cfg.mcp.get(name)
            if entry and entry.get("enabled"):
                cmd = entry.get("command")
                if cmd:
                    args += ["-c", f"mcp_servers.{name}.command={cmd}"]
        return args

    @staticmethod
    def _parse_events(stdout: str) -> tuple[str, int, int, int]:
        """Tolerant JSONL parse: return (final_text, input_tokens, output_tokens,
        cached_input_tokens). Codex event schema varies across releases, so we
        scan defensively and keep the last text-bearing event as the final
        message. `cached` is 0 when the release reports no cache field — never
        guessed, so a 0 hit-rate in the ledger reads as "unknown or cold", which
        is the honest reading either way."""
        text = ""
        in_tok = out_tok = cached = 0
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(ev, dict):
                continue
            usage = ev.get("usage") or (ev.get("item") or {}).get("usage") or {}
            if isinstance(usage, dict) and usage:
                in_tok = int(usage.get("input_tokens",
                             usage.get("prompt_tokens", in_tok)) or in_tok)
                out_tok = int(usage.get("output_tokens",
                              usage.get("completion_tokens", out_tok)) or out_tok)
                cached = _cached_input_tokens(usage) or cached
            cand = None
            if isinstance(ev.get("text"), str):
                cand = ev["text"]
            elif isinstance(ev.get("message"), str):
                cand = ev["message"]
            else:
                item = ev.get("item") or {}
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    cand = item["text"]
                msg = ev.get("msg") or {}
                if isinstance(msg, dict):
                    if isinstance(msg.get("message"), str):
                        cand = msg["message"]
                    elif isinstance(msg.get("text"), str):
                        cand = msg["text"]
            if cand is not None:
                text = cand
        return text, in_tok, out_tok, cached

    async def complete(self, *, model: str, system: str, user: str,
                       cwd: str | None = None,
                       params: dict | None = None,
                       session: str | None = None,
                       effort: str | None = None,
                       allowed_tools: str | None = None,
                       mcp_config: str | None = None,
                       on_progress: Callable[[dict], None] | None = None
                       ) -> LLMResult:
        # `session` is accepted for signature parity and ignored: `codex exec` is
        # one-shot, and session_active() correctly reports False, so callers
        # always send the full self-contained payload on this provider.
        # `effort` likewise: `codex exec` has no verified equivalent flag, so a
        # level configured for the claude_cli tier is inapplicable here rather
        # than mistranslated into a setting we have not checked. Flipping
        # smart_provider to codex_cli therefore drops effort control, by design.
        # `params` (sampling overrides) is accepted for signature parity and
        # ignored: `codex exec` has no convenient temperature flag, and the spec
        # is explicit about not forcing one on the CLI tier.
        binary = os.environ.get("CODEX_BIN", self.pcfg.binary)
        prompt = f"{system}\n\n{user}" if system else user

        last_msg = tempfile.NamedTemporaryFile(
            mode="r", suffix=".txt", delete=False)
        last_msg_path = last_msg.name
        last_msg.close()

        # MCP bypass (Fix 2): `codex exec` cannot approve MCP tool calls
        # non-interactively without --dangerously-bypass-approvals-and-sandbox
        # (openai/codex#24135, open). Gated behind BOTH enable_mcp AND the
        # loudly-named mcp_bypass, OFF by default. The bypass REMOVES the sandbox,
        # so we must NOT also pass --sandbox (a flag conflict) — the two are
        # mutually exclusive here.
        bypass = bool(self.pcfg.get("enable_mcp")) and bool(self.pcfg.get("mcp_bypass"))
        args = [binary, "exec", "--json", "--skip-git-repo-check"]
        if bypass:
            args += ["--dangerously-bypass-approvals-and-sandbox"]  # TEMPORARY: until #24135
        else:
            args += ["--sandbox", self._sandbox_mode()]
        args += ["--model", model, "--output-last-message", last_msg_path]
        args += self._mcp_config_args()
        args += [prompt]

        # Idle, not wall-clock — the same reasoning as claude_cli (see
        # _process.stream_cli): `codex exec` runs a tool loop whose length the
        # model decides, and it streams progress to stderr throughout, so
        # silence is the only honest signal that it has stopped working.
        idle_s = float(self.pcfg.get("idle_timeout_s", DEFAULT_IDLE_TIMEOUT_S))
        total_s = self.pcfg.get("timeout_s", DEFAULT_TOTAL_TIMEOUT_S)
        total_s = float(total_s) if total_s else None
        log.debug("codex_cli: %s exec ... (cwd=%s, sandbox=%s)",
                  binary, cwd, "bypass" if bypass else self._sandbox_mode())
        # `codex exec --json` emits JSONL events. They are collected HERE rather
        # than read back off `run.stdout`, which keeps only a bounded head for
        # error messages — `_parse_events` needs every event to find the usage
        # numbers, and a long run would otherwise silently lose its token ledger.
        # They double as coarse progress for a caller with somewhere to show it.
        events: list[str] = []

        def on_line(tag: str, line: str) -> None:
            if tag != "stdout" or not line.startswith("{"):
                return
            events.append(line)
            if on_progress is not None:
                on_progress({"phase": "event", "text": line[:200]})

        try:
            run = await stream_cli(args, cwd=cwd, idle_timeout_s=idle_s,
                                   total_timeout_s=total_s, on_line=on_line)
        except CliTimeout as e:
            _safe_unlink(last_msg_path)
            raise OrchestratorError(f"codex CLI {e}") from e

        out = "\n".join(events)
        combined = run.combined

        if run.returncode != 0:
            _safe_unlink(last_msg_path)
            if LIMIT_PATTERNS.search(combined):
                raise LimitExhausted(f"codex CLI limit: {combined.strip()[:500]}")
            raise OrchestratorError(
                f"codex CLI exited {run.returncode}: {combined.strip()[:1000]}")

        text, in_tok, out_tok, cached = self._parse_events(out)
        # --output-last-message is the most reliable source of the final text.
        file_text = _read_and_unlink(last_msg_path)
        if file_text:
            text = file_text

        # Subscription auth (default): logged, not counted (cost 0, like
        # claude_cli). API auth: try the price table; unknown model -> 0.
        cost = 0.0
        if self.pcfg.get("auth", "subscription") == "api":
            cost = self._priced(model, in_tok, out_tok)

        # Codex reports `input_tokens` as the TOTAL prompt (cached bytes included),
        # unlike the Anthropic shape — so hit is the reported cached count and miss
        # is the remainder, preserving hit + miss == input_tokens.
        hit = min(cached, in_tok)
        return LLMResult(text=text, input_tokens=in_tok, output_tokens=out_tok,
                         cost_usd=cost, model=model, raw=out,
                         cache_hit_tokens=hit, cache_miss_tokens=max(in_tok - hit, 0))

    def _priced(self, model: str, in_tok: int, out_tok: int) -> float:
        for wm in (self.cfg.get("worker_models") or {}).keys():
            entry = self.cfg.worker_models.get(wm)
            if entry and entry.get("provider") == self.name and entry.get("model") == model:
                return (in_tok / 1e6 * float(entry.get("input_per_mtok", 0))
                        + out_tok / 1e6 * float(entry.get("output_per_mtok", 0)))
        return 0.0


def _cached_input_tokens(usage: dict) -> int:
    """Cached-prefix input tokens from a codex `usage` object, tolerating the
    shapes seen across releases (flat `cached_input_tokens`, the Anthropic-style
    `cache_read_input_tokens`, or OpenAI's nested
    `input_tokens_details.cached_tokens`). 0 when the release reports none."""
    for key in ("cached_input_tokens", "cache_read_input_tokens"):
        if usage.get(key) is not None:
            return int(usage[key] or 0)
    details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        return int(details.get("cached_tokens") or 0)
    return 0


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _read_and_unlink(path: str) -> str:
    try:
        with open(path) as f:
            text = f.read().strip()
    except OSError:
        text = ""
    _safe_unlink(path)
    return text
