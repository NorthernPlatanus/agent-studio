"""Claude Code subscription provider — `claude -p` subprocess.

Uses the installed Claude Code CLI, which picks up subscription auth on its
own; no API key, no per-token cash cost (reported cost is logged as
subscription usage). Detects weekly/usage limit exhaustion and raises
LimitExhausted so the run can checkpoint and pause.

Output is consumed as `--output-format stream-json` rather than `json`. The
final `result` event is byte-identical to what `json` returns, so nothing
downstream changed; what streaming buys is (a) an IDLE timeout instead of a
wall-clock one — see `_process.stream_cli` for why that distinction is the whole
ballgame on a tool-using call — (b) progress the operator can watch, and (c) the
CLI's own session id, which is what makes a retry able to resume rather than
restart.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..core.errors import LimitExhausted, OrchestratorError, SessionLost
from ._process import CliTimeout, stream_cli
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
# Infrastructure failures that say nothing about the request: retrying the same
# prompt is the correct response. Consulted ONLY on a non-zero exit, and only
# after the limit check — "rate limit" is a 429 that must NOT be retried here,
# it has to reach the run's own pause/degrade ladder as LimitExhausted.
TRANSIENT_PATTERNS = re.compile(
    r"(ECONNRESET|ETIMEDOUT|ENOTFOUND|EAI_AGAIN|EPIPE|socket hang ?up"
    r"|fetch failed|network (error|timeout)|connection (reset|refused|closed|error)"
    r"|\b50[234]\b|bad gateway|service unavailable|gateway time-?out"
    r"|overloaded|internal server error)",
    re.I,
)


# Accepted `--effort` values, from this CLI's own `--help` (and its warning text
# on a bad value). Kept as a literal set because the CLI does NOT reject an
# unknown level — it prints a warning to stderr, ignores it, and runs at the
# default effort. That would silently change the smart tier's behavior mid-study,
# so a typo has to fail at config load instead.
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

#: Fallbacks when the provider config sets neither. The idle default is ~27x the
#: largest inter-event gap measured on a real planner turn (11s over 135 events),
#: so it has room for a slow first call or a network stall and still kills a
#: wedged process sooner than the old 600s wall clock did.
DEFAULT_IDLE_TIMEOUT_S = 300
DEFAULT_TOTAL_TIMEOUT_S = 3600
DEFAULT_RETRY_ATTEMPTS = 2

#: Floor between progress callbacks. The CLI emits a `thinking_tokens` heartbeat
#: per few tokens; forwarding those one-for-one would push thousands of frames
#: through the API's fan-out for one planner turn.
PROGRESS_MIN_INTERVAL_S = 2.0

#: How long a pinned session may sit idle before `--resume` stops being a saving
#: and becomes a penalty.
#:
#: Resuming is only cheap while the conversation's prefix is still in the prompt
#: cache. Anthropic's cache TTL for these sessions is one hour; past it, a resume
#: replays the ENTIRE accumulated conversation as fresh input, at cache-WRITE
#: weight (1.25x) rather than cache-read weight (0.1x). Measured on a real
#: planner turn: the prefix was 47k tokens and the whole invocation billed 326k
#: input — a cold resume of that conversation costs more than starting over,
#: because starting over at least re-sends only the 47k payload.
#:
#: 50 minutes leaves ten minutes of headroom under the one-hour TTL for a turn
#: that is itself minutes long. 0 disables expiry (resume regardless of age).
DEFAULT_SESSION_MAX_IDLE_S = 50 * 60

#: What a RETRY sends when it resumes the killed attempt's session, in place of
#: the original prompt. Verified against the CLI: a session that was SIGKILLed
#: mid-turn still holds the prompt it was started with (a resumed session recalls
#: a codeword from the interrupted turn and places it "in this same
#: conversation"). Re-sending the real prompt would therefore append a second
#: copy of it — on the planner that is ~70KB — to a context that already carries
#: the whole interrupted exploration.
RESUME_NUDGE = (
    "Your previous attempt in this session was interrupted before it returned an "
    "answer. Do not start over and do not re-read what you have already read: "
    "continue from what you established, and reply now with the final answer in "
    "exactly the format the system prompt requires."
)


@dataclass
class _LiveSession:
    """A pinned CLI conversation and when we last exchanged tokens with it.

    `last_seen` is monotonic, not wall-clock: the whole point of the timestamp is
    to reason about a cache TTL, and a laptop that sleeps or an NTP correction
    must not make a cold session look warm (or vice versa).
    """

    sid: str
    last_seen: float


class ClaudeCliProvider(LLMProvider):
    type = "claude_cli"

    def __init__(self, name, pcfg, cfg):
        super().__init__(name, pcfg, cfg)
        # continuity key -> live CLI session. Instance state, and provider
        # instances are cached per-Config (providers/__init__), so sessions are
        # scoped to one run and never leak across projects or tests.
        self._sessions: dict[str, _LiveSession] = {}

    def session_max_idle_s(self) -> float:
        """Idle ceiling for `--resume` (see DEFAULT_SESSION_MAX_IDLE_S)."""
        try:
            return float(self.pcfg.get("session_max_idle_s",
                                       DEFAULT_SESSION_MAX_IDLE_S))
        except (AttributeError, TypeError, ValueError):
            return DEFAULT_SESSION_MAX_IDLE_S

    def _live(self, key: str | None) -> _LiveSession | None:
        """The session for `key`, or None — DROPPING it first if it has gone cold.

        Expiry has to happen here rather than in `session_active` alone, because
        `complete` is also reached without that check (the first turn of a loop
        passes a session key with no delta). Both paths must agree, or the caller
        abbreviates its payload for a session this method is about to discard.
        """
        if key is None:
            return None
        entry = self._sessions.get(key)
        if entry is None:
            return None
        max_idle = self.session_max_idle_s()
        idle = time.monotonic() - entry.last_seen
        if max_idle > 0 and idle > max_idle:
            self._sessions.pop(key, None)
            log.info("planner session %s idle %.0fs (> %.0fs) — its prompt cache "
                     "has expired, so resuming would replay the whole "
                     "conversation at full price; starting fresh instead",
                     key, idle, max_idle)
            return None
        return entry

    def session_active(self, key: str) -> bool:
        return self._live(key) is not None

    def end_session(self, key: str) -> None:
        self._sessions.pop(key, None)

    async def complete(self, *, model: str, system: str, user: str,
                       cwd: str | None = None,
                       params: dict | None = None,
                       session: str | None = None,
                       effort: str | None = None,
                       allowed_tools: str | None = None,
                       mcp_config: str | None = None,
                       on_progress: Callable[[dict], None] | None = None
                       ) -> LLMResult:
        # `params` (sampling overrides) is accepted for signature parity and
        # ignored: `claude -p` has no convenient temperature flag, and the spec
        # is explicit about not forcing one on the CLI tier.
        level = effort or self.pcfg.get("effort")
        if level and level not in EFFORT_LEVELS:
            raise OrchestratorError(
                f"effort={level!r} is not a valid level. Use one of "
                f"{', '.join(EFFORT_LEVELS)} — the CLI would ignore an "
                f"unknown value and silently run at its default effort. "
                f"(Set per role as roles.<role>.effort / run.escalate_effort, "
                f"or tier-wide as providers.{self.name}.effort.)")

        # Session continuity: the first call with a key pins a uuid via
        # --session-id, later calls continue it with --resume. Verified against
        # this CLI's `--help` (both flags exist and --session-id takes a uuid) —
        # per the project rule about never guessing external CLI surface.
        pinned: str | None = None
        resuming = False
        if session is not None:
            entry = self._live(session)         # expires a cold session for us
            if entry is None:
                pinned = str(uuid.uuid4())
                self._sessions[session] = _LiveSession(pinned, time.monotonic())
            else:
                pinned, resuming = entry.sid, True

        idle_s = float(self.pcfg.get("idle_timeout_s", DEFAULT_IDLE_TIMEOUT_S))
        total_s = self.pcfg.get("timeout_s", DEFAULT_TOTAL_TIMEOUT_S)
        total_s = float(total_s) if total_s else None
        attempts = max(1, int(self.pcfg.get("retry_attempts", DEFAULT_RETRY_ATTEMPTS)))

        # The id to --resume on a retry. Starts as the session's pinned id (if
        # we are continuing one) and is otherwise learned from the CLI's own
        # `system/init` event on the failed attempt.
        resume_id: str | None = pinned if resuming else None
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            # ►► A retry NEVER re-passes --session-id. Verified against the CLI:
            # once an attempt has started, its id exists, and reusing it fails
            # outright with "Session ID <uuid> is already in use." --resume is
            # both the working flag and the better one — the killed attempt's
            # exploration is still in that session (verified: a session survives
            # SIGKILL and resumes), so the retry continues warm instead of
            # re-reading the repo from scratch.
            if attempt == 1:
                use_session_id = None if resuming else pinned
                use_resume = pinned if resuming else None
            elif resume_id:
                use_session_id, use_resume = None, resume_id
            else:
                # The failed attempt died before the CLI announced a session, so
                # there is nothing to resume. Re-pin a FRESH id rather than the
                # old one: we cannot know whether the CLI got far enough to
                # create it, and reusing a live id fails outright.
                use_resume = None
                use_session_id = str(uuid.uuid4()) if session is not None else None
                if session is not None:
                    self._sessions[session] = _LiveSession(use_session_id,
                                                           time.monotonic())

            # Resuming our own killed attempt: the session still holds the
            # original prompt (see RESUME_NUDGE), so send the nudge instead of a
            # second copy of it. The caller's own `session` continuity is not
            # touched — on that path `user` is already the abbreviated delta that
            # plan_or_ask chose to send.
            prompt = RESUME_NUDGE if (attempt > 1 and use_resume) else user
            args = self._build_args(
                model=model, system=system, user=prompt, level=level,
                allowed_tools=allowed_tools, mcp_config=mcp_config,
                session_id=use_session_id, resume=use_resume)

            sink = _EventSink(on_progress, root=cwd)
            try:
                run = await stream_cli(
                    args, cwd=cwd, idle_timeout_s=idle_s,
                    total_timeout_s=total_s, on_line=sink.feed)
            except CliTimeout as e:
                # An idle kill means the process wedged and produced nothing —
                # the one timeout worth retrying. Blowing the absolute ceiling
                # means it was working the whole time and would do it again.
                resume_id = sink.session_id or resume_id
                last_error = OrchestratorError(f"claude CLI {e}")
                if e.retryable and attempt < attempts:
                    log.warning("claude CLI wedged (%s); retrying %d/%d%s",
                                e, attempt + 1, attempts,
                                f" by resuming {resume_id}" if resume_id else "")
                    await asyncio.sleep(_backoff(attempt))
                    continue
                raise last_error from e

            resume_id = sink.session_id or resume_id
            payload = sink.result

            if run.returncode != 0:
                combined = run.combined
                if LIMIT_PATTERNS.search(combined):
                    raise _limit_exhausted(combined.strip()[:500], sink.rate_limit)
                # Only the CALLER's session (run.session_reuse) turns a dead
                # resume into SessionLost — that signal means "resend the full
                # payload you abbreviated away", which is meaningless for a
                # resume this loop chose on its own.
                self._raise_if_session_lost(session, resuming, combined)
                last_error = OrchestratorError(
                    f"claude CLI exited {run.returncode}: {combined.strip()[:1000]}")
                if (use_resume and not resuming
                        and SESSION_LOST_PATTERNS.search(combined)
                        and attempt < attempts):
                    # Our own retry-resume found nothing to resume. The prompt is
                    # self-contained, so drop the id and let the next attempt
                    # start clean rather than failing on our optimization.
                    log.warning("claude CLI could not resume %s on retry; the "
                                "next attempt starts fresh", use_resume)
                    resume_id = None
                    await asyncio.sleep(_backoff(attempt))
                    continue
                if TRANSIENT_PATTERNS.search(combined) and attempt < attempts:
                    log.warning("claude CLI failed transiently (exit %s); retrying "
                                "%d/%d: %s", run.returncode, attempt + 1, attempts,
                                combined.strip()[:200])
                    await asyncio.sleep(_backoff(attempt))
                    continue
                raise last_error

            if payload is None:
                raise OrchestratorError(
                    "claude CLI produced no result event: "
                    f"{run.combined.strip()[:500]}")

            if payload.get("is_error"):
                msg = str(payload.get("result", ""))
                if LIMIT_PATTERNS.search(msg):
                    raise _limit_exhausted(msg[:500], sink.rate_limit)
                self._raise_if_session_lost(session, resuming, msg)
                last_error = OrchestratorError(f"claude CLI error: {msg[:1000]}")
                if TRANSIENT_PATTERNS.search(msg) and attempt < attempts:
                    log.warning("claude CLI reported a transient error; retrying "
                                "%d/%d: %s", attempt + 1, attempts, msg[:200])
                    await asyncio.sleep(_backoff(attempt))
                    continue
                raise last_error

            # The turn landed, so the conversation is warm again as of NOW. Only
            # the clock moves: the id stays the one WE pinned (--session-id) or
            # resumed (--resume). Adopting the stream's `session_id` here instead
            # would be a guess about whether this CLI forks on resume, and that
            # is precisely the kind of unverified assumption about external CLI
            # surface this provider is written to avoid.
            entry = self._sessions.get(session) if session is not None else None
            if entry is not None:
                entry.last_seen = time.monotonic()

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

        # Unreachable: every path above either returns or raises. Kept as a
        # belt-and-braces guard so a future edit to the loop cannot silently
        # fall out of it returning None.
        raise last_error or OrchestratorError("claude CLI made no attempts")

    def _build_args(self, *, model: str, system: str, user: str,
                    level: str | None, allowed_tools: str | None,
                    mcp_config: str | None, session_id: str | None,
                    resume: str | None) -> list[str]:
        binary = os.environ.get("CLAUDE_BIN", self.pcfg.binary)
        args = [
            binary, "-p", user,
            # stream-json needs --verbose in print mode (verified: without it the
            # CLI refuses with "requires --verbose").
            "--output-format", "stream-json", "--verbose",
            # ►► NOT optional, despite looking like a nicety. Without it the one
            # systematically silent window in a planner turn is the final answer
            # itself: a `text` block is only emitted once COMPLETE, so writing
            # 26KB of spec JSON was measured as 174 seconds of total silence —
            # by far the largest gap in the run, and one that grows with the size
            # of the plan. An idle clock would eventually kill exactly the calls
            # that were doing the most work. With deltas streaming, the same
            # window measured a 2s maximum gap.
            "--include-partial-messages",
            "--model", model,
        ]
        # Reasoning effort. The caller passes a per-ROLE level (planner and
        # reviewer are not equally hard); `providers.<name>.effort` is the
        # tier-wide fallback when a role sets none. Validated by the caller.
        if level:
            args += ["--effort", str(level)]
        if resume:
            args += ["--resume", resume]
        elif session_id:
            args += ["--session-id", session_id]
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
        return args

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


def _limit_exhausted(detail: str, rate_limit: dict | None) -> LimitExhausted:
    """Build the limit error, carrying the reset window when the CLI told us one.

    `resetsAt` is unix seconds and `rateLimitType` names the window (`five_hour`,
    `weekly`, …). Both are taken from the stream's `rate_limit_event`, which the
    CLI emits on every call — so this is the number the provider reported, never
    an estimate of ours. A caller that can wait (the planner chat) freezes until
    then instead of failing.
    """
    info = rate_limit or {}
    resets_at = info.get("resetsAt")
    try:
        resets_at = float(resets_at) if resets_at is not None else None
    except (TypeError, ValueError):
        resets_at = None
    limit_type = info.get("rateLimitType")
    suffix = ""
    if resets_at is not None:
        suffix = (f" (the {limit_type or 'usage'} window resets at "
                  f"{time.strftime('%H:%M', time.localtime(resets_at))})")
    return LimitExhausted(f"claude CLI limit: {detail}{suffix}",
                          resets_at=resets_at,
                          limit_type=str(limit_type) if limit_type else None)


def _backoff(attempt: int) -> float:
    """1s, 2s, 4s … capped. Short on purpose: these retries sit in front of an
    operator watching a chat, and the failures being retried are wedges and
    5xx-class blips, not a queue we are being asked to back off from."""
    return min(2.0 ** (attempt - 1), 8.0)


class _EventSink:
    """Turns the CLI's stream-json lines into the three things we want from them:
    the final `result` payload, the session id, and operator-facing progress.

    Tolerant by construction — an unparseable or unfamiliar line is skipped, not
    an error. The stream is a debug-shaped surface that changes between CLI
    releases, and a call that has already been paid for must not be thrown away
    because a new event type showed up in it.
    """

    def __init__(self, on_progress: Callable[[dict], None] | None,
                 root: str | None = None):
        self._on_progress = on_progress
        self._root = root
        self.result: dict | None = None
        self.session_id: str | None = None
        #: The newest `rate_limit_event.rate_limit_info`. The CLI emits one on
        #: EVERY call, not only when it refuses, so the window's reset time is
        #: known before it is needed.
        self.rate_limit: dict | None = None
        self._last_emit = 0.0

    def feed(self, tag: str, line: str) -> None:
        if tag == "stderr" or not line.startswith("{"):
            return
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return
        if not isinstance(event, dict):
            return

        if self.session_id is None and event.get("session_id"):
            self.session_id = str(event["session_id"])

        kind = event.get("type")
        if kind == "rate_limit_event":
            info = event.get("rate_limit_info")
            if isinstance(info, dict):
                self.rate_limit = info
        elif kind == "result":
            self.result = event
        elif kind == "assistant":
            self._assistant(event)
        elif kind == "system" and event.get("subtype") == "thinking_tokens":
            self._progress({"phase": "thinking",
                            "tokens": event.get("estimated_tokens")})

    def _assistant(self, event: dict) -> None:
        content = (event.get("message") or {}).get("content") or []
        if not isinstance(content, list):
            return
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                self._progress({"phase": "tool", "tool": block.get("name"),
                                "target": _tool_target(block.get("input"),
                                                       self._root)},
                               force=True)
            elif block.get("type") == "text":
                text = (block.get("text") or "").strip()
                if text:
                    self._progress({"phase": "text", "text": text[:200]},
                                   force=True)

    def _progress(self, event: dict, *, force: bool = False) -> None:
        """Throttled unless `force`. Tool calls and text are the events an
        operator actually reads, and there are tens of them per turn; the
        thinking heartbeat arrives thousands of times and is only there to say
        "still alive", which two seconds of resolution conveys perfectly well."""
        if self._on_progress is None:
            return
        now = time.monotonic()
        if not force and now - self._last_emit < PROGRESS_MIN_INTERVAL_S:
            return
        self._last_emit = now
        self._on_progress(event)


def _tool_target(payload: object, root: str | None = None) -> str | None:
    """The one field worth showing next to a tool name. Ordered by how much it
    tells an operator watching the planner work.

    Paths come back absolute, and the repo prefix is the same on every one of
    them — it is the part the reader already knows, and on a real checkout it
    consumed most of the line. Shown relative to the repo instead.
    """
    if not isinstance(payload, dict):
        return None
    for key in ("file_path", "path", "pattern", "command", "url", "prompt"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            target = value.strip()
            if root:
                prefix = root.rstrip("/") + "/"
                if target.startswith(prefix):
                    target = target[len(prefix):]
            return target[:120]
    return None


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
