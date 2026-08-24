from __future__ import annotations

import asyncio
import contextlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

log = logging.getLogger("orchestrator.providers")


@dataclass
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0     # provider-reported or price-table-derived
    model: str = ""
    raw: object = None
    # Prompt-cache telemetry (filled by providers that expose it in `usage`,
    # e.g. openai_compatible/DeepSeek). 0 when unknown — never a lifeline.
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0


class LLMProvider(ABC):
    """One configured provider (an endpoint / a CLI), possibly many models."""

    type: str

    #: Does a non-None `effort` actually reach the backend on this provider?
    #: False means the level is DROPPED — `openai_compatible` talks to
    #: /v1/chat/completions, which has no reasoning dial for the models it
    #: fronts, and `codex exec` has no verified effort flag. Both used to drop it
    #: with nothing but a code comment to say so, which is the whole "I set high
    #: reasoning and nothing happened" failure: every layer above (preset ->
    #: RunContext.worker_effort -> the call) faithfully carried a value that the
    #: last hop discarded in silence. Declaring it as a class attribute lets the
    #: base warn once per (provider, model) and lets `core.validate` refuse to
    #: let the binding ship quietly in the first place.
    supports_effort: bool = False

    #: Ceiling on simultaneous in-flight calls through ONE provider instance,
    #: when `providers.<name>.max_concurrency` says nothing. None = unbounded.
    #:
    #: Unbounded is right for an HTTP backend: the endpoint does its own
    #: admission control, a 429 comes back as a visible error, and the SDK
    #: already retries. It is wrong for a CLI backend, where every concurrent
    #: call is a whole Node runtime riding one subscription — see the override
    #: on the CLI provider classes for the arithmetic.
    default_max_concurrency: int | None = None

    def __init__(self, name: str, pcfg, cfg):
        self.name = name
        self.pcfg = pcfg
        self.cfg = cfg

    def _warn_unsupported_effort(self, model: str, effort: str | None) -> None:
        """Say out loud that a requested reasoning effort is being dropped.

        Called by providers whose `supports_effort` is False, from the call path
        rather than from __init__, because the level is a per-CALL argument (a
        role's effort, a candidate's effort) and provider config cannot see it.

        Throttled to one line per (provider, model): a run dispatches hundreds of
        worker calls through one provider instance, and a warning repeated per
        call is a warning nobody reads. The set lives on the instance, and
        instances are per-Config, so a new run says it again.
        """
        if effort is None or self.supports_effort:
            return
        seen = getattr(self, "_effort_warned", None)
        if seen is None:
            seen = self._effort_warned = set()
        key = (self.name, str(model))
        if key in seen:
            return
        seen.add(key)
        log.warning(
            "provider %s (%s) does not support reasoning effort: effort=%r "
            "requested for model %s is being IGNORED. Bind that role/worker to a "
            "preset on a provider that supports effort (claude_cli, "
            "openai_responses), or drop the effort key so the config stops "
            "promising something the backend cannot do.",
            self.name, type(self).type, effort, model)

    # ---- concurrency ceiling ---------------------------------------------
    # Nothing above this layer bounds how many calls are in flight. Effective
    # LLM concurrency is `run.max_parallel_tasks x len(roles.worker.candidates)`
    # — 3 x 3 = 9 by default — and there is no semaphore anywhere else in the
    # codebase. Nine simultaneous `claude -p` processes is nine Node runtimes on
    # one subscription: the rate limit is hit almost immediately, and because
    # `engine.runner._run_batch` waits with FIRST_EXCEPTION, the resulting
    # LimitExhausted cancels every sibling task in the batch. That turns an edge
    # case into the normal case, which is why the ceiling lives here (per
    # provider, where the shared resource actually is) rather than in the
    # scheduler, where it would also throttle the free deterministic work.

    @classmethod
    def configured_max_concurrency(cls, pcfg) -> int | None:
        """Resolve `providers.<name>.max_concurrency` for this provider class.

        A CLASSMETHOD because `core.validate` has to answer the same question at
        startup from config alone: constructing a provider to ask it opens an
        HTTP client and reads an API key out of the environment, which is far
        too much to do just to warn about a pool width.

        Missing or null falls back to the class default. A value <= 0 means
        "unbounded" explicitly, so an operator can switch the ceiling off on a
        CLI provider (a second machine, a higher plan) without editing code.
        """
        raw = None
        if pcfg is not None:
            try:
                raw = pcfg.get("max_concurrency")
            except (AttributeError, TypeError):
                raw = None
        if raw is None:
            return cls.default_max_concurrency
        try:
            limit = int(raw)
        except (TypeError, ValueError):
            log.warning("max_concurrency=%r is not an integer; using the "
                        "default for %s (%s)", raw, cls.__name__,
                        cls.default_max_concurrency)
            return cls.default_max_concurrency
        return limit if limit > 0 else None

    def max_concurrency(self) -> int | None:
        """This instance's ceiling, or None for unbounded."""
        return type(self).configured_max_concurrency(self.pcfg)

    def _concurrency_gate(self):
        """`async with` guard around the part of a call that holds the resource.

        Providers wrap the ACTUAL backend call (the subprocess, the HTTP
        request) rather than the whole method, so a retry's backoff sleep does
        not sit on a slot that another candidate could be using.

        The semaphore is created lazily, on first use inside a running loop, for
        the same reason `RunContext.inspector_lock` is: an asyncio primitive
        binds to the loop that is running when it is created, and provider
        instances are cached per Config, which outlives any one `asyncio.run`.
        It is rebuilt when the loop changes (a degraded run's fresh context, the
        next unit test) or when the configured limit does, so a stale primitive
        can never silently serialize — or fail to bound — a later run.
        """
        limit = self.max_concurrency()
        if not limit:
            return contextlib.nullcontext()
        loop = asyncio.get_running_loop()
        sem = getattr(self, "_sem", None)
        if (sem is None or getattr(self, "_sem_loop", None) is not loop
                or getattr(self, "_sem_limit", None) != limit):
            sem = asyncio.Semaphore(limit)
            # A strong reference to the loop on purpose: it keeps the object
            # alive, so the identity check above can never be fooled by a new
            # loop landing at a recycled address.
            self._sem, self._sem_loop, self._sem_limit = sem, loop, limit
        return sem

    # ---- optional session continuity -------------------------------------
    # A provider that can continue a prior conversation server-side (the Claude
    # CLI's --session-id/--resume) overrides these. The default is "no sessions",
    # so callers can always ask and simply get the stateless behavior.

    def session_active(self, key: str) -> bool:
        """True if a later call with this `session` key would CONTINUE an existing
        conversation. Callers ask BEFORE building the prompt, because the answer
        decides whether to send the full context or just the new turn."""
        return False

    def end_session(self, key: str) -> None:
        """Forget a session key (scope boundary reached — e.g. the task is done),
        so the next call with it starts fresh."""

    @abstractmethod
    async def complete(self, *, model: str, system: str, user: str,
                       cwd: str | None = None,
                       params: dict | None = None,
                       session: str | None = None,
                       effort: str | None = None,
                       allowed_tools: str | None = None,
                       mcp_config: str | None = None,
                       on_progress: Callable[[dict], None] | None = None
                       ) -> LLMResult:
        """Single-turn completion. `cwd` only matters for CLI providers with
        repo read access. `params` carries optional per-candidate sampling
        overrides (temperature/top_p/seed/...); CLI providers accept and ignore
        it (no convenient temperature flag on `claude -p` / `codex exec`), while
        openai_compatible spreads it into the chat-completions call.

        `session` is an opaque continuity key (see session_active). Providers
        without server-side sessions ignore it and stay stateless.

        `effort` is the reasoning-depth level for reasoning-capable providers,
        resolved per ROLE by the caller (RunContext.role_effort) rather than read
        from provider config, so planner and reviewer can differ on one tier.
        None means "leave the provider's own default alone"; providers with no
        effort concept ignore it.

        `allowed_tools` and `mcp_config` are per-ROLE tool policy for CLI
        providers, again resolved by the caller. They exist so one role can be
        granted an inspector's MCP tools without handing the same tools to every
        other role on the tier — provider-level config cannot express that.

        `on_progress` receives coarse in-flight events (`{phase, ...}`) while the
        call runs, for callers that have somewhere to show them — the planner
        chat, where one turn is minutes long and a silent spinner is
        indistinguishable from a hang. It is advisory: providers that cannot
        stream simply never call it, and a callback that raises is logged and
        ignored rather than failing a call that has already been paid for."""

    async def complete_chat(self, *, model: str, system: str,
                            messages: list[dict], cwd: str | None = None,
                            params: dict | None = None,
                            session: str | None = None,
                            effort: str | None = None,
                            allowed_tools: str | None = None,
                            mcp_config: str | None = None,
                            on_progress: Callable[[dict], None] | None = None
                            ) -> LLMResult:
        """Multi-turn completion over role/content dicts.

        Default implementation is the stable-prefix passthrough: it flattens the
        turns into one user string and delegates to `complete`. That keeps the
        CLI providers (claude_cli/codex_cli) working unchanged — they are
        single-turn — while openai_compatible overrides this to pass the native
        `messages` array through (so automatic prefix caching applies).
        """
        user = "\n\n".join(
            m["content"] for m in messages if m.get("role") != "system")
        return await self.complete(model=model, system=system, user=user,
                                   cwd=cwd, params=params, session=session,
                                   effort=effort, allowed_tools=allowed_tools,
                                   mcp_config=mcp_config,
                                   on_progress=on_progress)
