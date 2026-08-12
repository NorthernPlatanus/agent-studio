from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass


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

    def __init__(self, name: str, pcfg, cfg):
        self.name = name
        self.pcfg = pcfg
        self.cfg = cfg

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
