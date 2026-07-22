from __future__ import annotations

from abc import ABC, abstractmethod
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

    @abstractmethod
    async def complete(self, *, model: str, system: str, user: str,
                       cwd: str | None = None,
                       params: dict | None = None) -> LLMResult:
        """Single-turn completion. `cwd` only matters for CLI providers with
        repo read access. `params` carries optional per-candidate sampling
        overrides (temperature/top_p/seed/...); CLI providers accept and ignore
        it (no convenient temperature flag on `claude -p` / `codex exec`), while
        openai_compatible spreads it into the chat-completions call."""

    async def complete_chat(self, *, model: str, system: str,
                            messages: list[dict], cwd: str | None = None,
                            params: dict | None = None) -> LLMResult:
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
                                   cwd=cwd, params=params)
