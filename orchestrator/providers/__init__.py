"""Provider registry. Adding a provider type = one class + one registry line."""

from __future__ import annotations

import weakref

from .anthropic_api import AnthropicApiProvider
from .base import LLMProvider, LLMResult
from .claude_cli import ClaudeCliProvider
from .codex_cli import CodexCliProvider
from .openai_compatible import OpenAICompatibleProvider
from .openai_responses import OpenAIResponsesProvider

PROVIDER_TYPES: dict[str, type[LLMProvider]] = {
    "anthropic_api": AnthropicApiProvider,
    "claude_cli": ClaudeCliProvider,
    "codex_cli": CodexCliProvider,
    "openai_compatible": OpenAICompatibleProvider,
    "openai_responses": OpenAIResponsesProvider,
}

# Provider instances are cached PER Config (not process-globally by name): a
# provider binds its Config's base_url / price table / env-derived key at
# construction, so a bare name->instance cache would hand a second Config (a
# different project under LangGraph Studio, a resumed/degraded run, or the next
# unit test) the FIRST Config's stale provider. A WeakKeyDictionary keyed on the
# Config keeps the intended one-instance-per-run caching while isolating configs
# and auto-evicting when a Config is garbage-collected (no id() recycling risk).
_caches: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def get_provider(cfg, name: str) -> LLMProvider:
    cache = _caches.get(cfg)
    if cache is None:
        cache = {}
        _caches[cfg] = cache
    inst = cache.get(name)
    if inst is None:
        pcfg = cfg.providers.get(name)
        if pcfg is None:
            raise ValueError(f"Unknown provider: {name}")
        cls = PROVIDER_TYPES.get(pcfg.type)
        if cls is None:
            raise ValueError(f"Unknown provider type: {pcfg.type}")
        inst = cls(name, pcfg, cfg)
        cache[name] = inst
    return inst


__all__ = ["LLMProvider", "LLMResult", "get_provider", "PROVIDER_TYPES"]
