"""Provider registry. Adding a provider type = one class + one registry line."""

from __future__ import annotations

from .base import LLMProvider, LLMResult
from .claude_cli import ClaudeCliProvider
from .codex_cli import CodexCliProvider
from .openai_compatible import OpenAICompatibleProvider

PROVIDER_TYPES: dict[str, type[LLMProvider]] = {
    "claude_cli": ClaudeCliProvider,
    "codex_cli": CodexCliProvider,
    "openai_compatible": OpenAICompatibleProvider,
}

_instances: dict[str, LLMProvider] = {}


def get_provider(cfg, name: str) -> LLMProvider:
    if name not in _instances:
        pcfg = cfg.providers.get(name)
        if pcfg is None:
            raise ValueError(f"Unknown provider: {name}")
        ptype = pcfg.type
        cls = PROVIDER_TYPES.get(ptype)
        if cls is None:
            raise ValueError(f"Unknown provider type: {ptype}")
        _instances[name] = cls(name, pcfg, cfg)
    return _instances[name]


__all__ = ["LLMProvider", "LLMResult", "get_provider", "PROVIDER_TYPES"]
