"""OpenAI-compatible endpoint provider (CometAPI, OpenRouter, vLLM, ...).

Cost is derived from the project's worker_models price table (the aggregator
response usually doesn't carry pricing).
"""

from __future__ import annotations

import logging
import os

from openai import AsyncOpenAI

from ..core.errors import OrchestratorError
from .base import LLMProvider, LLMResult

log = logging.getLogger("orchestrator.openai_compatible")


class OpenAICompatibleProvider(LLMProvider):
    type = "openai_compatible"

    def __init__(self, name, pcfg, cfg):
        super().__init__(name, pcfg, cfg)
        key_env = pcfg.get("api_key_env", "")
        api_key = os.environ.get(key_env, "") if key_env else ""
        if not api_key:
            log.warning("provider %s: env %s is empty (calls will fail)", name, key_env)
        self.client = AsyncOpenAI(
            base_url=pcfg.base_url,
            api_key=api_key or "missing-key",
            timeout=float(pcfg.get("timeout_s", 300)),
            max_retries=2,
        )
        # model id -> (in $/Mtok, out $/Mtok)
        self.prices: dict[str, tuple[float, float]] = {}
        for wm in (cfg.get("worker_models") or {}).keys():
            entry = cfg.worker_models.get(wm)
            if entry.provider == name:
                self.prices[entry.model] = (
                    float(entry.get("input_per_mtok", 0)),
                    float(entry.get("output_per_mtok", 0)),
                )

    async def complete(self, *, model: str, system: str, user: str,
                       cwd: str | None = None) -> LLMResult:
        return await self._chat(model, [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])

    async def complete_chat(self, *, model: str, system: str,
                            messages: list[dict], cwd: str | None = None) -> LLMResult:
        """Pass the native OpenAI messages array through so the endpoint's
        automatic prefix caching sees a byte-stable prefix across turns."""
        wire = [{"role": "system", "content": system}] + [
            {"role": m["role"], "content": m["content"]} for m in messages
            if m.get("role") != "system"]
        return await self._chat(model, wire)

    async def _chat(self, model: str, messages: list[dict]) -> LLMResult:
        try:
            resp = await self.client.chat.completions.create(
                model=model, messages=messages)
        except Exception as e:  # surfaced as a candidate failure, not a crash
            raise OrchestratorError(f"{self.name}/{model} call failed: {e}") from e

        text = (resp.choices[0].message.content or "") if resp.choices else ""
        usage = resp.usage
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0
        hit, miss = _cache_tokens(usage, in_tok)
        pin, pout = self.prices.get(model, (0.0, 0.0))
        cost = in_tok / 1e6 * pin + out_tok / 1e6 * pout
        return LLMResult(text=text, input_tokens=in_tok, output_tokens=out_tok,
                         cost_usd=cost, model=model, raw=resp,
                         cache_hit_tokens=hit, cache_miss_tokens=miss)


def _cache_tokens(usage, in_tok: int) -> tuple[int, int]:
    """Extract cache-hit/miss input tokens from a usage object, tolerating the
    several shapes providers use (DeepSeek prompt_cache_hit_tokens; OpenAI-style
    prompt_tokens_details.cached_tokens). Returns (hit, miss); 0/in_tok if absent.
    """
    if usage is None:
        return 0, 0
    hit = getattr(usage, "prompt_cache_hit_tokens", None)
    miss = getattr(usage, "prompt_cache_miss_tokens", None)
    if hit is None:
        details = getattr(usage, "prompt_tokens_details", None)
        if isinstance(details, dict):
            hit = details.get("cached_tokens")
        elif details is not None:
            hit = getattr(details, "cached_tokens", None)
    hit = int(hit or 0)
    miss = int(miss) if miss is not None else max(in_tok - hit, 0)
    return hit, miss
