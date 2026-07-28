"""OpenAI-compatible endpoint provider (CometAPI, OpenRouter, vLLM, ...).

Cost is derived from the project's worker_models price table (the aggregator
response usually doesn't carry pricing).
"""

from __future__ import annotations

import logging
import os
import re

from openai import AsyncOpenAI

from ..core.errors import OrchestratorError
from .base import LLMProvider, LLMResult

log = logging.getLogger("orchestrator.openai_compatible")

# Sampling/generation knobs we forward to chat.completions.create(). A stray key
# outside this set is dropped BEFORE the call (an unknown kwarg is a client-side
# TypeError otherwise); a whitelisted key the endpoint still rejects at runtime
# (e.g. `seed` on a provider that doesn't implement it) is dropped by the
# tolerant 400 retry in _chat. `extra_body` is the escape hatch for
# provider-specific fields (top_k, min_p, repetition_penalty, ...).
_SAMPLING_KEYS = frozenset({
    "temperature", "top_p", "frequency_penalty", "presence_penalty", "seed",
    "stop", "max_tokens", "max_completion_tokens", "logit_bias", "n",
    "logprobs", "top_logprobs", "response_format", "extra_body",
})


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
                       cwd: str | None = None,
                       params: dict | None = None,
                       session: str | None = None,
                       effort: str | None = None,
                       allowed_tools: str | None = None,
                       mcp_config: str | None = None) -> LLMResult:
        # `session` is ignored: a chat-completions endpoint is stateless, so the
        # full messages array IS the context (and session_active() says False, so
        # callers never abbreviate on this provider). `effort` is ignored too —
        # the cheap worker models have no reasoning-effort dial, and callers pass
        # None for them anyway (RunContext.worker_effort).
        return await self._chat(model, [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ], params)

    async def complete_chat(self, *, model: str, system: str,
                            messages: list[dict], cwd: str | None = None,
                            params: dict | None = None,
                            session: str | None = None,
                            effort: str | None = None,
                            allowed_tools: str | None = None,
                            mcp_config: str | None = None) -> LLMResult:
        """Pass the native OpenAI messages array through so the endpoint's
        automatic prefix caching sees a byte-stable prefix across turns.
        `session`/`effort` are accepted for signature parity and ignored."""
        wire = [{"role": "system", "content": system}] + [
            {"role": m["role"], "content": m["content"]} for m in messages
            if m.get("role") != "system"]
        return await self._chat(model, wire, params)

    async def _chat(self, model: str, messages: list[dict],
                    params: dict | None = None) -> LLMResult:
        # Only forward recognized sampling keys; a whitelisted key the endpoint
        # rejects at runtime (HTTP 400 naming the field) is dropped and retried
        # so one unsupported knob (e.g. `seed`) never fails the whole candidate.
        attempted = _safe_params(params)
        while True:
            try:
                resp = await self.client.chat.completions.create(
                    model=model, messages=messages, **attempted)
                break
            except Exception as e:  # surfaced as a candidate failure, not a crash
                rejected = ([k for k in attempted if _names_param(e, k)]
                            if getattr(e, "status_code", None) == 400 else [])
                if not rejected:
                    raise OrchestratorError(
                        f"{self.name}/{model} call failed: {e}") from e
                for k in rejected:
                    attempted.pop(k, None)
                log.warning("%s/%s rejected sampling param(s) %s; retrying without",
                            self.name, model, ", ".join(rejected))

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


def _safe_params(params: dict | None) -> dict:
    """Keep only recognized sampling keys; warn on anything dropped (a stray
    key would raise a client-side TypeError inside create())."""
    if not params:
        return {}
    safe: dict = {}
    unknown: list[str] = []
    for k, v in dict(params).items():
        if k in _SAMPLING_KEYS:
            safe[k] = v
        else:
            unknown.append(k)
    if unknown:
        log.warning("dropping unrecognized sampling param(s): %s", ", ".join(unknown))
    return safe


def _names_param(err: Exception, key: str) -> bool:
    """True if the error text names `key` as a standalone token — used to drop a
    param the endpoint rejected (e.g. \"Unsupported parameter: 'seed'\") and retry."""
    return re.search(rf"\b{re.escape(key)}\b", str(err)) is not None


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
