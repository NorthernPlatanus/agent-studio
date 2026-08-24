"""OpenAI-compatible endpoint provider (CometAPI, OpenRouter, vLLM, ...).

Cost is derived from the config's price table (the aggregator response usually
doesn't carry pricing) — see ops/pricing.py, which builds it from
presets ∪ worker_models ∪ roles and owns the cache-tier and long-context rules.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

from openai import AsyncOpenAI

from ..core.errors import OrchestratorError
from ..ops.pricing import build_price_table, price
from ._openai_shared import (CHAT_SAMPLING_KEYS, _cache_tokens, _names_param,
                             _safe_params)
from .base import LLMProvider, LLMResult

log = logging.getLogger("orchestrator.openai_compatible")

# The forwarded-param whitelist and the three helpers below it moved to
# _openai_shared when openai_responses needed the same mechanics; the comment
# explaining each one moved with it. Re-exported under the old name because it
# reads as this provider's own vocabulary at the call site (and because
# `_safe_params` defaults to exactly this set).
_SAMPLING_KEYS = CHAT_SAMPLING_KEYS


class OpenAICompatibleProvider(LLMProvider):
    type = "openai_compatible"
    # /v1/chat/completions has no reasoning dial for the models this endpoint
    # fronts, so a level passed here is dropped — loudly now (see
    # LLMProvider._warn_unsupported_effort) instead of by a code comment alone.
    supports_effort = False

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
        # model id -> pricing entry. Built by ops.pricing from presets ∪
        # worker_models ∪ roles: this table used to be built here from
        # worker_models alone, so a planner or reviewer pointed at an HTTP
        # provider recorded every call at $0.00 — a whole tier of real spend
        # missing from the ledger and from the budget guard that reads it.
        self.prices: dict[str, dict] = build_price_table(cfg, name)

    async def complete(self, *, model: str, system: str, user: str,
                       cwd: str | None = None,
                       params: dict | None = None,
                       session: str | None = None,
                       effort: str | None = None,
                       allowed_tools: str | None = None,
                       mcp_config: str | None = None,
                       on_progress: Callable[[dict], None] | None = None
                       ) -> LLMResult:
        # `on_progress` is ignored: this provider awaits one non-streaming HTTP
        # response, so there is no in-flight event to report. Callers get the
        # documented "never called" behavior rather than a fabricated heartbeat.
        # `session` is ignored: a chat-completions endpoint is stateless, so the
        # full messages array IS the context (and session_active() says False, so
        # callers never abbreviate on this provider). `effort` is ignored too —
        # the cheap worker models have no reasoning-effort dial, and callers pass
        # None for them anyway (RunContext.worker_effort) — but when one does not,
        # the drop is now announced once per (provider, model) rather than
        # inferred from reading this comment.
        self._warn_unsupported_effort(model, effort)
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
                            mcp_config: str | None = None,
                            on_progress: Callable[[dict], None] | None = None
                            ) -> LLMResult:
        """Pass the native OpenAI messages array through so the endpoint's
        automatic prefix caching sees a byte-stable prefix across turns.
        `session`/`effort`/`on_progress` are accepted for parity and ignored —
        `effort` audibly so (a warning per provider/model), since a dropped
        reasoning level is invisible in the result."""
        self._warn_unsupported_effort(model, effort)
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
                # Unbounded by default (see LLMProvider.default_max_concurrency)
                # — the endpoint does its own admission control. The gate is here
                # so `providers.<name>.max_concurrency` can still bound a proxy
                # with a per-key concurrency limit of its own.
                async with self._concurrency_gate():
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
        # Cache-aware: a hit is priced at the entry's cache_read rate when it
        # declares one. With no cache rates configured this is exactly the old
        # flat `in_tok * pin + out_tok * pout`.
        cost = price(self.prices.get(model), in_tok, out_tok, hit, miss)
        return LLMResult(text=text, input_tokens=in_tok, output_tokens=out_tok,
                         cost_usd=cost, model=model, raw=resp,
                         cache_hit_tokens=hit, cache_miss_tokens=miss)

