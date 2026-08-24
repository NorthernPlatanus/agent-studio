"""Helpers shared by the two OpenAI-SDK providers.

`openai_compatible` (chat completions: CometAPI/DeepSeek/OpenRouter/vLLM) and
`openai_responses` (the Responses API: the GPT-5.6 reasoning family) differ in
request shape, parameter names, reasoning carriage and usage shape — which is
why they are two classes rather than one class with four branches. What they do
NOT differ in is the three small mechanics below: whitelist the sampling params
before the call, recognize a 400 that names a param so it can be dropped and
retried, and read whatever cache-token shape the endpoint happened to use.

Those three were duplicated in `openai_compatible` alone until the Responses
provider needed them too. They live here so the second copy never gets written:
a tolerant-400 retry that drifts between the two classes would mean one provider
silently keeps failing a candidate the other recovers.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger("orchestrator.openai_shared")

# Sampling/generation knobs we forward to chat.completions.create(). A stray key
# outside this set is dropped BEFORE the call (an unknown kwarg is a client-side
# TypeError otherwise); a whitelisted key the endpoint still rejects at runtime
# (e.g. `seed` on a provider that doesn't implement it) is dropped by the
# tolerant 400 retry in _chat. `extra_body` is the escape hatch for
# provider-specific fields (top_k, min_p, repetition_penalty, ...).
CHAT_SAMPLING_KEYS = frozenset({
    "temperature", "top_p", "frequency_penalty", "presence_penalty", "seed",
    "stop", "max_tokens", "max_completion_tokens", "logit_bias", "n",
    "logprobs", "top_logprobs", "response_format", "extra_body",
})


def _safe_params(params: dict | None, allowed=None) -> dict:
    """Keep only recognized sampling keys; warn on anything dropped (a stray
    key would raise a client-side TypeError inside create()).

    `allowed` defaults to the chat-completions set for callers that predate the
    Responses provider; that provider passes its own, narrower set — the two
    APIs genuinely disagree about which knobs exist (`max_tokens` vs
    `max_output_tokens`, no `logit_bias` on Responses at all), and a shared
    whitelist would have to be the union, i.e. wrong for both.
    """
    if not params:
        return {}
    keys = CHAT_SAMPLING_KEYS if allowed is None else allowed
    safe: dict = {}
    unknown: list[str] = []
    for k, v in dict(params).items():
        if k in keys:
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
    prompt_tokens_details.cached_tokens; the Responses API's
    input_tokens_details.cached_tokens). Returns (hit, miss); 0/in_tok if absent.
    """
    if usage is None:
        return 0, 0
    hit = getattr(usage, "prompt_cache_hit_tokens", None)
    miss = getattr(usage, "prompt_cache_miss_tokens", None)
    if hit is None:
        # Chat completions call this bucket `prompt_tokens_details`; the
        # Responses API renames it `input_tokens_details` along with the token
        # counts themselves. Same field inside, so try both rather than making
        # the Responses provider carry its own near-identical copy of this.
        for attr in ("prompt_tokens_details", "input_tokens_details"):
            details = getattr(usage, attr, None)
            if isinstance(details, dict):
                hit = details.get("cached_tokens")
            elif details is not None:
                hit = getattr(details, "cached_tokens", None)
            if hit is not None:
                break
    hit = int(hit or 0)
    miss = int(miss) if miss is not None else max(in_tok - hit, 0)
    return hit, miss
