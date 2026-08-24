"""OpenAI **Responses** API provider — the reasoning path (GPT-5.6 / Luna).

Why this is a second class rather than a flag on `openai_compatible`: the two
APIs disagree on all four things a provider has to get right.

    | thing            | chat completions        | responses                    |
    |------------------|-------------------------|------------------------------|
    | request shape    | `messages=[...]`        | `instructions=` + `input=`   |
    | reasoning        | `reasoning_effort=`     | `reasoning={"effort": ...}`  |
    | output cap       | `max_tokens`            | `max_output_tokens`          |
    | usage            | prompt_/completion_     | input_/output_tokens, plus   |
    |                  | tokens                  | *_tokens_details buckets     |

Branching all four inside one class would make the DeepSeek path (the one that
has been carrying every worker call to date) harder to reason about for no gain.
`openai_compatible` stays exactly as it was and keeps serving
CometAPI/DeepSeek/OpenRouter; the mechanics both need live in `_openai_shared`.

The concrete bug this class fixes: `openai_compatible` **ignores `effort`
entirely**, so a worker bound to "Luna at high reasoning" was issuing a plain
call at the model's default effort and charging for it. Everything above the
provider carried the level correctly; the last hop dropped it.

**This provider deliberately never attaches tools.** Workers here answer in the
plain-text block protocol (`<file>` / `<edit>` blocks parsed by
`ops/patch.py:parse_worker_response`), so there is nothing for a function tool to
do — and the GPT-5.6 family rejects function tools combined with reasoning on
/v1/chat/completions with a 400 that points at /v1/responses. Attaching tools
here would walk that failure right back into the class that exists to avoid it.
Leave `tools=` off.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

from openai import AsyncOpenAI

from ..core.errors import OrchestratorError
from ..ops.pricing import build_price_table, price
from ._openai_shared import _cache_tokens, _names_param, _safe_params
from .base import LLMProvider, LLMResult

log = logging.getLogger("orchestrator.openai_responses")

# Sampling knobs forwarded to responses.create(). NOT the chat-completions set:
# `max_tokens` and `logit_bias` do not exist on this API (the cap is
# `max_output_tokens`), and a profile written for the DeepSeek era carries
# `max_tokens` in `params` — dropping it here with a warning is the whole point
# of keeping a separate whitelist, rather than sending it and finding out from a
# 400. `extra_body` stays as the escape hatch for anything the SDK version in
# use does not model yet.
_SAMPLING_KEYS = frozenset({
    "temperature", "top_p", "max_output_tokens", "truncation", "store",
    "metadata", "text", "top_logprobs", "parallel_tool_calls", "extra_body",
})

#: Accepted `reasoning.effort` levels for the GPT-5.6 family (default: medium).
#: Validated rather than passed through, for the same reason claude_cli
#: validates its own: an unrecognized level is a config typo, and the tolerant
#: retry below would turn the resulting 400 into a silent no-effort call — the
#: exact failure this provider was written to end.
EFFORT_LEVELS = ("none", "low", "medium", "high", "xhigh", "max")


class OpenAIResponsesProvider(LLMProvider):
    type = "openai_responses"
    # The reason this class exists: `effort` reaches the model here, as
    # `reasoning={"effort": <level>}`.
    supports_effort = True

    def __init__(self, name, pcfg, cfg):
        super().__init__(name, pcfg, cfg)
        key_env = pcfg.get("api_key_env", "")
        api_key = os.environ.get(key_env, "") if key_env else ""
        if not api_key:
            # Same degrade-to-a-401 shape as openai_compatible, and the same
            # answer: core/validate.py escalates this to a startup ERROR when an
            # active binding actually routes here, so the warning is only ever
            # seen for a provider nothing calls.
            log.warning("provider %s: env %s is empty (calls will fail)", name, key_env)
        self.client = AsyncOpenAI(
            # base_url is optional here: unlike the aggregators, this provider's
            # natural default IS api.openai.com, so a config that omits it still
            # works instead of failing on a missing key.
            base_url=pcfg.get("base_url") or None,
            api_key=api_key or "missing-key",
            timeout=float(pcfg.get("timeout_s", 300)),
            max_retries=2,
        )
        # model id -> pricing entry, from presets ∪ worker_models ∪ roles. This
        # is the provider whose pricing has a cliff in it (>272K input re-prices
        # the WHOLE request), so the shared table in ops/pricing is doing real
        # work here rather than multiplying two numbers.
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
        # `on_progress` is ignored: one non-streaming HTTP response, so there is
        # no in-flight event to report and callers get the documented "never
        # called" behavior rather than a fabricated heartbeat. `session` is
        # ignored too — this call does not set `store`/`previous_response_id`, so
        # the request IS the context, and session_active() says False so callers
        # never abbreviate. `allowed_tools`/`mcp_config` are ignored on purpose:
        # see the class docstring on why no tools are ever attached.
        return await self._respond(model, system,
                                   [{"role": "user", "content": user}],
                                   params, effort)

    async def complete_chat(self, *, model: str, system: str,
                            messages: list[dict], cwd: str | None = None,
                            params: dict | None = None,
                            session: str | None = None,
                            effort: str | None = None,
                            allowed_tools: str | None = None,
                            mcp_config: str | None = None,
                            on_progress: Callable[[dict], None] | None = None
                            ) -> LLMResult:
        """Pass the turns through as the `input` array so the endpoint's
        automatic prefix caching sees a byte-stable prefix across turns.

        A system turn inside `messages` is dropped rather than forwarded: on this
        API the system prompt is the top-level `instructions` parameter, and
        sending it twice would both waste input tokens and break the stable
        prefix the cache keys on."""
        wire = [{"role": m["role"], "content": m["content"]} for m in messages
                if m.get("role") != "system"]
        return await self._respond(model, system, wire, params, effort)

    async def _respond(self, model: str, system: str, input_items: list[dict],
                       params: dict | None, effort: str | None) -> LLMResult:
        attempted = _safe_params(params, _SAMPLING_KEYS)
        request: dict = {"input": input_items}
        if system:
            request["instructions"] = system

        # Per-call effort wins over the tier-wide providers.<name>.effort, the
        # same precedence claude_cli uses: the roles are not equally hard, and
        # the per-role/per-candidate value is the more specific statement.
        level = effort or self.pcfg.get("effort")
        if level:
            if level not in EFFORT_LEVELS:
                raise OrchestratorError(
                    f"effort={level!r} is not a valid level for "
                    f"{self.name}/{model}. Use one of {', '.join(EFFORT_LEVELS)} "
                    f"(set per role as roles.<role>.effort, per candidate on a "
                    f"worker_models entry or its preset, or tier-wide as "
                    f"providers.{self.name}.effort).")
            # Deliberately NOT part of `attempted`: the tolerant-400 retry below
            # drops whatever the endpoint names, and letting it drop `reasoning`
            # would silently downgrade the call to default effort and bill for it
            # — indistinguishable, from the ledger, from the bug this class
            # fixes. A 400 on `reasoning` should fail the candidate loudly.
            request["reasoning"] = {"effort": str(level)}

        while True:
            try:
                # Unbounded by default (see LLMProvider.default_max_concurrency);
                # `providers.<name>.max_concurrency` bounds it when the account's
                # own rate limit is the tighter constraint.
                async with self._concurrency_gate():
                    resp = await self.client.responses.create(
                        model=model, **request, **attempted)
                break
            except Exception as e:  # surfaced as a candidate failure, not a crash
                # Same tolerant-400 as openai_compatible._chat: one unsupported
                # knob must not fail the whole candidate. It matters MORE here —
                # the reasoning models reject `temperature` outright, and the
                # shipped Luna presets are diversified by effort/approach for
                # exactly that reason.
                rejected = ([k for k in attempted if _names_param(e, k)]
                            if getattr(e, "status_code", None) == 400 else [])
                if not rejected:
                    raise OrchestratorError(
                        f"{self.name}/{model} call failed: {e}") from e
                for k in rejected:
                    attempted.pop(k, None)
                log.warning("%s/%s rejected sampling param(s) %s; retrying without",
                            self.name, model, ", ".join(rejected))

        text = _output_text(resp)
        usage = getattr(resp, "usage", None)
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        # REPORTED AS-IS. `output_tokens` on this API already INCLUDES
        # `output_tokens_details.reasoning_tokens` — the reasoning trace is
        # billed as output, it is not a fourth bucket. Adding the two would
        # inflate every recorded cost by the reasoning share (large by
        # construction at effort=high) and the pre-flight budget guard reads the
        # same ledger, so the run would throttle itself on money it never spent.
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        hit, miss = _cache_tokens(usage, in_tok)
        cost = price(self.prices.get(model), in_tok, out_tok, hit, miss)
        reasoning_tok = _reasoning_tokens(usage)
        if reasoning_tok:
            # Visible for diagnosis (how much of the output was thinking), never
            # added to a total. The count also rides along on LLMResult.raw.
            log.debug("%s/%s: %d of %d output tokens were reasoning",
                      self.name, model, reasoning_tok, out_tok)
        return LLMResult(text=text, input_tokens=in_tok, output_tokens=out_tok,
                         cost_usd=cost, model=model, raw=resp,
                         cache_hit_tokens=hit, cache_miss_tokens=miss)


def _output_text(resp) -> str:
    """The assistant's text, preferring the SDK's flattened `output_text`.

    `output_text` is a convenience property on the SDK's response object; a fake,
    an older SDK, or a raw dict may not have it, and a response whose only items
    are reasoning summaries legitimately has none. The fallback walks
    `output[].content[].text` — the shape `output_text` itself concatenates — so
    a missing convenience property costs the call nothing.
    """
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text:
        return text
    parts: list[str] = []
    for item in (getattr(resp, "output", None) or []):
        for chunk in (getattr(item, "content", None) or []):
            value = getattr(chunk, "text", None)
            if isinstance(value, str):
                parts.append(value)
    return "".join(parts)


def _reasoning_tokens(usage) -> int:
    """`usage.output_tokens_details.reasoning_tokens`, 0 when absent. For
    reporting only — never summed into output_tokens (see _respond)."""
    details = getattr(usage, "output_tokens_details", None)
    if isinstance(details, dict):
        return int(details.get("reasoning_tokens") or 0)
    return int(getattr(details, "reasoning_tokens", 0) or 0)
