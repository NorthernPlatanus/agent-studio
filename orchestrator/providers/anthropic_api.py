"""Anthropic **Messages** API provider — the metered half of the Claude tier.

`claude_cli` already talks to these models, but through the Claude Code CLI on a
subscription: no API key, no cash cost, one Node runtime per call and a hard
concurrency ceiling because of it. This class is the same models over HTTP,
billed per token. Both ship so the operator can move a role or a worker between
"quota I already pay for" and "cash I can parallelize" with a one-line preset
change — which is the entire point of the preset layer.

Four things the Messages API does differently from the OpenAI shapes, and each
one is a silent failure if it is got wrong:

    | thing         | chat completions / responses  | messages                    |
    |---------------|-------------------------------|-----------------------------|
    | system prompt | a `system` turn in the array  | TOP-LEVEL `system=` param   |
    | output cap    | optional `max_tokens`         | **REQUIRED** `max_tokens=`  |
    | reasoning     | `reasoning_effort` / `.effort`| `thinking={type, budget}`   |
    | usage         | prompt_/completion_ inclusive | `input_tokens` EXCLUDES the |
    |               | of cached prefix              | two cache buckets           |

The usage difference is the one that quietly corrupts the ledger: on this API
`input_tokens` counts only the uncached remainder, and the cached prefix is
reported separately as `cache_read_input_tokens` / `cache_creation_input_tokens`.
Reading the bare `input_tokens` would understate the prompt weight of exactly the
warm-cache calls this codebase is built to produce. `claude_cli.cache_tokens`
already parses this shape (the CLI mirrors the API), so it is imported rather
than reinterpreted — one reading of these four fields, not two that can drift.

**This provider deliberately never attaches tools.** Workers here answer in the
plain-text block protocol (`<file>` / `<edit>` blocks parsed by
`ops/patch.py:parse_worker_response`); every write is routed through
`ops.patch.apply_response` so the spec's `files_write` allowlist is enforceable.
A `tools=` block would hand the model a way around that. Leave it off.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from ..core.errors import OrchestratorError
from ..core.presets import as_dict, resolve_entry, section
from ..ops.pricing import build_price_table, price
from ._openai_shared import _safe_params
from .base import LLMProvider, LLMResult
from .claude_cli import cache_tokens

log = logging.getLogger("orchestrator.anthropic_api")

# Sampling knobs forwarded to messages.create(). `_safe_params` lives in
# `_openai_shared` but is not OpenAI-specific — it is a whitelist filter that
# warns about what it drops — so it is imported rather than copied. The SET is
# what differs: `top_k` and `stop_sequences` exist here and nowhere else in the
# repo, `frequency_penalty`/`presence_penalty`/`logit_bias`/`seed` do not exist
# at all, and `max_tokens` is in the list only so that a profile carrying it
# does NOT trip the "unrecognized param" warning — it is popped straight back
# out below and sent as the required top-level argument.
_SAMPLING_KEYS = frozenset({
    "temperature", "top_p", "top_k", "stop_sequences", "metadata",
    "max_tokens", "extra_body",
})

#: Params extended thinking FORBIDS. The API 400s on either one when a
#: `thinking` block is present — see `_thinking` for why they are stripped
#: proactively instead of being discovered from the error.
_THINKING_FORBIDS = ("temperature", "top_p")

#: effort level -> `thinking.budget_tokens`. A table rather than a passthrough
#: because effort is the vocabulary every other tier speaks (`roles.<role>.effort`,
#: a worker_models entry, a preset) and this API wants a token count; without the
#: mapping, moving a role from claude_cli to this provider would mean rewriting
#: the config in a second unit. Override per preset/entry/tier with
#: `thinking_budget_tokens` (see `_resolved_int`).
#:
#: `none` is present and maps to no thinking at all, so a preset can move between
#: this provider and openai_responses (whose level set includes `none`) without
#: the level suddenly becoming invalid.
THINKING_BUDGETS: dict[str, int | None] = {
    "none": None,
    "low": 4096,
    "medium": 8192,
    "high": 16384,
    "xhigh": 32768,
    "max": 65536,
}

EFFORT_LEVELS = tuple(THINKING_BUDGETS)

#: The API's own floor for `thinking.budget_tokens`. Below this the request is
#: rejected, so a budget clamped under it (a very small `max_tokens`) means
#: "thinking is not possible on this call" rather than "think a little".
_MIN_THINKING_BUDGET = 1024

#: Fallback when nothing anywhere declares `max_tokens`. Mirrors the default in
#: `ops/budget.py` for the same key, so the pre-flight estimate and the actual
#: cap agree when neither is configured.
_DEFAULT_MAX_TOKENS = 8000


class AnthropicApiProvider(LLMProvider):
    type = "anthropic_api"
    # `effort` reaches the model here, as a thinking budget (see THINKING_BUDGETS).
    supports_effort = True

    def __init__(self, name, pcfg, cfg):
        super().__init__(name, pcfg, cfg)
        key_env = pcfg.get("api_key_env", "")
        self._api_key = os.environ.get(key_env, "") if key_env else ""
        if not self._api_key:
            # Same degrade-to-a-401 shape as the other HTTP providers, and the
            # same answer: core/validate.py escalates this to a startup ERROR
            # when an active binding actually routes here, so the warning is
            # only ever seen for a provider nothing calls.
            log.warning("provider %s: env %s is empty (calls will fail)", name, key_env)
        self.prices: dict[str, dict] = build_price_table(cfg, name)
        # model id -> the resolved preset/entry that binds it, so per-model
        # settings written on a PRESET (`max_tokens`, `thinking_budget_tokens`)
        # can reach a call site that only ever receives a model id. Same
        # recovery `build_price_table` does for prices, and for the same reason:
        # nothing between the config and `complete()` carries the entry itself.
        self.bindings: dict[str, dict] = _binding_table(cfg, name)
        self._budget_max_tokens = int(
            section(cfg, "budget").get("assumed_max_output_tokens",
                                       _DEFAULT_MAX_TOKENS) or _DEFAULT_MAX_TOKENS)
        # Built on first use, NOT here: the SDK import is module-local so that a
        # machine without the `anthropic` package still starts, still validates
        # its config, and still runs every other backend. See `_client`.
        self.client = None

    def _client(self):
        """The AsyncAnthropic client, constructed on first call.

        The import is deliberately inside the function. `anthropic` is a main
        dependency (a shipped preset must work out of the box), but a top-level
        import would turn "this one optional backend is not installed" into an
        ImportError at `orchestrator.providers` import time — i.e. every other
        provider, the CLI, and the config validator all fail to start because of
        a backend nobody in this project may even reference. Degrading to a
        clear OrchestratorError at the moment of the call keeps the blast radius
        to the binding that actually asked for it.
        """
        if self.client is not None:
            return self.client
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:
            raise OrchestratorError(
                f"provider {self.name} (anthropic_api) needs the `anthropic` "
                f"package, which is not installed in this environment. Install "
                f"it (`pip install 'anthropic>=0.40'`, or reinstall the "
                f"orchestrator so the [project] dependency is picked up), or "
                f"bind that role/worker to a preset on another provider."
            ) from e
        self.client = AsyncAnthropic(
            # Optional: this provider's natural default IS api.anthropic.com, so
            # a config that omits base_url still works (a gateway fronting the
            # Messages API can still set one).
            base_url=self.pcfg.get("base_url") or None,
            api_key=self._api_key or "missing-key",
            timeout=float(self.pcfg.get("timeout_s", 300)),
            max_retries=2,
        )
        return self.client

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
        # ignored too — this API is stateless, the request IS the context, and
        # session_active() says False so callers never abbreviate a turn.
        # `allowed_tools`/`mcp_config` are ignored on purpose: see the class
        # docstring on why no tools are ever attached.
        return await self._message(model, system,
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
        """Pass the turns through as the native `messages` array.

        A system turn inside `messages` is DROPPED rather than forwarded: on this
        API the system prompt is the top-level `system=` parameter, and the array
        accepts only `user`/`assistant`. Forwarding one would be a 400, and
        forwarding it *as well as* the top-level parameter would pay for it
        twice and break the byte-stable prefix prompt caching keys on.
        """
        wire = [{"role": m["role"], "content": m["content"]} for m in messages
                if m.get("role") != "system"]
        return await self._message(model, system, wire, params, effort)

    async def _message(self, model: str, system: str, messages: list[dict],
                       params: dict | None, effort: str | None) -> LLMResult:
        attempted = _safe_params(params, _SAMPLING_KEYS)
        # REQUIRED by the API — there is no server-side default and a call
        # without it is a 400, so this resolves to a number on every path.
        max_tokens = self._max_tokens(model, attempted.pop("max_tokens", None))

        request: dict[str, Any] = {"messages": messages, "max_tokens": max_tokens}
        if system:
            # Top level, not a message. This is the single most common way to
            # port an OpenAI-shaped call to this API and get a 400 for it.
            request["system"] = system

        thinking = self._thinking(model, effort, max_tokens)
        if thinking is not None:
            request["thinking"] = thinking
            # Extended thinking forbids BOTH of these outright. Stripped here,
            # loudly, rather than left for the API to reject: this is the Luna
            # trap in a second costume — a best-of-N pool whose candidates differ
            # only by `temperature` collapses into N identical calls at N times
            # the price, and the operator has to be able to SEE that in the log.
            # `core.validate._check_pool_diversity` warns about the same shape at
            # startup; this is the runtime half of the same statement.
            dropped = [k for k in _THINKING_FORBIDS if k in attempted]
            for k in dropped:
                attempted.pop(k, None)
            if dropped:
                log.warning(
                    "%s/%s: extended thinking is enabled, which forbids %s — "
                    "dropping them from this call. Sampling params are NOT "
                    "best-of-N diversity here: candidates that differ only by "
                    "temperature become identical requests. Differentiate by "
                    "`effort` or `approach:` instead.",
                    self.name, model, ", ".join(dropped))

        try:
            # Unbounded by default (see LLMProvider.default_max_concurrency);
            # `providers.<name>.max_concurrency` bounds it when the account's own
            # rate limit is the tighter constraint.
            async with self._concurrency_gate():
                resp = await self._client().messages.create(
                    model=model, **request, **attempted)
        except OrchestratorError:
            raise                      # the missing-SDK message, already clear
        except Exception as e:         # surfaced as a candidate failure, not a crash
            # No tolerant-400-and-retry here, unlike the OpenAI providers. The
            # one param collision this API actually has (thinking vs
            # temperature/top_p) is stripped proactively above, so a 400 that
            # gets this far is a real problem with the request and re-issuing it
            # minus a guessed key would just pay for a second failure.
            raise OrchestratorError(f"{self.name}/{model} call failed: {e}") from e

        usage = _usage_dict(getattr(resp, "usage", None))
        # `input_tokens` on this API EXCLUDES the cache buckets; cache_tokens
        # sums all three so the recorded prompt weight is the real one and
        # hit + miss == input_tokens, the invariant every other tier maintains.
        in_tok, hit, miss = cache_tokens(usage)
        out_tok = int(usage.get("output_tokens", 0) or 0)
        cost = price(self.prices.get(model), in_tok, out_tok, hit, miss)
        return LLMResult(text=_output_text(resp), input_tokens=in_tok,
                         output_tokens=out_tok, cost_usd=cost, model=model,
                         raw=resp, cache_hit_tokens=hit, cache_miss_tokens=miss)

    # ---- request pieces ---------------------------------------------------

    def _max_tokens(self, model: str, from_params: Any) -> int:
        """The required output cap, in precedence order.

        1. `params.max_tokens` on the worker_models entry — the per-candidate
           knob, and the same one `ops.budget.estimate_and_check` already reads
           for the pre-flight estimate, so the guard and the call agree.
        2. `max_tokens` on the preset/entry that binds this model id.
        3. `providers.<name>.max_tokens` — the tier-wide default.
        4. `budget.assumed_max_output_tokens`.

        There is a default at all — rather than a hard failure — because a role
        binding never passes `params` (see nodes/planner.py), so a preset-only
        route has no way to supply one per call. `core.validate` still asks for
        an explicit value on any anthropic_api binding, since 8000 tokens is a
        guess about a model the operator knows better than we do.
        """
        for value in (from_params,
                      self.bindings.get(model, {}).get("max_tokens"),
                      self.pcfg.get("max_tokens")):
            resolved = _positive_int(value)
            if resolved:
                return resolved
        return self._budget_max_tokens

    def _thinking(self, model: str, effort: str | None,
                  max_tokens: int) -> dict | None:
        """The `thinking` block for this call, or None to leave thinking off.

        None (not `{"type": "disabled"}`) so a call with no effort sends no
        thinking key at all — the model's own default behavior, byte-identical
        to what it received before this provider existed.
        """
        level = effort or self.pcfg.get("effort")
        if not level:
            return None
        if level not in THINKING_BUDGETS:
            # Validated rather than passed through, for the same reason
            # claude_cli and openai_responses validate theirs: an unrecognized
            # level is a config typo, and silently running at the model's
            # default depth while the config, the UI and the operator all
            # believe otherwise is the exact failure the effort plumbing exists
            # to end.
            raise OrchestratorError(
                f"effort={level!r} is not a valid level for {self.name}/{model}. "
                f"Use one of {', '.join(EFFORT_LEVELS)} (set per role as "
                f"roles.<role>.effort, per candidate on a worker_models entry or "
                f"its preset, or tier-wide as providers.{self.name}.effort).")

        budget = _positive_int(self.bindings.get(model, {}).get(
            "thinking_budget_tokens")) or _positive_int(
            self.pcfg.get("thinking_budget_tokens")) or THINKING_BUDGETS[level]
        if not budget:
            return None               # effort: none — thinking explicitly off

        # HARD API CONSTRAINT: budget_tokens must be strictly less than
        # max_tokens (the thinking tokens are spent out of the same output
        # allowance). Clamped here with a warning rather than left to 400,
        # because the default budget for effort=max (65536) exceeds the default
        # max_tokens (8000) — i.e. the collision is the NORMAL case, not an edge
        # one, and an operator who set `effort: max` should be told the depth
        # they asked for is not what they got.
        if budget >= max_tokens:
            clamped = max_tokens - 1
            if clamped < _MIN_THINKING_BUDGET:
                # Below the API's own floor there is no legal budget at all, so
                # the only correct request is one with no thinking in it.
                log.warning(
                    "%s/%s: effort=%r wants a %d-token thinking budget but "
                    "max_tokens is only %d, leaving less than the API's %d-token "
                    "minimum — sending NO thinking block. Raise max_tokens "
                    "(params.max_tokens, the preset, or "
                    "providers.%s.max_tokens) to actually get extended thinking.",
                    self.name, model, level, budget, max_tokens,
                    _MIN_THINKING_BUDGET, self.name)
                return None
            log.warning(
                "%s/%s: thinking budget_tokens=%d is >= max_tokens=%d, which the "
                "API rejects (thinking is spent out of the output allowance); "
                "clamping to %d. Raise max_tokens to get the full effort=%r "
                "depth you asked for.",
                self.name, model, budget, max_tokens, clamped, level)
            budget = clamped
        return {"type": "enabled", "budget_tokens": budget}


# ---- module helpers ---------------------------------------------------------

def _positive_int(value: Any) -> int | None:
    """`value` as a positive int, or None for absent/zero/garbage. Garbage is
    logged rather than raised: a typo'd cap must not kill a run that has a
    perfectly good fallback one line down."""
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        log.warning("ignoring non-integer token count %r", value)
        return None
    return number if number > 0 else None


def _binding_table(cfg: Any, provider_name: str) -> dict[str, dict]:
    """model id -> the resolved preset / worker_models / roles entry that binds it.

    The same recovery `ops.pricing.build_price_table` performs for prices, minus
    the "skip entries with no prices" rule: what is being recovered here
    (`max_tokens`, `thinking_budget_tokens`) is not a price and an entry that
    declares one but no prices is perfectly ordinary.

    Later sources win, so a worker_models entry overrides the preset it names.
    Two entries for the SAME model id with different caps collapse to the last
    one — the same caveat the price table carries, and the same answer: per-call
    `params.max_tokens` is the way to differ per candidate.
    """
    table: dict[str, dict] = {}

    def add(entry: Any, provider_default: str | None = None) -> None:
        resolved = resolve_entry(cfg, entry)
        if not resolved:
            return
        provider = resolved.get("provider") or provider_default
        model = resolved.get("model")
        if provider != provider_name or not model:
            return
        table[str(model)] = resolved

    for preset in section(cfg, "presets").values():
        add(preset)
    for entry in section(cfg, "worker_models").values():
        add(entry)

    roles = section(cfg, "roles")
    # A role with `provider: null` inherits roles.smart_provider (see
    # RunContext.role_target), so resolve that here too.
    smart = roles.get("smart_provider") or "claude_cli"
    for entry in roles.values():
        data = as_dict(entry)
        if not data.get("model") and not data.get("preset"):
            continue      # skips smart_provider (a string) and roles.worker (a pool)
        add(entry, provider_default=smart)
    return table


def _usage_dict(usage) -> dict:
    """The four token counts as a plain dict, whatever shape `usage` arrived in.

    `claude_cli.cache_tokens` — the single reading of this field set in the repo
    — takes a dict, because the CLI hands it decoded JSON. The SDK hands back a
    pydantic model. Normalizing here is what lets the mapping stay in one place
    instead of being re-derived from the same four field names a second time.
    """
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    return {key: getattr(usage, key, 0) or 0 for key in (
        "input_tokens", "output_tokens",
        "cache_creation_input_tokens", "cache_read_input_tokens")}


def _output_text(resp) -> str:
    """The assistant's text, from the `content` block list.

    Only `text` blocks are joined. A response with extended thinking enabled ALSO
    carries `thinking` blocks (with a `.thinking` attribute, not `.text`), and
    those must not end up in the string the worker's block protocol is parsed
    out of — a reasoning trace containing something that looks like a `<file>`
    block would otherwise be applied as a patch.
    """
    parts: list[str] = []
    for block in (getattr(resp, "content", None) or []):
        kind = getattr(block, "type", None)
        if kind is None and isinstance(block, dict):
            kind, block = block.get("type"), _AttrView(block)
        if kind and kind != "text":
            continue
        value = getattr(block, "text", None)
        if isinstance(value, str):
            parts.append(value)
    return "".join(parts)


class _AttrView:
    """Attribute access over a dict, so `_output_text` can read a raw-dict
    response (an older SDK, a fake, a replayed fixture) with one code path."""

    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name):
        return self._data.get(name)
