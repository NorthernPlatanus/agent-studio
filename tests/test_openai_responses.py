"""openai_responses — the provider where `effort` actually reaches the model.

Everything asserted here is something that fails SILENTLY if it regresses:
a dropped `reasoning` block runs at default depth and bills for it, a
double-counted reasoning token inflates the ledger the budget guard reads, a
`max_tokens` left in a profile 400s the call, and a cache hit priced as fresh
input costs 10x what it should.

The OpenAI SDK is faked the way tests/test_worker_params.py fakes chat
completions: replace `p.client` wholesale with a SimpleNamespace whose
`responses.create` records its kwargs, so no HTTP client, key or network is
involved.
"""

import logging
import types

import pytest

from orchestrator.core.config import Section
from orchestrator.core.errors import OrchestratorError
from orchestrator.providers import PROVIDER_TYPES
from orchestrator.providers.openai_compatible import OpenAICompatibleProvider
from orchestrator.providers.openai_responses import OpenAIResponsesProvider


# ---- fakes ------------------------------------------------------------------

def _usage(in_tok=0, out_tok=0, cached=None, reasoning=None):
    """A Responses-shaped usage object: input_/output_tokens plus the two
    *_tokens_details buckets (absent unless the test asks for them)."""
    u = types.SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok)
    if cached is not None:
        u.input_tokens_details = types.SimpleNamespace(cached_tokens=cached)
    if reasoning is not None:
        u.output_tokens_details = types.SimpleNamespace(reasoning_tokens=reasoning)
    return u


def _resp(text="OUT", usage=None):
    return types.SimpleNamespace(output_text=text, output=[], usage=usage)


class _FakeResponses:
    def __init__(self, exc_seq=None, resp=None):
        self.calls: list[dict] = []
        self.exc_seq = list(exc_seq or [])
        self.resp = resp

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc_seq:
            exc = self.exc_seq.pop(0)
            if exc is not None:
                raise exc
        return self.resp if self.resp is not None else _resp()


class _Fake400(Exception):
    status_code = 400


#: The shipped luna_high price shape (config/default.yaml), inlined so the test
#: pins the arithmetic rather than whatever the config happens to say today.
_LUNA = {
    "provider": "openai", "model": "gpt-5.6-luna", "effort": "high",
    "input_per_mtok": 0.20, "output_per_mtok": 1.20,
    "cache_read_per_mtok": 0.02,
    "long_context": {"threshold_in_tok": 272000,
                     "input_multiplier": 2.0, "output_multiplier": 1.5},
}


def _oai(rec, presets=None, pcfg_extra=None):
    pcfg = Section({"type": "openai_responses",
                    "base_url": "https://api.openai.com/v1",
                    "api_key_env": "MISSING_ENV", "timeout_s": 5,
                    **(pcfg_extra or {})})
    cfg = Section({"presets": presets or {}, "worker_models": {}, "roles": {}})
    p = OpenAIResponsesProvider("openai", pcfg, cfg)
    p.client = types.SimpleNamespace(responses=rec)
    return p


# ---- registration -----------------------------------------------------------

def test_registered_as_a_provider_type():
    """config/default.yaml ships providers.openai with this type; until it is
    registered, validate_config warns about it on every spending command."""
    assert PROVIDER_TYPES["openai_responses"] is OpenAIResponsesProvider


# ---- reasoning effort -------------------------------------------------------

async def test_effort_is_sent_as_a_reasoning_block():
    rec = _FakeResponses()
    p = _oai(rec)
    await p.complete(model="gpt-5.6-luna", system="s", user="u", effort="high")
    assert rec.calls[0]["reasoning"] == {"effort": "high"}
    assert "reasoning_effort" not in rec.calls[0]      # that is the chat spelling


async def test_no_effort_means_no_reasoning_key_at_all():
    """None means 'leave the provider's own default alone' — an explicit
    reasoning block would pin the model to a level the config never chose."""
    rec = _FakeResponses()
    p = _oai(rec)
    await p.complete(model="gpt-5.6-luna", system="s", user="u", effort=None)
    assert "reasoning" not in rec.calls[0]


async def test_tier_wide_effort_applies_and_the_call_wins_over_it():
    rec = _FakeResponses()
    p = _oai(rec, pcfg_extra={"effort": "low"})
    await p.complete(model="gpt-5.6-luna", system="s", user="u")
    assert rec.calls[0]["reasoning"] == {"effort": "low"}
    await p.complete(model="gpt-5.6-luna", system="s", user="u", effort="xhigh")
    assert rec.calls[1]["reasoning"] == {"effort": "xhigh"}


async def test_an_invalid_effort_level_raises_instead_of_400ing():
    rec = _FakeResponses()
    p = _oai(rec)
    with pytest.raises(OrchestratorError):
        await p.complete(model="gpt-5.6-luna", system="s", user="u", effort="turbo")
    assert rec.calls == []


# ---- request shape ----------------------------------------------------------

async def test_instructions_and_input_not_messages():
    rec = _FakeResponses()
    p = _oai(rec)
    await p.complete(model="gpt-5.6-luna", system="SYS", user="USR")
    call = rec.calls[0]
    assert call["instructions"] == "SYS"
    assert call["input"] == [{"role": "user", "content": "USR"}]
    assert "messages" not in call


async def test_complete_chat_passes_turns_and_hoists_the_system_prompt():
    rec = _FakeResponses()
    p = _oai(rec)
    await p.complete_chat(model="gpt-5.6-luna", system="SYS", messages=[
        {"role": "system", "content": "IGNORED"},
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"}])
    call = rec.calls[0]
    assert call["instructions"] == "SYS"
    assert call["input"] == [{"role": "user", "content": "one"},
                             {"role": "assistant", "content": "two"},
                             {"role": "user", "content": "three"}]


# ---- params -----------------------------------------------------------------

async def test_max_tokens_dropped_and_max_output_tokens_forwarded():
    """A profile written for the DeepSeek era carries `max_tokens` in params;
    on this API the cap is `max_output_tokens` and `max_tokens` is a 400."""
    rec = _FakeResponses()
    p = _oai(rec)
    await p.complete(model="gpt-5.6-luna", system="s", user="u",
                     params={"max_tokens": 4000, "max_output_tokens": 8000,
                             "logit_bias": {"1": 1}})
    call = rec.calls[0]
    assert "max_tokens" not in call
    assert "logit_bias" not in call
    assert call["max_output_tokens"] == 8000


async def test_temperature_dropped_and_retried_on_a_400_naming_it():
    """The Luna case: reasoning models reject `temperature`, and one rejected
    knob must not fail the candidate."""
    err = _Fake400("Unsupported parameter: 'temperature' is not supported "
                   "with this model")
    rec = _FakeResponses(exc_seq=[err, None])
    p = _oai(rec)
    res = await p.complete(model="gpt-5.6-luna", system="s", user="u",
                           effort="high", params={"temperature": 0.3})
    assert res.text == "OUT"
    assert len(rec.calls) == 2
    assert rec.calls[0]["temperature"] == 0.3          # first attempt included it
    assert "temperature" not in rec.calls[1]           # retry dropped it
    assert rec.calls[1]["reasoning"] == {"effort": "high"}   # effort survived


async def test_a_400_naming_reasoning_fails_loudly_rather_than_downgrading():
    """`reasoning` is deliberately outside the droppable set: retrying without
    it would run at default depth and bill for it, which is exactly the silent
    failure this provider exists to end."""
    rec = _FakeResponses(exc_seq=[_Fake400("reasoning is not supported")])
    p = _oai(rec)
    with pytest.raises(OrchestratorError):
        await p.complete(model="gpt-5.6-luna", system="s", user="u", effort="high")
    assert len(rec.calls) == 1


# ---- usage and cost ---------------------------------------------------------

async def test_reasoning_tokens_are_not_added_to_output_tokens():
    """`output_tokens` already INCLUDES the reasoning trace. Adding it again
    inflates the cost ledger — and the budget guard reads that ledger."""
    rec = _FakeResponses(resp=_resp(usage=_usage(in_tok=1000, out_tok=400,
                                                 reasoning=300)))
    p = _oai(rec, presets={"luna_high": _LUNA})
    res = await p.complete(model="gpt-5.6-luna", system="s", user="u",
                           effort="high")
    assert res.output_tokens == 400                    # not 700
    assert res.input_tokens == 1000
    # The count is still reachable for diagnosis — via raw, never via a total.
    assert res.raw.usage.output_tokens_details.reasoning_tokens == 300


async def test_cached_input_tokens_land_in_cache_hit_tokens():
    rec = _FakeResponses(resp=_resp(usage=_usage(in_tok=1000, out_tok=100,
                                                 cached=800)))
    p = _oai(rec, presets={"luna_high": _LUNA})
    res = await p.complete(model="gpt-5.6-luna", system="s", user="u")
    assert res.cache_hit_tokens == 800
    assert res.cache_miss_tokens == 200
    # 800 @ $0.02 + 200 @ $0.20 + 100 @ $1.20, all per Mtok.
    expected = 800 / 1e6 * 0.02 + 200 / 1e6 * 0.20 + 100 / 1e6 * 1.20
    assert res.cost_usd == pytest.approx(expected)


async def test_cost_comes_from_the_price_table_including_the_272k_cliff():
    """>272K input re-prices the WHOLE request: 2x input, 1.5x output."""
    rec = _FakeResponses(resp=_resp(usage=_usage(in_tok=300_000, out_tok=10_000)))
    p = _oai(rec, presets={"luna_high": _LUNA})
    res = await p.complete(model="gpt-5.6-luna", system="s", user="u")
    expected = (300_000 / 1e6 * 0.20) * 2.0 + (10_000 / 1e6 * 1.20) * 1.5
    assert res.cost_usd == pytest.approx(expected)


async def test_an_unpriced_model_costs_zero_rather_than_a_guess():
    rec = _FakeResponses(resp=_resp(usage=_usage(in_tok=1000, out_tok=100)))
    p = _oai(rec)                                       # no presets -> no table
    res = await p.complete(model="gpt-5.6-luna", system="s", user="u")
    assert res.cost_usd == 0.0


async def test_text_falls_back_to_the_output_items_when_output_text_is_absent():
    """`output_text` is an SDK convenience property; an older SDK (or a
    response whose items we have to walk) must still yield the text."""
    resp = types.SimpleNamespace(
        output=[types.SimpleNamespace(content=[
            types.SimpleNamespace(text="hello "),
            types.SimpleNamespace(text="world")])],
        usage=_usage(in_tok=1, out_tok=2))
    rec = _FakeResponses(resp=resp)
    p = _oai(rec)
    res = await p.complete(model="gpt-5.6-luna", system="s", user="u")
    assert res.text == "hello world"


# ---- the base-class effort warning -----------------------------------------

async def test_a_provider_without_effort_support_warns_when_given_one(caplog):
    """openai_compatible drops `effort` (chat completions has no dial for the
    models it fronts). It used to do so with only a code comment, which is the
    whole 'I set high reasoning and nothing happened' failure."""
    pcfg = Section({"type": "openai_compatible", "base_url": "http://x",
                    "api_key_env": "MISSING_ENV", "timeout_s": 5})
    p = OpenAICompatibleProvider("cometapi", pcfg, Section({"worker_models": {}}))
    assert p.supports_effort is False
    calls: list[dict] = []

    class _Rec:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content="OUT"))], usage=None)

    p.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=_Rec()))

    with caplog.at_level(logging.WARNING, logger="orchestrator.providers"):
        await p.complete(model="deepseek-v4-flash", system="s", user="u",
                         effort="high")
        await p.complete(model="deepseek-v4-flash", system="s", user="u",
                         effort="high")
        await p.complete(model="other-model", system="s", user="u", effort="high")
        await p.complete(model="deepseek-v4-flash", system="s", user="u")

    warnings = [r.getMessage() for r in caplog.records
                if "does not support reasoning effort" in r.getMessage()]
    # Throttled to one line per (provider, model): a run makes hundreds of
    # worker calls, and a per-call warning is a warning nobody reads.
    assert len(warnings) == 2
    assert all("cometapi" in w for w in warnings)
    assert any("deepseek-v4-flash" in w for w in warnings)
    assert len(calls) == 4                              # every call still ran


def test_the_responses_provider_declares_effort_support():
    assert OpenAIResponsesProvider.supports_effort is True


# ---- config validation ------------------------------------------------------

def _validate(**overrides):
    from pathlib import Path

    from orchestrator.core.config import Config
    from orchestrator.core.validate import validate_config
    data = {
        "presets": {},
        "providers": {"claude_cli": {"type": "claude_cli", "binary": "claude"},
                      "cometapi": {"type": "openai_compatible",
                                   "base_url": "http://x"},
                      "openai": {"type": "openai_responses",
                                 "api_key_env": "MISSING_ENV"}},
        "worker_models": {"w1": {"provider": "claude_cli", "model": "sonnet"}},
        "roles": {"smart_provider": "claude_cli",
                  "worker": {"default": "w1", "candidates": ["w1"]}},
        "run": {}, "domains": {},
    }
    data.update(overrides)
    return validate_config(Config(data, "proj", Path("/tmp")),
                           will_spend=False)


def test_validate_warns_when_a_binding_sets_effort_on_a_backend_without_one():
    """The config can say 'this worker runs at high reasoning' against a
    backend with no dial at all. It resolves, it dispatches, and the level is
    dropped at the last hop — so the binding gets named at startup instead."""
    report = _validate(worker_models={
        "w1": {"provider": "cometapi", "model": "deepseek-v4-flash",
               "effort": "high"}})
    assert report.errors == []
    assert any("worker_models.w1" in w and "openai_compatible" in w
               for w in report.warnings)


def test_validate_is_silent_about_effort_on_an_effort_capable_backend():
    report = _validate(worker_models={
        "w1": {"provider": "openai", "model": "gpt-5.6-luna", "effort": "high"}})
    assert not any("reasoning-effort dial" in w for w in report.warnings)
