"""anthropic_api — the Messages API provider, and the four ways it can go wrong.

Every assertion here covers something that fails SILENTLY or expensively if it
regresses:

* `system` sent as a message instead of the top-level parameter is a 400 — the
  single most common way an OpenAI-shaped call gets ported to this API wrong.
* a missing `max_tokens` is a 400 on EVERY call: the API has no default.
* `thinking.budget_tokens >= max_tokens` is a 400 too, and with the shipped
  defaults (effort=max wants 65536, the budget default is 8000) it is the normal
  case rather than an edge one.
* `temperature`/`top_p` alongside extended thinking is a 400 as well — and the
  interesting failure is not the 400, it is a best-of-N pool whose candidates
  differ only by temperature quietly becoming N identical calls.
* `input_tokens` on this API EXCLUDES the cache buckets, so reading it bare
  understates the prompt weight of exactly the warm-cache calls this codebase
  is built to produce.

The SDK is faked the way tests/test_worker_params.py and
tests/test_openai_responses.py fake theirs: replace `p.client` wholesale with a
SimpleNamespace whose `messages.create` records its kwargs. Nothing here needs
the real `anthropic` package installed — which is also what the last test
asserts, from the other direction.
"""

import logging
import sys
import types

import pytest

from orchestrator.core.config import Section
from orchestrator.core.errors import OrchestratorError
from orchestrator.ops.pricing import price
from orchestrator.providers import PROVIDER_TYPES
from orchestrator.providers.anthropic_api import AnthropicApiProvider
from orchestrator.providers.claude_cli import cache_tokens


# ---- fakes ------------------------------------------------------------------

def _usage(input_tokens=0, output_tokens=0, cache_creation=0, cache_read=0):
    """A Messages-shaped usage object. `input_tokens` is the UNCACHED remainder
    only — the two cache buckets are separate, which is the whole point."""
    return types.SimpleNamespace(
        input_tokens=input_tokens, output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read)


def _resp(text="OUT", usage=None, blocks=None):
    """A Messages response: a list of content blocks plus usage."""
    if blocks is None:
        blocks = [types.SimpleNamespace(type="text", text=text)]
    return types.SimpleNamespace(content=blocks, usage=usage)


class _FakeMessages:
    def __init__(self, resp=None, exc=None):
        self.calls: list[dict] = []
        self.resp = resp
        self.exc = exc

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return self.resp if self.resp is not None else _resp()

    @property
    def last(self) -> dict:
        return self.calls[-1]


#: A Claude-shaped price entry (list rates, inlined so the test pins the
#: arithmetic rather than whatever config/default.yaml happens to say today).
#: Unlike the Luna shape this one DOES carry a cache-write rate, because this
#: provider reports a distinct `cache_creation_input_tokens` count.
_CLAUDE = {
    "provider": "anthropic", "model": "claude-opus-5",
    "input_per_mtok": 3.00, "output_per_mtok": 15.00,
    "cache_read_per_mtok": 0.30, "cache_write_per_mtok": 3.75,
}


def _anthropic(rec, presets=None, pcfg_extra=None, budget=None,
               worker_models=None):
    pcfg = Section({"type": "anthropic_api", "api_key_env": "MISSING_ENV",
                    "timeout_s": 5, **(pcfg_extra or {})})
    cfg = Section({"presets": presets or {}, "roles": {},
                   "worker_models": worker_models or {},
                   "budget": budget if budget is not None else {}})
    p = AnthropicApiProvider("anthropic", pcfg, cfg)
    # Replacing the client wholesale is what keeps the real SDK out of the test:
    # `_client()` hands back an already-set client without importing anything.
    p.client = types.SimpleNamespace(messages=rec)
    return p


# ---- registration -----------------------------------------------------------

def test_registered_as_a_provider_type():
    """config/default.yaml ships providers.anthropic with this type; until it is
    registered, validate_config warns about it on every spending command."""
    assert PROVIDER_TYPES["anthropic_api"] is AnthropicApiProvider


def test_supports_effort():
    """The reason this class exists rather than routing through
    openai_compatible: the level reaches the model, as a thinking budget."""
    assert AnthropicApiProvider.supports_effort is True


# ---- the system prompt ------------------------------------------------------

async def test_system_is_a_top_level_param_not_a_message():
    rec = _FakeMessages()
    p = _anthropic(rec)
    await p.complete(model="claude-opus-5", system="SYS", user="U")
    assert rec.last["system"] == "SYS"
    assert rec.last["messages"] == [{"role": "user", "content": "U"}]
    assert all(m["role"] != "system" for m in rec.last["messages"])


async def test_complete_chat_lifts_a_system_turn_out_of_the_array():
    """A system turn left in `messages` is a 400; sent twice it is paid for
    twice and breaks the stable prefix prompt caching keys on."""
    rec = _FakeMessages()
    p = _anthropic(rec)
    await p.complete_chat(model="claude-opus-5", system="SYS", messages=[
        {"role": "system", "content": "STRAY"},
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "U2"}])
    assert rec.last["system"] == "SYS"
    assert rec.last["messages"] == [{"role": "user", "content": "U1"},
                                    {"role": "assistant", "content": "A1"},
                                    {"role": "user", "content": "U2"}]


# ---- max_tokens -------------------------------------------------------------

async def test_max_tokens_defaults_from_the_budget_assumption():
    rec = _FakeMessages()
    p = _anthropic(rec, budget={"assumed_max_output_tokens": 4321})
    await p.complete(model="claude-opus-5", system="", user="U")
    assert rec.last["max_tokens"] == 4321


async def test_max_tokens_is_always_sent_even_with_no_budget_section():
    """The API has no server-side default: a call without max_tokens is a 400,
    so there is no configuration in which this key may be absent."""
    rec = _FakeMessages()
    p = _anthropic(rec)
    await p.complete(model="claude-opus-5", system="", user="U")
    assert rec.last["max_tokens"] == 8000      # ops/budget.py's own default


async def test_a_preset_max_tokens_reaches_the_call():
    """A role binding never passes `params` (nodes/planner.py), so the preset is
    the only place a role-level cap can be written — it has to be recovered from
    the model id at the call site."""
    rec = _FakeMessages()
    p = _anthropic(rec, presets={"claude_api": dict(_CLAUDE, max_tokens=20000)},
                   budget={"assumed_max_output_tokens": 4321})
    await p.complete(model="claude-opus-5", system="", user="U")
    assert rec.last["max_tokens"] == 20000


async def test_params_max_tokens_wins_over_the_preset():
    rec = _FakeMessages()
    p = _anthropic(rec, presets={"claude_api": dict(_CLAUDE, max_tokens=20000)})
    await p.complete(model="claude-opus-5", system="", user="U",
                     params={"max_tokens": 999})
    assert rec.last["max_tokens"] == 999


async def test_max_tokens_in_params_is_consumed_not_reported_as_unknown(caplog):
    """A profile written for the DeepSeek era carries `params: {max_tokens: N}`,
    and here that is exactly the right value — so it is lifted out as the
    required top-level argument rather than warned about and dropped."""
    rec = _FakeMessages()
    p = _anthropic(rec)
    with caplog.at_level(logging.WARNING, logger="orchestrator.openai_shared"):
        await p.complete(model="claude-opus-5", system="", user="U",
                         params={"max_tokens": 512, "seed": 7})
    assert rec.last["max_tokens"] == 512
    assert "seed" in caplog.text            # no such knob on this API: dropped
    assert "max_tokens" not in caplog.text


# ---- effort -> thinking -----------------------------------------------------

async def test_effort_high_becomes_a_thinking_budget():
    rec = _FakeMessages()
    p = _anthropic(rec, budget={"assumed_max_output_tokens": 32000})
    await p.complete(model="claude-opus-5", system="", user="U", effort="high")
    assert rec.last["thinking"] == {"type": "enabled", "budget_tokens": 16384}


async def test_no_effort_sends_no_thinking_key_at_all():
    """None, not `{"type": "disabled"}` — the request is byte-identical to what
    the model received before this provider learned about effort."""
    rec = _FakeMessages()
    p = _anthropic(rec)
    await p.complete(model="claude-opus-5", system="", user="U")
    assert "thinking" not in rec.last


async def test_effort_none_disables_thinking():
    """`none` exists so a preset can move between this provider and
    openai_responses (whose level set has it) without becoming invalid."""
    rec = _FakeMessages()
    p = _anthropic(rec)
    await p.complete(model="claude-opus-5", system="", user="U", effort="none")
    assert "thinking" not in rec.last


async def test_a_preset_thinking_budget_overrides_the_table():
    rec = _FakeMessages()
    p = _anthropic(rec, presets={"claude_api": dict(
        _CLAUDE, max_tokens=40000, thinking_budget_tokens=12345)})
    await p.complete(model="claude-opus-5", system="", user="U", effort="high")
    assert rec.last["thinking"]["budget_tokens"] == 12345


async def test_an_unknown_effort_level_is_a_loud_error():
    rec = _FakeMessages()
    p = _anthropic(rec)
    with pytest.raises(OrchestratorError) as e:
        await p.complete(model="claude-opus-5", system="", user="U",
                         effort="ludicrous")
    assert "ludicrous" in str(e.value)
    assert rec.calls == []      # never dispatched


async def test_a_budget_at_or_above_max_tokens_is_clamped_with_a_warning(caplog):
    """The API rejects budget_tokens >= max_tokens outright, and with the
    defaults (effort=max wants 65536, max_tokens defaults to 8000) that is the
    NORMAL case — so it is clamped rather than allowed to 400."""
    rec = _FakeMessages()
    p = _anthropic(rec, budget={"assumed_max_output_tokens": 8000})
    with caplog.at_level(logging.WARNING, logger="orchestrator.anthropic_api"):
        await p.complete(model="claude-opus-5", system="", user="U", effort="max")
    assert rec.last["max_tokens"] == 8000
    assert rec.last["thinking"] == {"type": "enabled", "budget_tokens": 7999}
    assert "clamping to 7999" in caplog.text


async def test_thinking_is_dropped_when_no_legal_budget_fits(caplog):
    """Under the API's 1024-token floor there is no legal budget at all, so the
    only correct request is one with no thinking block in it."""
    rec = _FakeMessages()
    p = _anthropic(rec, budget={"assumed_max_output_tokens": 500})
    with caplog.at_level(logging.WARNING, logger="orchestrator.anthropic_api"):
        await p.complete(model="claude-opus-5", system="", user="U", effort="low")
    assert "thinking" not in rec.last
    assert "NO thinking block" in caplog.text


# ---- the best-of-N trap -----------------------------------------------------

async def test_temperature_and_top_p_are_stripped_when_thinking_is_on(caplog):
    """Extended thinking forbids both. Stripped LOUDLY, because the expensive
    failure is not the 400 — it is three candidates that differ only by
    temperature becoming three identical calls at 3x the price."""
    rec = _FakeMessages()
    p = _anthropic(rec, budget={"assumed_max_output_tokens": 32000})
    with caplog.at_level(logging.WARNING, logger="orchestrator.anthropic_api"):
        await p.complete(model="claude-opus-5", system="", user="U",
                         effort="high",
                         params={"temperature": 0.3, "top_p": 0.9, "top_k": 20})
    assert "temperature" not in rec.last
    assert "top_p" not in rec.last
    assert rec.last["top_k"] == 20        # not forbidden, so not touched
    assert "forbids temperature, top_p" in caplog.text


async def test_temperature_and_top_p_survive_when_thinking_is_off():
    rec = _FakeMessages()
    p = _anthropic(rec)
    await p.complete(model="claude-opus-5", system="", user="U",
                     params={"temperature": 0.3, "top_p": 0.9})
    assert rec.last["temperature"] == 0.3
    assert rec.last["top_p"] == 0.9


# ---- usage and cost ---------------------------------------------------------

async def test_cache_tokens_use_the_claude_cli_mapping():
    """`cache_read_input_tokens` is the hit side; the miss side is the fresh
    remainder PLUS the cache-creation write, and `input_tokens` is the sum of
    all three — the same reading claude_cli does, so the ledger is comparable
    across the subscription and metered halves of the same tier."""
    usage = _usage(input_tokens=1000, output_tokens=200,
                   cache_creation=500, cache_read=4000)
    rec = _FakeMessages(resp=_resp("OUT", usage))
    p = _anthropic(rec)
    r = await p.complete(model="claude-opus-5", system="", user="U")

    assert r.cache_hit_tokens == 4000
    assert r.cache_miss_tokens == 1500          # fresh 1000 + created 500
    assert r.input_tokens == 5500               # NOT the bare 1000
    assert r.output_tokens == 200
    assert r.cache_hit_tokens + r.cache_miss_tokens == r.input_tokens
    # And it is literally claude_cli's function, not a second reading of it.
    assert cache_tokens({"input_tokens": 1000, "cache_creation_input_tokens": 500,
                         "cache_read_input_tokens": 4000}) == (5500, 4000, 1500)


async def test_cost_is_computed_through_ops_pricing():
    usage = _usage(input_tokens=1000, output_tokens=200,
                   cache_creation=500, cache_read=4000)
    rec = _FakeMessages(resp=_resp("OUT", usage))
    p = _anthropic(rec, presets={"claude_api": _CLAUDE})
    r = await p.complete(model="claude-opus-5", system="", user="U")

    # 4000 hits @ $0.30 + 1500 misses @ $3.75 + 200 out @ $15.00 per Mtok
    assert r.cost_usd == pytest.approx(0.009825)
    assert r.cost_usd == pytest.approx(price(_CLAUDE, 5500, 200, 4000, 1500))


async def test_an_unpriced_model_costs_zero_rather_than_a_guess():
    rec = _FakeMessages(resp=_resp("OUT", _usage(input_tokens=1000,
                                                 output_tokens=200)))
    p = _anthropic(rec)
    r = await p.complete(model="claude-opus-5", system="", user="U")
    assert r.cost_usd == 0.0


async def test_only_text_blocks_become_the_answer():
    """A thinking block carries `.thinking`, not `.text`. Letting a reasoning
    trace into the returned string would let something that merely LOOKS like a
    `<file>` block be applied as a patch."""
    blocks = [types.SimpleNamespace(type="thinking", thinking="<file>nope</file>"),
              types.SimpleNamespace(type="text", text="REAL")]
    rec = _FakeMessages(resp=_resp(blocks=blocks))
    p = _anthropic(rec)
    r = await p.complete(model="claude-opus-5", system="", user="U")
    assert r.text == "REAL"


# ---- failure shapes ---------------------------------------------------------

async def test_an_api_failure_surfaces_as_an_orchestrator_error():
    rec = _FakeMessages(exc=RuntimeError("boom"))
    p = _anthropic(rec)
    with pytest.raises(OrchestratorError) as e:
        await p.complete(model="claude-opus-5", system="", user="U")
    assert "anthropic/claude-opus-5" in str(e.value)


async def test_a_missing_sdk_is_an_orchestrator_error_not_an_import_error(
        monkeypatch):
    """The whole reason the import is module-local. A top-level import would
    turn "this one optional backend is absent" into an ImportError at
    `orchestrator.providers` import time — taking down every other backend, the
    CLI, and the config validator with it.

    `sys.modules["anthropic"] = None` makes the import raise whether or not the
    real package is installed, so this test says the same thing on a machine
    that has it and one that does not.
    """
    monkeypatch.setitem(sys.modules, "anthropic", None)
    p = _anthropic(_FakeMessages())
    p.client = None                     # force the lazy construction path
    with pytest.raises(OrchestratorError) as e:
        await p.complete(model="claude-opus-5", system="", user="U")
    message = str(e.value)
    assert "anthropic" in message and "anthropic_api" in message
    assert "install" in message.lower()
