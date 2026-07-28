"""Fix 1: per-candidate sampling params + approach hints for best-of-N.

Two layers:
  * worker.run_candidate pulls `params`/`approach` from the worker_models entry,
    threads `params` into the provider call and `approach` into the FROZEN stable
    prefix; the `senior` pseudo-candidate has neither (and must not crash).
  * openai_compatible._chat forwards whitelisted sampling params into create()
    and drops a runtime-rejected key (HTTP 400) with a tolerant retry.
"""

import types

import pytest

import orchestrator.nodes.worker as worker_mod
from orchestrator.core.config import Section, load_config
from orchestrator.core.context import RunContext
from orchestrator.core.errors import OrchestratorError
from orchestrator.providers.base import LLMResult
from orchestrator.providers.openai_compatible import (OpenAICompatibleProvider,
                                                      _safe_params)

from tests.test_worker_loop import FakeBudget, FakeGit, FakeStore, _spec


# ---- worker layer: params/approach wiring -----------------------------------

class RecordingProvider:
    """Captures what run_candidate passes, applies a trivial patch so the empty
    gate passes (exercises the full first-turn path)."""
    type = "fake"

    def __init__(self, text='<file path="a.txt">\nx\n</file>'):
        self.text = text
        self.calls: list[dict] = []

    async def complete(self, *, model, system, user, cwd=None, params=None,
                       session=None, effort=None):
        self.calls.append({"params": params, "user": user})
        return LLMResult(text=self.text)

    async def complete_chat(self, *, model, system, messages, cwd=None,
                            params=None, session=None, effort=None):
        self.calls.append({"params": params, "effort": effort,
                           "messages": [dict(m) for m in messages]})
        return LLMResult(text=self.text)


def _ctx(tmp_path, worker_models, candidates, default=None):
    cfg = load_config()                              # generic defaults
    d = cfg._data
    d["project"]["repo_path"] = str(tmp_path / "repo")
    (tmp_path / "repo").mkdir(exist_ok=True)
    d["gate"]["commands"] = []                       # empty gate => passes
    d["roles"]["worker"] = {"default": default or candidates[0],
                            "candidates": candidates}
    d["worker_models"] = worker_models
    return RunContext(cfg=cfg, store=FakeStore(), git=FakeGit(tmp_path / "wt"),
                      budget=FakeBudget(), run_id="r")


async def _run(ctx, provider, monkeypatch, cand_id, *, attempt=1, feedback="",
               messages=None):
    monkeypatch.setattr(worker_mod, "get_provider", lambda cfg, name: provider)
    payload = {"run_id": "r", "task_id": "T-1", "spec": _spec(), "cand_id": cand_id,
               "attempt": attempt, "feedback": feedback, "messages": messages or []}
    return await worker_mod.run_candidate(ctx, payload)


async def test_approach_and_params_reach_worker_call(tmp_path, monkeypatch):
    wm = {"flash": {"provider": "fake", "model": "m",
                    "params": {"temperature": 0.7},
                    "approach": "prioritize robustness and edge cases"}}
    ctx = _ctx(tmp_path, wm, ["flash"])
    provider = RecordingProvider()
    out = await _run(ctx, provider, monkeypatch, "flash")
    cand = out["candidates"][0]
    assert cand["status"] == "gate_passed"
    # params threaded verbatim into the provider call
    assert provider.calls[0]["params"] == {"temperature": 0.7}
    # approach lands in the FROZEN prefix (first user turn), at the end
    prefix = cand["messages"][0]["content"]
    assert "# APPROACH" in prefix
    assert "prioritize robustness and edge cases" in prefix
    assert prefix.index("# APPROACH") > prefix.index("# FILES")


async def test_senior_has_no_params_or_approach(tmp_path, monkeypatch):
    # The escalation senior is NOT a worker_models key: params/approach are None,
    # nothing is appended, and the extra kwarg must not crash the CLI senior.
    wm = {"flash": {"provider": "fake", "model": "m",
                    "params": {"temperature": 0.7}, "approach": "x"}}
    ctx = _ctx(tmp_path, wm, ["flash"])
    provider = RecordingProvider()
    out = await _run(ctx, provider, monkeypatch, "senior", attempt=5,
                     feedback="ESCALATED: prior workers failed")
    cand = out["candidates"][0]
    assert cand["status"] == "gate_passed"            # no crash
    assert provider.calls[0]["params"] is None        # senior: no sampling params
    assert "# APPROACH" not in cand["messages"][0]["content"]
    # The senior is the one candidate that DOES carry an effort level, from
    # run.escalate_effort (unset in this generic config -> falls through to None).
    ctx.cfg._data["run"]["escalate_effort"] = "medium"
    provider2 = RecordingProvider()
    await _run(ctx, provider2, monkeypatch, "senior", attempt=5, feedback="f")
    assert provider2.calls[0]["effort"] == "medium"


async def test_cheap_worker_is_sent_no_effort(tmp_path, monkeypatch):
    """A chat-completions worker has no reasoning-effort dial. Even with an
    effort configured for the smart tier, the cheap candidate must be called
    with None — otherwise the flag reads as if it applied to both tiers."""
    wm = {"flash": {"provider": "fake", "model": "m"}}
    ctx = _ctx(tmp_path, wm, ["flash"])
    ctx.cfg._data["providers"]["claude_cli"]["effort"] = "high"
    ctx.cfg._data["run"]["escalate_effort"] = "medium"
    provider = RecordingProvider()
    await _run(ctx, provider, monkeypatch, "flash")
    assert provider.calls[0]["effort"] is None


async def test_three_keys_one_model_distinct_candidates(tmp_path, monkeypatch):
    # Best-of-N: three keys pointing at ONE model, differing only in sampling
    # params + approach. Each resolves to a distinct candidate with its own
    # frozen prefix; the price table (keyed on model) still sees one model.
    wm = {
        "flash_safe": {"provider": "fake", "model": "one",
                       "params": {"temperature": 0.2}, "approach": "simplicity"},
        "flash_mid":  {"provider": "fake", "model": "one",
                       "params": {"temperature": 0.7}, "approach": "balanced"},
        "flash_wild": {"provider": "fake", "model": "one",
                       "params": {"temperature": 1.0, "top_p": 0.95},
                       "approach": "robustness"},
    }
    order = ["flash_safe", "flash_mid", "flash_wild"]
    ctx = _ctx(tmp_path, wm, order, default="flash_mid")

    seen_params, seen_prefixes = [], []
    for cid in order:
        provider = RecordingProvider()
        out = await _run(ctx, provider, monkeypatch, cid)
        assert out["candidates"][0]["status"] == "gate_passed"
        seen_params.append(provider.calls[0]["params"])
        seen_prefixes.append(out["candidates"][0]["messages"][0]["content"])

    # all three resolve to the same model...
    assert [ctx.worker_target(c)[1] for c in order] == ["one", "one", "one"]
    # ...but carry distinct params and distinct frozen prefixes
    assert seen_params == [{"temperature": 0.2}, {"temperature": 0.7},
                           {"temperature": 1.0, "top_p": 0.95}]
    assert len(set(seen_prefixes)) == 3
    for frag in ("simplicity", "balanced", "robustness"):
        assert any(frag in pre for pre in seen_prefixes)


# ---- provider layer: params -> create() + tolerant unknown-key guard --------

def _resp(text="OUT", usage=None):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))],
        usage=usage)


class _FakeCompletions:
    def __init__(self, exc_seq=None):
        self.calls: list[dict] = []
        self.exc_seq = list(exc_seq or [])

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc_seq:
            exc = self.exc_seq.pop(0)
            if exc is not None:
                raise exc
        return _resp()


class _Fake400(Exception):
    status_code = 400


class _Fake500(Exception):
    status_code = 500


def _oai(rec):
    pcfg = Section({"type": "openai_compatible", "base_url": "http://x",
                    "api_key_env": "MISSING_ENV", "timeout_s": 5})
    cfg = Section({"worker_models": {}})
    p = OpenAICompatibleProvider("cometapi", pcfg, cfg)
    p.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=rec))
    return p


async def test_params_reach_create():
    rec = _FakeCompletions()
    p = _oai(rec)
    res = await p.complete(model="m", system="s", user="u",
                           params={"temperature": 0.7, "top_p": 0.95})
    assert res.text == "OUT"
    assert rec.calls[0]["temperature"] == 0.7
    assert rec.calls[0]["top_p"] == 0.95
    assert rec.calls[0]["model"] == "m"


async def test_unknown_param_not_forwarded():
    rec = _FakeCompletions()
    p = _oai(rec)
    await p.complete(model="m", system="s", user="u",
                     params={"bogus_key": 1, "temperature": 0.3})
    assert "bogus_key" not in rec.calls[0]
    assert rec.calls[0]["temperature"] == 0.3


async def test_rejected_seed_dropped_and_retried():
    err = _Fake400("Unsupported parameter: 'seed' is not supported by this model")
    rec = _FakeCompletions(exc_seq=[err, None])
    p = _oai(rec)
    res = await p.complete(model="m", system="s", user="u",
                           params={"temperature": 0.2, "seed": 5})
    assert res.text == "OUT"
    assert len(rec.calls) == 2
    assert rec.calls[0].get("seed") == 5              # first attempt included it
    assert "seed" not in rec.calls[1]                 # retry dropped the bad key
    assert rec.calls[1]["temperature"] == 0.2         # kept the good param


async def test_non_400_error_surfaces_without_retry():
    rec = _FakeCompletions(exc_seq=[_Fake500("temperature server exploded")])
    p = _oai(rec)
    with pytest.raises(OrchestratorError):
        await p.complete(model="m", system="s", user="u",
                         params={"temperature": 0.2})
    assert len(rec.calls) == 1                         # no key-dropping on a non-400


async def test_no_params_calls_create_cleanly():
    rec = _FakeCompletions()
    p = _oai(rec)
    await p.complete(model="m", system="s", user="u")
    assert set(rec.calls[0]) == {"model", "messages"}  # nothing spurious spread in


def test_safe_params_whitelist():
    assert _safe_params({"temperature": 0.5, "bogus": 1, "top_p": 0.9}) == \
        {"temperature": 0.5, "top_p": 0.9}
    assert _safe_params(None) == {}
    assert _safe_params({}) == {}
