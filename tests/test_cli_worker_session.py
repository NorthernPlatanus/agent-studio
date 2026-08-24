"""A CLI backend as an ordinary WORKER, not just a smart-tier role.

Nothing ever asserted that a worker must be an HTTP provider — the `senior`
escalation pseudo-candidate has always been a `claude -p` implementer — but the
worker path degraded badly when you pointed a normal candidate at it, in two
ways this module pins down.

**Sessions.** `nodes.worker` used to call `complete_chat` without a `session`,
and the inherited implementation flattens every turn into one `-p` prompt. So a
candidate's retrieval rounds and retry feedback each re-sent the entire
accumulated conversation — frozen file prefix, every grep result, every prior
answer — as fresh input, on the tier whose binding constraint is subscription
tokens. The machinery to avoid that (`--session-id` / `--resume`) was already in
the provider; the worker simply never asked for it.

**Concurrency.** Effective LLM concurrency is
`run.max_parallel_tasks x len(roles.worker.candidates)` — nine by default — and
there was no semaphore anywhere in the codebase. Nine simultaneous Node runtimes
on one subscription hit the rate limit almost at once, and `_run_batch` waits
with FIRST_EXCEPTION, so the resulting LimitExhausted cancels every sibling.
"""

import asyncio
import time
from pathlib import Path

import pytest

from orchestrator.core.config import Config, Section
from orchestrator.core.context import RunContext
from orchestrator.core.validate import validate_config
from orchestrator.nodes import integrator
from orchestrator.providers import get_provider
from orchestrator.providers.claude_cli import ClaudeCliProvider, _LiveSession
from tests.conftest import FakeCli, FakeProc, stream_json


_OK = stream_json({"result": "ok", "usage": {"input_tokens": 1,
                                             "output_tokens": 1}})
_LOST = "Error: No conversation found with session ID 1234"

SESSION = "run-1:T-1:w"


def _provider(monkeypatch, *specs: dict, **pcfg) -> ClaudeCliProvider:
    """A claude_cli provider whose subprocess is scripted by `specs`."""
    cli = FakeCli(*(specs or ({"out": _OK},)))
    cli.install(monkeypatch)
    base = {"type": "claude_cli", "binary": "claude", "timeout_s": 600}
    provider = ClaudeCliProvider("claude_cli", Section({**base, **pcfg}),
                                 Section({"mcp": {}}))
    provider.cli = cli                       # tests read argv off this
    return provider


def _prompt(cli: FakeCli, call: int) -> str:
    """The `-p` payload of call `call` (argv is [binary, '-p', prompt, ...])."""
    return cli.argv[call][2]


async def _turn(provider, messages, session=SESSION):
    return await provider.complete_chat(model="opus", system="SYS",
                                        messages=messages, session=session)


# ---- session continuity ------------------------------------------------------

async def test_turn_two_resumes_and_sends_only_the_newest_turn(monkeypatch):
    """The whole point: turn 2 costs the delta, not the conversation."""
    p = _provider(monkeypatch, {"out": _OK}, {"out": _OK})
    messages = [{"role": "user", "content": "FROZEN PREFIX"}]
    await _turn(p, messages)

    # Turn 1 pins a fresh conversation and sends the full payload.
    assert p.cli.flag("--session-id", 0) is not None
    assert p.cli.flag("--resume", 0) is None
    assert "FROZEN PREFIX" in _prompt(p.cli, 0)

    messages += [{"role": "assistant", "content": "PRIOR ANSWER"},
                 {"role": "user", "content": "GREP RESULTS"}]
    await _turn(p, messages)

    # Turn 2 continues it and carries ONLY what was appended since the CLI's own
    # reply. The frozen prefix and the model's prior answer are already in that
    # session; re-sending them is exactly the cost this change exists to remove.
    assert p.cli.flag("--resume", 1) == p.cli.flag("--session-id", 0)
    assert p.cli.flag("--session-id", 1) is None
    assert _prompt(p.cli, 1) == "GREP RESULTS"
    assert "FROZEN PREFIX" not in _prompt(p.cli, 1)
    assert "PRIOR ANSWER" not in _prompt(p.cli, 1)


async def test_two_turns_appended_since_the_last_reply_both_travel(monkeypatch):
    """`nodes.worker` can append twice between calls — retrieval results and
    then the 'retrieval exhausted, implement now' instruction. Taking only
    `messages[-1]` would send the instruction without the results it refers to."""
    p = _provider(monkeypatch, {"out": _OK}, {"out": _OK})
    messages = [{"role": "user", "content": "FROZEN PREFIX"}]
    await _turn(p, messages)
    messages += [{"role": "assistant", "content": "A"},
                 {"role": "user", "content": "RESULTS"},
                 {"role": "user", "content": "EXHAUSTED"}]
    await _turn(p, messages)
    assert _prompt(p.cli, 1) == "RESULTS\n\nEXHAUSTED"


async def test_a_lost_session_resends_the_full_history_exactly_once(monkeypatch):
    """A dead `--resume` is recoverable, not a candidate failure.

    The abbreviated payload means nothing without the conversation it referred
    to, so the whole history goes out again — once, into a fresh session. Twice
    would be the expensive way to discover that a second failure is real.
    """
    p = _provider(monkeypatch,
                  {"out": _OK},                       # turn 1: pin
                  {"out": "", "err": _LOST, "rc": 1},  # turn 2: resume fails
                  {"out": _OK})                       # the one resend
    messages = [{"role": "user", "content": "FROZEN PREFIX"}]
    await _turn(p, messages)
    messages += [{"role": "assistant", "content": "PRIOR ANSWER"},
                 {"role": "user", "content": "GREP RESULTS"}]
    result = await _turn(p, messages)

    assert result.text == "ok"                        # the run continues
    assert len(p.cli.argv) == 3                       # one resend, not a loop
    assert p.cli.flag("--resume", 1) is not None      # the attempt that died
    # The recovery opens a NEW conversation (the dead id was dropped) and carries
    # everything the abbreviated turn assumed the old one still held.
    assert p.cli.flag("--session-id", 2) is not None
    assert p.cli.flag("--resume", 2) is None
    assert "FROZEN PREFIX" in _prompt(p.cli, 2)
    assert "GREP RESULTS" in _prompt(p.cli, 2)


async def test_end_session_makes_the_next_call_start_fresh(monkeypatch):
    """`nodes.integrator.finalize` ends a finished task's sessions. After that
    the next call with the same key must pin a new conversation, not resume a
    worktree that has already been deleted."""
    p = _provider(monkeypatch, {"out": _OK}, {"out": _OK})
    messages = [{"role": "user", "content": "FROZEN PREFIX"}]
    await _turn(p, messages)
    assert p.session_active(SESSION)

    p.end_session(SESSION)
    assert not p.session_active(SESSION)

    messages += [{"role": "assistant", "content": "A"},
                 {"role": "user", "content": "NEXT"}]
    await _turn(p, messages)
    assert p.cli.flag("--session-id", 1) is not None
    assert p.cli.flag("--resume", 1) is None
    assert "FROZEN PREFIX" in _prompt(p.cli, 1)       # full payload again


async def test_no_session_key_keeps_the_stateless_behaviour(monkeypatch):
    """Providers that are handed no key behave exactly as before: one flattened
    prompt, no session flags. This is the HTTP tier's path through the same
    override, and the first turn of every CLI loop."""
    p = _provider(monkeypatch, {"out": _OK})
    await _turn(p, [{"role": "user", "content": "A"},
                    {"role": "assistant", "content": "B"},
                    {"role": "user", "content": "C"}], session=None)
    assert p.cli.flag("--session-id", 0) is None
    assert p.cli.flag("--resume", 0) is None
    assert _prompt(p.cli, 0) == "A\n\nB\n\nC"


# ---- the concurrency ceiling -------------------------------------------------

class _ConcurrencyProbe:
    """Stands in for `create_subprocess_exec` and measures REAL overlap.

    Counting in `FakeCli` would not do: it hands back a finished process
    immediately, so nothing would ever be in flight at the same time as anything
    else. This one holds each "process" open for a beat, which is the window
    during which the provider is holding its semaphore slot.

    `peak` is a LOWER bound on true concurrency — the slot is released a little
    after this returns — which is the safe direction for both assertions below.
    """

    def __init__(self, hold_s: float = 0.05):
        self.hold_s = hold_s
        self.live = 0
        self.peak = 0
        self.started = 0

    async def __call__(self, *args, **kwargs):
        self.started += 1
        self.live += 1
        self.peak = max(self.peak, self.live)
        try:
            await asyncio.sleep(self.hold_s)
        finally:
            self.live -= 1
        return FakeProc(out=_OK)


async def _fan_out(monkeypatch, n: int, **pcfg) -> _ConcurrencyProbe:
    probe = _ConcurrencyProbe()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", probe)
    base = {"type": "claude_cli", "binary": "claude", "timeout_s": 600}
    p = ClaudeCliProvider("claude_cli", Section({**base, **pcfg}),
                          Section({"mcp": {}}))
    await asyncio.gather(*(p.complete(model="opus", system="s", user=f"u{i}")
                           for i in range(n)))
    return probe


async def test_the_semaphore_bounds_processes_in_flight(monkeypatch):
    probe = await _fan_out(monkeypatch, 6, max_concurrency=2)
    assert probe.started == 6          # every call still runs...
    assert probe.peak == 2             # ...but never more than two at a time


async def test_zero_means_unbounded(monkeypatch):
    """The control, and the escape hatch: 0 switches the ceiling off, so an
    operator on a bigger plan does not have to edit code to use it."""
    probe = await _fan_out(monkeypatch, 6, max_concurrency=0)
    assert probe.peak == 6


async def test_the_cli_default_ceiling_applies_with_no_config(monkeypatch):
    """Unset must not mean unbounded on a CLI backend — that is the shipped
    default that would otherwise start nine Node runtimes on one subscription."""
    probe = await _fan_out(monkeypatch, 6)
    assert ClaudeCliProvider.default_max_concurrency == 2
    assert probe.peak == 2


# ---- session state is scoped to the Config-bound provider instance -----------

def test_a_fresh_config_does_not_resurrect_stale_session_ids():
    """Provider instances are cached per Config, and sessions live on the
    instance. Any path that rebuilds the run from a freshly loaded Config — a
    resume, the degrade re-entry in `engine.runner`, the next unit test — must
    therefore start with no sessions at all.

    Asserted rather than assumed because the failure mode of a "harmless"
    caching change (keying provider instances by NAME) is a run that resumes a
    conversation rooted in a worktree that no longer exists, and the symptom is
    a confusing CLI error minutes into the next run, not here.
    """
    data = {
        "providers": {"claude_cli": {"type": "claude_cli", "binary": "claude"}},
        "worker_models": {"w": {"provider": "claude_cli", "model": "opus"}},
        "roles": {"smart_provider": "claude_cli",
                  "worker": {"default": "w", "candidates": ["w"]}},
    }
    old_cfg = Config(dict(data), "proj", Path("/tmp"))
    old_ctx = RunContext(cfg=old_cfg, store=None, git=None, budget=None,
                         run_id="run-1")
    key = old_ctx.worker_session_key("T-1", "w")
    old = get_provider(old_cfg, "claude_cli")
    old._sessions[key] = old._sessions.get(key) or _live_session()
    assert old.session_active(key)

    # A degraded run resumes under the SAME run id, so the key is byte-identical
    # — the isolation has to come from the provider instance, not the key.
    new_cfg = Config(dict(data), "proj", Path("/tmp"))
    new_ctx = RunContext(cfg=new_cfg, store=None, git=None, budget=None,
                         run_id="run-1")
    assert new_ctx.worker_session_key("T-1", "w") == key
    new = get_provider(new_cfg, "claude_cli")
    assert new is not old
    assert not new.session_active(key)


def _live_session() -> _LiveSession:
    return _LiveSession("sid-1", time.monotonic())


# ---- the wiring: worker dispatches with a key, finalize ends it --------------

def _wired_ctx() -> RunContext:
    data = {
        "providers": {"claude_cli": {"type": "claude_cli", "binary": "claude"}},
        "worker_models": {"w": {"provider": "claude_cli", "model": "opus"}},
        "roles": {"smart_provider": "claude_cli",
                  "worker": {"default": "w", "candidates": ["w"]}},
    }
    cfg = Config(data, "proj", Path("/tmp"))
    return RunContext(cfg=cfg, store=None, git=None, budget=None, run_id="run-1")


def test_finalize_ends_the_candidates_session():
    """A finished task must not leave a pinned conversation behind. The provider
    instance lives as long as the Config — i.e. the whole run — so without this
    every candidate of every task holds a session id until `session_max_idle_s`
    eventually expires it, and the worktree it was rooted at is already gone."""
    ctx = _wired_ctx()
    provider = get_provider(ctx.cfg, "claude_cli")
    key = ctx.worker_session_key("T-1", "w")
    provider._sessions[key] = _live_session()

    integrator._end_worker_session(ctx, "T-1", "w")
    assert not provider.session_active(key)


def test_ending_a_session_never_raises_on_the_terminal_path():
    """Cleanup runs after the task has already succeeded or failed. An unusual
    candidate id must not turn a finished task into a crash."""
    integrator._end_worker_session(_wired_ctx(), "T-1", "no-such-candidate")


# ---- validation: a pool wider than the backend will run ----------------------

def _cfg(candidates, **pover) -> Config:
    data = {
        "presets": {"cli_opus": {"label": "CLI · opus", "provider": "claude_cli",
                                 "model": "opus"}},
        "providers": {"claude_cli": {"type": "claude_cli", "binary": "claude",
                                     **pover}},
        "worker_models": {c: {"preset": "cli_opus", "approach": c}
                          for c in candidates},
        "roles": {"smart_provider": "claude_cli",
                  "worker": {"default": candidates[0],
                             "candidates": list(candidates)}},
        "run": {"degrade_model": candidates[0]},
        "domains": {},
    }
    return Config(data, "proj", Path("/tmp"))


def test_a_cli_backed_pool_wider_than_max_concurrency_warns():
    report = validate_config(_cfg(["a", "b", "c"], max_concurrency=2))
    assert report.ok                                   # a warning, never fatal
    hit = [w for w in report.warnings if "max_concurrency" in w]
    assert len(hit) == 1
    assert "3 candidates" in hit[0] and "claude_cli" in hit[0]


def test_a_pool_that_fits_the_ceiling_is_silent():
    report = validate_config(_cfg(["a", "b"], max_concurrency=2))
    assert [w for w in report.warnings if "max_concurrency" in w] == []


def test_an_unbounded_provider_never_warns_about_pool_width():
    report = validate_config(_cfg(["a", "b", "c"], max_concurrency=0))
    assert [w for w in report.warnings if "max_concurrency" in w] == []


@pytest.mark.parametrize("raw,expected", [
    (None, 2), (1, 1), ("3", 3), (0, None), (-1, None), ("nonsense", 2),
])
def test_max_concurrency_is_resolved_without_building_a_provider(raw, expected):
    """`core.validate` asks this question at startup, from config alone —
    constructing a provider to ask would open an HTTP client and read an API key
    out of the environment."""
    pcfg = Section({} if raw is None else {"max_concurrency": raw})
    assert ClaudeCliProvider.configured_max_concurrency(pcfg) == expected
