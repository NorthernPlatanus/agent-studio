"""Item 2 steps 3-4: stable review prefixes, and CLI session continuity.

The smart tier ran fully cold — every planner/reviewer call a fresh process with
a fresh context. Step 3 makes the payload prefix-stable (helps on every
provider); step 4 lets the Claude CLI continue one conversation across the turns
of a `discuss` loop instead of re-sending backlog + map + specs each time.
"""

from pathlib import Path

import pytest

import orchestrator.nodes.planner as planner_mod
from tests.conftest import FakeCli, stream_json
from orchestrator.core.config import Config, Section, load_config
from orchestrator.core.context import RunContext
from orchestrator.core.errors import SessionLost
from orchestrator.nodes.reviewer import build_review_prompt
from orchestrator.providers.base import LLMProvider, LLMResult
from orchestrator.providers.claude_cli import ClaudeCliProvider
from orchestrator.ops.store import Store


# ---- step 3: the review payload is stable-first ------------------------------

def _spec():
    return {"id": "T-1", "title": "do the thing", "description": "a description",
            "acceptance": ["it works", "tests pass"]}


def _cand(cid, diff, notes="n"):
    return {"cand_id": cid, "model": "m", "diff": diff, "notes": notes}


def test_review_prefix_is_byte_identical_across_review_rounds():
    spec = _spec()
    round1 = build_review_prompt(spec, {"w": _cand("w", "diff v1")})
    round2 = build_review_prompt(spec, {"w": _cand("w", "diff v2 much longer")})
    head = f"# TASK {spec['id']}: {spec['title']}"
    # everything up to the first candidate block is invariant
    prefix1 = round1[:round1.index("## Candidate")]
    prefix2 = round2[:round2.index("## Candidate")]
    assert prefix1 == prefix2 and prefix1.startswith(head)
    assert "it works" in prefix1                 # acceptance is inside the prefix
    assert "diff v1" not in prefix1              # the diff is not


def test_candidate_order_does_not_depend_on_dict_order():
    spec = _spec()
    a, b = _cand("alpha", "A"), _cand("beta", "B")
    assert build_review_prompt(spec, {"alpha": a, "beta": b}) == \
        build_review_prompt(spec, {"beta": b, "alpha": a})


# ---- step 4: claude_cli sessions ---------------------------------------------

_OK = {"out": stream_json({"result": "ok",
                           "usage": {"input_tokens": 1, "output_tokens": 1}})}


def _cli(monkeypatch, *specs):
    """Provider whose subprocess is scripted by `specs`; records argv."""
    cli = FakeCli(*specs).install(monkeypatch)
    p = ClaudeCliProvider("claude_cli",
                          Section({"type": "claude_cli", "binary": "claude",
                                   "timeout_s": 600, "retry_attempts": 1}),
                          Section({"mcp": {}}))
    return p, cli


async def test_no_session_key_means_no_session_flags(monkeypatch):
    p, cli = _cli(monkeypatch, _OK)
    await p.complete(model="opus", system="S", user="U")
    assert "--session-id" not in cli.argv[0] and "--resume" not in cli.argv[0]
    assert p.session_active("anything") is False


async def test_first_call_pins_a_session_then_resumes_it(monkeypatch):
    p, cli = _cli(monkeypatch, _OK)
    await p.complete(model="opus", system="S", user="turn 1", session="discuss:r1")
    sid = cli.flag("--session-id", 0)
    assert sid and len(sid) == 36 and sid.count("-") == 4   # a uuid, as required
    assert p.session_active("discuss:r1") is True

    await p.complete(model="opus", system="S", user="turn 2", session="discuss:r1")
    assert cli.flag("--resume", 1) == sid
    assert "--session-id" not in cli.argv[1]


async def test_distinct_keys_get_distinct_sessions(monkeypatch):
    p, cli = _cli(monkeypatch, _OK)
    await p.complete(model="opus", system="S", user="a", session="k1")
    await p.complete(model="opus", system="S", user="b", session="k2")
    assert cli.flag("--session-id", 0) != cli.flag("--session-id", 1)


async def test_end_session_forgets_the_key(monkeypatch):
    p, _ = _cli(monkeypatch, _OK)
    await p.complete(model="opus", system="S", user="a", session="k")
    p.end_session("k")
    assert p.session_active("k") is False


async def test_failed_resume_raises_session_lost_and_drops_the_id(monkeypatch):
    lost = {"rc": 1, "err": "No conversation found with session ID abc"}
    p, _ = _cli(monkeypatch, _OK, lost)
    await p.complete(model="opus", system="S", user="a", session="k")
    with pytest.raises(SessionLost):
        await p.complete(model="opus", system="S", user="b", session="k")
    assert p.session_active("k") is False      # dropped, so a retry starts fresh


async def test_other_failures_are_not_mistaken_for_a_lost_session(monkeypatch):
    from orchestrator.core.errors import LimitExhausted, OrchestratorError
    boom = {"rc": 1, "err": "something else went wrong"}
    p, _ = _cli(monkeypatch, _OK, boom)
    await p.complete(model="opus", system="S", user="a", session="k")
    with pytest.raises(OrchestratorError) as e:
        await p.complete(model="opus", system="S", user="b", session="k")
    assert not isinstance(e.value, SessionLost)
    assert p.session_active("k") is True        # session left intact

    limit = {"rc": 1, "err": "weekly limit reached; no conversation"}
    p2, _ = _cli(monkeypatch, _OK, limit)
    await p2.complete(model="opus", system="S", user="a", session="k")
    with pytest.raises(LimitExhausted):         # limit wins over the session read
        await p2.complete(model="opus", system="S", user="b", session="k")


# ---- the planner sends a delta only when the session is live -----------------

class SessionProvider(LLMProvider):
    type = "fake_cli"

    def __init__(self, *, active=False, raise_lost=False):
        super().__init__("p", None, None)
        self._active = active
        self._raise_lost = raise_lost
        self.users: list[str] = []
        self.sessions: list[str | None] = []

    def session_active(self, key):
        return self._active

    async def complete(self, *, model, system, user, cwd=None, params=None,
                       session=None, effort=None, allowed_tools=None,
                       mcp_config=None, on_progress=None):
        self.users.append(user)
        self.sessions.append(session)
        if self._raise_lost and len(self.users) == 1:
            self._active = False
            raise SessionLost("gone")
        return LLMResult(text='{"specs": []}')


def _ctx(tmp_path, session_reuse):
    cfg = load_config()
    cfg._data["project"]["repo_path"] = str(tmp_path)
    cfg._data["run"]["session_reuse"] = session_reuse
    (tmp_path / "BACKLOG.md").write_text("- [ ] **T-1** thing\n")
    return RunContext(cfg=cfg, store=Store(tmp_path / "s.sqlite3"), git=None,
                      budget=_NullBudget(), run_id="r")


class _NullBudget:
    def record(self, **kw):
        pass


async def _ask(ctx, provider, monkeypatch, **kw):
    monkeypatch.setattr(planner_mod, "get_provider", lambda cfg, name: provider)
    return await planner_mod.plan_or_ask(ctx, **kw)


async def test_full_payload_when_session_reuse_is_off(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, session_reuse=False)
    p = SessionProvider(active=True)
    await _ask(ctx, p, monkeypatch, session="discuss:r", delta="use sqlite")
    assert "# BACKLOG" in p.users[0]            # nothing abbreviated
    assert p.sessions == [None]                 # and no session key forwarded


async def test_first_turn_sends_full_payload_even_with_reuse_on(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, session_reuse=True)
    p = SessionProvider(active=False)           # no live session yet
    await _ask(ctx, p, monkeypatch, session="discuss:r", delta="")
    assert "# BACKLOG" in p.users[0]
    assert p.sessions == ["discuss:r"]


async def test_later_turn_sends_only_the_delta(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, session_reuse=True)
    p = SessionProvider(active=True)
    await _ask(ctx, p, monkeypatch, session="discuss:r", delta="use sqlite")
    assert "# BACKLOG" not in p.users[0]        # the session already holds it
    assert "use sqlite" in p.users[0]


async def test_lost_session_falls_back_to_the_full_payload(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, session_reuse=True)
    p = SessionProvider(active=True, raise_lost=True)
    env = await _ask(ctx, p, monkeypatch, session="discuss:r", delta="use sqlite")
    assert env["specs"] == []                   # the turn still succeeded
    assert len(p.users) == 2
    assert "# BACKLOG" not in p.users[0]        # abbreviated attempt
    assert "# BACKLOG" in p.users[1]            # resent whole, exactly once
