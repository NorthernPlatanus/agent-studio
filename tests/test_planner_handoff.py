"""Session expiry, the handoff digest, and the specs-block budget.

Together these are one mechanism: a planner session is reused while its prompt
cache is warm, abandoned when it is not, and what survives the abandonment is a
digest rather than a replay of the dead conversation.
"""

import json

import pytest

import orchestrator.nodes.planner as planner_mod
from orchestrator.core.config import Section
from orchestrator.ops import handoff
from orchestrator.providers.claude_cli import (
    DEFAULT_SESSION_MAX_IDLE_S, ClaudeCliProvider, _LiveSession)
from tests.conftest import FakeCli, stream_json
from tests.test_session_reuse import SessionProvider, _ask, _ctx

_OK = {"out": stream_json({"result": '{"specs": []}',
                           "usage": {"input_tokens": 1, "output_tokens": 1}})}


def _cli(monkeypatch, *specs, **pcfg):
    cli = FakeCli(*specs).install(monkeypatch)
    base = {"type": "claude_cli", "binary": "claude", "timeout_s": 600,
            "retry_attempts": 1}
    base.update(pcfg)
    return ClaudeCliProvider("claude_cli", Section(base), Section({"mcp": {}})), cli


# ---- session expiry ----------------------------------------------------------

async def test_warm_session_is_resumed(monkeypatch):
    p, cli = _cli(monkeypatch, _OK, _OK)
    await p.complete(model="opus", system="S", user="1", session="k")
    sid = cli.flag("--session-id", 0)
    assert p.session_active("k") is True
    await p.complete(model="opus", system="S", user="2", session="k")
    assert cli.flag("--resume", 1) == sid


async def test_cold_session_is_dropped_not_resumed(monkeypatch):
    p, cli = _cli(monkeypatch, _OK, _OK, session_max_idle_s=1800)
    await p.complete(model="opus", system="S", user="1", session="k")
    first = cli.flag("--session-id", 0)

    # 31 minutes of silence: past the window, so the id must not be reused.
    p._sessions["k"] = _LiveSession(p._sessions["k"].sid,
                                    p._sessions["k"].last_seen - 1860)
    assert p.session_active("k") is False        # and the check itself expires it

    await p.complete(model="opus", system="S", user="2", session="k")
    assert "--resume" not in cli.argv[1]
    second = cli.flag("--session-id", 1)
    assert second and second != first


async def test_expiry_applies_even_without_a_session_active_check(monkeypatch):
    """`complete` is reached directly on a loop's first turn — it must expire the
    session itself rather than trusting the caller to have asked."""
    p, cli = _cli(monkeypatch, _OK, _OK, session_max_idle_s=1800)
    await p.complete(model="opus", system="S", user="1", session="k")
    p._sessions["k"] = _LiveSession(p._sessions["k"].sid,
                                    p._sessions["k"].last_seen - 99999)
    await p.complete(model="opus", system="S", user="2", session="k")
    assert "--resume" not in cli.argv[1]


async def test_zero_disables_expiry(monkeypatch):
    p, cli = _cli(monkeypatch, _OK, _OK, session_max_idle_s=0)
    await p.complete(model="opus", system="S", user="1", session="k")
    p._sessions["k"] = _LiveSession(p._sessions["k"].sid, -99999.0)
    assert p.session_active("k") is True
    await p.complete(model="opus", system="S", user="2", session="k")
    assert cli.flag("--resume", 1)


async def test_a_successful_turn_refreshes_the_clock(monkeypatch):
    p, cli = _cli(monkeypatch, _OK, _OK, session_max_idle_s=1800)
    await p.complete(model="opus", system="S", user="1", session="k")
    sid = p._sessions["k"].sid
    aged = p._sessions["k"].last_seen - 1700          # old, but still inside
    p._sessions["k"] = _LiveSession(sid, aged)
    await p.complete(model="opus", system="S", user="2", session="k")
    assert p._sessions["k"].last_seen > aged          # stamped by the new turn
    assert p._sessions["k"].sid == sid                # and the id is not re-guessed


async def test_default_window_sits_under_the_cache_ttl():
    assert 0 < DEFAULT_SESSION_MAX_IDLE_S < 3600


async def test_config_ttl_stays_above_the_provider_window():
    """Both clocks measure the same silence; the API's must be the outer one."""
    from orchestrator.api.discuss import IDLE_TTL_S
    assert IDLE_TTL_S > DEFAULT_SESSION_MAX_IDLE_S


# ---- the specs block ---------------------------------------------------------

def _specs(n, **extra):
    return [dict({"id": f"T-{i:03d}", "title": f"task {i}" + "x" * 200,
                  "description": "d" * 400, "status": "ready",
                  "files_write": [f"src/{i}.ts"],
                  "acceptance": ["a" * 300]}, **extra) for i in range(n)]


def test_specs_block_is_always_valid_json():
    block = planner_mod._specs_block(_specs(60), None, budget=4000)
    payload = block.split("\n\n(")[0]
    assert isinstance(json.loads(payload), list)      # never sliced mid-token


def test_specs_block_announces_what_it_dropped():
    block = planner_mod._specs_block(_specs(60), None, budget=4000)
    assert "omitted" in block and "60 in total" in block


def test_specs_block_keeps_focus_specs_whole():
    specs = _specs(30)
    block = planner_mod._specs_block(specs, ["T-005"], budget=20000)
    kept = json.loads(block.split("\n\n(")[0])
    focus = [s for s in kept if s["id"] == "T-005"]
    assert focus and "acceptance" in focus[0]         # full detail retained
    others = [s for s in kept if s["id"] != "T-005"]
    assert others and all("acceptance" not in s for s in others)


def test_specs_block_keeps_needs_plan_whole_without_only_ids():
    specs = _specs(4) + _specs(1, status="needs_plan")
    block = planner_mod._specs_block(specs, None, budget=20000)
    kept = json.loads(block.split("\n\n(")[0])
    pending = [s for s in kept if s.get("status") == "needs_plan"]
    assert pending and "acceptance" in pending[0]


def test_specs_block_fits_the_real_board_budget():
    block = planner_mod._specs_block(_specs(42), None)
    assert len(block) <= planner_mod.SPECS_BUDGET_CHARS + 200   # + the note


def test_specs_block_survives_an_impossible_budget():
    block = planner_mod._specs_block(_specs(5), None, budget=10)
    assert "omitted" in block                          # degrades, never raises


# ---- the handoff -------------------------------------------------------------

_ENV = {"questions": [{"id": "q1", "q": "which db?"}],
        "assumptions": ["sqlite is fine"],
        "specs": [{"id": "T-1", "title": "db layer", "complexity": "m",
                   "risk": "low", "description": "d", "files_write": ["db.py"]}]}


class EnvProvider(SessionProvider):
    """SessionProvider answering with real envelopes instead of `{specs: []}`.

    `envs` is a sequence consumed one per call; the last one repeats, so a
    discuss loop always reaches a terminating state.
    """

    def __init__(self, env=None, envs=None, **kw):
        super().__init__(**kw)
        self._envs = list(envs) if envs else [env if env is not None else _ENV]

    async def complete(self, **kw):
        self.users.append(kw["user"])
        self.sessions.append(kw.get("session"))
        from orchestrator.providers.base import LLMResult
        env = self._envs[min(len(self.users) - 1, len(self._envs) - 1)]
        return LLMResult(text=json.dumps(env))


_ASK = {"questions": [{"id": "q1", "q": "which db?"}], "assumptions": [],
        "specs": []}
_PROPOSE = {"questions": [], "assumptions": [],
            "specs": [{"id": "T-9", "title": "t", "description": "d",
                       "files_write": ["a.py"]}]}


async def test_a_turn_records_a_digest_and_a_snapshot(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, session_reuse=False)
    p = EnvProvider()
    await _ask(ctx, p, monkeypatch, session="s", delta="")

    row = ctx.store.load_handoff(ctx.cfg.project_name)
    assert row and "sqlite is fine" in row["digest"]
    assert "T-1: db layer" in row["digest"]
    saved = json.loads(handoff.snapshot_path(ctx).read_text())
    assert saved["specs"][0]["description"] == "d"    # full detail on disk only


async def test_the_digest_reaches_a_cold_start_prompt(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, session_reuse=False)
    p = EnvProvider()
    await _ask(ctx, p, monkeypatch, session="s", delta="")
    await _ask(ctx, p, monkeypatch, session="s", delta="")

    assert "# PREVIOUS PLANNING SESSION" in p.users[1]
    assert "sqlite is fine" in p.users[1]
    assert "Read it ONLY if" in p.users[1]             # don't open it by reflex


async def test_no_digest_block_before_any_turn_has_run(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, session_reuse=False)
    p = SessionProvider()
    await _ask(ctx, p, monkeypatch, session="s", delta="")
    assert "# PREVIOUS PLANNING SESSION" not in p.users[0]


def test_digest_is_bounded(tmp_path):
    env = {"questions": [], "assumptions": ["a" * 9000],
           "specs": [{"id": f"T-{i}", "title": "t"} for i in range(500)]}
    text = handoff._digest(env, env["specs"], 0)
    assert len(text) <= handoff.MAX_DIGEST_CHARS + 40


async def test_a_failed_snapshot_write_does_not_fail_the_turn(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, session_reuse=False)

    def boom(_ctx):
        raise OSError("disk is on fire")
    monkeypatch.setattr(handoff, "snapshot_path", boom)

    env = await _ask(ctx, EnvProvider(), monkeypatch, session="s", delta="")
    assert env["specs"][0]["id"] == "T-1"              # the turn still returned


# ---- regressions from defaulting session_reuse ON ----------------------------

async def _drive(ctx, monkeypatch, provider, opts, replies):
    """Run one discuss loop with scripted replies, mutating `opts` as we go."""
    import orchestrator.nodes.discuss as discuss_mod
    monkeypatch.setattr(planner_mod, "get_provider", lambda cfg, name: provider)
    it = iter(replies)

    def read(_prompt):
        step = next(it)
        if callable(step):
            return step()
        return step

    return await discuss_mod.run_discuss(
        ctx, "plan it", read=read, write=lambda *_: None, settings=lambda: opts)


async def test_attachments_added_mid_conversation_survive_a_resumed_turn(
        tmp_path, monkeypatch):
    """A file the operator attaches on turn N has never been seen by a session
    opened on turn 1 — a resumed turn that sends only the delta drops it."""
    from orchestrator.nodes.discuss import DiscussSettings, PinnedFile

    ctx = _ctx(tmp_path, session_reuse=True)
    p = EnvProvider(envs=[_ASK, _PROPOSE], active=True)
    opts = DiscussSettings()

    def attach_then_answer():
        opts.pinned = [PinnedFile(path="uploaded/crash.log",
                                  text="SEGFAULT at frame 91")]
        return "here is the log"

    await _drive(ctx, monkeypatch, p, opts, [attach_then_answer, "abort"])
    assert any("SEGFAULT at frame 91" in u for u in p.users[1:]), \
        "the attachment never reached the planner"


async def test_a_changed_note_survives_a_resumed_turn(tmp_path, monkeypatch):
    from orchestrator.nodes.discuss import DiscussSettings

    ctx = _ctx(tmp_path, session_reuse=True)
    p = EnvProvider(envs=[_ASK, _PROPOSE], active=True)
    opts = DiscussSettings()

    def set_note_then_answer():
        opts.note = "ship behind a feature flag"
        return "ok"

    await _drive(ctx, monkeypatch, p, opts, [set_note_then_answer, "abort"])
    assert any("ship behind a feature flag" in u for u in p.users[1:]), \
        "the operator's new note never reached the planner"


async def test_a_question_only_turn_does_not_erase_the_recorded_specs(
        tmp_path, monkeypatch):
    """Turn 1 proposes, turn 2 asks a follow-up. The digest is what survives an
    expired session, so it must not forget the proposal on the way."""
    ctx = _ctx(tmp_path, session_reuse=False)
    p = EnvProvider(envs=[_ENV, _ASK])
    await _ask(ctx, p, monkeypatch, session="s", delta="")
    await _ask(ctx, p, monkeypatch, session="s", delta="")

    row = ctx.store.load_handoff(ctx.cfg.project_name)
    assert "T-1: db layer" in row["digest"], "the proposal was forgotten"
    assert "which db?" in row["digest"]      # and the new question is there too


async def test_a_concluded_conversation_leaves_no_handoff(tmp_path, monkeypatch):
    """The digest bridges an INTERRUPTED conversation. Once specs are applied the
    conversation is over, and replaying its open questions into the next one
    would have the planner re-answering settled ground."""
    from orchestrator.nodes.discuss import DiscussSettings

    ctx = _ctx(tmp_path, session_reuse=True)
    p = EnvProvider(envs=[_ASK, _PROPOSE], active=True)
    await _drive(ctx, monkeypatch, p, DiscussSettings(), ["an answer", "y"])

    row = ctx.store.load_handoff(ctx.cfg.project_name)
    assert not (row and row["digest"].strip()), "stale digest outlived its session"


async def test_an_aborted_conversation_leaves_no_handoff(tmp_path, monkeypatch):
    from orchestrator.nodes.discuss import DiscussSettings

    ctx = _ctx(tmp_path, session_reuse=True)
    p = EnvProvider(envs=[_ASK, _PROPOSE], active=True)
    await _drive(ctx, monkeypatch, p, DiscussSettings(), ["an answer", "abort"])

    row = ctx.store.load_handoff(ctx.cfg.project_name)
    assert not (row and row["digest"].strip())


async def test_a_non_oserror_snapshot_failure_does_not_fail_the_turn(
        tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, session_reuse=False)

    def boom(_ctx):
        raise AttributeError("store has no path")
    monkeypatch.setattr(handoff, "snapshot_path", boom)

    env = await _ask(ctx, EnvProvider(), monkeypatch, session="s", delta="")
    assert env["specs"][0]["id"] == "T-1"


def test_digest_respects_its_cap_with_no_newlines():
    env = {"questions": [], "assumptions": ["a" * 20000], "specs": []}
    assert len(handoff._digest(env, [], 0)) <= handoff.MAX_DIGEST_CHARS


async def test_a_question_only_turn_does_not_erase_the_assumptions(
        tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, session_reuse=False)
    p = EnvProvider(envs=[_ENV, _ASK])
    await _ask(ctx, p, monkeypatch, session="s", delta="")
    await _ask(ctx, p, monkeypatch, session="s", delta="")
    row = ctx.store.load_handoff(ctx.cfg.project_name)
    assert "sqlite is fine" in row["digest"]


async def test_stale_questions_are_always_replaced(tmp_path, monkeypatch):
    """Unlike specs and assumptions, a question is a live prompt to the operator
    — carrying an answered one forward would ask it twice."""
    ctx = _ctx(tmp_path, session_reuse=False)
    p = EnvProvider(envs=[_ENV, _PROPOSE])
    await _ask(ctx, p, monkeypatch, session="s", delta="")
    await _ask(ctx, p, monkeypatch, session="s", delta="")
    row = ctx.store.load_handoff(ctx.cfg.project_name)
    assert "which db?" not in row["digest"]


# ---- backlog: completed items are archaeology, not planning surface ----------

_BOARD = """# Backlog

## M3 — The World
- [x] **T-101** Load the GLB via GLTFLoader; render it in features/track/ui. {done1} See ADR-0023 for the reader contract.
- [ ] **T-140** Tune drift scoring on the real road. {open1}
- [~] **T-120** Load the low-poly AE86 behind the render seam. {open2}

## M4 — Gameplay
- [!] **T-112** Blocked on the asset pipeline.
- [x] **T-099** Short done item.
- [x] **T-098** Another finished one. {done2}
"""


def _board(tmp_path, ctx):
    text = _BOARD.format(done1="D1" * 450, done2="D2" * 450,
                         open1="O1" * 450, open2="O2" * 450)
    (tmp_path / "BACKLOG.md").write_text(text)
    ctx.cfg._data["project"]["backlog_file"] = "BACKLOG.md"
    return text


def test_done_bodies_are_dropped_and_open_bodies_are_not(tmp_path):
    ctx = _ctx(tmp_path, session_reuse=False)
    raw = _board(tmp_path, ctx)
    out = planner_mod._backlog_excerpt(ctx, None)

    assert "D1" * 450 not in out and "D2" * 450 not in out   # finished: dropped
    assert "O1" * 450 in out and "O2" * 450 in out            # open: byte-identical
    assert len(out) < len(raw) * 0.7                          # and it is the bulk
    for line in raw.splitlines():
        if line.startswith(("- [ ]", "- [~]", "- [!]", "#")):
            assert line in out


def test_every_id_survives_the_collapse(tmp_path):
    ctx = _ctx(tmp_path, session_reuse=False)
    _board(tmp_path, ctx)
    out = planner_mod._backlog_excerpt(ctx, None)
    for tid in ("T-101", "T-140", "T-120", "T-112", "T-099"):
        assert f"**{tid}**" in out


def test_references_are_rescued_from_the_dropped_tail(tmp_path):
    ctx = _ctx(tmp_path, session_reuse=False)
    _board(tmp_path, ctx)
    out = planner_mod._backlog_excerpt(ctx, None)
    assert "ADR-0023" in out, "the pointer worth keeping was truncated away"


def test_short_done_items_are_left_alone(tmp_path):
    ctx = _ctx(tmp_path, session_reuse=False)
    _board(tmp_path, ctx)
    out = planner_mod._backlog_excerpt(ctx, None)
    assert "- [x] **T-099** Short done item." in out


def test_the_collapse_is_announced(tmp_path):
    ctx = _ctx(tmp_path, session_reuse=False)
    _board(tmp_path, ctx)
    out = planner_mod._backlog_excerpt(ctx, None)
    assert "abbreviated" in out and "BACKLOG.md" in out


def test_headings_and_order_are_preserved(tmp_path):
    ctx = _ctx(tmp_path, session_reuse=False)
    raw = _board(tmp_path, ctx)
    out = planner_mod._backlog_excerpt(ctx, None)
    ids = lambda t: [m.group(1) for m in
                     __import__("re").finditer(r"\*\*([A-Z]+-\d+)\*\*", t)]
    assert ids(out) == ids(raw)
    assert [l for l in out.splitlines() if l.startswith("#")] == \
           [l for l in raw.splitlines() if l.startswith("#")]


def test_only_ids_path_is_unaffected(tmp_path):
    ctx = _ctx(tmp_path, session_reuse=False)
    _board(tmp_path, ctx)
    out = planner_mod._backlog_excerpt(ctx, ["T-101"])
    assert "D1" * 450 in out          # requested explicitly, so shown in full
    assert "T-140" not in out


def test_an_unparseable_backlog_config_is_left_alone(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, session_reuse=False)
    raw = _board(tmp_path, ctx)
    monkeypatch.setattr(planner_mod, "make_backlog",
                        lambda _cfg: (_ for _ in ()).throw(TypeError("bad")))
    assert planner_mod._backlog_excerpt(ctx, None) == raw


# ---- shape tolerance --------------------------------------------------------

def test_specs_block_handles_an_empty_queue():
    assert planner_mod._specs_block([], None) == "[]"


def test_specs_block_tolerates_malformed_specs():
    for specs in ([{"title": "no id"}], [{"id": None}], [{"id": "T-1"}],
                  [{"id": "T-1", "deps": {"nested": 1}}]):
        block = planner_mod._specs_block(specs, None)
        json.loads(block.split("\n\n(")[0])          # still valid JSON


def test_digest_tolerates_bare_string_questions():
    """A model that answers with strings instead of {id, q} must thin the digest,
    not lose it — `record` would otherwise swallow the error and save nothing."""
    text = handoff._digest({"questions": ["which db?"], "assumptions": [],
                            "specs": []}, [], 0)
    assert "which db?" in text


def test_only_ids_promotes_a_child_spec_by_its_parent():
    specs = [{"id": "T-131a", "parent_id": "T-131", "title": "child",
              "description": "d", "acceptance": ["a"]},
             {"id": "T-200", "title": "other", "description": "d",
              "acceptance": ["a"]}]
    kept = json.loads(planner_mod._specs_block(specs, ["T-131"]).split("\n\n(")[0])
    assert "acceptance" in kept[0] and "acceptance" not in kept[1]
