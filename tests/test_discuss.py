"""Phase 3b: interactive discuss loop with a scripted planner + scripted stdin."""

import time
from pathlib import Path

import pytest

import orchestrator.nodes.discuss as discuss_mod
from orchestrator.core.config import load_config
from orchestrator.core.context import RunContext
from orchestrator.core.errors import LimitExhausted, OrchestratorError, SessionLost
from orchestrator.ops.store import Store


class ScriptedPlanner:
    """Stands in for plan_or_ask, returning envelopes in sequence."""
    def __init__(self, envelopes):
        self.envelopes = list(envelopes)
        self.i = 0
        self.transcripts = []
        self.deltas = []
        self.sessions = []
        self.overrides = []

    async def __call__(self, ctx, *, transcript="", discussion="", only_ids=None,
                       session=None, delta="", effort=None, model=None,
                       session_reuse=None, on_progress=None):
        self.on_progress = on_progress
        self.transcripts.append(transcript)
        self.deltas.append(delta)
        self.sessions.append(session)
        self.overrides.append({"discussion": discussion, "only_ids": only_ids,
                               "effort": effort, "model": model,
                               "session_reuse": session_reuse})
        env = self.envelopes[min(self.i, len(self.envelopes) - 1)]
        self.i += 1
        return env


def _ctx(tmp_path):
    cfg = load_config()
    cfg._data["project"]["repo_path"] = str(tmp_path)
    store = Store(tmp_path / "s.sqlite3")
    return RunContext(cfg=cfg, store=store, git=None, budget=None, run_id="r")


async def test_clarify_then_approve_persists(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    planner = ScriptedPlanner([
        {"questions": [{"id": "q1", "q": "which db?", "why": "changes files"}],
         "assumptions": [], "specs": []},
        {"questions": [], "assumptions": ["use sqlite"],
         "specs": [{"id": "T-1", "title": "db layer", "description": "d",
                    "files_write": ["db.py"]}]},
    ])
    monkeypatch.setattr(discuss_mod, "plan_or_ask", planner)

    reads = iter(["sqlite", "y"])
    out = []
    specs = await discuss_mod.run_discuss(
        ctx, "build a db layer", read=lambda p: next(reads), write=out.append)

    assert [s["id"] for s in specs] == ["T-1"]
    assert ctx.store.get_task("T-1") is not None            # persisted
    assert ctx.store.load_discussion(ctx.cfg.project_name)   # transcript saved
    # the human's answer was folded into the transcript on the 2nd planner call
    assert "sqlite" in planner.transcripts[1]


async def test_abort_persists_nothing(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    planner = ScriptedPlanner([
        {"questions": [], "assumptions": [],
         "specs": [{"id": "T-9", "title": "x", "description": "d"}]},
    ])
    monkeypatch.setattr(discuss_mod, "plan_or_ask", planner)

    reads = iter(["abort"])
    specs = await discuss_mod.run_discuss(
        ctx, "do something", read=lambda p: next(reads), write=lambda *_: None)

    assert specs == []
    assert ctx.store.get_task("T-9") is None


# ---- session continuity through the loop (item 2 step 4) --------------------

async def test_loop_passes_a_scoped_session_and_the_newest_answer_as_delta(
        tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    planner = ScriptedPlanner([
        {"questions": [{"id": "q1", "q": "which db?"}], "assumptions": [], "specs": []},
        {"questions": [{"id": "q2", "q": "which orm?"}], "assumptions": [], "specs": []},
        {"questions": [], "assumptions": [],
         "specs": [{"id": "T-1", "title": "db", "description": "d",
                    "files_write": ["db.py"]}]},
    ])
    monkeypatch.setattr(discuss_mod, "plan_or_ask", planner)
    reads = iter(["sqlite", "none", "y"])
    await discuss_mod.run_discuss(ctx, "build a db layer",
                                  read=lambda p: next(reads), write=lambda s: None)

    assert planner.sessions == ["discuss:r"] * 3          # one key, scoped to the run
    # turn 1 has no delta (nothing to continue from); later turns carry only the
    # human's newest answer — the rest of the context lives in the session.
    assert planner.deltas == ["", "sqlite", "none"]


async def test_loop_ends_the_provider_session_when_it_finishes(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    ended = []
    monkeypatch.setattr(discuss_mod, "plan_or_ask", ScriptedPlanner([
        {"questions": [], "assumptions": [], "specs": []}]))
    monkeypatch.setattr(discuss_mod, "get_provider",
                        lambda cfg, name: type("P", (), {
                            "end_session": lambda self, k: ended.append(k)})())
    reads = iter(["abort"])
    await discuss_mod.run_discuss(ctx, "x", read=lambda p: next(reads),
                                  write=lambda s: None)
    assert ended == ["discuss:r"]     # not left open for the next discuss to inherit


async def test_typing_edit_still_asks_for_the_note(tmp_path, monkeypatch):
    """The CLI's two-step form: `edit` at the prompt, then the note.

    Kept working alongside the chat's one-shot form (see `run_discuss`), because
    the CLI's prompt literally offers the word `edit` as an answer.
    """
    ctx = _ctx(tmp_path)
    proposal = {"questions": [], "assumptions": [],
                "specs": [{"id": "T-1", "title": "db layer", "description": "d",
                           "files_write": ["db.py"]}]}
    planner = ScriptedPlanner([proposal, proposal])
    monkeypatch.setattr(discuss_mod, "plan_or_ask", planner)

    reads = iter(["edit", "split it in two", "y"])
    await discuss_mod.run_discuss(ctx, "build a db layer",
                                  read=lambda p: next(reads), write=lambda *_: None)

    assert "EDIT: split it in two" in planner.transcripts[1]
    assert planner.deltas[1] == "EDIT: split it in two"


async def test_a_revision_typed_in_one_go_is_the_note(tmp_path, monkeypatch):
    """A chat client has no `[y/edit/abort]` prompt to type at — it sends the
    revision itself. Reading that as a *choice* and then asking again threw the
    note away and blocked on a read the operator had already answered."""
    ctx = _ctx(tmp_path)
    proposal = {"questions": [], "assumptions": [],
                "specs": [{"id": "T-1", "title": "db layer", "description": "d",
                           "files_write": ["db.py"]}]}
    planner = ScriptedPlanner([proposal, proposal])
    monkeypatch.setattr(discuss_mod, "plan_or_ask", planner)

    # Two reads, not three: the revision is the answer to the only question asked.
    reads = iter(["Split it in two", "y"])
    await discuss_mod.run_discuss(ctx, "build a db layer",
                                  read=lambda p: next(reads), write=lambda *_: None)

    # Verbatim, not lower-cased: the note is prose for the planner, not a keyword.
    assert "EDIT: Split it in two" in planner.transcripts[1]
    assert planner.deltas[1] == "EDIT: Split it in two"


# ---- a failed planner turn does not end the conversation ---------------------
# Before this, ANY exception out of plan_or_ask propagated to the API's `_drive`,
# which marked the session `failed` and closed it. A planner turn is minutes long
# and fails for reasons that have nothing to do with the operator (a wedge, a
# 5xx), and the transcript was saved only AFTER a successful call — so a failure
# on turn 1 threw away the conversation and persisted nothing.

class FlakyPlanner:
    """Raises on the first N calls, then behaves."""

    def __init__(self, failures: int, envelope: dict, exc=None):
        self.failures = failures
        self.envelope = envelope
        self.exc = exc or OrchestratorError("claude CLI produced no output for 300s")
        self.calls = 0
        self.transcripts: list[str] = []

    async def __call__(self, ctx, *, transcript="", **kw):
        self.calls += 1
        self.transcripts.append(transcript)
        if self.calls <= self.failures:
            raise self.exc
        return self.envelope


_ONE_SPEC = {"questions": [], "assumptions": [],
             "specs": [{"id": "T-5", "title": "x", "description": "d",
                        "files_write": ["a.py"]}]}


async def test_a_failed_turn_keeps_the_session_and_retries(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    planner = FlakyPlanner(1, _ONE_SPEC)
    monkeypatch.setattr(discuss_mod, "plan_or_ask", planner)

    frames: list[dict] = []
    reads = iter(["", "y"])          # empty answer at the retry prompt = retry
    specs = await discuss_mod.run_discuss(
        ctx, "build it", read=lambda p: next(reads), write=lambda _l: None,
        emit=frames.append)

    assert [s["id"] for s in specs] == ["T-5"]        # it recovered
    assert planner.calls == 2
    kinds = [f["kind"] for f in frames]
    assert "turn_failed" in kinds
    assert {"kind": "awaiting", "expects": "retry"} in frames


async def test_the_transcript_is_saved_before_the_call_not_after(tmp_path, monkeypatch):
    """The operator's opening message has to survive a turn that never returns."""
    ctx = _ctx(tmp_path)
    planner = FlakyPlanner(1, _ONE_SPEC)
    monkeypatch.setattr(discuss_mod, "plan_or_ask", planner)

    saved: list[str] = []
    real_save = ctx.store.save_discussion

    def spy(session, text):
        saved.append(text)
        return real_save(session, text)

    monkeypatch.setattr(ctx.store, "save_discussion", spy)
    reads = iter(["", "y"])
    await discuss_mod.run_discuss(ctx, "build it", read=lambda p: next(reads),
                                  write=lambda _l: None)
    # saved once BEFORE the failing first call, carrying the opening message
    assert saved and "build it" in saved[0]


async def test_typing_at_the_retry_prompt_adds_context_for_the_next_attempt(
        tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    planner = FlakyPlanner(1, _ONE_SPEC)
    monkeypatch.setattr(discuss_mod, "plan_or_ask", planner)

    reads = iter(["only look at the api dir", "y"])
    await discuss_mod.run_discuss(ctx, "build it", read=lambda p: next(reads),
                                  write=lambda _l: None)
    assert "only look at the api dir" in planner.transcripts[1]


async def test_aborting_a_failed_turn_stops_cleanly(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    planner = FlakyPlanner(5, _ONE_SPEC)
    monkeypatch.setattr(discuss_mod, "plan_or_ask", planner)

    frames: list[dict] = []
    specs = await discuss_mod.run_discuss(
        ctx, "build it", read=lambda p: "abort", write=lambda _l: None,
        emit=frames.append)
    assert specs == []
    assert planner.calls == 1
    assert [f["kind"] for f in frames][-1] == "aborted"


async def test_control_flow_errors_still_propagate(tmp_path, monkeypatch):
    """SessionLost is the provider telling plan_or_ask to resend a full payload,
    not a turn the operator should be asked about."""
    ctx = _ctx(tmp_path)
    planner = FlakyPlanner(1, _ONE_SPEC, exc=SessionLost("gone"))
    monkeypatch.setattr(discuss_mod, "plan_or_ask", planner)
    with pytest.raises(SessionLost):
        await discuss_mod.run_discuss(ctx, "build it", read=lambda p: "",
                                      write=lambda _l: None)


async def test_progress_events_reach_the_emitter(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)

    async def planner(ctx_, *, on_progress=None, **kw):
        on_progress({"phase": "tool", "tool": "Read", "target": "a.py"})
        return _ONE_SPEC

    monkeypatch.setattr(discuss_mod, "plan_or_ask", planner)
    frames: list[dict] = []
    await discuss_mod.run_discuss(ctx, "build it", read=lambda p: "y",
                                  write=lambda _l: None, emit=frames.append)
    assert {"kind": "progress", "phase": "tool", "tool": "Read",
            "target": "a.py"} in frames


async def test_a_failed_turn_drops_provider_continuity(tmp_path, monkeypatch):
    """`session_reuse` sends only the newest human turn once the provider says a
    session is live. After a failed turn that assumption is unsafe — the call may
    have died before its payload landed — so the retry must be self-contained."""
    ctx = _ctx(tmp_path)
    planner = FlakyPlanner(1, _ONE_SPEC)
    monkeypatch.setattr(discuss_mod, "plan_or_ask", planner)
    ended: list[str] = []
    monkeypatch.setattr(discuss_mod, "get_provider",
                        lambda cfg, name: type("P", (), {
                            "end_session": lambda self, k: ended.append(k)})())

    reads = iter(["more context", "y"])
    await discuss_mod.run_discuss(ctx, "build it", read=lambda p: next(reads),
                                  write=lambda _l: None)
    assert "discuss:r" in ended          # continuity dropped after the failure


# ---- freezing until the usage window resets ---------------------------------
# A planning session can outlast the five-hour quota it is spending (one measured
# turn: ~520k subscription tokens). Failing at the limit throws away a
# conversation whose only problem is that the clock has not rolled over.

class LimitedPlanner:
    """Raises LimitExhausted once, then behaves."""

    def __init__(self, envelope, *, resets_in: float | None, limit_type="five_hour"):
        self.envelope = envelope
        self.resets_in = resets_in
        self.limit_type = limit_type
        self.calls = 0

    async def __call__(self, ctx, **kw):
        self.calls += 1
        if self.calls == 1:
            raise LimitExhausted(
                "claude CLI limit: usage limit reached",
                resets_at=(time.time() + self.resets_in
                           if self.resets_in is not None else None),
                limit_type=self.limit_type)
        return self.envelope


async def test_the_loop_freezes_until_the_window_resets_then_retries(
        tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    planner = LimitedPlanner(_ONE_SPEC, resets_in=0.05)
    monkeypatch.setattr(discuss_mod, "plan_or_ask", planner)

    frames: list[dict] = []
    specs = await discuss_mod.run_discuss(
        ctx, "build it", read=lambda p: "y", write=lambda _l: None,
        emit=frames.append)

    assert [s["id"] for s in specs] == ["T-5"]      # it waited, then succeeded
    assert planner.calls == 2
    paused = next(f for f in frames if f["kind"] == "limit_paused")
    assert paused["limit_type"] == "five_hour" and paused["seconds"] > 0
    assert {"kind": "awaiting", "expects": "frozen"} in frames


async def test_a_limit_with_no_reported_reset_is_never_waited_on(
        tmp_path, monkeypatch):
    """Inventing a reset time would freeze the session for a guess."""
    ctx = _ctx(tmp_path)
    planner = LimitedPlanner(_ONE_SPEC, resets_in=None)
    monkeypatch.setattr(discuss_mod, "plan_or_ask", planner)

    frames: list[dict] = []
    reads = iter(["", "y"])
    await discuss_mod.run_discuss(ctx, "build it", read=lambda p: next(reads),
                                  write=lambda _l: None, emit=frames.append)
    kinds = [f["kind"] for f in frames]
    assert "limit_paused" not in kinds     # fell through to the ordinary path
    assert "turn_failed" in kinds


async def test_a_reset_further_out_than_the_cap_is_not_waited_on(
        tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    ctx.cfg._data["run"]["limit_freeze_max_s"] = 60
    planner = LimitedPlanner(_ONE_SPEC, resets_in=3600)   # an hour away
    monkeypatch.setattr(discuss_mod, "plan_or_ask", planner)

    frames: list[dict] = []
    reads = iter(["", "y"])
    await discuss_mod.run_discuss(ctx, "build it", read=lambda p: next(reads),
                                  write=lambda _l: None, emit=frames.append)
    assert "limit_paused" not in [f["kind"] for f in frames]
    assert "turn_failed" in [f["kind"] for f in frames]


async def test_freezing_can_be_disabled(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    ctx.cfg._data["run"]["limit_freeze_max_s"] = 0
    planner = LimitedPlanner(_ONE_SPEC, resets_in=0.05)
    monkeypatch.setattr(discuss_mod, "plan_or_ask", planner)

    frames: list[dict] = []
    reads = iter(["", "y"])
    await discuss_mod.run_discuss(ctx, "build it", read=lambda p: next(reads),
                                  write=lambda _l: None, emit=frames.append)
    assert "limit_paused" not in [f["kind"] for f in frames]


async def test_the_operator_can_abort_a_freeze(tmp_path, monkeypatch):
    """The wait must not be a black hole: a closed tab delivers an abort onto
    the same queue the loop reads, and it has to be heard mid-freeze."""
    ctx = _ctx(tmp_path)
    planner = LimitedPlanner(_ONE_SPEC, resets_in=30)     # long enough to matter
    monkeypatch.setattr(discuss_mod, "plan_or_ask", planner)

    async def read(_prompt):
        return "abort"

    frames: list[dict] = []
    specs = await discuss_mod.run_discuss(ctx, "build it", read=read,
                                          write=lambda _l: None,
                                          emit=frames.append)
    assert specs == [] and planner.calls == 1
    assert [f["kind"] for f in frames][-1] == "aborted"
