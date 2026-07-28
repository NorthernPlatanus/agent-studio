"""Phase 3b: interactive discuss loop with a scripted planner + scripted stdin."""

from pathlib import Path

import orchestrator.nodes.discuss as discuss_mod
from orchestrator.core.config import load_config
from orchestrator.core.context import RunContext
from orchestrator.ops.store import Store


class ScriptedPlanner:
    """Stands in for plan_or_ask, returning envelopes in sequence."""
    def __init__(self, envelopes):
        self.envelopes = list(envelopes)
        self.i = 0
        self.transcripts = []
        self.deltas = []
        self.sessions = []

    async def __call__(self, ctx, *, transcript="", discussion="", only_ids=None,
                       session=None, delta=""):
        self.transcripts.append(transcript)
        self.deltas.append(delta)
        self.sessions.append(session)
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
