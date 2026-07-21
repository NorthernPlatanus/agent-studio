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

    async def __call__(self, ctx, *, transcript="", discussion="", only_ids=None):
        self.transcripts.append(transcript)
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
