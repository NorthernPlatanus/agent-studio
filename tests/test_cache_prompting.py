"""Phase 2.5: stable-prefix prompting, per-candidate warm messages, escalation
routing, and the usage-table migration."""

from pathlib import Path

import pytest

import orchestrator.nodes.worker as worker_mod
from orchestrator.core.config import Config, load_config
from orchestrator.core.context import RunContext
from orchestrator.engine.graph import (decide_after_collect, escalation_ready,
                                        plan_dispatch)
from orchestrator.ops.store import Store
from orchestrator.providers.base import LLMProvider, LLMResult

from tests.test_worker_loop import (FakeBudget, FakeGit, FakeStore,
                                    ScriptedProvider, _spec)


# ---- base complete_chat flatten (CLI providers) -----------------------------

class _Cliish(LLMProvider):
    type = "cli_fake"

    def __init__(self):
        super().__init__("p", None, None)
        self.seen = None

    async def complete(self, *, model, system, user, cwd=None, params=None):
        self.seen = {"system": system, "user": user, "cwd": cwd, "params": params}
        return LLMResult(text="ok")


async def test_base_complete_chat_flattens_for_cli():
    p = _Cliish()
    msgs = [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"}]
    res = await p.complete_chat(model="m", system="SYS", messages=msgs, cwd="/wt")
    assert res.text == "ok"
    assert p.seen["system"] == "SYS"
    assert p.seen["user"] == "A\n\nB\n\nC"     # turns flattened, system separate
    assert p.seen["cwd"] == "/wt"


# ---- stable prefix + warm messages ------------------------------------------

def _ctx(tmp_path, **run_over):
    cfg = load_config()
    d = cfg._data
    d["project"]["repo_path"] = str(tmp_path / "repo")
    (tmp_path / "repo").mkdir(exist_ok=True)
    d["gate"]["commands"] = []
    d["roles"]["worker"] = {"default": "w", "candidates": ["w"]}
    d["worker_models"] = {"w": {"provider": "fake", "model": "m"}}
    d["run"].update(run_over)
    return RunContext(cfg=cfg, store=FakeStore(), git=FakeGit(tmp_path / "wt"),
                      budget=FakeBudget(), run_id="r")


async def _run(ctx, provider, monkeypatch, *, attempt=1, messages=None, feedback=""):
    monkeypatch.setattr(worker_mod, "get_provider", lambda cfg, name: provider)
    spec = _spec()
    spec["files_read"] = ["z.txt", "a.txt"]     # deliberately unsorted
    payload = {"run_id": "r", "task_id": "T-1", "spec": spec, "cand_id": "w",
               "attempt": attempt, "feedback": feedback, "messages": messages or []}
    return await worker_mod.run_candidate(ctx, payload)


async def test_files_sorted_once_in_stable_prefix(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    provider = ScriptedProvider(['<file path="a.txt">\nx\n</file>'])
    out = await _run(ctx, provider, monkeypatch)
    prefix = out["candidates"][0]["messages"][0]["content"]
    # a.txt listed before z.txt regardless of files_read order
    assert prefix.index('<source path="a.txt">') < prefix.index('<source path="z.txt">')


async def test_prefix_byte_identical_across_attempts(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    p1 = ScriptedProvider(['<file path="a.txt">\nx\n</file>'])
    out1 = await _run(ctx, p1, monkeypatch)
    msgs1 = out1["candidates"][0]["messages"]
    prefix1 = msgs1[0]["content"]

    # retry (attempt 2) continues the SAME candidate's messages with feedback
    p2 = ScriptedProvider(['<file path="a.txt">\ny\n</file>'])
    out2 = await _run(ctx, p2, monkeypatch, attempt=2, messages=msgs1, feedback="fix it")
    msgs2 = out2["candidates"][0]["messages"]
    assert msgs2[0]["content"] == prefix1          # frozen prefix, byte-identical
    assert any("RETRY FEEDBACK" in m["content"] and "fix it" in m["content"]
               for m in msgs2 if m["role"] == "user")
    assert len(msgs2) > len(msgs1)                  # grew by append


async def test_input_messages_not_mutated(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    p1 = ScriptedProvider(['<file path="a.txt">\nx\n</file>'])
    out1 = await _run(ctx, p1, monkeypatch)
    msgs1 = out1["candidates"][0]["messages"]
    snapshot = [dict(m) for m in msgs1]
    p2 = ScriptedProvider(['<file path="a.txt">\ny\n</file>'])
    await _run(ctx, p2, monkeypatch, attempt=2, messages=msgs1, feedback="fix")
    assert msgs1 == snapshot                        # earlier turns never mutated


async def test_retrieved_files_not_folded_into_prefix(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    # worktree has a neighbor file the worker reads mid-chain
    wt = tmp_path / "wt" / "t-1-w"
    provider = ScriptedProvider([
        "<read>neighbor.txt</read>",
        '<file path="a.txt">\nx\n</file>',
    ])
    monkeypatch.setattr(worker_mod, "get_provider", lambda cfg, name: provider)
    # create the neighbor after worktree exists (FakeGit makes the dir on call 1)
    orig = ctx.git.create_worktree
    def make(name):
        p = orig(name)
        (p / "neighbor.txt").write_text("SECRET-NEIGHBOR")
        return p
    ctx.git.create_worktree = make
    out = await _run(ctx, provider, monkeypatch)
    msgs = out["candidates"][0]["messages"]
    assert "SECRET-NEIGHBOR" not in msgs[0]["content"]        # not in frozen prefix
    assert any("SECRET-NEIGHBOR" in m["content"]
               for m in msgs if m["role"] == "user")          # in a later suffix turn


async def test_fresh_chain_with_feedback_includes_it(tmp_path, monkeypatch):
    # Regression (F2): the escalation senior is a FRESH candidate (empty prior
    # `messages`) dispatched WITH feedback (the prior-failure log). The feedback
    # must reach the prompt — an `elif feedback` after the prefix branch silently
    # drops it, leaving the expensive senior blind to what the cheap workers hit.
    ctx = _ctx(tmp_path)
    provider = ScriptedProvider(['<file path="a.txt">\nx\n</file>'])
    out = await _run(ctx, provider, monkeypatch, attempt=5, messages=[],
                     feedback="ESCALATED: prior workers failed:\nboom")
    msgs = out["candidates"][0]["messages"]
    assert "ESCALATED" not in msgs[0]["content"]              # frozen prefix untouched
    assert any("ESCALATED" in m["content"]
               for m in msgs[1:] if m["role"] == "user")      # feedback in a later turn


# ---- escalation routing ------------------------------------------------------

def _cfg_escalation(max_fix=2, on=True):
    return Config({"run": {"escalate_on_exhaustion": on, "max_fix_rounds": max_fix,
                           "max_retries": 5, "n_candidates": 1},
                   "roles": {"worker": {"default": "w", "candidates": ["w"]}},
                   "gate": {"log_tail_chars": 500}}, "p", Path("/tmp"))


def _red(attempt):
    return {"candidates": [{"cand_id": "w", "attempt": attempt,
                            "status": "gate_failed", "gate_log": "boom"}],
            "attempt": attempt}


def test_escalation_ready_only_after_max_fix():
    cfg = _cfg_escalation(max_fix=2)
    latest = {"w": {"cand_id": "w", "attempt": 1, "status": "gate_failed"}}
    assert escalation_ready(cfg, {"attempt": 1}, latest, 1) is False
    assert escalation_ready(cfg, {"attempt": 2}, latest, 2) is True


def test_escalation_once_guard():
    cfg = _cfg_escalation(max_fix=2)
    latest = {"w": {"cand_id": "w", "attempt": 2, "status": "gate_failed"}}
    assert escalation_ready(cfg, {"escalated": True}, latest, 3) is False


def test_dispatch_routes_to_senior_on_escalation():
    cfg = _cfg_escalation(max_fix=2)
    update = plan_dispatch(cfg, _red(2))
    assert update["to_run"] == ["senior"]
    assert update["escalated"] is True
    assert "ESCALATED" in update["feedback"]


def test_route_escalates_then_finalizes_after_senior():
    cfg = _cfg_escalation(max_fix=2)
    # red and out of fix rounds, not yet escalated -> dispatch (to escalate)
    assert decide_after_collect(cfg, _red(2)) == "dispatch"
    # senior ran and also failed -> finalize (marked for human), fires once
    st = _red(3)
    st["escalated"] = True
    assert decide_after_collect(cfg, st) == "finalize"


def test_no_escalation_when_disabled_uses_retries():
    cfg = _cfg_escalation(max_fix=2, on=False)
    # falls back to legacy: attempt(2) < max_retries(5) -> dispatch (ordinary retry)
    assert decide_after_collect(cfg, _red(2)) == "dispatch"
    assert plan_dispatch(cfg, _red(2))["to_run"] == ["w"]


def test_escalation_reachable_when_max_fix_exceeds_max_retries():
    # Regression (F1): with the shipped-default shape max_fix_rounds (4) >
    # max_retries (3), escalation must still be reachable. When the ladder is on
    # the cheap loop is bounded by max_fix_rounds, not the smaller max_retries;
    # otherwise a red task finalizes at attempt==max_retries and never escalates.
    cfg = Config({"run": {"escalate_on_exhaustion": True, "max_fix_rounds": 4,
                          "max_retries": 3, "n_candidates": 1},
                  "roles": {"worker": {"default": "w", "candidates": ["w"]}},
                  "gate": {"log_tail_chars": 500}}, "p", Path("/tmp"))
    # cheap loop keeps retrying past max_retries(3), up to max_fix_rounds(4)
    assert decide_after_collect(cfg, _red(3)) == "dispatch"
    # at max_fix_rounds the still-red task escalates to the senior (not finalize)
    assert decide_after_collect(cfg, _red(4)) == "dispatch"
    assert plan_dispatch(cfg, _red(4))["to_run"] == ["senior"]
    # after the senior also fails, the once-guard finalizes it
    st = _red(5)
    st["escalated"] = True
    assert decide_after_collect(cfg, st) == "finalize"


# ---- usage-table migration ---------------------------------------------------

def test_usage_migration_on_preexisting_db(tmp_path):
    db = tmp_path / "old.sqlite3"
    import sqlite3
    conn = sqlite3.connect(db)
    # an OLD usage table without the cache columns
    conn.executescript("""
        CREATE TABLE usage (ts REAL, run_id TEXT, task_id TEXT, role TEXT,
            provider TEXT, model TEXT, input_tokens INTEGER, output_tokens INTEGER,
            cost_usd REAL, cash INTEGER);
        INSERT INTO usage VALUES(1,'r','t','worker','p','m',10,5,0.0,0);
    """)
    conn.commit()
    conn.close()

    store = Store(db)          # __init__ runs the tolerant migration
    cols = {r["name"] for r in store._conn.execute("PRAGMA table_info(usage)")}
    assert {"cache_hit_tokens", "cache_miss_tokens"} <= cols
    # new inserts carry cache tokens
    store.record_usage("r", "t", "worker", "p", "m", 20, 4, 0.0, False,
                       cache_hit_tokens=16, cache_miss_tokens=4)
    row = store._conn.execute(
        "SELECT SUM(cache_hit_tokens) h FROM usage").fetchone()
    assert row["h"] == 16
    store.close()
