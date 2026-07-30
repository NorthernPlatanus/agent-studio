"""Phase 2: worker retrieval loop — parse new blocks, execute read-only, and
never apply retrieval-only output as a patch. Drives run_candidate with a
scripted provider and lightweight git/store/budget fakes.
"""

import orchestrator.nodes.worker as worker_mod
from orchestrator.core.config import load_config
from orchestrator.core.context import RunContext
from orchestrator.ops.patch import apply_response, parse_worker_response
from orchestrator.providers.base import LLMResult


# ---- parse-level invariants -------------------------------------------------

def test_parse_retrieval_blocks():
    p = parse_worker_response("<grep>foo</grep>\n<read>src/a.ts</read>\n<ls>src</ls>")
    assert p.grep == ["foo"]
    assert p.need_files == ["src/a.ts"]
    assert p.ls == ["src"]
    assert p.has_retrieval and p.touched_paths == set()


def test_need_files_alias_still_parses():
    p = parse_worker_response("<need_files>\nsrc/x.ts\n</need_files>")
    assert p.need_files == ["src/x.ts"]


def test_retrieval_only_output_applies_nothing(tmp_path):
    parsed = parse_worker_response("<grep>foo</grep>")
    assert apply_response(tmp_path, parsed, None) == []   # no <file>/<edit> => no writes


# ---- loop-level integration -------------------------------------------------

class FakeGit:
    def __init__(self, base):
        self.work_dir = base

    def create_worktree(self, name):
        wt = self.work_dir / name
        wt.mkdir(parents=True, exist_ok=True)
        return wt

    async def acreate_worktree(self, name, from_branch=None):
        # Mirrors Git.acreate_worktree (locked async wrapper). Tests that swap in
        # their own create_worktree still work: this dispatches through the
        # attribute, not the class body.
        return self.create_worktree(name)

    def wt_branch(self, name):
        return f"wt/{name}"

    def commit_all(self, wt, msg):
        pass

    async def acommit_all(self, wt, msg):
        # Mirrors Git.acommit_all (locked async wrapper); dispatches through the
        # attribute so a test that swaps commit_all still sees its own version.
        return self.commit_all(wt, msg)

    def diff_against_feature(self, wt):
        return "DIFF"


class FakeStore:
    def __init__(self):
        self.events = []

    def log_event(self, run_id, task_id, kind, msg):
        self.events.append((kind, msg))

    def set_task_status(self, *a, **k):
        pass


class FakeBudget:
    def record(self, **k):
        pass

    def estimate_and_check(self, **k):
        return 0.0


class ScriptedProvider:
    type = "fake"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.last_messages = None

    async def complete(self, *, model, system, user, cwd=None, params=None,
                       session=None, effort=None):
        text = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return LLMResult(text=text)

    async def complete_chat(self, *, model, system, messages, cwd=None,
                            params=None, session=None, effort=None):
        self.last_messages = messages
        self.last_params = params
        text = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return LLMResult(text=text)


def _ctx(tmp_path, retrieval_rounds=3):
    cfg = load_config()                          # generic defaults
    d = cfg._data
    d["project"]["repo_path"] = str(tmp_path / "repo")
    (tmp_path / "repo").mkdir(exist_ok=True)
    d["gate"]["commands"] = []                   # empty gate => passes
    d["run"]["retrieval_rounds"] = retrieval_rounds
    d["roles"]["worker"] = {"default": "w", "candidates": ["w"]}
    d["worker_models"] = {"w": {"provider": "fake", "model": "m"}}
    store = FakeStore()
    ctx = RunContext(cfg=cfg, store=store, git=FakeGit(tmp_path / "wt"),
                     budget=FakeBudget(), run_id="r")
    return ctx, store


def _spec():
    return {"id": "T-1", "title": "t", "description": "d",
            "files_read": [], "files_write": ["a.txt"]}


async def _run(ctx, provider, monkeypatch, attempt=1, retrieval_rounds=3):
    monkeypatch.setattr(worker_mod, "get_provider", lambda cfg, name: provider)
    payload = {"run_id": "r", "task_id": "T-1", "spec": _spec(),
               "cand_id": "w", "attempt": attempt, "feedback": ""}
    return await worker_mod.run_candidate(ctx, payload)


async def test_grep_then_patch_passes(tmp_path, monkeypatch):
    ctx, store = _ctx(tmp_path)
    provider = ScriptedProvider([
        "<grep>foo</grep>",
        '<file path="a.txt">\nhello\n</file>',
    ])
    out = await _run(ctx, provider, monkeypatch)
    cand = out["candidates"][0]
    assert cand["status"] == "gate_passed"
    assert provider.calls == 2
    assert any(k == "retrieval" for k, _ in store.events)
    assert (tmp_path / "wt" / "t-1-w" / "a.txt").read_text() == "hello\n"


async def test_retrieval_exhaustion_forces_final_and_logs(tmp_path, monkeypatch):
    ctx, store = _ctx(tmp_path, retrieval_rounds=1)
    # never emits a patch: 1 real round, then a forced-final round, then break
    provider = ScriptedProvider(["<grep>x</grep>"])
    out = await _run(ctx, provider, monkeypatch)
    cand = out["candidates"][0]
    assert cand["status"] == "patch_failed"          # no patch ever produced
    kinds = [k for k, _ in store.events]
    assert "retrieval" in kinds
    assert "retrieval_exhausted" in kinds


# ---- prose instead of a patch (defect-plan #2 item 6) -----------------------

async def test_a_prose_reply_is_flagged_so_the_fix_round_is_refunded(tmp_path, monkeypatch):
    """Observed twice in a row after a review `revise`: 831 then 1,241 output
    tokens of something that was not a patch, two of four fix rounds gone, an
    escalation forced — having attempted nothing."""
    ctx, store = _ctx(tmp_path)
    provider = ScriptedProvider(
        ["Good catch. I would start by extracting the event writer, then..."])
    out = await _run(ctx, provider, monkeypatch)
    cand = out["candidates"][0]
    assert cand["status"] == "patch_failed"
    assert cand["no_patch"] is True             # graph.unproductive_attempts reads this
    assert "no <file>/<edit> blocks" in cand["error"]
    # its own event kind, so `metrics` can show a model that keeps doing this
    assert any(k == "no_patch" for k, _ in store.events)


async def test_a_failed_patch_that_DID_produce_blocks_is_not_refunded(tmp_path, monkeypatch):
    """Writing outside the allowlist is a real attempt that failed. Only an empty
    response is free, or every rejected patch would stop costing a round."""
    ctx, _ = _ctx(tmp_path)
    provider = ScriptedProvider(['<file path="not-allowed.txt">\nx\n</file>'])
    out = await _run(ctx, provider, monkeypatch)
    cand = out["candidates"][0]
    assert cand["status"] == "patch_failed"
    assert cand.get("no_patch") is not True


async def test_retry_feedback_restates_the_output_contract(tmp_path, monkeypatch):
    """The hypothesis this fixes: the feedback turn arrives thousands of tokens
    after the contract in the frozen prefix and reads as a discussion prompt, so
    the model answers in kind."""
    ctx, _ = _ctx(tmp_path)
    provider = ScriptedProvider(['<file path="a.txt">\nfixed\n</file>'])
    monkeypatch.setattr(worker_mod, "get_provider", lambda cfg, name: provider)
    out = await worker_mod.run_candidate(ctx, {
        "run_id": "r", "task_id": "T-1", "spec": _spec(), "cand_id": "w",
        "attempt": 2, "feedback": "REVIEW NOTES:\nassert the collapsed form",
        "messages": [{"role": "user", "content": "FROZEN PREFIX"}]})
    assert out["candidates"][0]["status"] == "gate_passed"
    turn = [m for m in provider.last_messages if m["role"] == "user"][-1]["content"]
    assert "assert the collapsed form" in turn        # the actual instruction
    assert "<file>" in turn and "<edit>" in turn      # ...and the contract
    assert "no prose" in turn.lower()
    assert "<grep>" in turn                           # the one legal alternative
    # The frozen prefix is untouched — the contract arrives as a later turn.
    assert provider.last_messages[0]["content"] == "FROZEN PREFIX"


async def test_the_first_attempt_gets_no_retry_contract(tmp_path, monkeypatch):
    """Attempt 1 has no feedback, so nothing may be appended after the prefix —
    that block IS the cache key for the whole warm chain."""
    ctx, _ = _ctx(tmp_path)
    provider = ScriptedProvider(['<file path="a.txt">\nx\n</file>'])
    await _run(ctx, provider, monkeypatch)
    assert len(provider.last_messages) == 2           # prefix + the assistant reply
    assert "OUTPUT CONTRACT" not in provider.last_messages[0]["content"]


async def test_worker_refuses_spec_without_files_write(tmp_path, monkeypatch):
    """Belt for specs planned before persist_specs validated files_write: an
    absent allowlist must fail closed, before any tokens are spent."""
    ctx, store = _ctx(tmp_path)
    provider = ScriptedProvider(['<file path="anywhere.txt">\nx\n</file>'])
    monkeypatch.setattr(worker_mod, "get_provider", lambda cfg, name: provider)
    spec = _spec()
    del spec["files_write"]
    out = await worker_mod.run_candidate(ctx, {
        "run_id": "r", "task_id": "T-1", "spec": spec, "cand_id": "w",
        "attempt": 1, "feedback": ""})
    cand = out["candidates"][0]
    assert cand["status"] == "patch_failed"
    assert "files_write" in cand["error"]
    assert provider.calls == 0            # refused before the LLM call
