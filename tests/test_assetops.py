"""Deterministic asset ops — the named-command runner, its place in the
candidate pipeline, and the plan-time name check.

Same shape as tests/test_visualgate.py: unit-test the runner against real
subprocesses in a tmp worktree, then drive the node that uses it.

The invariant under test throughout: an asset op is a FIXED command a human
wrote in the profile, referenced by name. No LLM role gains shell, and a name no
human configured never runs anything and never passes silently.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

import orchestrator.nodes.worker as worker_mod
from orchestrator.core.config import Config, load_config
from orchestrator.core.context import RunContext
from orchestrator.nodes.planner import validate_spec
from orchestrator.ops.assetops import (UnknownAssetOp, available_ops,
                                       resolve_op, run_asset_op)


def _cfg(ops=None, *, timeout_s=1200, tail=6000):
    return Config({"asset_ops": ops if ops is not None else {},
                   "gate": {"timeout_s": timeout_s, "log_tail_chars": tail}},
                  "p", Path("/tmp"))


# ---- resolution -------------------------------------------------------------

def test_no_ops_configured_is_the_default():
    assert available_ops(load_config()) == []      # generic defaults: empty map


def test_available_ops_is_sorted():
    assert available_ops(_cfg({"b": "true", "a": "true"})) == ["a", "b"]


def test_string_entry_inherits_the_gate_timeout():
    cfg = _cfg({"reduce": "gltf-transform x"}, timeout_s=900)
    assert resolve_op(cfg, "reduce") == ("gltf-transform x", 900)


def test_mapping_entry_may_pin_its_own_timeout():
    cfg = _cfg({"bake": {"cmd": "blender --background", "timeout_s": 3600}})
    assert resolve_op(cfg, "bake") == ("blender --background", 3600)


def test_unknown_name_raises_and_lists_what_exists():
    with pytest.raises(UnknownAssetOp) as e:
        resolve_op(_cfg({"reduce_ae86_poly": "x"}), "reduce_ae87_poly")
    assert "reduce_ae86_poly" in str(e.value)


def test_mapping_entry_without_cmd_raises():
    with pytest.raises(UnknownAssetOp):
        resolve_op(_cfg({"bake": {"timeout_s": 60}}), "bake")


# ---- the runner -------------------------------------------------------------

def test_spec_without_asset_op_is_a_noop(tmp_path):
    res = run_asset_op(tmp_path, _cfg(), {"id": "T-1"})
    assert res.passed is True and res.ran is False


def test_runs_the_command_in_the_candidate_worktree(tmp_path):
    cfg = _cfg({"make_asset": "printf 'processed' > out.bin"})
    res = run_asset_op(tmp_path, cfg, {"id": "T-1", "asset_op": "make_asset"})
    assert res.passed is True and res.ran is True and res.op == "make_asset"
    assert (tmp_path / "out.bin").read_text() == "processed"


def test_nonzero_exit_fails_with_the_output(tmp_path):
    cfg = _cfg({"reduce": "echo 'no such file: model.glb' >&2; exit 2"})
    res = run_asset_op(tmp_path, cfg, {"id": "T-1", "asset_op": "reduce"})
    assert res.passed is False and res.ran is True
    assert "exited 2" in res.log_tail and "no such file" in res.log_tail


def test_timeout_fails(tmp_path):
    cfg = _cfg({"slow": {"cmd": "sleep 5", "timeout_s": 1}})
    res = run_asset_op(tmp_path, cfg, {"id": "T-1", "asset_op": "slow"})
    assert res.passed is False and "TIMEOUT" in res.log_tail


def test_unknown_op_fails_the_candidate_rather_than_skipping(tmp_path):
    """The whole point of naming ops: a name nobody configured must be loud.
    Skipping would hand the reviewer a diff whose asset was never processed."""
    cfg = _cfg({"reduce": "true"})
    res = run_asset_op(tmp_path, cfg, {"id": "T-1", "asset_op": "typo"})
    assert res.passed is False and res.ran is True
    assert "unknown asset_op" in res.log_tail and "reduce" in res.log_tail


def test_log_tail_is_capped(tmp_path):
    cfg = _cfg({"loud": "python3 -c \"print('x'*9000)\"; exit 1"}, tail=200)
    res = run_asset_op(tmp_path, cfg, {"id": "T-1", "asset_op": "loud"})
    assert res.passed is False and len(res.log_tail) < 300


# ---- inside the candidate pipeline ------------------------------------------

class FakeGit:
    """Records commits; worktrees are plain directories (see test_worker_loop)."""

    def __init__(self, base):
        self.work_dir = base
        self.commits: list[str] = []

    def create_worktree(self, name):
        wt = self.work_dir / name
        wt.mkdir(parents=True, exist_ok=True)
        return wt

    async def acreate_worktree(self, name, from_branch=None):
        return self.create_worktree(name)

    def wt_branch(self, name):
        return f"wt/{name}"

    async def acommit_all(self, wt, msg):
        self.commits.append(msg)

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

    def __init__(self, text):
        self.text = text

    async def complete_chat(self, *, model, system, messages, cwd=None,
                            params=None, session=None, effort=None):
        from orchestrator.providers.base import LLMResult
        return LLMResult(text=self.text)


def _run_ctx(tmp_path, ops, gate_commands):
    cfg = load_config()
    d = cfg._data
    (tmp_path / "repo").mkdir(exist_ok=True)
    d["project"]["repo_path"] = str(tmp_path / "repo")
    d["gate"]["commands"] = gate_commands
    d["asset_ops"] = ops
    d["roles"]["worker"] = {"default": "w", "candidates": ["w"]}
    d["worker_models"] = {"w": {"provider": "fake", "model": "m"}}
    store = FakeStore()
    git = FakeGit(tmp_path / "wt")
    ctx = RunContext(cfg=cfg, store=store, git=git, budget=FakeBudget(), run_id="r")
    return ctx, store, git


async def _candidate(tmp_path, monkeypatch, *, ops, asset_op,
                     gate_commands=("printf gate-ran > gate.marker",)):
    ctx, store, git = _run_ctx(tmp_path, ops, list(gate_commands))
    monkeypatch.setattr(worker_mod, "get_provider",
                        lambda cfg, name: ScriptedProvider(
                            '<file path="a.txt">\nhello\n</file>'))
    spec = {"id": "T-1", "title": "t", "description": "d",
            "files_read": [], "files_write": ["a.txt"], "asset_op": asset_op}
    out = await worker_mod.run_candidate(ctx, {
        "run_id": "r", "task_id": "T-1", "spec": spec, "cand_id": "w",
        "attempt": 1, "feedback": ""})
    return out["candidates"][0], store, git, tmp_path / "wt" / "t-1-w"


async def test_op_output_lands_in_the_worktree_and_is_committed(tmp_path, monkeypatch):
    """The output has to be COMMITTED, not merely written: the candidate's diff
    is `merge-base..HEAD`, so an uncommitted .glb would never reach the reviewer
    or the merge."""
    cand, store, git, wt = await _candidate(
        tmp_path, monkeypatch,
        ops={"reduce": "printf low-poly > model.low.glb"}, asset_op="reduce")
    assert cand["status"] == "gate_passed"
    assert (wt / "model.low.glb").read_text() == "low-poly"
    assert (wt / "a.txt").read_text() == "hello\n"       # the worker's patch too
    # two commits: the worker's patch, then the op's output
    assert len(git.commits) == 2 and "asset(T-1): reduce" in git.commits[1]
    assert any(k == "asset_op" for k, _ in store.events)


async def test_the_op_runs_before_the_gate(tmp_path, monkeypatch):
    """The gate has to build against the PROCESSED asset — a gate that ran first
    would typecheck code importing a file that doesn't exist yet."""
    cand, _, _, wt = await _candidate(
        tmp_path, monkeypatch,
        ops={"reduce": "printf low > model.low.glb"}, asset_op="reduce",
        gate_commands=("test -f model.low.glb",))
    assert cand["status"] == "gate_passed"


async def test_a_failed_op_fails_the_candidate_exactly_like_a_red_gate(tmp_path,
                                                                       monkeypatch):
    cand, store, _, wt = await _candidate(
        tmp_path, monkeypatch,
        ops={"reduce": "echo 'gltf-transform: invalid glb' >&2; exit 1"},
        asset_op="reduce")
    assert cand["status"] == "gate_failed"           # the existing retry branch
    assert "asset_op reduce" in cand["gate_log"]
    assert "invalid glb" in cand["gate_log"]
    # own event kind so `metrics` counts a broken tool separately from bad code
    assert [k for k, _ in store.events] == ["asset_op_failed"]
    assert not (wt / "gate.marker").exists()         # the gate never ran


async def test_an_unknown_op_name_never_reaches_the_gate(tmp_path, monkeypatch):
    cand, store, _, wt = await _candidate(
        tmp_path, monkeypatch, ops={"reduce": "true"}, asset_op="ghost_op")
    assert cand["status"] == "gate_failed"
    assert "unknown asset_op" in cand["gate_log"]
    assert not (wt / "gate.marker").exists()


async def test_specs_without_an_op_are_untouched(tmp_path, monkeypatch):
    """The default path for every project that never configures this."""
    cand, store, git, wt = await _candidate(tmp_path, monkeypatch, ops={},
                                            asset_op=None)
    assert cand["status"] == "gate_passed"
    assert len(git.commits) == 1                     # no extra asset commit
    assert not any(k.startswith("asset_op") for k, _ in store.events)


# ---- plan time --------------------------------------------------------------

def _spec(**over):
    spec = {"id": "T-1", "title": "t", "description": "d", "files_write": ["a.ts"]}
    spec.update(over)
    return spec


def test_validate_accepts_a_configured_op():
    validate_spec(_spec(asset_op="reduce"), asset_ops=["reduce"])


def test_validate_rejects_an_unknown_op_at_plan_time():
    """Caught here, or every candidate of the task fails later for a reason the
    retry feedback cannot fix (the worker cannot add an op to the profile)."""
    with pytest.raises(ValueError) as e:
        validate_spec(_spec(asset_op="ghost"), asset_ops=["reduce"])
    assert "ghost" in str(e.value) and "reduce" in str(e.value)


def test_validate_rejects_any_op_when_none_are_configured():
    with pytest.raises(ValueError):
        validate_spec(_spec(asset_op="reduce"), asset_ops=[])


def test_validate_is_unchanged_for_specs_without_an_op():
    validate_spec(_spec(), asset_ops=[])
    validate_spec(_spec(asset_op="anything"))        # no list passed => not checked


def test_human_only_specs_are_exempt():
    validate_spec(_spec(agent_able=False, asset_op="ghost", files_write=None),
                  asset_ops=[])


def test_planner_payload_lists_the_available_ops(tmp_path, monkeypatch):
    """The planner can only reference a name it was shown — this is where the
    names come from (it has no read access to the profile)."""
    from orchestrator.nodes import planner

    cfg = load_config()
    cfg._data["project"]["repo_path"] = str(tmp_path)
    cfg._data["project"]["backlog_file"] = "BACKLOG.md"
    cfg._data["asset_ops"] = {"reduce_ae86_poly": "npx @gltf-transform/cli simplify a b"}
    (tmp_path / "BACKLOG.md").write_text("- [ ] **T-1** do a thing\n")
    ctx = SimpleNamespace(cfg=cfg, store=SimpleNamespace(all_tasks=lambda: []))

    text = planner._full_prompt(ctx, discussion="", transcript="", only_ids=None)
    assert "ASSET OPS AVAILABLE" in text
    assert "reduce_ae86_poly" in text and "gltf-transform" in text


def test_planner_payload_says_nothing_when_no_ops_exist(tmp_path):
    from orchestrator.nodes import planner

    cfg = load_config()
    cfg._data["project"]["repo_path"] = str(tmp_path)
    cfg._data["project"]["backlog_file"] = "BACKLOG.md"
    (tmp_path / "BACKLOG.md").write_text("- [ ] **T-1** do a thing\n")
    ctx = SimpleNamespace(cfg=cfg, store=SimpleNamespace(all_tasks=lambda: []))

    assert "ASSET OPS" not in planner._full_prompt(
        ctx, discussion="", transcript="", only_ids=None)
