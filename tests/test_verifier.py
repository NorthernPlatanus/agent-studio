"""Visual verdict phase: an agent inspects the candidate's RUNNING scene via MCP.

Covers the parts that can fail silently and dangerously: a verdict that can't be
parsed must not read as a pass, the inspector must look at the CANDIDATE's
worktree rather than the primary checkout, only the verifier role gets MCP tools,
and the phase must be serialized because the dev/bridge ports are singletons.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from orchestrator.core.config import Config
from orchestrator.core.context import RunContext
from orchestrator.engine.graph import decide_after_review, decide_after_verify
from orchestrator.ops import verifier


# ---- verdict parsing: fail closed -------------------------------------------

def test_parses_a_clean_verdict():
    v = verifier.parse_verdict('{"ok": true, "findings": ["light intensity 0.9"],'
                               ' "facts": {"fps": 60}}')
    assert v["ok"] is True
    assert v["findings"] == ["light intensity 0.9"]
    assert v["facts"] == {"fps": 60}


def test_prose_instead_of_json_is_a_failure_not_a_pass():
    """The phase exists to catch what assertions miss; 'the verifier rambled' is
    not evidence that the scene is correct."""
    v = verifier.parse_verdict("Looks great to me, shipping it!")
    assert v["ok"] is False and "no JSON verdict" in v["findings"][0]


def test_malformed_json_is_a_failure():
    v = verifier.parse_verdict('{"ok": true, "findings": [unclosed')
    assert v["ok"] is False


def test_missing_ok_field_is_a_failure():
    v = verifier.parse_verdict('{"findings": ["something"]}')
    assert v["ok"] is False and "no `ok` field" in v["findings"][0]


def test_empty_response_is_a_failure():
    assert verifier.parse_verdict("").ok if False else True
    assert verifier.parse_verdict("")["ok"] is False


def test_string_findings_are_normalized_to_a_list():
    v = verifier.parse_verdict('{"ok": false, "findings": "one thing"}')
    assert v["findings"] == ["one thing"]


# ---- unverifiable: a structural dead end, not a defect -----------------------

def test_unverifiable_criteria_are_parsed_separately_from_findings():
    """"I measured this and it is wrong" and "no tool I have can observe this" are
    both `ok: false`, but only the first is worth another ~400k-token attempt."""
    v = verifier.parse_verdict(
        '{"ok": false, "findings": ["bounds are exact"],'
        ' "unverifiable": ["frame 1 — cannot reload the running app"]}')
    assert v["ok"] is False
    assert v["unverifiable"] == ["frame 1 — cannot reload the running app"]
    assert v["findings"] == ["bounds are exact"]


def test_an_ok_verdict_with_an_unmeasured_criterion_fails_closed():
    """Self-contradictory: the prompt is explicit that unverifiable is not
    verified. Merging on the optimistic half of that would be the worst outcome."""
    v = verifier.parse_verdict('{"ok": true, "findings": [], '
                               '"unverifiable": ["startup transient"]}')
    assert v["ok"] is False


def test_a_clean_verdict_has_no_unverifiable_criteria():
    v = verifier.parse_verdict('{"ok": true, "findings": []}')
    assert v["unverifiable"] == [] and v["ok"] is True


def test_string_unverifiable_is_normalized():
    v = verifier.parse_verdict('{"ok": false, "unverifiable": "frame 1"}')
    assert v["unverifiable"] == ["frame 1"]


def test_unverifiable_verdict_ends_the_task_for_a_human():
    from orchestrator.engine.graph import apply_scene_verdict

    cand = {"cand_id": "w", "attempt": 1, "status": "gate_passed", "diff": "d"}
    update = apply_scene_verdict(cand, {
        "ok": False, "findings": ["bounds exact"],
        "unverifiable": ["the first rendered frame"]})
    nc = update["candidates"][0]
    assert nc["status"] == "visual_unverifiable"
    assert "UNVERIFIABLE" in nc["gate_log"]
    assert "bounds exact" in nc["gate_log"]     # keep what WAS confirmed
    assert update["outcome"] == "needs_human"
    assert "the first rendered frame" in update["blocked_reason"]


def test_a_measured_rejection_still_rejoins_the_retry_ladder():
    from orchestrator.engine.graph import apply_scene_verdict

    cand = {"cand_id": "w", "attempt": 1, "status": "gate_passed"}
    update = apply_scene_verdict(cand, {"ok": False, "findings": ["light is black"]})
    assert update["candidates"][0]["status"] == "visual_failed"
    assert "outcome" not in update                    # the ladder decides, not this


def test_a_pass_carries_measured_facts_onto_the_candidate():
    from orchestrator.engine.graph import apply_scene_verdict

    cand = {"cand_id": "w", "attempt": 1, "status": "gate_passed",
            "visual_facts": {"fromGate": 1}}
    update = apply_scene_verdict(cand, {"ok": True, "findings": [],
                                        "facts": {"fov": 50}})
    nc = update["candidates"][0]
    assert nc["status"] == "gate_passed"
    assert nc["visual_facts"] == {"fromGate": 1, "fov": 50}


def test_unverifiable_candidate_routes_straight_to_finalize():
    """Retrying buys another full inspection with the same answer — the loop the
    scene verdict burned ~797k subscription tokens in."""
    cfg = _cfg()
    state = {"attempt": 1, "spec": {"visual": True},
             "candidates": [{"cand_id": "w", "attempt": 1,
                             "status": "visual_unverifiable"}]}
    assert decide_after_verify(cfg, state) == "finalize"    # despite retries left


# ---- gating ------------------------------------------------------------------

def _cfg(**over):
    verify = {"enabled": True, "run_cmd": "true", "dev_port": 5199,
              "bridge_port": 9333, "mcp_server": "threejs"}
    verify.update(over)
    return Config({
        "verify": verify,
        "paths": {"state_dir": "./state"},
        "roles": {"smart_provider": "claude_cli",
                  "verifier": {"provider": None, "model": "claude-opus-5",
                               "effort": "medium",
                               "allowed_tools": "mcp__threejs__scene_tree"},
                  "reviewer": {"provider": None, "model": "claude-opus-5"}},
        "providers": {"claude_cli": {"type": "claude_cli",
                                     "allowed_tools": "Read,Grep,Glob"}},
        "run": {"max_retries": 3, "escalate_on_exhaustion": False},
    }, "p", Path("/tmp"))


def test_only_visual_specs_are_verified():
    """A pure-logic task would pay a browser launch and ~60 tool schemas to look
    at a scene its diff never touched."""
    cfg = _cfg()
    assert verifier.verify_needed(cfg, {"visual": True}) is True
    assert verifier.verify_needed(cfg, {"visual": False}) is False
    assert verifier.verify_needed(cfg, None) is False


def test_disabled_verify_never_runs():
    assert verifier.verify_needed(_cfg(enabled=False), {"visual": True}) is False


def test_approved_visual_spec_routes_to_verify_not_integrate():
    state = {"verdict": {"decision": "approve"}, "attempt": 1,
             "spec": {"visual": True}}
    assert decide_after_review(_cfg(), state) == "verify"
    state["spec"] = {"visual": False}
    assert decide_after_review(_cfg(), state) == "integrate"
    # With the phase off, a visual spec still goes straight to integrate.
    state["spec"] = {"visual": True}
    assert decide_after_review(_cfg(enabled=False), state) == "integrate"


def test_rejected_scene_rejoins_the_ordinary_retry_path():
    """A rejected candidate is `visual_failed` — not green — so it reuses the
    existing retry/escalate/finalize machinery rather than a parallel path."""
    cfg = _cfg()
    ok = {"attempt": 1, "candidates": [{"cand_id": "w", "attempt": 1,
                                        "status": "gate_passed"}]}
    assert decide_after_verify(cfg, ok) == "integrate"
    bad = {"attempt": 1, "candidates": [{"cand_id": "w", "attempt": 1,
                                         "status": "visual_failed"}]}
    assert decide_after_verify(cfg, bad) == "dispatch"      # retries left
    bad["attempt"] = 3
    assert decide_after_verify(cfg, bad) == "finalize"      # exhausted


# ---- generated MCP config ----------------------------------------------------

def test_generated_mcp_config_carries_the_dev_port(tmp_path):
    """The port is a runtime fact, so the config is generated rather than
    hand-maintained — otherwise it would depend on the agent remembering to call
    set_dev_port."""
    dest = verifier.build_mcp_config(_cfg(), tmp_path / "sub" / "mcp.json")
    data = json.loads(dest.read_text())
    srv = data["mcpServers"]["threejs"]
    assert srv["command"] == "npx"
    assert srv["args"] == ["-y", "threejs-devtools-mcp"]
    assert srv["env"]["DEV_PORT"] == "5199"
    assert srv["env"]["BRIDGE_PORT"] == "9333"
    assert srv["env"]["HEADLESS"] == "true"       # no human, no open tab
    assert srv["env"]["THREEJS_DEVTOOLS_NO_OVERLAY"] == "1"


def test_headless_false_is_honoured(tmp_path):
    dest = verifier.build_mcp_config(_cfg(headless=False), tmp_path / "m.json")
    env = json.loads(dest.read_text())["mcpServers"]["threejs"]["env"]
    assert env["HEADLESS"] == "false"


def test_mcp_env_extras_override_defaults(tmp_path):
    dest = verifier.build_mcp_config(
        _cfg(mcp_env={"CHROME_PATH": "/custom/chrome", "DEV_PORT": "6000"}),
        tmp_path / "m.json")
    env = json.loads(dest.read_text())["mcpServers"]["threejs"]["env"]
    assert env["CHROME_PATH"] == "/custom/chrome"
    assert env["DEV_PORT"] == "6000"


def test_port_templating():
    assert verifier._fmt("npm run dev -- --port {port}", 5199).endswith("5199")
    # A stray brace must not raise (str.format would KeyError here).
    assert verifier._fmt("echo ${HOME} {port}", 42) == "echo ${HOME} 42"


# ---- role-scoped tool policy -------------------------------------------------

def test_only_the_verifier_gets_mcp_tools():
    """Granting the inspector tier-wide would hand a live scene to the planner and
    reviewer too — the global `mcp:` section's exact failure mode."""
    ctx = RunContext(cfg=_cfg(), store=None, git=None, budget=None, run_id="r")
    assert ctx.role_allowed_tools("verifier") == "mcp__threejs__scene_tree"
    assert ctx.role_allowed_tools("reviewer") is None    # -> provider-wide default


# ---- the call: cwd, budget, teardown ----------------------------------------

class _RecordingProvider:
    type = "claude_cli"

    def __init__(self, text):
        self.text = text
        self.call = None

    async def complete(self, **kw):
        self.call = kw
        return SimpleNamespace(text=self.text, input_tokens=10, output_tokens=2,
                               cost_usd=0.0, cache_hit_tokens=0,
                               cache_miss_tokens=10)


class _Budget:
    def __init__(self):
        self.records = []

    def record(self, **kw):
        self.records.append(kw)


def _run_verification(tmp_path, monkeypatch, response, **cfg_over):
    """Drive run_verification with the app spawn/probe stubbed out."""
    spawned, torn = [], []
    monkeypatch.setattr(verifier, "_spawn",
                        lambda cmd, wt: spawned.append((cmd, str(wt))) or "PROC")
    monkeypatch.setattr(verifier, "_teardown", lambda p: torn.append(p))
    monkeypatch.setattr(verifier, "wait_for_probe", lambda url, t: None)
    provider = _RecordingProvider(response)
    monkeypatch.setattr("orchestrator.providers.get_provider",
                        lambda cfg, name: provider)

    cfg = _cfg(**cfg_over)
    cfg._data["paths"]["state_dir"] = str(tmp_path / "state")
    monkeypatch.setattr(type(cfg), "prompt", lambda self, n, **k: "SYSTEM")
    ctx = RunContext(cfg=cfg, store=None, git=None, budget=_Budget(), run_id="r")
    spec = {"id": "T-1", "title": "t", "description": "d", "visual": True,
            "acceptance": ["a light exists"]}
    cand = {"cand_id": "w", "worktree": str(tmp_path / "wt"), "diff": "D",
            "model": "m"}
    verdict = asyncio.run(verifier.run_verification(ctx, spec, cand))
    return verdict, provider, spawned, torn, ctx


def test_inspects_the_candidate_worktree_not_the_primary_checkout(tmp_path, monkeypatch):
    """The load-bearing detail: the reviewer uses repo_path, and a verifier
    pointed there would grade this candidate's diff against another build."""
    v, provider, spawned, torn, ctx = _run_verification(
        tmp_path, monkeypatch, '{"ok": true, "findings": []}')
    assert v["ok"] is True
    assert provider.call["cwd"] == str(tmp_path / "wt")
    assert spawned[0][1] == str(tmp_path / "wt")
    assert torn == ["PROC"]                       # app always torn down
    assert ctx.budget.records[0]["role"] == "verifier"


def test_per_role_effort_and_tools_reach_the_provider(tmp_path, monkeypatch):
    _, provider, *_ = _run_verification(tmp_path, monkeypatch,
                                        '{"ok": true, "findings": []}')
    assert provider.call["effort"] == "medium"
    assert provider.call["allowed_tools"] == "mcp__threejs__scene_tree"
    assert provider.call["mcp_config"].endswith("-mcp.json")


def test_app_teardown_happens_even_when_the_call_raises(tmp_path, monkeypatch):
    """A leaked dev server would hold the port and wedge every later
    verification in the run."""
    torn = []
    monkeypatch.setattr(verifier, "_spawn", lambda cmd, wt: "PROC")
    monkeypatch.setattr(verifier, "_teardown", lambda p: torn.append(p))
    monkeypatch.setattr(verifier, "wait_for_probe", lambda url, t: None)

    class _Boom:
        type = "claude_cli"
        async def complete(self, **kw):
            raise RuntimeError("cli exploded")
    monkeypatch.setattr("orchestrator.providers.get_provider",
                        lambda cfg, name: _Boom())
    cfg = _cfg()
    cfg._data["paths"]["state_dir"] = str(tmp_path / "state")
    monkeypatch.setattr(type(cfg), "prompt", lambda self, n, **k: "S")
    ctx = RunContext(cfg=cfg, store=None, git=None, budget=_Budget(), run_id="r")
    try:
        asyncio.run(verifier.run_verification(
            ctx, {"id": "T-1", "title": "t", "description": "d"},
            {"cand_id": "w", "worktree": str(tmp_path), "diff": "D"}))
    except RuntimeError:
        pass
    assert torn == ["PROC"]


def test_app_never_ready_is_a_failure_not_a_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(verifier, "_spawn", lambda cmd, wt: "PROC")
    monkeypatch.setattr(verifier, "_teardown", lambda p: None)
    def never(url, t):
        raise verifier.FactSourceError("ready_probe timed out")
    monkeypatch.setattr(verifier, "wait_for_probe", never)
    cfg = _cfg(ready_probe="http://localhost:{port}/")
    cfg._data["paths"]["state_dir"] = str(tmp_path / "state")
    ctx = RunContext(cfg=cfg, store=None, git=None, budget=_Budget(), run_id="r")
    v = asyncio.run(verifier.run_verification(
        ctx, {"id": "T-1", "title": "t", "description": "d"},
        {"cand_id": "w", "worktree": str(tmp_path), "diff": "D"}))
    assert v["ok"] is False and "did not become ready" in v["findings"][0]


def test_enabled_without_run_cmd_is_a_loud_error(tmp_path, monkeypatch):
    """Silently skipping would mark an unverified task verified."""
    cfg = _cfg(run_cmd=None)
    cfg._data["paths"]["state_dir"] = str(tmp_path / "state")
    ctx = RunContext(cfg=cfg, store=None, git=None, budget=_Budget(), run_id="r")
    try:
        asyncio.run(verifier.run_verification(
            ctx, {"id": "T-1", "title": "t", "description": "d"},
            {"cand_id": "w", "worktree": str(tmp_path), "diff": "D"}))
        raise AssertionError("should have raised")
    except Exception as e:
        assert "run_cmd" in str(e)


# ---- serialization -----------------------------------------------------------

async def test_inspector_lock_serializes_the_phase():
    """The dev server and bridge are bound to fixed ports; two concurrent
    verifications would inspect each other's app."""
    ctx = RunContext(cfg=_cfg(), store=None, git=None, budget=None, run_id="r")
    order: list[str] = []

    async def phase(name):
        async with ctx.inspector_lock:
            order.append(f"{name}-start")
            await asyncio.sleep(0.01)
            order.append(f"{name}-end")

    await asyncio.gather(phase("a"), phase("b"))
    # Strictly non-overlapping: no start appears between another's start and end.
    assert order in (["a-start", "a-end", "b-start", "b-end"],
                     ["b-start", "b-end", "a-start", "a-end"])


async def test_inspector_lock_is_one_object_per_context():
    ctx = RunContext(cfg=_cfg(), store=None, git=None, budget=None, run_id="r")
    assert ctx.inspector_lock is ctx.inspector_lock


# ---- best-of-N: only the winner is inspected, so only the winner may decide ---

def _boN(winner_status, loser_status="gate_passed", **extra):
    """Two green candidates, `a` selected by review, `a` then carrying whatever
    the scene verdict made of it. `b` was never inspected — one verdict per task
    is what makes the phase affordable — and review never demotes it."""
    return {"attempt": 1, "spec": {"visual": True},
            "verdict": {"decision": "approve", "winner": "a"},
            "candidates": [
                {"cand_id": "a", "attempt": 1, "status": "gate_passed"},
                {"cand_id": "b", "attempt": 1, "status": loser_status},
                {"cand_id": "a", "attempt": 1, "status": winner_status}],
            **extra}


def test_an_unverifiable_winner_ends_the_task_even_with_a_green_sibling():
    """`any(gate_passed)` saw the never-inspected loser and called it a pass, so
    the task went to integrate — which merges latest[winner], i.e. the candidate
    the verdict could not observe — and overwrote needs_human with done."""
    state = _boN("visual_unverifiable", outcome="needs_human")
    assert decide_after_verify(_cfg(), state) == "finalize"


def test_a_rejected_winner_does_not_integrate_because_a_sibling_is_green():
    """Same hole, the measured half: the winner was inspected and found wrong,
    and its diff is the one integrate would merge."""
    state = _boN("visual_failed")
    assert decide_after_verify(_cfg(), state) == "dispatch"     # retries left


def test_a_verified_winner_still_integrates_at_best_of_n():
    assert decide_after_verify(_cfg(), _boN("gate_passed")) == "integrate"


def test_a_red_sibling_never_blocks_a_verified_winner():
    """The converse: routing on the winner must not let a losing candidate's
    failure hold back a winner the verifier confirmed."""
    state = _boN("gate_passed", loser_status="gate_failed")
    assert decide_after_verify(_cfg(), state) == "integrate"


def test_single_candidate_routing_is_unchanged():
    for status, expected in (("gate_passed", "integrate"),
                             ("visual_unverifiable", "finalize"),
                             ("visual_failed", "dispatch")):
        state = {"attempt": 1, "spec": {"visual": True},
                 "verdict": {"decision": "approve", "winner": "a"},
                 "candidates": [{"cand_id": "a", "attempt": 1, "status": status}]}
        assert decide_after_verify(_cfg(), state) == expected


# ---- what the human backlog is told about a scene rejection -------------------

def test_a_rejected_scene_records_why_for_the_backlog():
    """`finalize` writes `blocked_reason` to the board. Without one, a task that
    exhausted its retries after a scene rejection wrote `agent run failed — needs
    human`, and the measured findings lived only in the gate log and the event
    table — nowhere the human looks."""
    from orchestrator.engine.graph import apply_scene_verdict

    update = apply_scene_verdict(
        {"cand_id": "a", "attempt": 1, "status": "gate_passed"},
        {"ok": False, "findings": ["ground plane renders black", "fps 4"]})
    assert update["candidates"][0]["status"] == "visual_failed"
    assert update["blocked_reason"] == ("scene verdict rejected: ground plane "
                                        "renders black; fps 4")


def test_the_reason_is_cleared_when_the_next_attempt_starts():
    """It describes a judgment about one attempt. Left standing, the previous
    round's scene findings would end up on the backlog next to an unrelated later
    failure."""
    from orchestrator.core.config import Config as _C
    from orchestrator.engine.graph import plan_dispatch

    cfg = _C({"run": {"n_candidates": 1, "max_retries": 3,
                      "escalate_on_exhaustion": False,
                      "senior_first_for_high_risk": False},
              "roles": {"worker": {"default": "cheap", "candidates": ["cheap"]}},
              "gate": {"log_tail_chars": 500}, "domains": {}}, "p", Path("/tmp"))
    state = {"attempt": 1, "spec": {"id": "T-1"},
             "blocked_reason": "scene verdict rejected: ground plane renders black",
             "candidates": [{"cand_id": "cheap", "attempt": 1,
                             "status": "visual_failed", "gate_log": "L"}]}
    assert plan_dispatch(cfg, state)["blocked_reason"] == ""
