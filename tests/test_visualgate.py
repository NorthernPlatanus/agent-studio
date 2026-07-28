"""Phase 6: visual gate — safe assertion evaluator, routing, and the
visual_failed retry path (no empty-Send livelock)."""

import shlex
from pathlib import Path

import pytest

import orchestrator.ops.visualgate as vg
from orchestrator.core.config import Config
from orchestrator.engine.graph import (decide_after_collect, decide_after_visual,
                                        visual_needed)
from orchestrator.ops.visualgate import (AssertionError_, evaluate_assertion,
                                         run_assertions)


# ---- safe evaluator ---------------------------------------------------------

FACTS = {
    "scene": {"visibleMeshCount": 3, "boundingBox": {"empty": False}},
    "lights": [{"intensity": 0.0}, {"intensity": 1.5}],
    "camera": {"seesSceneBounds": True},
}


def test_truthy_and_comparison():
    assert evaluate_assertion("scene.visibleMeshCount > 0", FACTS) is True
    assert evaluate_assertion("scene.visibleMeshCount > 5", FACTS) is False
    assert evaluate_assertion("camera.seesSceneBounds", FACTS) is True


def test_comprehension_any():
    assert evaluate_assertion("any(l.intensity > 0 for l in lights)", FACTS) is True
    assert evaluate_assertion("all(l.intensity > 0 for l in lights)", FACTS) is False


def test_subscript_and_len():
    assert evaluate_assertion("len(lights) == 2", FACTS) is True
    assert evaluate_assertion("lights[1].intensity > 1", FACTS) is True


def test_refuses_arbitrary_code():
    for bad in ("__import__('os')", "open('/etc/passwd')", "scene.__class__",
                "().__class__.__bases__"):
        with pytest.raises(AssertionError_):
            evaluate_assertion(bad, FACTS)


def test_unknown_field_is_error():
    with pytest.raises(AssertionError_):
        evaluate_assertion("scene.nope > 0", FACTS)


def test_run_assertions_collects_failures():
    failures = run_assertions(
        ["scene.visibleMeshCount > 0", "scene.visibleMeshCount > 99",
         "any(l.intensity > 0 for l in lights)"], FACTS)
    assert failures == ["scene.visibleMeshCount > 99"]


def test_invalid_assertion_counts_as_failure():
    failures = run_assertions(["__import__('os')"], FACTS)
    assert len(failures) == 1 and "invalid" in failures[0]


# ---- check() lifecycle ------------------------------------------------------

def _cfg(**vg_over):
    base = {"enabled": True, "assertions": ["scene.visibleMeshCount > 0"],
            "run_cmd": None}
    base.update(vg_over)
    return Config({"visual_gate": base, "run": {"max_retries": 3}}, "p", Path("/tmp"))


def test_check_disabled_passes_through():
    cfg = Config({"visual_gate": {"enabled": False}}, "p", Path("/tmp"))
    assert vg.check(cfg, Path("/tmp")).passed is True


def test_check_passes_and_fails_via_injected_facts(tmp_path):
    cfg = _cfg()
    ok = vg.check(cfg, tmp_path, fetch_facts=lambda: {"scene": {"visibleMeshCount": 2}})
    assert ok.passed is True and ok.enforced is True
    bad = vg.check(cfg, tmp_path, fetch_facts=lambda: {"scene": {"visibleMeshCount": 0}})
    assert bad.passed is False and bad.failures == ["scene.visibleMeshCount > 0"]
    assert bad.enforced is True


def test_check_enabled_without_inspector_is_unenforced_passthrough(tmp_path):
    # Regression (F3): enabled but no fact source -> passes through, but flagged
    # enforced=False so the node can log an auditable `visual_gate_skipped` event
    # instead of masquerading as a real visual pass.
    cfg = _cfg()
    res = vg.check(cfg, tmp_path)          # no fetch_facts wired (production reality)
    assert res.passed is True and res.enforced is False


# ---- routing ----------------------------------------------------------------

def _cfg_route(enabled=True):
    return Config({"visual_gate": {"enabled": enabled},
                   "run": {"max_retries": 3, "escalate_on_exhaustion": False}},
                  "p", Path("/tmp"))


def _state(status, visual=True, attempt=1):
    return {"attempt": attempt, "spec": {"visual": visual},
            "candidates": [{"cand_id": "w", "attempt": attempt, "status": status}]}


def test_green_visual_routes_to_visual_gate():
    cfg = _cfg_route(enabled=True)
    assert decide_after_collect(cfg, _state("gate_passed", visual=True)) == "visual_gate"


def test_green_nonvisual_routes_to_review():
    cfg = _cfg_route(enabled=True)
    assert decide_after_collect(cfg, _state("gate_passed", visual=False)) == "review"


def test_disabled_gate_routes_to_review():
    cfg = _cfg_route(enabled=False)
    assert decide_after_collect(cfg, _state("gate_passed", visual=True)) == "review"
    assert visual_needed(cfg, _state("gate_passed", visual=True)) is False


def test_visual_failed_reruns_not_livelock():
    cfg = _cfg_route(enabled=True)
    # after the visual gate, the only candidate is visual_failed (not green)
    st = _state("visual_failed", visual=True, attempt=1)
    # not green -> retry (attempt 1 < max_retries 3), so dispatch has work to do
    assert decide_after_visual(cfg, st) == "dispatch"
    # and a passing candidate after visual -> review
    assert decide_after_visual(cfg, _state("gate_passed", visual=True)) == "review"


def test_visual_failed_out_of_retries_finalizes():
    cfg = _cfg_route(enabled=True)
    assert decide_after_visual(cfg, _state("visual_failed", attempt=3)) == "finalize"


# ---- facts_cmd: the real fact source (item 4) --------------------------------

def _json_cmd(payload: str) -> str:
    """A one-liner that prints `payload` verbatim on stdout."""
    return f"printf '%s' {shlex.quote(payload)}"


def test_facts_cmd_provides_facts_and_enforces(tmp_path):
    cfg = _cfg(facts_cmd=_json_cmd('{"scene": {"visibleMeshCount": 3}}'))
    res = vg.check(cfg, tmp_path)
    assert res.passed is True and res.enforced is True
    assert res.facts == {"scene": {"visibleMeshCount": 3}}


def test_facts_cmd_failing_assertion_fails_the_candidate(tmp_path):
    cfg = _cfg(facts_cmd=_json_cmd('{"scene": {"visibleMeshCount": 0}}'))
    res = vg.check(cfg, tmp_path)
    assert res.passed is False and res.enforced is True
    assert res.failures == ["scene.visibleMeshCount > 0"]


def test_malformed_json_is_not_silently_green(tmp_path):
    cfg = _cfg(facts_cmd=_json_cmd("not json at all"))
    res = vg.check(cfg, tmp_path)
    assert res.passed is False          # the whole point: a broken inspector fails
    assert res.enforced is True
    assert "did not print JSON" in res.failures[0]


def test_non_object_json_rejected(tmp_path):
    cfg = _cfg(facts_cmd=_json_cmd("[1, 2, 3]"))
    res = vg.check(cfg, tmp_path)
    assert res.passed is False and "expected a JSON object" in res.failures[0]


def test_nonzero_exit_fails_with_stderr(tmp_path):
    cfg = _cfg(facts_cmd="echo 'boom: no renderer' >&2; exit 3")
    res = vg.check(cfg, tmp_path)
    assert res.passed is False
    assert "exited 3" in res.failures[0] and "no renderer" in res.failures[0]


def test_facts_cmd_timeout_fails(tmp_path):
    cfg = _cfg(facts_cmd="sleep 5", facts_timeout_s=1)
    res = vg.check(cfg, tmp_path)
    assert res.passed is False and "timed out" in res.failures[0]


def test_facts_cmd_runs_in_the_candidate_worktree(tmp_path):
    (tmp_path / "marker.txt").write_text("here")
    cfg = _cfg(facts_cmd='python3 -c "import os,json;'
                         'print(json.dumps({\'scene\':{\'visibleMeshCount\':'
                         'os.path.exists(\'marker.txt\')*2}}))"')
    res = vg.check(cfg, tmp_path)
    assert res.passed is True and res.facts["scene"]["visibleMeshCount"] == 2


def test_injected_fetch_facts_still_wins_over_facts_cmd(tmp_path):
    # The injection point stays open for a future MCP client.
    cfg = _cfg(facts_cmd=_json_cmd('{"scene": {"visibleMeshCount": 0}}'))
    res = vg.check(cfg, tmp_path, fetch_facts=lambda: {"scene": {"visibleMeshCount": 9}})
    assert res.passed is True


def test_ready_probe_timeout_fails_before_facts(tmp_path):
    called = []
    # run_cmd is required for the probe to be live at all — it exists to wait for
    # the app we start (see test_ready_probe_without_run_cmd_is_skipped).
    cfg = _cfg(run_cmd="sleep 30",
               facts_cmd=_json_cmd('{"scene": {"visibleMeshCount": 1}}'),
               ready_probe="http://127.0.0.1:1/never", ready_timeout_s=1)
    res = vg.check(cfg, tmp_path, fetch_facts=lambda: called.append(1) or {})
    assert res.passed is False and "not ready" in res.failures[0]
    assert called == []                 # facts never fetched against a dead app


def test_ready_probe_without_run_cmd_is_skipped(tmp_path):
    """A probe with nothing to probe must not block. Otherwise this misconfig
    burns the full ready_timeout per candidate and then fails all of them, which
    reads like a broken scene rather than a broken config."""
    cfg = _cfg(ready_probe="http://127.0.0.1:1/never", ready_timeout_s=30)
    res = vg.check(cfg, tmp_path, fetch_facts=lambda: {"scene": {"visibleMeshCount": 3}})
    assert res.passed is True and res.enforced is True


# ---- the graph node: a crashed inspector is not a pass ----------------------

class _EventStore:
    def __init__(self):
        self.events = []

    def log_event(self, run_id, task_id, kind, detail=""):
        self.events.append((kind, detail))

    def set_task_status(self, *a, **k):
        pass


def _visual_node(cfg, store):
    from orchestrator.core.context import RunContext
    from orchestrator.engine.graph import build_task_graph
    ctx = RunContext(cfg=cfg, store=store, git=None, budget=None, run_id="r",
                     dry_run=True)
    return build_task_graph(ctx).nodes["visual_gate"].runnable


def _green_state(tmp_path):
    return {"run_id": "r", "task_id": "T-1", "attempt": 1,
            "spec": {"id": "T-1", "visual": True},
            "candidates": [{"cand_id": "w", "attempt": 1, "status": "gate_passed",
                            "worktree": str(tmp_path), "model": "m", "diff": "d",
                            "gate_log": "", "error": "", "notes": "",
                            "branch": "b", "messages": []}]}


async def test_inspector_crash_marks_candidate_failed_not_green(tmp_path, monkeypatch):
    """Regression: the node swallowed any inspector exception with `continue`,
    leaving the candidate `gate_passed` — a crash looked exactly like a pass."""
    store = _EventStore()
    def boom(*a, **k):
        raise RuntimeError("inspector segfaulted")
    monkeypatch.setattr(vg, "check", boom)

    out = await _visual_node(_cfg(facts_cmd="x"), store).ainvoke(_green_state(tmp_path))
    assert out["candidates"][0]["status"] == "visual_failed"
    assert "inspector segfaulted" in out["candidates"][0]["gate_log"]
    assert any(k == "visual_gate_error" for k, _ in store.events)


async def test_skipped_passthrough_is_logged_distinctly(tmp_path):
    # Enabled, no fact source: candidate stays green but the event says skipped.
    store = _EventStore()
    out = await _visual_node(_cfg(facts_cmd=None), store).ainvoke(_green_state(tmp_path))
    assert out["candidates"] == []                       # nothing demoted
    assert [k for k, _ in store.events] == ["visual_gate_skipped"]


async def test_enforced_failure_is_logged_as_a_real_gate(tmp_path):
    store = _EventStore()
    cfg = _cfg(facts_cmd=_json_cmd('{"scene": {"visibleMeshCount": 0}}'))
    out = await _visual_node(cfg, store).ainvoke(_green_state(tmp_path))
    assert out["candidates"][0]["status"] == "visual_failed"
    assert [k for k, _ in store.events] == ["visual_gate"]


# ---- measured facts reach the reviewer --------------------------------------

async def test_enforced_pass_carries_facts_onto_the_candidate(tmp_path):
    """A visual PASS used to discard res.facts, so the measured scene never
    reached the reviewer — which observes nothing itself and would grade a
    visual change on the diff alone."""
    store = _EventStore()
    cfg = _cfg(facts_cmd=_json_cmd('{"scene": {"visibleMeshCount": 7}}'))
    out = await _visual_node(cfg, store).ainvoke(_green_state(tmp_path))
    cand = out["candidates"][0]
    assert cand["status"] == "gate_passed"          # still green, just enriched
    assert cand["visual_facts"] == {"scene": {"visibleMeshCount": 7}}
    assert [k for k, _ in store.events] == ["visual_gate"]


async def test_skipped_pass_carries_no_facts(tmp_path):
    """An unenforced pass-through measured nothing — attaching facts there would
    dress up a skipped gate as evidence."""
    store = _EventStore()
    out = await _visual_node(_cfg(facts_cmd=None), store).ainvoke(_green_state(tmp_path))
    assert out["candidates"] == []


def test_review_prompt_includes_measured_facts():
    from orchestrator.nodes.reviewer import build_review_prompt

    spec = {"id": "T-1", "title": "t", "description": "d", "acceptance": ["a"]}
    passed = {"w": {"model": "m", "diff": "D", "notes": "n",
                    "visual_facts": {"fps": 58, "scene": {"visibleMeshCount": 7}}}}
    text = build_review_prompt(spec, passed)
    assert "Measured scene facts" in text
    assert '"fps": 58' in text
    # Stable-first invariant holds: the spec head is unchanged by facts.
    bare = build_review_prompt(spec, {"w": {**passed["w"], "visual_facts": None}})
    head = "# TASK T-1: t"
    assert text.index(head) == bare.index(head) == 0
    assert "Measured scene facts" not in bare


def test_facts_json_is_deterministic_and_capped():
    from orchestrator.nodes.reviewer import _facts_json

    a = _facts_json({"b": 1, "a": 2})
    b = _facts_json({"a": 2, "b": 1})
    assert a == b                                  # sorted keys, stable payload
    big = _facts_json({"k": "x" * 9000}, max_chars=200)
    assert len(big) < 300 and big.endswith("(truncated)")
