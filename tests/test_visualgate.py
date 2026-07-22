"""Phase 6: visual gate — safe assertion evaluator, routing, and the
visual_failed retry path (no empty-Send livelock)."""

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
