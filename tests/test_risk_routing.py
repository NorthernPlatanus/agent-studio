"""O1: route on the planner's `risk` / `complexity` labels.

The planner writes both into every spec and nothing read either one. Both
routings are opt-in — with the flags off, every assertion here must describe the
old behavior exactly.
"""

from pathlib import Path

from orchestrator.core.config import Config
from orchestrator.engine.graph import (auto_integrate, decide_after_collect,
                                       plan_dispatch, senior_first)


def _cfg(**run_over):
    run = {"n_candidates": 1, "max_retries": 3, "max_fix_rounds": 4,
           "escalate_on_exhaustion": False, "auto_integrate_low_risk": False,
           "senior_first_for_high_risk": False}
    run.update(run_over)
    return Config({"run": run,
                   "roles": {"worker": {"default": "cheap", "candidates": ["cheap"]}},
                   "gate": {"log_tail_chars": 500},
                   "visual_gate": {"enabled": False},
                   "domains": {}}, "p", Path("/tmp"))


def _state(spec, status="gate_passed", attempt=1, n=1):
    cands = [{"cand_id": f"c{i}", "attempt": attempt, "status": status,
              "gate_log": "", "error": ""} for i in range(n)]
    return {"attempt": attempt, "spec": spec, "candidates": cands}


LOW = {"id": "T-1", "risk": "low", "complexity": "s"}
HIGH = {"id": "T-2", "risk": "high", "complexity": "m"}
BIG = {"id": "T-3", "risk": "med", "complexity": "l"}


# ---- skip the reviewer for low-risk singles ---------------------------------

def test_green_low_risk_single_goes_to_review_when_flag_is_off():
    assert decide_after_collect(_cfg(), _state(LOW)) == "review"


def test_green_low_risk_single_integrates_directly_when_enabled():
    cfg = _cfg(auto_integrate_low_risk=True)
    assert decide_after_collect(cfg, _state(LOW)) == "integrate"


def test_auto_integrate_requires_low_risk():
    cfg = _cfg(auto_integrate_low_risk=True)
    assert decide_after_collect(cfg, _state(HIGH)) == "review"
    assert decide_after_collect(cfg, _state({"id": "T"})) == "review"   # unlabelled


def test_auto_integrate_requires_a_single_candidate():
    # With N candidates there IS a selection to make — that is the reviewer's job.
    cfg = _cfg(auto_integrate_low_risk=True)
    assert decide_after_collect(cfg, _state(LOW, n=2)) == "review"


def test_auto_integrate_never_skips_the_visual_gate():
    cfg = _cfg(auto_integrate_low_risk=True)
    cfg._data["visual_gate"] = {"enabled": True}
    state = _state({**LOW, "visual": True})
    assert decide_after_collect(cfg, state) == "visual_gate"


def test_auto_integrate_predicate_ignores_red_candidates():
    cfg = _cfg(auto_integrate_low_risk=True)
    red = {"c0": {"cand_id": "c0", "attempt": 1, "status": "gate_failed"}}
    assert auto_integrate(cfg, LOW, red) is False


# ---- start hard tasks at the senior -----------------------------------------

def test_senior_first_is_off_by_default():
    assert senior_first(_cfg(escalate_on_exhaustion=True), HIGH) is False
    assert plan_dispatch(_cfg(), _state(HIGH, attempt=0))["to_run"] == ["cheap"]


def test_senior_first_needs_the_escalation_ladder_configured():
    # Without escalate_on_exhaustion there is no senior target to route to.
    cfg = _cfg(senior_first_for_high_risk=True, escalate_on_exhaustion=False)
    assert senior_first(cfg, HIGH) is False
    assert plan_dispatch(cfg, {"attempt": 0, "spec": HIGH})["to_run"] == ["cheap"]


def test_high_risk_and_large_complexity_start_at_the_senior():
    cfg = _cfg(senior_first_for_high_risk=True, escalate_on_exhaustion=True)
    assert plan_dispatch(cfg, {"attempt": 0, "spec": HIGH})["to_run"] == ["senior"]
    assert plan_dispatch(cfg, {"attempt": 0, "spec": BIG})["to_run"] == ["senior"]
    assert plan_dispatch(cfg, {"attempt": 0, "spec": LOW})["to_run"] == ["cheap"]


def test_senior_first_does_not_burn_the_once_guard():
    """Starting at the senior must not mark the task `escalated` — that guard
    finalizes on the next red result, which would give a hard task ONE attempt."""
    cfg = _cfg(senior_first_for_high_risk=True, escalate_on_exhaustion=True)
    update = plan_dispatch(cfg, {"attempt": 0, "spec": HIGH})
    assert "escalated" not in update
    # so a failed first attempt still retries rather than finalizing
    red = _state(HIGH, status="gate_failed", attempt=1)
    assert decide_after_collect(cfg, red) == "dispatch"


# ---- integrate without a reviewer verdict -----------------------------------

class _Git:
    feature = "agents/feature"
    work_dir = Path("/tmp")

    async def amerge_into_feature(self, branch, message):
        self.merged = (branch, message)
        return "deadbeefcafe"


class _Store:
    def __init__(self):
        self.events = []

    def log_event(self, run_id, task_id, kind, detail=""):
        self.events.append((kind, detail))


async def test_integrate_picks_the_sole_green_candidate_and_logs_it(tmp_path):
    from orchestrator.core.context import RunContext
    from orchestrator.nodes.integrator import integrate

    git, store = _Git(), _Store()
    ctx = RunContext(cfg=_cfg(auto_integrate_low_risk=True), store=store, git=git,
                     budget=None, run_id="r")
    state = {"run_id": "r", "task_id": "T-1",
             "spec": {"id": "T-1", "title": "t", "risk": "low"},
             "candidates": [{"cand_id": "c0", "attempt": 1, "status": "gate_passed",
                             "branch": "agents/wt/t-1-c0", "diff": "d",
                             "gate_log": "", "error": ""}]}
    out = await integrate(ctx, state)          # no "verdict" key at all

    assert out["outcome"] == "done"
    assert out["integration"]["winner"] == "c0"     # named for the writeback
    assert git.merged[0] == "agents/wt/t-1-c0"
    # merging without review must be visible, not inferred from a missing verdict
    assert any(k == "auto_integrated" for k, _ in store.events)
