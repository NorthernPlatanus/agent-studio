"""Defect-plan #2 items 3, 4 and 6: who gets another attempt, and when.

Three separate holes in the retry ladder, all observed in live runs:

  3. `escalation_ready` demanded a RED gate, so a candidate that passed the gate
     and kept drawing `revise` finalized `failed` at max_fix_rounds with
     escalate_on_exhaustion on the whole time — the senior never consulted.
  4. The senior got exactly one attempt (`escalated` -> finalize), and died on a
     trivial missing import.
  6. A worker that answers a `revise` with prose spent a fix round having
     attempted nothing, twice in a row, which forced the escalation.
"""

from pathlib import Path

from orchestrator.core.config import Config
from orchestrator.engine.graph import (attempts_used, decide_after_collect,
                                       decide_after_review, decide_after_verify,
                                       is_stuck, plan_dispatch, retry_ceiling,
                                       senior_rounds_left, unproductive_attempts)

SPEC = {"id": "T-1", "title": "t", "risk": "med"}


def _cfg(**run_over):
    run = {"n_candidates": 1, "max_retries": 3, "max_fix_rounds": 2,
           "escalate_on_exhaustion": True, "senior_fix_rounds": 1,
           "max_unproductive_attempts": 2, "auto_integrate_low_risk": False,
           "senior_first_for_high_risk": False}
    run.update(run_over)
    return Config({"run": run,
                   "roles": {"worker": {"default": "cheap", "candidates": ["cheap"]}},
                   "gate": {"log_tail_chars": 500},
                   "visual_gate": {"enabled": False},
                   "verify": {"enabled": False},
                   "domains": {}}, "p", Path("/tmp"))


def _cand(cand_id, attempt, status, **extra):
    return {"cand_id": cand_id, "attempt": attempt, "status": status,
            "gate_log": "typecheck exploded", "error": "", **extra}


# ---- 3. escalation must be reachable from a REVIEW failure -------------------

def _green_revise(attempt, notes="add the tests you were asked for"):
    return {"attempt": attempt, "spec": SPEC,
            "candidates": [_cand("cheap", attempt, "gate_passed")],
            "verdict": {"decision": "revise", "winner": "cheap", "notes": notes}}


def test_review_revise_still_retries_while_rounds_remain():
    assert decide_after_review(_cfg(), _green_revise(1)) == "dispatch"


def test_review_revise_out_of_rounds_escalates_instead_of_finalizing():
    """The regression: a green gate the reviewer keeps rejecting IS the task the
    ladder exists for, and it used to fall straight through to finalize."""
    cfg, state = _cfg(), _green_revise(2)          # max_fix_rounds == 2
    assert decide_after_review(cfg, state) == "dispatch"
    update = plan_dispatch(cfg, state)
    assert update["to_run"] == ["senior"]
    assert update["escalated"] is True
    assert update["escalated_attempt"] == 3        # bounds the senior's retries
    # The senior needs the objection it has to answer — a gate log would be green.
    assert "reviewer" in update["feedback"]
    assert "add the tests you were asked for" in update["feedback"]


def test_review_revise_still_finalizes_with_the_ladder_off():
    """Unchanged behavior when there is no senior configured: bounded by
    max_retries, and nowhere to escalate to."""
    cfg = _cfg(escalate_on_exhaustion=False)       # ceiling == max_retries == 3
    assert retry_ceiling(cfg) == 3
    assert decide_after_review(cfg, _green_revise(2)) == "dispatch"
    assert decide_after_review(cfg, _green_revise(3)) == "finalize"


def test_a_red_gate_still_escalates_exactly_as_before():
    cfg = _cfg()
    red = {"attempt": 2, "spec": SPEC,
           "candidates": [_cand("cheap", 2, "gate_failed")]}
    assert decide_after_collect(cfg, red) == "dispatch"
    update = plan_dispatch(cfg, red)
    assert update["to_run"] == ["senior"] and update["escalated"] is True
    assert "pass the gate" in update["feedback"]
    assert "typecheck exploded" in update["feedback"]


def test_an_approved_task_never_escalates():
    cfg = _cfg()
    state = {"attempt": 2, "spec": SPEC,
             "candidates": [_cand("cheap", 2, "gate_passed")],
             "verdict": {"decision": "approve", "winner": "cheap"}}
    assert decide_after_review(cfg, state) == "integrate"


# ---- 4. the senior gets senior_fix_rounds retries ----------------------------

def _senior_red(attempt, escalated_attempt=3):
    return {"attempt": attempt, "spec": SPEC, "escalated": True,
            "escalated_attempt": escalated_attempt,
            "candidates": [_cand("senior", attempt, "gate_failed")]}


def test_the_senior_gets_one_retry_then_a_human():
    cfg = _cfg()
    assert decide_after_collect(cfg, _senior_red(3)) == "dispatch"
    assert decide_after_collect(cfg, _senior_red(4)) == "finalize"


def test_the_seniors_retry_carries_the_gate_log_and_does_not_re_escalate():
    cfg = _cfg()
    update = plan_dispatch(cfg, _senior_red(3))
    assert update["to_run"] == ["senior"]
    assert "escalated" not in update          # the once-guard is already set
    assert "typecheck exploded" in update["feedback"]


def test_senior_fix_rounds_zero_restores_the_old_one_shot_behavior():
    cfg = _cfg(senior_fix_rounds=0)
    assert senior_rounds_left(cfg, _senior_red(3)) is False
    assert decide_after_collect(cfg, _senior_red(3)) == "finalize"


def test_more_rounds_are_honoured():
    cfg = _cfg(senior_fix_rounds=2)
    assert decide_after_collect(cfg, _senior_red(4)) == "dispatch"
    assert decide_after_collect(cfg, _senior_red(5)) == "finalize"


def test_a_checkpoint_without_escalated_attempt_falls_back_to_one_shot():
    """State written before the field existed (or senior_first, which skips the
    once-guard): guessing a start attempt could loop, so don't."""
    cfg = _cfg()
    state = _senior_red(3)
    del state["escalated_attempt"]
    assert senior_rounds_left(cfg, state) is False
    assert decide_after_collect(cfg, state) == "finalize"


def test_the_senior_also_gets_its_retry_after_a_review_revise():
    cfg = _cfg()
    state = {**_green_revise(3), "escalated": True, "escalated_attempt": 3}
    state["candidates"] = [_cand("senior", 3, "gate_passed")]
    state["verdict"]["winner"] = "senior"
    assert decide_after_review(cfg, state) == "dispatch"
    state["attempt"] = 4
    assert decide_after_review(cfg, state) == "finalize"


# ---- 6. an attempt that attempted nothing is refunded -----------------------

def _prose(cand_id, attempt):
    return _cand(cand_id, attempt, "patch_failed", no_patch=True,
                 error="worker returned no <file>/<edit> blocks")


def test_a_prose_reply_does_not_cost_a_fix_round():
    cfg = _cfg()                                   # ceiling 2
    state = {"attempt": 2, "spec": SPEC,
             "candidates": [_prose("cheap", 1), _prose("cheap", 2)]}
    assert unproductive_attempts(cfg, state) == 2
    assert attempts_used(cfg, state) == 0
    assert decide_after_collect(cfg, state) == "dispatch"   # not escalate, not finalize


def test_the_refund_is_bounded_so_a_chatty_model_cannot_loop_forever():
    cfg = _cfg()                                   # cap 2
    state = {"attempt": 4, "spec": SPEC,
             "candidates": [_prose("cheap", n) for n in (1, 2, 3, 4)]}
    assert unproductive_attempts(cfg, state) == 2
    assert attempts_used(cfg, state) == 2          # the ceiling is reached anyway
    assert decide_after_collect(cfg, state) == "dispatch"   # -> escalation branch
    assert plan_dispatch(cfg, state)["to_run"] == ["senior"]


def test_a_mixed_attempt_is_productive():
    """One candidate rambled, another produced a patch that failed its gate. Work
    happened, so the round counts."""
    cfg = _cfg()
    state = {"attempt": 1, "spec": SPEC,
             "candidates": [_prose("a", 1), _cand("b", 1, "gate_failed")]}
    assert unproductive_attempts(cfg, state) == 0
    assert attempts_used(cfg, state) == 1


def test_the_refund_can_be_disabled():
    cfg = _cfg(max_unproductive_attempts=0)
    state = {"attempt": 2, "spec": SPEC,
             "candidates": [_prose("cheap", 1), _prose("cheap", 2)]}
    assert unproductive_attempts(cfg, state) == 0
    assert attempts_used(cfg, state) == 2
    # ceiling reached -> the ladder takes over exactly as it did before
    assert plan_dispatch(cfg, state)["to_run"] == ["senior"]


def test_ordinary_failures_are_never_refunded():
    cfg = _cfg()
    state = {"attempt": 2, "spec": SPEC,
             "candidates": [_cand("cheap", 1, "gate_failed"),
                            _cand("cheap", 2, "gate_failed")]}
    assert unproductive_attempts(cfg, state) == 0
    assert attempts_used(cfg, state) == 2


# ---- 4 x 6: the senior's rounds are refunded on the same terms ---------------

def _senior_prose(*attempts, escalated_attempt=3, last="patch_failed"):
    """The senior escalated at `escalated_attempt` and answered every attempt
    listed with prose instead of `<file>`/`<edit>` blocks."""
    cands = [_cand("senior", a, last, no_patch=True,
                   error="worker returned no <file>/<edit> blocks")
             for a in attempts]
    return {"attempt": max(attempts), "spec": SPEC, "escalated": True,
            "escalated_attempt": escalated_attempt, "candidates": cands}


def test_the_seniors_round_is_not_spent_on_prose():
    """Item 6 refunds an attempt that attempted nothing — but senior_rounds_left
    measured the RAW attempt, so the most expensive tier could burn its entire
    allowance on two conversational replies without emitting a single patch,
    while the cheap tier was forgiven exactly that."""
    cfg = _cfg()
    assert senior_rounds_left(cfg, _senior_prose(3)) is True
    assert senior_rounds_left(cfg, _senior_prose(3, 4)) is True
    assert decide_after_collect(cfg, _senior_prose(3, 4)) == "dispatch"


def test_the_seniors_refund_is_bounded_too():
    """`max_unproductive_attempts` caps this window as it caps the cheap tier's,
    so a permanently chatty senior still terminates."""
    cfg = _cfg()                                   # cap 2, senior_fix_rounds 1
    assert senior_rounds_left(cfg, _senior_prose(3, 4, 5)) is True
    assert senior_rounds_left(cfg, _senior_prose(3, 4, 5, 6)) is False
    assert decide_after_collect(cfg, _senior_prose(3, 4, 5, 6)) == "finalize"


def test_a_senior_round_spent_on_a_real_patch_is_still_spent():
    """The refund is for attempts that attempted nothing. A red gate is work."""
    cfg = _cfg()
    state = {"attempt": 4, "spec": SPEC, "escalated": True, "escalated_attempt": 3,
             "candidates": [_cand("senior", 3, "gate_failed"),
                            _cand("senior", 4, "gate_failed")]}
    assert senior_rounds_left(cfg, state) is False


def test_the_senior_is_not_handed_the_cheap_tiers_refunds():
    """`since=escalated_attempt`: the rounds the cheap tier wasted on prose were
    already refunded against retry_ceiling and must not be spent twice."""
    cfg = _cfg()
    state = {"attempt": 4, "spec": SPEC, "escalated": True, "escalated_attempt": 3,
             "candidates": [_cand("cheap", 1, "patch_failed", no_patch=True),
                            _cand("cheap", 2, "patch_failed", no_patch=True),
                            _cand("senior", 3, "gate_failed"),
                            _cand("senior", 4, "gate_failed")]}
    assert unproductive_attempts(cfg, state) == 2          # global view unchanged
    assert unproductive_attempts(cfg, state, since=3) == 0
    assert senior_rounds_left(cfg, state) is False         # both rounds were work


# ---- 3 (again). "stuck" has exactly one definition -------------------------

def _scene_rejected(n, attempt=2):
    """The scene verdict inspected the winner and rejected it. At best-of-N the
    loser was never inspected — one verdict per task is what makes the phase
    affordable — so it is still sitting there `gate_passed`."""
    cands = [_cand("a", attempt, "visual_failed")]
    if n > 1:
        cands.append(_cand("b", attempt, "gate_passed"))
    return {"attempt": attempt, "spec": SPEC, "candidates": cands,
            "verdict": {"decision": "approve", "winner": "a"}}


def test_is_stuck_covers_all_three_shapes():
    """One function, because the gate path, the review path and the verify path
    each learned this separately and disagreed in between."""
    red = {"candidates": [_cand("cheap", 1, "gate_failed")]}
    assert is_stuck(red, {"cheap": _cand("cheap", 1, "gate_failed")}) is True
    green_revise = _green_revise(1)
    assert is_stuck(green_revise, {"cheap": _cand("cheap", 1, "gate_passed")}) is True
    st = _scene_rejected(2)
    assert is_stuck(st, {"a": _cand("a", 2, "visual_failed"),
                         "b": _cand("b", 2, "gate_passed")}) is True
    # ...and a healthy green task is not stuck.
    ok = {"verdict": {"decision": "approve", "winner": "a"}}
    assert is_stuck(ok, {"a": _cand("a", 1, "gate_passed")}) is False


def test_a_scene_rejection_escalates_even_with_a_stale_green_sibling():
    """`not any(gate_passed)` saw the never-inspected loser and called the task
    healthy, so the senior was never consulted on a task the cheap tier could not
    get past the scene verdict — item 3's complaint, in the verify router."""
    cfg = _cfg()                                   # ceiling 2, ladder on
    assert decide_after_verify(cfg, _scene_rejected(1)) == "dispatch"
    assert decide_after_verify(cfg, _scene_rejected(2)) == "dispatch"


def test_the_verify_router_and_dispatch_agree_on_escalating():
    """The failure mode this guards is silent: the router says "dispatch, we are
    escalating" and dispatch takes the ordinary retry branch instead, so the
    ladder does nothing and nothing logs it."""
    cfg = _cfg()
    for n in (1, 2):
        state = _scene_rejected(n)
        assert decide_after_verify(cfg, state) == "dispatch"
        update = plan_dispatch(cfg, state)
        assert update["to_run"] == ["senior"], f"n={n}"
        assert update["escalated"] is True, f"n={n}"


def test_a_scene_rejection_below_the_ceiling_just_retries():
    cfg = _cfg()                                   # max_fix_rounds 2
    state = _scene_rejected(2, attempt=1)
    assert decide_after_verify(cfg, state) == "dispatch"
    update = plan_dispatch(cfg, state)
    assert update["to_run"] == ["a"]               # the rejected winner, warm
    assert "escalated" not in update


def test_the_ladder_off_still_finalizes_a_scene_rejection():
    cfg = _cfg(escalate_on_exhaustion=False)       # ceiling falls to max_retries
    state = _scene_rejected(2, attempt=3)
    assert decide_after_verify(cfg, state) == "finalize"
