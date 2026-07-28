"""Phase 3c: reviewer scoring rubric — threshold guard + max-weighted winner."""

from orchestrator.nodes.reviewer import (_parse_verdict, _select_winner,
                                         passes_threshold, weighted_score)


def test_scores_parsed():
    v = _parse_verdict('{"decision":"approve","scores":{"acceptance":5,"tests":4,'
                       '"minimality":5,"protocol_fit":4}}')
    assert v["decision"] == "approve"
    assert v["scores"]["acceptance"] == 5


def test_threshold_guard_demotes_low_acceptance_approve():
    # high notes but acceptance 3 -> approve flips to revise
    v = _parse_verdict('{"decision":"approve","scores":{"acceptance":3,"tests":5,'
                       '"minimality":5,"protocol_fit":5},"notes":"lgtm"}')
    assert v["decision"] == "revise"


def test_threshold_guard_demotes_any_dim_below_two():
    v = _parse_verdict('{"decision":"approve","scores":{"acceptance":5,"tests":1,'
                       '"minimality":5,"protocol_fit":5}}')
    assert v["decision"] == "revise"


def test_threshold_passes_clean_approve():
    assert passes_threshold({"acceptance": 4, "tests": 2, "minimality": 3,
                             "protocol_fit": 2}) is True
    assert passes_threshold({"acceptance": 4, "tests": 1, "minimality": 3,
                             "protocol_fit": 2}) is False


def test_missing_scorecard_fails_closed():
    # An approve with no scores is the exact hole the rubric closes: it merges
    # on the reviewer's word alone, with nothing deterministic to check.
    v = _parse_verdict('{"decision":"approve","notes":"lgtm"}')
    assert v["decision"] == "revise"
    assert "no scorecard" in v["notes"]     # the note must explain the demotion
    assert "lgtm" in v["notes"]             # ...without dropping the model's own


def test_missing_scorecard_pass_through_when_rubric_not_required():
    v = _parse_verdict('{"decision":"approve","notes":"ok"}', require_rubric=False)
    assert v["decision"] == "approve"       # explicit opt-out (review.require_rubric)
    assert passes_threshold({}, require_rubric=False) is True
    assert passes_threshold({}) is False     # default is fail-closed


def test_bad_json_defaults_revise():
    assert _parse_verdict("not json")["decision"] == "revise"


def test_winner_is_max_weighted():
    passed = {"a": {}, "b": {}, "c": {}}
    verdict = {"winner": "a", "scores_by_candidate": {
        "a": {"acceptance": 3, "tests": 5, "minimality": 5, "protocol_fit": 5},
        "b": {"acceptance": 5, "tests": 5, "minimality": 5, "protocol_fit": 5},
        "c": {"acceptance": 4, "tests": 3, "minimality": 3, "protocol_fit": 3}}}
    # b has the highest weighted score despite the model naming "a" as winner
    assert _select_winner(verdict, passed) == "b"
    assert weighted_score(verdict["scores_by_candidate"]["b"]) > \
        weighted_score(verdict["scores_by_candidate"]["a"])


def test_winner_falls_back_to_stated_then_lowest_id():
    passed = {"x": {}, "y": {}}
    assert _select_winner({"winner": "y"}, passed) == "y"      # no per-cand scores
    assert _select_winner({}, passed) == "x"                    # last resort


def test_winner_tiebreak_is_order_independent():
    # Same candidates, opposite insertion order, identical (tied) scores: the
    # winner must not depend on which dict happened to be built first.
    tie = {"acceptance": 5, "tests": 5, "minimality": 5, "protocol_fit": 5}
    verdict = {"scores_by_candidate": {"b": dict(tie), "a": dict(tie)}}
    assert _select_winner(verdict, {"a": {}, "b": {}}) == \
        _select_winner(verdict, {"b": {}, "a": {}}) == "a"
    # ...and the no-scores last resort is equally stable
    assert _select_winner({}, {"z": {}, "a": {}}) == "a"
