"""Phase 3a: planner output parsing — object envelope, questions shape, and
legacy bare-array back-compat."""

import pytest

from orchestrator.nodes.planner import parse_planner_output


def test_bare_array_legacy():
    env = parse_planner_output('[{"id":"T-1","title":"t","description":"d"}]')
    assert env["questions"] == []
    assert len(env["specs"]) == 1
    assert env["specs"][0]["id"] == "T-1"


def test_object_with_specs():
    text = '''ignored prose
    {"assumptions":["a"],"specs":[{"id":"T-2","title":"t","description":"d",
     "complexity":"m","risk":"low","domain":"physics","visual":true}]}'''
    env = parse_planner_output(text)
    assert env["questions"] == []
    assert env["assumptions"] == ["a"]
    assert env["specs"][0]["visual"] is True
    assert env["specs"][0]["domain"] == "physics"


def test_object_with_questions_is_ask():
    text = '{"questions":[{"id":"q1","q":"which db?","why":"changes files"}],"specs":[]}'
    env = parse_planner_output(text)
    assert env["questions"] and env["questions"][0]["id"] == "q1"
    assert env["specs"] == []


def test_bare_single_spec_object():
    env = parse_planner_output('{"id":"T-3","title":"t","description":"d"}')
    assert env["specs"] == [{"id": "T-3", "title": "t", "description": "d"}]


def test_no_json_raises():
    with pytest.raises(ValueError):
        parse_planner_output("no json here")
