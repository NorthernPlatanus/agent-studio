"""Phase 3a: planner output parsing — object envelope, questions shape, and
legacy bare-array back-compat."""

import pytest

from orchestrator.nodes.planner import parse_planner_output, validate_spec


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


# ---- files_write validation (item 6) ----------------------------------------

def _spec(**over):
    s = {"id": "T-9", "title": "t", "description": "d", "files_write": ["a.py"]}
    s.update(over)
    return s


def test_valid_spec_accepted():
    validate_spec(_spec())


def test_spec_without_files_write_rejected():
    # No allowlist at all => apply_response would permit any path in the worktree.
    with pytest.raises(ValueError, match="no 'files_write'"):
        validate_spec(_spec(files_write=None))
    s = _spec()
    del s["files_write"]
    with pytest.raises(ValueError, match="no 'files_write'"):
        validate_spec(s)


def test_spec_with_empty_files_write_rejected():
    # Empty allowlist => every write rejected => the task can never go green.
    with pytest.raises(ValueError, match="empty/invalid"):
        validate_spec(_spec(files_write=[]))
    with pytest.raises(ValueError, match="empty/invalid"):
        validate_spec(_spec(files_write="a.py"))     # string, not a list


def test_human_only_spec_exempt_from_files_write():
    # agent_able:false never reaches a worker, so it needs no write allowlist.
    validate_spec(_spec(agent_able=False, files_write=None))


def test_missing_core_fields_still_rejected():
    with pytest.raises(ValueError, match="missing 'title'"):
        validate_spec(_spec(title=""))
