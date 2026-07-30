"""Phase 3a: planner output parsing — object envelope, questions shape, and
legacy bare-array back-compat."""

from types import SimpleNamespace

import pytest

from orchestrator.nodes.planner import (needs_plan_ids, parse_planner_output,
                                        persist_specs, validate_spec)


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


# ---- decomposition + batching (defect-plan #2 items 5 and 8) ----------------

class _Store:
    def __init__(self, tasks=()):
        self.saved = []
        self.tasks = list(tasks)

    def upsert_task(self, spec):
        self.saved.append(spec)

    def log_event(self, *a, **k):
        pass

    def all_tasks(self):
        return self.tasks


def _persist(specs):
    store = _Store()
    ctx = SimpleNamespace(store=store, run_id="r")
    persist_specs(ctx, specs)
    return {s["id"]: s for s in store.saved}


def test_persist_derives_parent_id_for_sub_tasks():
    """T-131 on the board planned as T-131a/T-131b. Without this the integrator's
    writeback has no line to fall back to and the board keeps no trace of the run."""
    saved = _persist([_spec(id="T-131a"), _spec(id="T-131b")])
    assert saved["T-131a"]["parent_id"] == "T-131"
    assert saved["T-131b"]["parent_id"] == "T-131"


def test_persist_keeps_an_explicit_parent_id():
    saved = _persist([_spec(id="T-500", parent_id="T-131")])
    assert saved["T-500"]["parent_id"] == "T-131"


def test_persist_adds_no_parent_id_to_an_ordinary_spec():
    assert "parent_id" not in _persist([_spec(id="T-140")])["T-140"]


def test_needs_plan_ids_batches_the_whole_queue():
    """Planner cost is per invocation, not per task: 385k for one, 425k for two."""
    store = _Store([{"id": "T-1", "status": "needs_plan"},
                    {"id": "T-2", "status": "ready"},
                    {"id": "T-3", "status": "needs_plan"},
                    {"id": "T-4", "status": "done"}])
    assert needs_plan_ids(store) == ["T-1", "T-3"]


def test_needs_plan_ids_honours_a_limit():
    store = _Store([{"id": f"T-{n}", "status": "needs_plan"} for n in (1, 2, 3)])
    assert needs_plan_ids(store, 2) == ["T-1", "T-2"]
    assert needs_plan_ids(store, 0) == ["T-1", "T-2", "T-3"]     # 0 == no limit


def test_needs_plan_ids_empty_queue():
    assert needs_plan_ids(_Store([{"id": "T-1", "status": "done"}])) == []
