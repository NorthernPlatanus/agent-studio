"""Shape and content of every GET endpoint against the seeded fixture store."""

from __future__ import annotations

from tests.api.fixtures.seed_store import (CHECKPOINT_TASK, PAUSE_NOTE, PROJECT,
                                           RUN_DONE, RUN_PAUSED, TASKS)

BASE = f"/api/projects/{PROJECT}"


def test_projects_lists_the_fixture_project(client):
    body = client.get("/api/projects").json()
    entry = next(p for p in body["projects"] if p["name"] == PROJECT)
    assert entry["has_store"] and entry["has_checkpoints"]
    # The template profile has project.repo_path: null — reportable, not an error.
    assert entry["repo_path"] is None


def test_summary(client):
    body = client.get(f"{BASE}/summary").json()
    assert body["project"] == PROJECT
    assert body["task_count"] == len(TASKS)
    assert body["queue_stats"]["done"] == 2
    assert body["queue_stats"]["needs_plan"] == 1
    assert body["active_run"]["id"] == RUN_PAUSED
    assert body["active_run"]["note"] == PAUSE_NOTE
    assert body["last_run"]["id"] == RUN_PAUSED
    assert body["totals"]["subscription"]["calls"] == 5
    assert body["totals"]["cash"]["calls"] == 6
    assert body["cash_spend_usd"] > 0
    assert body["event_counts"]["escalated"] == 1
    assert body["event_counts"]["visual_gate_skipped"] == 1
    # 0-filled, so the dashboard never has to guess a missing key's meaning.
    assert body["event_counts"]["retrieval_exhausted"] == 0
    assert body["max_event_rowid"] > 0


def test_tasks_list_and_filters(client):
    body = client.get(f"{BASE}/tasks").json()
    assert body["total"] == len(TASKS)
    assert {t["status"] for t in body["tasks"]} == {
        "ready", "running", "needs_human", "done", "failed", "rejected",
        "human_only", "needs_plan"}
    # The list projection carries the filterable spec fields but no spec blob.
    row = next(t for t in body["tasks"] if t["id"] == "T-102")
    assert row["risk"] == "medium" and row["deps"] == ["T-101"]
    assert "spec" not in row

    assert client.get(f"{BASE}/tasks?status=done").json()["total"] == 2
    assert client.get(f"{BASE}/tasks?domain=seam").json()["total"] == 1
    assert client.get(f"{BASE}/tasks?parent_id=T-131").json()["total"] == 2
    assert client.get(f"{BASE}/tasks?q=timeline").json()["total"] == 3
    assert client.get(f"{BASE}/tasks?milestone=M1").json()["total"] == len(TASKS)
    assert client.get(f"{BASE}/tasks?status=nonesuch").json() == {
        "tasks": [], "total": 0, "queue_stats": {}}


def test_task_detail(client):
    body = client.get(f"{BASE}/tasks/T-102").json()
    assert body["id"] == "T-102" and body["status"] == "done"
    assert body["files_write"] == ["src/widgets/task-table.tsx"]
    assert body["acceptance"]
    assert body["cash_spend_usd"] > 0
    assert body["spec"]["n_candidates"] == 3
    kinds = [e["kind"] for e in body["events"]]
    assert kinds == ["gate", "gate", "visual_gate_skipped", "review", "merged",
                     "finalized"]
    assert all(e["task_id"] == "T-102" for e in body["events"])


def test_task_detail_lists_children_of_a_decomposed_parent(client):
    assert client.get(f"{BASE}/tasks/T-131").json()["children"] == ["T-131a", "T-131b"]
    assert client.get(f"{BASE}/tasks/T-131a").json()["parent_id"] == "T-131"


def test_unknown_task_is_404(client):
    assert client.get(f"{BASE}/tasks/T-999").status_code == 404
    assert client.get(f"{BASE}/tasks/T-999/candidates").status_code == 404


def test_candidates_come_from_the_checkpoint_when_there_is_one(client):
    body = client.get(f"{BASE}/tasks/{CHECKPOINT_TASK}/candidates").json()
    assert body["source"] == "checkpoint"
    assert body["run_id"] == RUN_PAUSED
    by_id = {c["cand_id"]: c for c in body["candidates"]}
    assert by_id["deepseek"]["status"] == "gate_passed"
    assert by_id["deepseek"]["attempt"] == 2
    # The diff itself is never shipped — it can be hundreds of KB on a polled route.
    assert by_id["deepseek"]["has_diff"] is True
    assert "diff" not in by_id["deepseek"]
    assert by_id["kimi"]["no_patch"] is True
    assert "prose" in by_id["kimi"]["error"]


def test_candidates_fall_back_to_the_event_log(client):
    body = client.get(f"{BASE}/tasks/T-102/candidates").json()
    assert body["source"] == "events"
    by_key = {(c["cand_id"], c["attempt"]): c for c in body["candidates"]}
    assert by_key[("deepseek", 1)]["status"] == "gate_failed"
    assert "TS2345" in by_key[("deepseek", 1)]["gate_log"]
    assert by_key[("kimi", 1)]["status"] == "gate_passed"


def test_candidates_report_none_when_nothing_was_ever_attempted(client):
    body = client.get(f"{BASE}/tasks/T-130/candidates").json()
    assert body["source"] == "none" and body["candidates"] == []


def test_waves_is_a_zero_cost_schedule_preview(client):
    body = client.get(f"{BASE}/waves").json()
    assert body["max_parallel_tasks"] >= 1
    first = body["waves"][0]["tasks"]
    assert first and all(t["candidates"] for t in first)
    # files_write disjointness inside a wave is the scheduler's core invariant.
    for wave in body["waves"]:
        writes = [w for t in wave["tasks"] for w in t["files_write"]]
        assert len(writes) == len(set(writes))
    scheduled = {t["id"] for w in body["waves"] for t in w["tasks"]}
    # T-131b depends on T-131a, so it can only land in a later wave — never the
    # same one.
    assert "T-131a" in scheduled and "T-131b" in scheduled
    wave_of = {t["id"]: w["index"] for w in body["waves"] for t in w["tasks"]}
    assert wave_of["T-131b"] > wave_of["T-131a"]
    # The seam task declares no deps: a planner mistake the preview must surface.
    assert body["seam_missing_deps"] == ["T-112"]
    assert body["queue_stats"]["ready"] == 5


def test_runs_list_carries_token_totals_per_channel(client):
    runs = client.get(f"{BASE}/runs").json()["runs"]
    assert [r["id"] for r in runs] == [RUN_PAUSED, RUN_DONE]
    done = next(r for r in runs if r["id"] == RUN_DONE)
    assert done["tokens"]["subscription"]["in_tok"] == 402_000 + 62_000 + 71_000 + 24_000
    assert done["tokens"]["cash"]["calls"] == 3
    assert client.get(f"{BASE}/runs?limit=1").json()["runs"][0]["id"] == RUN_PAUSED


def test_run_detail(client):
    body = client.get(f"{BASE}/runs/{RUN_PAUSED}").json()
    assert body["status"] == "paused" and body["note"] == PAUSE_NOTE
    assert body["task_ids"] == ["T-120", "T-121", "T-122"]
    assert {e["kind"] for e in body["events"]} == {
        "gate", "candidate_failed", "no_patch", "escalated", "verify_unverifiable",
        "crashed", "finalized"}
    assert client.get(f"{BASE}/runs/nope").status_code == 404


def test_usage_rollups(client):
    body = client.get(f"{BASE}/usage").json()
    assert body["group_by"] == "role"
    rows = {(r["role"], r["cash"]): r for r in body["rows"]}
    assert rows[("planner", False)]["in_tok"] == 402_000
    assert 0 < rows[("planner", False)]["cache_hit_rate"] < 1
    # kimi reported cache_miss but never a hit: a measured 0 %, not "unknown".
    assert rows[("verifier", False)]["cache_hit_rate"] is None
    assert body["totals"]["cash"]["calls"] == 6

    for group_by in ("model", "provider", "day"):
        rolled = client.get(f"{BASE}/usage?group_by={group_by}").json()
        assert rolled["group_by"] == group_by
        assert rolled["rows"] and all(r["key"] for r in rolled["rows"])
    days = {r["day"] for r in client.get(f"{BASE}/usage?group_by=day").json()["rows"]}
    assert len(days) == 2
    assert client.get(f"{BASE}/usage?group_by=bogus").status_code == 422


def test_metrics_matches_the_cli_report(client):
    body = client.get(f"{BASE}/metrics").json()
    assert body["completed_tasks"] == 2
    outcomes = {(o["cand_id"], o["first_attempt"]): o for o in body["gate_outcomes"]}
    assert outcomes[("deepseek", True)] == {"cand_id": "deepseek",
                                           "first_attempt": True, "passed": 1,
                                           "failed": 2, "pass_rate": 1 / 3}
    assert outcomes[("kimi", True)]["passed"] == 1
    assert body["event_counts"]["no_patch"] == 1
    assert body["event_counts"]["verify_unverifiable"] == 1
    roles = {r["role"]: r for r in body["subscription_tokens_by_role"]}
    assert roles["planner"]["in_tok_per_completed_task"] == 402_000 / 2
    assert body["subscription_in_tok_per_completed_task"] > 0
    assert body["cash_usd_per_completed_task"] > 0


def test_events_pages_by_rowid(client):
    first = client.get(f"{BASE}/events?limit=3").json()
    assert len(first["events"]) == 3
    assert [e["rowid"] for e in first["events"]] == [1, 2, 3]
    cursor = first["next_since_rowid"]
    second = client.get(f"{BASE}/events?since_rowid={cursor}&limit=3").json()
    assert [e["rowid"] for e in second["events"]] == [4, 5, 6]

    tail = client.get(f"{BASE}/events?since_rowid={first['max_rowid']}").json()
    # An empty page keeps the caller's cursor; rewinding to 0 would replay the log.
    assert tail["events"] == [] and tail["next_since_rowid"] == first["max_rowid"]

    filtered = client.get(f"{BASE}/events?kind=gate").json()["events"]
    assert filtered and all(e["kind"] == "gate" for e in filtered)
    scoped = client.get(f"{BASE}/events?task_id=T-121&run_id={RUN_PAUSED}").json()
    assert {e["kind"] for e in scoped["events"]} == {"escalated", "verify_unverifiable",
                                                    "finalized"}


def test_events_desc_returns_the_newest_rows_and_is_exact_under_a_filter(client):
    """The reason `order=` exists: the client-side workaround it replaces was
    approximate, and silently so."""
    head = client.get(f"{BASE}/events?limit=3").json()
    tail = client.get(f"{BASE}/events?limit=3&order=desc").json()
    assert tail["order"] == "desc"
    rowids = [e["rowid"] for e in tail["events"]]
    assert rowids == sorted(rowids, reverse=True)
    assert rowids[0] == head["max_rowid"]
    # The cursor is the highest rowid in EITHER order, so a poller that switched
    # orders mid-stream still moves forward rather than replaying the page.
    assert tail["next_since_rowid"] == head["max_rowid"]

    # Under a filter, `since_rowid = max_rowid - N` counts rows the filter
    # excludes; `LIMIT` after `ORDER BY rowid DESC` counts only matching ones.
    gates = client.get(f"{BASE}/events?kind=gate").json()["events"]
    newest = client.get(f"{BASE}/events?kind=gate&limit=2&order=desc").json()["events"]
    assert len(newest) == 2
    assert [e["rowid"] for e in newest] == [e["rowid"] for e in gates[::-1]][:2]

    approximate = client.get(
        f"{BASE}/events?kind=gate&since_rowid={head['max_rowid'] - 2}").json()
    assert len(approximate["events"]) < 2, (
        "fixture no longer demonstrates the short-page failure this replaces")


def test_jobs_are_empty_for_a_project_that_has_never_run_one(client):
    """The read shape when the supervisor has nothing — the rest of the job
    surface lives in `test_jobs.py`, which actually supervises processes."""
    assert client.get(f"{BASE}/jobs").json() == {"jobs": []}
    r = client.get(f"{BASE}/jobs/whatever")
    assert r.status_code == 404 and "unknown job" in r.json()["detail"]
    assert client.get(f"{BASE}/jobs/whatever/log?offset=0").status_code == 404
