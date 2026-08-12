"""The job supervisor and the control endpoints.

**No test here may spawn a real orchestrator command.** `run`/`plan`/`resume`
spend subscription quota and mutate git worktrees; PLAN §3.4 and `AGENTS.md` both
forbid it. Everything below is supervised through the exact code path a real job
takes, with a harmless child — which is the reason `JobSupervisor.spawn` takes an
argv instead of building one. The argv *builders* are tested separately, as pure
functions, so the real command lines are still pinned without being executed.
"""

from __future__ import annotations

import copy
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from orchestrator.api import deps, jobs
from orchestrator.api.app import create_app
from orchestrator.core.config import Config
from tests.api.fixtures.seed_store import PROJECT, RUN_PAUSED

BASE = f"/api/projects/{PROJECT}"

# Harmless children, all `sys.executable` so nothing depends on PATH.
ECHO = [sys.executable, "-c", "print('hello from a fake job')"]
SLEEPER = [sys.executable, "-c", "import time; time.sleep(30)"]
FAILER = [sys.executable, "-c", "import sys; sys.exit(3)"]


# ---- fixtures ------------------------------------------------------------
@pytest.fixture
def job_state(tmp_path: Path, store_path: Path) -> Path:
    """A per-test state dir holding a *copy* of the seeded store.

    A copy, not the session fixture, because these tests spawn processes that
    write logs beside it and one of them rewrites run statuses.
    """
    state = tmp_path / "state"
    state.mkdir()
    shutil.copy(store_path, state / f"{PROJECT}.sqlite3")
    return state


@pytest.fixture
def job_cfg(job_state: Path, tmp_path: Path, cfg: Config) -> Config:
    """A config that can host a job: a state dir for logs and a repo_path.

    `repo_path` points at an empty tmp dir. Nothing in these tests reads it — it
    exists so `require_repo_path` passes, since the template `example` profile
    deliberately leaves it null.
    """
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    data = copy.deepcopy(cfg.as_dict())
    data["paths"]["state_dir"] = str(job_state)
    data["project"]["repo_path"] = str(checkout)
    return Config(data, PROJECT, deps.REPO_ROOT)


@pytest.fixture
def sup() -> jobs.JobSupervisor:
    # Short graces: the tests' children exit on the first SIGINT, and a real
    # 15s escalation would only be waited out by a bug.
    return jobs.JobSupervisor(sigint_grace_s=5.0, sigterm_grace_s=2.0)


@pytest.fixture
def job_client(job_cfg: Config, sup: jobs.JobSupervisor):
    registry = deps.ProjectRegistry(root=Path(job_cfg.paths.state_dir),
                                    configs={PROJECT: job_cfg})
    app = create_app()
    app.dependency_overrides[deps.get_registry] = lambda: registry
    app.dependency_overrides[jobs.get_supervisor] = lambda: sup
    with TestClient(app) as client:
        yield client


def _wait_for(sup: jobs.JobSupervisor, record: jobs.JobRecord,
              timeout: float = 10.0) -> jobs.JobRecord:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if sup.refresh(record).status not in jobs.LIVE_STATUSES:
            return record
        time.sleep(0.02)
    raise AssertionError(f"job {record.job_id} never exited (status {record.status})")


# ---- argv builders (pure, and the only place real commands appear) --------
def test_argv_builders_produce_the_cli_command_lines():
    assert jobs.run_argv("example") == [
        sys.executable, "-m", "orchestrator", "run", "--project", "example"]
    assert jobs.run_argv("example", tasks=["T-1", "T-2"], n=3, dry_run=True) == [
        sys.executable, "-m", "orchestrator", "run", "--project", "example",
        "--dry-run", "--tasks", "T-1,T-2", "--n", "3"]
    assert jobs.plan_argv("example", all_needs_plan=True, limit=5, note="focus") == [
        sys.executable, "-m", "orchestrator", "plan", "--project", "example",
        "--all-needs-plan", "--limit", "5", "focus"]
    # The note is positional and last, so argparse cannot read a later flag as
    # part of it.
    assert jobs.plan_argv("example", note="a note")[-1] == "a note"
    assert jobs.resume_argv("example")[-3:] == ["resume", "--project", "example"]
    assert jobs.import_backlog_argv("example")[-3:] == [
        "import-backlog", "--project", "example"]


def test_argv_uses_this_interpreter_not_whatever_python_is_on_path():
    """A job that resolved a different interpreter would import a different
    orchestrator — and a different store schema."""
    assert jobs.run_argv("example")[0] == sys.executable


# ---- supervisor ----------------------------------------------------------
def test_a_job_runs_to_completion_and_keeps_its_log(sup, job_cfg):
    record = sup.spawn(job_cfg, "run", ECHO)
    assert record.status == "running" and record.pid
    _wait_for(sup, record)

    assert record.status == "exited" and record.exit_code == 0
    assert record.ended_at and record.ended_at >= record.started_at
    _, text, eof = sup.read_log(record)
    assert "hello from a fake job" in text and eof
    # Nothing is evicted on exit: the console can still be read afterwards.
    assert [r.job_id for r in sup.list(job_cfg)] == [record.job_id]


def test_a_nonzero_exit_is_failed_not_stopped(sup, job_cfg):
    record = _wait_for(sup, sup.spawn(job_cfg, "run", FAILER))
    assert record.status == "failed" and record.exit_code == 3


def test_only_one_job_at_a_time_per_project(sup, job_cfg):
    first = sup.spawn(job_cfg, "run", SLEEPER)
    with pytest.raises(jobs.JobError) as excinfo:
        sup.spawn(job_cfg, "plan", ECHO)
    assert excinfo.value.status_code == 409
    assert first.job_id in excinfo.value.detail

    # …and the slot frees up once it is gone, rather than needing a restart.
    first.proc.kill()
    _wait_for(sup, first)
    assert sup.spawn(job_cfg, "plan", ECHO).job_id != first.job_id


def test_concurrent_spawns_still_start_exactly_one_job(sup, job_cfg):
    """The one-in-flight rule has to hold under real concurrency, not just in
    sequence: FastAPI runs these `def` endpoints in a threadpool, so two POSTs
    genuinely race. Unlocked, eight threads started seven runs — seven real runs
    spending quota and fighting over the same git worktrees."""
    import concurrent.futures as cf

    def attempt(_):
        try:
            return sup.spawn(job_cfg, "run", SLEEPER).job_id
        except jobs.JobError as e:
            assert e.status_code == 409
            return None

    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(attempt, range(8)))

    started = [r for r in results if r]
    assert len(started) == 1, f"{len(started)} jobs started concurrently"
    assert len(sup.list(job_cfg)) == 1

    for record in sup.list(job_cfg):
        if record.proc:
            record.proc.kill()
        _wait_for(sup, record)


def test_the_log_tail_is_byte_offset_and_resumable(sup, job_cfg):
    record = _wait_for(sup, sup.spawn(job_cfg, "run", ECHO))
    head_offset, head, _ = sup.read_log(record, 0, max_bytes=5)
    assert (head, head_offset) == ("hello", 5)

    rest_offset, rest, eof = sup.read_log(record, head_offset)
    assert head + rest == sup.read_log(record, 0)[1]
    assert eof and rest_offset == record.log_path.stat().st_size

    # Reading past the end is empty, not an error, and does not rewind.
    past = sup.read_log(record, rest_offset + 1000)
    assert past[1] == "" and past[0] == rest_offset


def test_reading_a_log_that_does_not_exist_yet_is_empty_not_an_error(sup, job_cfg):
    record = _wait_for(sup, sup.spawn(job_cfg, "run", ECHO))
    record.log_path.unlink()
    assert sup.read_log(record, 0) == (0, "", True)


async def test_stop_is_a_signal_and_the_record_says_stopped(sup, job_cfg):
    record = sup.spawn(job_cfg, "run", SLEEPER)
    stopped = await sup.stop(record)
    assert stopped.status == "stopped"
    assert stopped.ended_at is not None
    # `stopped`, not `failed`, even though a signalled child exits non-zero: the
    # difference is whether a human asked for it.
    assert stopped.exit_code != 0


async def test_stopping_an_already_finished_job_is_a_no_op(sup, job_cfg):
    record = _wait_for(sup, sup.spawn(job_cfg, "run", ECHO))
    again = await sup.stop(record)
    assert again.status == "exited" and again.exit_code == 0


def test_jobs_are_scoped_to_their_project(sup, job_cfg, tmp_path):
    record = _wait_for(sup, sup.spawn(job_cfg, "run", ECHO))
    other = copy.deepcopy(job_cfg.as_dict())
    other_cfg = Config(other, "other-project", deps.REPO_ROOT)
    assert sup.list(other_cfg) == []
    with pytest.raises(jobs.JobError) as excinfo:
        sup.get(other_cfg, record.job_id)
    assert excinfo.value.status_code == 404


def test_a_restarted_api_rebuilds_the_registry_from_the_sidecars(sup, job_cfg):
    finished = _wait_for(sup, sup.spawn(job_cfg, "run", ECHO))
    assert finished.sidecar_path.exists()

    fresh = jobs.JobSupervisor()
    rebuilt = fresh.get(job_cfg, finished.job_id)
    assert rebuilt.adopted
    assert (rebuilt.status, rebuilt.exit_code) == ("exited", 0)
    assert "hello from a fake job" in fresh.read_log(rebuilt)[1]


def test_a_job_the_restart_lost_is_failed_with_an_unknown_exit_code(sup, job_cfg):
    """The process is gone but the sidecar still says `running`: we never waited
    on it, so the code is unrecoverable. `null` says that, and an interrupted run
    is not reported as a success."""
    record = sup.spawn(job_cfg, "run", SLEEPER)
    record.proc.kill()
    record.proc.wait()

    fresh = jobs.JobSupervisor()
    rebuilt = fresh.get(job_cfg, record.job_id)
    assert rebuilt.status == "failed" and rebuilt.exit_code is None


def test_the_cursor_changes_when_a_status_does(sup, job_cfg):
    empty = sup.cursor(job_cfg)
    record = sup.spawn(job_cfg, "run", ECHO)
    running = sup.cursor(job_cfg)
    _wait_for(sup, record)
    assert len({empty, running, sup.cursor(job_cfg)}) == 3


def test_a_project_without_a_state_dir_cannot_host_a_job(sup, cfg, tmp_path):
    broken = copy.deepcopy(cfg.as_dict())
    broken["paths"]["state_dir"] = None
    with pytest.raises(jobs.JobError) as excinfo:
        sup.log_dir(Config(broken, PROJECT, deps.REPO_ROOT))
    assert excinfo.value.status_code == 409


def test_resolve_run_id_attaches_the_run_the_child_minted(sup, job_cfg, job_state):
    record = _wait_for(sup, sup.spawn(job_cfg, "run", ECHO))
    store = job_state / f"{PROJECT}.sqlite3"
    conn = sqlite3.connect(store)
    with conn:
        conn.execute("INSERT INTO runs (id, started_at, status) VALUES (?,?,?)",
                     ("20260801-000000-cccccc", record.started_at + 1, "running"))
    conn.close()

    assert sup.resolve_run_id(record, store).run_id == "20260801-000000-cccccc"
    # A run that predates the job is not this job's run.
    other = _wait_for(sup, sup.spawn(job_cfg, "run", ECHO))
    other.started_at = time.time() + 3600
    assert sup.resolve_run_id(other, store).run_id is None


def test_resolve_run_id_never_raises_on_a_missing_store(sup, job_cfg, tmp_path):
    record = _wait_for(sup, sup.spawn(job_cfg, "run", ECHO))
    assert sup.resolve_run_id(record, tmp_path / "nope.sqlite3").run_id is None
    assert sup.resolve_run_id(record, None).run_id is None


# ---- HTTP: the spending gate ---------------------------------------------
def test_a_run_without_confirm_is_refused_before_anything_is_spawned(job_client, sup,
                                                                     job_cfg):
    r = job_client.post(f"{BASE}/jobs/run", json={})
    assert r.status_code == 422
    assert "confirm" in r.text
    assert sup.list(job_cfg) == []


def test_plan_and_resume_are_confirm_gated_too(job_client):
    assert job_client.post(f"{BASE}/jobs/plan", json={}).status_code == 422
    assert job_client.post(f"{BASE}/jobs/resume", json={}).status_code == 422


def test_a_task_id_that_could_be_read_as_a_flag_is_refused(job_client):
    r = job_client.post(f"{BASE}/jobs/run",
                        json={"confirm": True, "tasks": ["--dry-run=no"]})
    assert r.status_code == 422 and "unsafe task id" in r.text


def test_a_plan_note_starting_with_a_dash_is_refused(job_client):
    r = job_client.post(f"{BASE}/jobs/plan", json={"confirm": True, "note": "--all"})
    assert r.status_code == 422


def test_a_project_with_no_repo_path_cannot_start_a_job(client):
    """`client` is the plain fixture, whose config has repo_path: null."""
    r = client.post(f"{BASE}/jobs/run", json={"confirm": True})
    assert r.status_code == 409 and "incomplete" in r.json()["detail"]


def test_an_unknown_project_is_404_before_the_body_is_considered(job_client):
    r = job_client.post("/api/projects/nope/jobs/run", json={"confirm": True})
    assert r.status_code == 404


# ---- HTTP: the spawn path, with the real command lines swapped out --------
@pytest.fixture
def harmless_argv(monkeypatch):
    """Replace the four argv builders with a child that just exits.

    The endpoints are still exercised end to end — routing, gating,
    `require_repo_path`, the supervisor, the 202 body — but no orchestrator
    process is ever created.
    """
    for name in ("run_argv", "plan_argv", "resume_argv", "import_backlog_argv",
                 "reconcile_argv"):
        monkeypatch.setattr(jobs, name, lambda *a, **k: list(ECHO))


def test_starting_a_run_returns_202_and_the_job_becomes_readable(job_client, sup,
                                                                 job_cfg,
                                                                 harmless_argv):
    r = job_client.post(f"{BASE}/jobs/run", json={"confirm": True})
    assert r.status_code == 202
    body = r.json()
    assert body["command"] == "run" and body["project"] == PROJECT
    assert body["run_id"] is None      # the child has not minted one yet

    job_id = body["job_id"]
    listed = job_client.get(f"{BASE}/jobs").json()["jobs"]
    assert [j["job_id"] for j in listed] == [job_id]

    _wait_for(sup, sup.get(job_cfg, job_id))
    detail = job_client.get(f"{BASE}/jobs/{job_id}").json()
    assert detail["status"] == "exited" and detail["exit_code"] == 0
    assert detail["log_path"].endswith(f"{job_id}.log")

    log = job_client.get(f"{BASE}/jobs/{job_id}/log").json()
    assert "hello from a fake job" in log["text"] and log["eof"]
    assert log["offset"] == 0 and log["next_offset"] == len(log["text"])


def test_a_dry_run_needs_no_confirmation(job_client, harmless_argv):
    assert job_client.post(f"{BASE}/jobs/run",
                           json={"dry_run": True}).status_code == 202


def test_a_second_job_while_one_is_in_flight_is_409(job_client, sup, job_cfg,
                                                    monkeypatch):
    monkeypatch.setattr(jobs, "run_argv", lambda *a, **k: list(SLEEPER))
    monkeypatch.setattr(jobs, "plan_argv", lambda *a, **k: list(ECHO))
    first = job_client.post(f"{BASE}/jobs/run", json={"confirm": True}).json()

    clash = job_client.post(f"{BASE}/jobs/plan", json={"confirm": True})
    assert clash.status_code == 409
    assert first["job_id"] in clash.json()["detail"]

    stopped = job_client.post(f"{BASE}/jobs/{first['job_id']}/stop")
    assert stopped.status_code == 200 and stopped.json()["status"] == "stopped"
    # The 409 was a live-job rule, not a permanent one.
    assert job_client.post(f"{BASE}/jobs/plan",
                           json={"confirm": True}).status_code == 202


def test_import_backlog_takes_no_body_at_all(job_client, harmless_argv):
    r = job_client.post(f"{BASE}/jobs/import-backlog")
    assert r.status_code == 202 and r.json()["command"] == "import-backlog"


def test_every_spawnable_command_survives_being_serialized(job_client, sup, job_cfg,
                                                           harmless_argv):
    """Spawn each command and read it back, because that is the half that broke.

    `reconcile` was added to the router and to `reconcile_argv` but not to the
    `Job.command` Literal, so it spawned fine and then failed to serialize: a 500
    on its own 202, and — because nothing is evicted on exit, and the sidecar
    outlives a restart — a 500 on **every** `GET /jobs` for that project
    afterwards, which the browser could only report as a CORS failure. Asserting
    per-command 202s would not have caught it; the read-back is the test.
    """
    for command in jobs.COMMANDS:
        if command == "resume":
            continue          # needs a paused run; covered by its own test
        body = {} if command in ("import-backlog", "reconcile") else {"confirm": True}
        r = job_client.post(f"{BASE}/jobs/{command}", json=body)
        assert r.status_code == 202, (command, r.status_code, r.text)
        assert r.json()["command"] == command

        listed = job_client.get(f"{BASE}/jobs")
        assert listed.status_code == 200, (command, listed.text)
        assert listed.json()["jobs"][0]["command"] == command
        # One at a time, so let it go before spawning the next.
        _wait_for(sup, sup.get(job_cfg, r.json()["job_id"]))


def test_reconcile_never_claims_a_run_it_only_closed(sup, job_cfg, job_state):
    """`reconcile` mints no run row, so resolving one would link a stranger's.

    The match is "newest run started around when this job did" — for a job that
    only rewrites existing rows, any hit belongs to something else the operator
    is running, and the console would show it as this job's output.
    """
    record = _wait_for(sup, sup.spawn(job_cfg, "reconcile", ECHO))
    store = job_state / f"{PROJECT}.sqlite3"
    conn = sqlite3.connect(store)
    with conn:
        conn.execute("INSERT INTO runs (id, started_at, status) VALUES (?,?,?)",
                     ("run-not-mine", record.started_at + 1, "running"))
    conn.close()

    assert sup.resolve_run_id(record, store).run_id is None


def test_resume_spawns_when_a_paused_run_exists(job_client, harmless_argv):
    r = job_client.post(f"{BASE}/jobs/resume", json={"confirm": True})
    assert r.status_code == 202 and r.json()["command"] == "resume"


def test_resume_is_404_when_there_is_nothing_paused(job_client, job_state,
                                                    harmless_argv):
    conn = sqlite3.connect(job_state / f"{PROJECT}.sqlite3")
    with conn:
        conn.execute("UPDATE runs SET status='done' WHERE id=?", (RUN_PAUSED,))
    conn.close()

    r = job_client.post(f"{BASE}/jobs/resume", json={"confirm": True})
    assert r.status_code == 404 and "no paused run" in r.json()["detail"]


def test_stopping_an_unknown_job_is_404(job_client):
    assert job_client.post(f"{BASE}/jobs/nope/stop").status_code == 404


def test_reading_past_the_end_of_a_log_reports_the_clamped_offset(job_client, sup,
                                                                  job_cfg,
                                                                  harmless_argv):
    """`offset` must be where the chunk actually starts, so a console can trust
    `offset <= next_offset`. Echoing the request would report a rewind."""
    job_id = job_client.post(f"{BASE}/jobs/run",
                             json={"confirm": True}).json()["job_id"]
    _wait_for(sup, sup.get(job_cfg, job_id))

    end = job_client.get(f"{BASE}/jobs/{job_id}/log").json()["next_offset"]
    past = job_client.get(f"{BASE}/jobs/{job_id}/log?offset={end + 5000}").json()
    assert past["text"] == "" and past["eof"] is True
    assert past["offset"] == past["next_offset"] == end


def test_the_job_list_carries_run_id_not_just_the_detail_route(job_client, sup,
                                                              job_cfg, job_state,
                                                              harmless_argv):
    """The console renders from the list; a run_id that only appeared in the
    drawer could never be linked."""
    job_id = job_client.post(f"{BASE}/jobs/run",
                             json={"confirm": True}).json()["job_id"]
    record = _wait_for(sup, sup.get(job_cfg, job_id))

    conn = sqlite3.connect(job_state / f"{PROJECT}.sqlite3")
    with conn:
        conn.execute("INSERT INTO runs (id, started_at, status) VALUES (?,?,?)",
                     ("20260801-000000-dddddd", record.started_at + 1, "done"))
    conn.close()

    listed = job_client.get(f"{BASE}/jobs").json()["jobs"]
    assert [j["run_id"] for j in listed] == ["20260801-000000-dddddd"]


def test_a_finished_job_with_no_run_is_only_looked_up_once(sup, job_cfg, job_state):
    """Guards the `run_id_settled` bound: the list is polled, so a job that never
    made a run must not re-query the store forever."""
    store = job_state / f"{PROJECT}.sqlite3"
    record = _wait_for(sup, sup.spawn(job_cfg, "run", ECHO))
    record.started_at = time.time() + 3600      # no run can match

    assert sup.resolve_run_id(record, store).run_id is None
    assert record.run_id_settled

    # A run appearing later is not retro-attached — the record is settled.
    conn = sqlite3.connect(store)
    with conn:
        conn.execute("INSERT INTO runs (id, started_at, status) VALUES (?,?,?)",
                     ("20260801-000000-eeeeee", record.started_at + 1, "done"))
    conn.close()
    assert sup.resolve_run_id(record, store).run_id is None
