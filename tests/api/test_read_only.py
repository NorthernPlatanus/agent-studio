"""The load-bearing safety property: a GET sweep must not touch the state files.

`Store.__init__` runs `executescript(SCHEMA)` plus migrations, so a single
accidental `Store(...)` inside a read path would rewrite the user's live state on
a dashboard poll. Nothing but this test would notice.
"""

from __future__ import annotations

from tests.api.fixtures.seed_store import PROJECT, RUN_PAUSED

BASE = f"/api/projects/{PROJECT}"

SWEEP = (
    "/healthz",
    "/openapi.json",
    "/api/projects",
    f"{BASE}/summary",
    f"{BASE}/waves",
    f"{BASE}/tasks",
    f"{BASE}/tasks/T-120",
    f"{BASE}/tasks/T-120/candidates",
    f"{BASE}/tasks/T-102/candidates",
    f"{BASE}/runs",
    f"{BASE}/runs/{RUN_PAUSED}",
    f"{BASE}/usage?group_by=role",
    f"{BASE}/usage?group_by=day",
    f"{BASE}/metrics",
    f"{BASE}/events",
    f"{BASE}/jobs",
)

# GET routes the sweep cannot call, each for a reason that is not "we forgot".
# `test_every_get_route_is_swept_or_listed_here` fails when a new route joins the
# app without a decision being made about it — the sweep is the only thing
# standing between a read path and the user's live state, and a silent gap in it
# would be invisible.
NOT_SWEPT = {
    # Never terminates by design. Its reads go through the same
    # `deps.open_read_only`, asserted directly in
    # `test_the_live_stream_reads_are_read_only_too`.
    "/api/projects/{project}/stream",
    # Needs a job that exists; `tests/api/test_jobs.py` owns these, and they
    # touch the jobs dir rather than the state files.
    "/api/projects/{project}/jobs/{job_id}",
    "/api/projects/{project}/jobs/{job_id}/log",
}


def _snapshot(directory):
    """Size + mtime of the database files, and size only for a non-empty `-wal`.

    Attaching a READER to a WAL database is not free at the filesystem level:
    SQLite needs the shared-memory index, so a `mode=ro` connection *creates*
    zero-byte `-shm` and `-wal` files if they are absent and the directory is
    writable. That is bookkeeping about who is reading, not database content, so
    `-shm` is excluded outright and a zero-byte `-wal` with it — otherwise this
    test passes only when some earlier test happened to open the same file first,
    which is exactly how it behaved before (green in a full run, red on
    `pytest tests/api/test_read_only.py`).

    The property survives the exclusion: a real write either changes the main
    file or grows the WAL past zero, and both are still compared.
    """
    out = {}
    for p in sorted(directory.iterdir()):
        stat = p.stat()
        if p.name.endswith("-shm") or (p.name.endswith("-wal") and stat.st_size == 0):
            continue
        out[p.name] = (stat.st_size,
                       None if p.name.endswith("-wal") else stat.st_mtime_ns)
    return out


def test_a_full_get_sweep_leaves_the_state_dir_byte_identical(client, state_dir):
    before = _snapshot(state_dir)
    for path in SWEEP:
        assert client.get(path).status_code == 200, path
    assert _snapshot(state_dir) == before


def test_the_snapshot_still_notices_a_real_write(tmp_path):
    """Guards the exclusions above: `_snapshot` ignores reader bookkeeping, so it
    has to be shown that it does not ignore a write."""
    import shutil
    import sqlite3

    from tests.api.fixtures import seed_store

    scratch = tmp_path / "state"
    scratch.mkdir()
    store = scratch / f"{PROJECT}.sqlite3"
    seed_store.seed(store)

    # A reader alone: excused.
    from orchestrator.api import deps
    quiet = _snapshot(scratch)
    conn = deps.open_read_only(store)
    conn.execute("SELECT COUNT(*) FROM tasks").fetchone()
    conn.close()
    assert _snapshot(scratch) == quiet

    # One row: caught.
    writer = sqlite3.connect(store)
    with writer:
        writer.execute("INSERT INTO events (ts, kind) VALUES (1.0, 'gate')")
    writer.close()
    assert _snapshot(scratch) != quiet
    shutil.rmtree(scratch)


def test_every_get_route_is_swept_or_listed_here(client):
    """A new read endpoint must not be able to skip the sweep by accident."""
    templates = {r.path for r in client.app.routes
                 if "GET" in getattr(r, "methods", set())}
    swept = {p.split("?")[0] for p in SWEEP}
    for template in templates:
        if template in NOT_SWEPT or not template.startswith("/api/projects"):
            continue
        # The sweep uses concrete ids; compare on the shape by substituting them.
        concrete = (template.replace("{project}", PROJECT)
                    .replace("{task_id}", "T-120").replace("{run_id}", RUN_PAUSED))
        assert concrete in swept, f"{template} is neither swept nor in NOT_SWEPT"


def test_the_live_stream_reads_are_read_only_too(state_dir, cfg):
    """The stream opens its own connections outside any request, so it does not
    inherit `store_conn`'s guarantees — it has to be checked on its own."""
    from orchestrator.api import jobs, live

    store = state_dir / f"{PROJECT}.sqlite3"
    before = _snapshot(state_dir)
    cursors, rowid = live.snapshot(store, cfg, jobs.JobSupervisor())
    assert rowid > 0 and cursors.tasks != "-"
    assert live.new_events(store, 0)
    assert _snapshot(state_dir) == before


def test_the_connection_really_is_read_only(conn):
    import sqlite3

    import pytest
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("UPDATE tasks SET status='done' WHERE id='T-120'")


def test_reads_never_create_a_state_dir_for_a_project_that_never_ran(tmp_path, cfg):
    """`Config.state_dir()` mkdirs, so read paths resolve the path themselves."""
    import copy

    from orchestrator.api import deps
    from orchestrator.core.config import Config

    data = copy.deepcopy(cfg.as_dict())
    data["paths"]["state_dir"] = str(tmp_path / "never")
    fresh = Config(data, "example", deps.REPO_ROOT)
    assert deps.store_path(fresh) == tmp_path / "never" / "example.sqlite3"
    assert not (tmp_path / "never").exists()
