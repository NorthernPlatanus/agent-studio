"""The SSE cursor loop.

Driven directly against `live.stream` with a tiny poll interval and a `max_ticks`
bound, rather than through an HTTP client: the property under test is "which
frame is emitted when the database changes underneath a running stream", and a
`TestClient` SSE read would add a timing surface without adding coverage. The
router wrapper is covered separately, once, for the things only it can get wrong.
"""

from __future__ import annotations

import copy
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import pytest

from orchestrator.api import deps, jobs, live
from orchestrator.core.config import Config
from tests.api.fixtures.seed_store import PROJECT

ECHO = [sys.executable, "-c", "print('tick')"]

# Fast enough that a test is not a pause, slow enough that a tick is a tick.
FAST = {"poll_interval_s": 0.01, "heartbeat_s": 60.0}


@pytest.fixture
def live_store(tmp_path: Path, store_path: Path) -> Path:
    """A writable copy of the seeded store — these tests mutate it mid-stream."""
    state = tmp_path / "state"
    state.mkdir()
    target = state / f"{PROJECT}.sqlite3"
    shutil.copy(store_path, target)
    return target


@pytest.fixture
def live_cfg(live_store: Path, cfg: Config) -> Config:
    data = copy.deepcopy(cfg.as_dict())
    data["paths"]["state_dir"] = str(live_store.parent)
    return Config(data, PROJECT, deps.REPO_ROOT)


@pytest.fixture
def sup() -> jobs.JobSupervisor:
    return jobs.JobSupervisor()


def write(store: Path, sql: str, params: tuple = ()) -> None:
    conn = sqlite3.connect(store)
    with conn:
        conn.execute(sql, params)
    conn.close()


async def collect(cfg, store, sup, mutate=None, *, ticks: int = 2,
                  **kwargs) -> tuple[dict, list[dict]]:
    """(hello frame, frames produced by `ticks` ticks), applying `mutate` first."""
    gen = live.stream(cfg, store, sup, max_ticks=ticks, **{**FAST, **kwargs})
    hello = await anext(gen)
    if mutate is not None:
        mutate()
    return hello, [frame async for frame in gen]


def names(frames: list[dict]) -> list[str]:
    return [f["event"] for f in frames]


# ---- cursors -------------------------------------------------------------
def test_each_cursor_moves_only_for_its_own_table(live_store):
    conn = deps.open_read_only(live_store)
    before = live.table_cursors(conn)
    conn.close()

    def after() -> dict:
        c = deps.open_read_only(live_store)
        try:
            return live.table_cursors(c)
        finally:
            c.close()

    write(live_store, "UPDATE tasks SET status='done', updated_at=? WHERE id='T-102'",
          (time.time(),))
    moved = after()
    assert moved["tasks"] != before["tasks"]
    assert {k: moved[k] for k in ("runs", "usage", "events")} == \
           {k: before[k] for k in ("runs", "usage", "events")}

    write(live_store, "INSERT INTO events (ts, kind, detail) VALUES (?,?,?)",
          (time.time(), "gate", "{}"))
    assert after()["events"] != moved["events"]


def test_a_run_changing_status_in_place_moves_the_runs_cursor(live_store):
    """`runs` has no updated_at and its status mutates — the reason that cursor
    folds in a status roll-up rather than just MAX(started_at)."""
    def cursor() -> str:
        conn = deps.open_read_only(live_store)
        try:
            return live.table_cursors(conn)["runs"]
        finally:
            conn.close()

    before = cursor()
    write(live_store, "UPDATE runs SET status='done' WHERE status='paused'")
    assert cursor() != before
    # …and so does a cost tick, which moves no other column either.
    mid = cursor()
    write(live_store, "UPDATE runs SET cost_usd = cost_usd + 1.0")
    assert cursor() != mid


def test_a_missing_store_yields_cursors_without_dying(tmp_path, live_cfg, sup):
    cursors, rowid = live.snapshot(tmp_path / "nope.sqlite3", live_cfg, sup)
    assert rowid == 0
    assert (cursors.tasks, cursors.events) == ("-", "-")
    # Jobs are process state, not store state, so that one is still real.
    assert cursors.jobs == "-"
    assert live.snapshot(None, live_cfg, sup)[0].jobs == "-"


def test_cursors_are_opaque_but_stable(live_store):
    conn = deps.open_read_only(live_store)
    try:
        assert live.table_cursors(conn) == live.table_cursors(conn)
    finally:
        conn.close()


# ---- the stream ----------------------------------------------------------
async def test_hello_carries_every_cursor_and_the_catch_up_rowid(live_cfg, live_store,
                                                                 sup):
    hello, _ = await collect(live_cfg, live_store, sup, ticks=0)
    assert hello["event"] == "hello"
    data = hello["data"]
    assert set(data) == {"tasks", "runs", "usage", "events", "jobs",
                         "event_rowid", "poll_interval_s"}
    assert data["event_rowid"] > 0


async def test_an_appended_event_is_pushed_as_rows_not_a_signal(live_cfg, live_store,
                                                                sup):
    """The single exception to "the stream is only an invalidation signal": the
    event log is append-only, so its rows can be appended client-side safely."""
    hello, frames = await collect(
        live_cfg, live_store, sup,
        lambda: write(live_store,
                      "INSERT INTO events (ts, run_id, task_id, kind, detail) "
                      "VALUES (?,?,?,?,?)",
                      (time.time(), None, "T-999", "merged", "pushed")))

    assert "events" in names(frames)
    payload = next(f for f in frames if f["event"] == "events")["data"]
    assert [e["task_id"] for e in payload["events"]] == ["T-999"]
    assert payload["next_since_rowid"] == hello["data"]["event_rowid"] + 1
    assert payload["truncated"] is False


async def test_only_rows_the_client_has_not_seen_are_pushed(live_cfg, live_store, sup):
    def add(detail: str):
        write(live_store, "INSERT INTO events (ts, kind, detail) VALUES (?,?,?)",
              (time.time(), "gate", detail))

    gen = live.stream(live_cfg, live_store, sup, max_ticks=4, **FAST)
    await anext(gen)
    add("first")
    first = await anext(gen)
    add("second")
    second = await anext(gen)

    assert [e["detail"] for e in first["data"]["events"]] == ["first"]
    # The cursor advanced, so "first" is not re-delivered.
    assert [e["detail"] for e in second["data"]["events"]] == ["second"]
    await gen.aclose()


async def test_a_mutable_table_only_signals_and_never_ships_rows(live_cfg, live_store,
                                                                 sup):
    _, frames = await collect(
        live_cfg, live_store, sup,
        lambda: write(live_store,
                      "UPDATE tasks SET status='failed', updated_at=? WHERE id='T-102'",
                      (time.time(),)))
    tasks_frame = next(f for f in frames if f["event"] == "tasks")
    assert set(tasks_frame["data"]) == {"cursor"}


async def test_a_started_job_shows_up_on_the_stream(live_cfg, live_store, sup,
                                                    tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    data = copy.deepcopy(live_cfg.as_dict())
    data["project"]["repo_path"] = str(checkout)
    cfg = Config(data, PROJECT, deps.REPO_ROOT)

    _, frames = await collect(cfg, live_store, sup,
                              lambda: sup.spawn(cfg, "run", ECHO), ticks=6)
    assert "jobs" in names(frames)


async def test_nothing_happening_produces_a_heartbeat_and_nothing_else(live_cfg,
                                                                       live_store,
                                                                       sup):
    _, frames = await collect(live_cfg, live_store, sup, ticks=4,
                              poll_interval_s=0.01, heartbeat_s=0.02)
    assert set(names(frames)) == {"heartbeat"}
    assert frames[0]["data"]["event_rowid"] > 0


async def test_a_busy_stream_does_not_also_heartbeat(live_cfg, live_store, sup):
    """A heartbeat exists to prove liveness; any other frame proves it already."""
    _, frames = await collect(
        live_cfg, live_store, sup,
        lambda: write(live_store, "INSERT INTO events (ts, kind) VALUES (?,?)",
                      (time.time(), "gate")),
        ticks=1, poll_interval_s=0.01, heartbeat_s=0.005)
    assert names(frames) == ["events"]


async def test_a_store_that_appears_mid_stream_is_picked_up(live_cfg, live_store, sup,
                                                            tmp_path):
    """A project that has never run still gets a stream; the first job creates
    its store, and the client must not have to reconnect to see it."""
    missing = tmp_path / "later.sqlite3"
    gen = live.stream(live_cfg, missing, sup, max_ticks=3, **FAST)
    hello = await anext(gen)
    assert hello["data"]["event_rowid"] == 0 and hello["data"]["tasks"] == "-"

    shutil.copy(live_store, missing)
    frames = [f async for f in gen]
    assert {"tasks", "runs", "usage", "events"} <= set(names(frames))


async def test_a_burst_bigger_than_the_cap_says_so(live_cfg, live_store, sup,
                                                   monkeypatch):
    monkeypatch.setattr(live, "MAX_PUSHED_EVENTS", 3)

    def flood():
        conn = sqlite3.connect(live_store)
        with conn:
            conn.executemany("INSERT INTO events (ts, kind) VALUES (?,?)",
                             [(time.time(), "gate")] * 10)
        conn.close()

    _, frames = await collect(live_cfg, live_store, sup, flood, ticks=1)
    payload = next(f for f in frames if f["event"] == "events")["data"]
    assert len(payload["events"]) == 3 and payload["truncated"] is True
    # The cursor still points at the real end of the log, so the client that
    # refetches from it lands in the right place rather than replaying three rows.
    assert payload["next_since_rowid"] > payload["events"][-1]["rowid"]


# ---- the router ----------------------------------------------------------
def test_the_stream_route_rejects_an_unknown_project(client):
    assert client.get("/api/projects/nope/stream").status_code == 404


def test_the_stream_route_serves_sse_and_is_not_gated_on_the_store(client,
                                                                   monkeypatch):
    """Wiring only — routing, the media type, and that frames reach the wire.

    `live.stream` is swapped for a finite generator because a real one never
    ends: `TestClient` drives the app through a portal that never reports the
    client as disconnected, so an endless response deadlocks the test rather
    than exercising anything. Termination is a property of the ASGI server, not
    of this route; the loop's own behaviour is covered above.

    No 409 when the store is missing, unlike the read endpoints: for a project
    that has never run, the interesting moment is precisely when its store
    appears.
    """
    async def two_frames(cfg, store, sup, **kwargs):
        yield {"event": "hello", "data": {"event_rowid": 0, "jobs": "-"}}
        yield {"event": "events",
               "data": {"events": [{"run_id": None, "kind": "gate"}],
                        "truncated": False}}

    monkeypatch.setattr(live, "stream", two_frames)

    with client.stream("GET", f"/api/projects/{PROJECT}/stream") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    assert "event: hello" in body and "event: events" in body
    assert [json.loads(line.split(":", 1)[1]) for line in body.splitlines()
            if line.startswith("data:")] == [
        {"event_rowid": 0, "jobs": "-"},
        {"events": [{"run_id": None, "kind": "gate"}], "truncated": False}]


def test_frame_payloads_are_json_not_a_python_repr(client, monkeypatch):
    """`sse_starlette` stringifies a non-str `data` with `str()`, which emits
    `{'k': None}` — single quotes, `None`, `False`. Every `JSON.parse` in the
    browser fails on it, and nothing else in the suite would notice, because a
    repr still looks like a populated frame in a smoke test."""
    async def one_frame(cfg, store, sup, **kwargs):
        yield {"event": "hello", "data": {"note": None, "truncated": False}}

    monkeypatch.setattr(live, "stream", one_frame)

    with client.stream("GET", f"/api/projects/{PROJECT}/stream") as response:
        body = "".join(response.iter_text())

    assert "data: " + json.dumps({"note": None, "truncated": False}) in body
    assert "None" not in body and "'" not in body
