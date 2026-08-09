"""The live stream: cursor polling turned into SSE.

PLAN §3.1 rule 5 — **the stream is a cache-invalidation signal, not a second
source of truth.** When a cursor moves, the client is told *which entity* changed
and refetches it through the normal read endpoints. There is exactly one
exception, and it is principled: `events` is append-only, so its rows are pushed
in full and appended to the client's cache. Nothing else can be pushed safely,
because anything mutable would need reconciling against a refetch anyway.

Why polling rather than a bus: the writers are **other processes** (the CLI
subprocesses `jobs.py` supervises). An in-process pub/sub would only ever see
writes the API itself made, and the API makes none. `MAX(rowid)` on an
append-only table and `MAX(updated_at)` on a mutable one are what SQLite can
answer cheaply, and they survive an API restart, a client reconnect, and a
missing store file — none of which a bus would.

A fresh read-only connection per tick, not one held open for the life of the
stream: the store may not exist when a client subscribes (a project that has
never run) and appear a second later when the first job creates it. Reopening
costs microseconds at 1 Hz and removes that whole class of staleness.
"""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from pathlib import Path

from ..core.config import Config
from . import reads
from .deps import open_read_only
from .jobs import JobSupervisor

POLL_INTERVAL_S = 1.0
# Long enough not to be chatty, short enough that a proxy or a sleeping laptop
# is noticed before the user wonders why the dashboard is frozen.
HEARTBEAT_S = 15.0
# A burst bigger than this means the client is far behind; it gets told to
# refetch instead, which is one query rather than a flood of appends.
MAX_PUSHED_EVENTS = 200


@dataclass(frozen=True)
class Cursors:
    """One comparable value per entity the UI caches.

    Strings throughout, including for the append-only tables: the loop only ever
    tests equality, and a uniform type keeps the "did anything change" diff a
    plain dict comparison.
    """

    tasks: str = "-"
    runs: str = "-"
    usage: str = "-"
    events: str = "-"
    jobs: str = "-"


def _digest(row: sqlite3.Row | tuple | None) -> str:
    """A short stable string for a row of aggregate values.

    Hashed because one of these (`runs`) folds in a `group_concat` of every run's
    status — unbounded in length, but only ever compared for equality.
    """
    if row is None:
        return "-"
    raw = "|".join("" if v is None else str(v) for v in row)
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _one(conn: sqlite3.Connection, sql: str) -> str:
    try:
        return _digest(conn.execute(sql).fetchone())
    except sqlite3.Error:
        # A table missing on an old state file must degrade to "no news", never
        # kill the stream.
        return "-"


def table_cursors(conn: sqlite3.Connection) -> dict[str, str]:
    return {
        # `tasks` is mutable, so rowcount alone would miss a status change;
        # `updated_at` is stamped on every upsert.
        "tasks": _one(conn, "SELECT COUNT(*), COALESCE(MAX(updated_at),0) FROM tasks"),
        # `runs` has no updated_at and its status changes in place — hence the
        # status roll-up. cost_usd is in there because a run's cost ticks up
        # without any other column moving.
        "runs": _one(conn, "SELECT COUNT(*), COALESCE(MAX(started_at),0), "
                           "COALESCE(SUM(cost_usd),0), COALESCE(group_concat(status),'')"
                           " FROM runs"),
        # Append-only: MAX(rowid) is sufficient and cheap.
        "usage": _one(conn, "SELECT COALESCE(MAX(rowid),0) FROM usage"),
        "events": _one(conn, "SELECT COALESCE(MAX(rowid),0) FROM events"),
    }


def snapshot(store: Path | None, cfg: Config,
             supervisor: JobSupervisor) -> tuple[Cursors, int]:
    """Current cursors, plus the newest event rowid (0 when there is no store).

    The rowid is returned alongside rather than derived from the cursor because
    the cursor is a hash — deliberately opaque — while the event pusher needs the
    real number to page from.
    """
    jobs_cursor = supervisor.cursor(cfg)
    if store is None or not store.exists():
        return Cursors(jobs=jobs_cursor), 0
    try:
        conn = open_read_only(store)
    except sqlite3.Error:
        return Cursors(jobs=jobs_cursor), 0
    try:
        return Cursors(jobs=jobs_cursor, **table_cursors(conn)), \
            reads.max_event_rowid(conn)
    finally:
        conn.close()


def new_events(store: Path | None, since_rowid: int) -> list[dict]:
    if store is None or not store.exists():
        return []
    try:
        conn = open_read_only(store)
    except sqlite3.Error:
        return []
    try:
        return reads.events(conn, since_rowid=since_rowid, limit=MAX_PUSHED_EVENTS)
    finally:
        conn.close()


async def stream(cfg: Config, store: Path | None, supervisor: JobSupervisor, *,
                 poll_interval_s: float = POLL_INTERVAL_S,
                 heartbeat_s: float = HEARTBEAT_S,
                 max_ticks: int | None = None) -> AsyncIterator[dict]:
    """Yield `sse_starlette`-shaped dicts: `{"event": name, "data": {...}}`.

    The first tick emits a `hello` carrying every cursor and the event rowid the
    client should consider itself caught up to. That is what makes a reconnect
    cheap: the client refetches once, adopts the cursor, and receives only
    deltas afterwards — rather than either replaying the log or silently missing
    the rows written while it was disconnected.

    `max_ticks` exists for the tests. Nothing else should pass it: a real stream
    ends when the client disconnects, which surfaces here as cancellation.
    """
    cursors, event_rowid = snapshot(store, cfg, supervisor)
    yield {"event": "hello", "data": {**asdict(cursors),
                                      "event_rowid": event_rowid,
                                      "poll_interval_s": poll_interval_s}}

    since_heartbeat = 0.0
    ticks = 0
    while max_ticks is None or ticks < max_ticks:
        await asyncio.sleep(poll_interval_s)
        ticks += 1
        fresh, newest_rowid = snapshot(store, cfg, supervisor)

        changed = {k: v for k, v in asdict(fresh).items()
                   if v != getattr(cursors, k)}
        for name in changed:
            if name == "events":
                # The one pushed payload. Rows, not a signal, because the log is
                # append-only — see the module docstring.
                rows = new_events(store, event_rowid)
                yield {"event": "events",
                       "data": {"events": rows, "next_since_rowid": newest_rowid,
                                "truncated": len(rows) >= MAX_PUSHED_EVENTS}}
            else:
                yield {"event": name, "data": {"cursor": changed[name]}}

        cursors, event_rowid = fresh, newest_rowid
        since_heartbeat = 0.0 if changed else since_heartbeat + poll_interval_s
        if since_heartbeat >= heartbeat_s:
            # Only when nothing else was sent: any other event proves liveness
            # just as well, and an unconditional heartbeat is pure noise.
            yield {"event": "heartbeat", "data": {"event_rowid": event_rowid}}
            since_heartbeat = 0.0
