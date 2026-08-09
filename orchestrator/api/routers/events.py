"""The event log, paged by sqlite `rowid`.

`rowid` is the cursor because `events` is append-only and has no id column: it is
monotonic, it survives an API restart (an in-process counter would not), and it
lets the UI's stream consumer append rather than refetch.
"""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, Query

from .. import reads
from ..deps import store_conn
from ..errors import READ_ERRORS
from ..schemas import Event, EventOrder, Events

router = APIRouter(prefix="/api/projects/{project}", tags=["events"],
                   responses=READ_ERRORS)


@router.get("/events", response_model=Events)
def list_events(since_rowid: int | None = Query(None, ge=0),
                kind: str | None = None, task_id: str | None = None,
                run_id: str | None = None,
                limit: int = Query(200, ge=1, le=2000),
                order: EventOrder = Query(
                    "asc", description="asc pages forward from since_rowid; desc "
                                       "returns the newest matching rows"),
                conn: sqlite3.Connection = Depends(store_conn)) -> Events:
    rows = reads.events(conn, since_rowid=since_rowid, kind=kind, task_id=task_id,
                        run_id=run_id, limit=limit, order=order)
    return Events(
        events=[Event(**r) for r in rows],
        order=order,
        # An empty page must return the cursor it was given, not 0 — otherwise a
        # poller that catches up rewinds to the start of the log. The highest
        # rowid in the page is the right cursor in BOTH orders: under `desc` the
        # page's first row is the newest one, and paging forward from anything
        # lower would re-deliver rows the client already has.
        next_since_rowid=max((r["rowid"] for r in rows), default=(since_rowid or 0)),
        max_rowid=reads.max_event_rowid(conn),
    )
