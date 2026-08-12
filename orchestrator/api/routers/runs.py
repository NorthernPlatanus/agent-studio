"""Runs: the list with per-run token totals, and one run's detail.

Token totals are reported per run and per channel even though the budget only
counts cash: on a subscription plan the input-token total is the resource that
actually runs out, so `cash spend: $0.02` alone describes ~1 % of a run.
"""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query

from ...ops import liveness
from .. import reads
from ..deps import store_conn
from ..errors import READ_ERRORS, ROW_ERRORS
from ..schemas import Event, RunDetail, RunListItem, Runs, TokenChannels

router = APIRouter(prefix="/api/projects/{project}", tags=["runs"],
                   responses=READ_ERRORS)


def run_item(conn: sqlite3.Connection, run: dict) -> RunListItem:
    # A `running` row is a claim, not an observation: the runner writes `done` and
    # `paused` on the paths that unwind, and nothing at all when the process is
    # killed. So every run carries the evidence for the claim — when it last
    # touched the store — and a flag for when the claim has expired
    # (`ops/liveness.py`). The status column is left as recorded; only `reconcile`
    # rewrites it, because only the CLI may write.
    last = reads.run_last_activity(conn, run["id"]) or run["started_at"]
    return RunListItem(
        id=run["id"], started_at=run["started_at"], status=run["status"],
        note=run.get("note"), cost_usd=run.get("cost_usd") or 0.0,
        tokens=TokenChannels(**reads.run_token_totals(conn, run["id"])),
        last_activity_at=last,
        stale=liveness.is_stale(run["status"], last),
    )


@router.get("/runs", response_model=Runs)
def list_runs(limit: int = Query(50, ge=1, le=500),
              conn: sqlite3.Connection = Depends(store_conn)) -> Runs:
    return Runs(runs=[run_item(conn, r) for r in reads.runs(conn, limit)])


@router.get("/runs/{run_id}", response_model=RunDetail, responses=ROW_ERRORS)
def get_run(run_id: str,
            event_limit: int = Query(500, ge=1, le=5000),
            conn: sqlite3.Connection = Depends(store_conn)) -> RunDetail:
    run = reads.get_run(conn, run_id)
    if run is None:
        raise HTTPException(404, f"unknown run: {run_id!r}")
    item = run_item(conn, run)
    return RunDetail(
        **item.model_dump(),
        task_ids=reads.run_task_ids(conn, run_id),
        events=[Event(**e) for e in reads.events(conn, run_id=run_id,
                                                 limit=event_limit)],
    )
