"""Tasks: filtered list, full detail, and the per-candidate board.

Filtering happens in Python rather than SQL because everything worth filtering on
(`domain`, `risk`, `visual`, `agent_able`) lives inside the `spec_json` blob, not
in a column. The tasks table is a few hundred rows at most, so a scan is cheaper
than teaching the store a new schema.
"""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query

from ...core.config import Config
from ...engine.scheduler import queue_stats
from .. import reads
from ..deps import checkpoint_path, resolve_project, store_conn
from ..errors import READ_ERRORS, ROW_ERRORS
from ..schemas import (Candidate, Candidates, Event, TaskDetail, TaskListItem,
                       Tasks)

router = APIRouter(prefix="/api/projects/{project}", tags=["tasks"],
                   responses=READ_ERRORS)

# Enough of a failing gate to diagnose it in a drawer; the full log can be
# megabytes and the panel polls this endpoint.
GATE_LOG_TAIL = 4000


def _list_item(task: dict) -> TaskListItem:
    return TaskListItem(
        id=task["id"], title=task.get("title") or task["id"],
        status=task["status"], milestone=task.get("milestone"),
        retries=task.get("retries") or 0, cost_usd=task.get("cost_usd") or 0.0,
        updated_at=task.get("updated_at"), domain=task.get("domain"),
        risk=task.get("risk"), complexity=task.get("complexity"),
        visual=task.get("visual"), agent_able=task.get("agent_able"),
        n_candidates=task.get("n_candidates"), parent_id=task.get("parent_id"),
        deps=list(task.get("deps") or []),
    )


@router.get("/tasks", response_model=Tasks)
def list_tasks(status: str | None = None, milestone: str | None = None,
               domain: str | None = None, parent_id: str | None = None,
               q: str | None = Query(None, description="substring of id or title"),
               conn: sqlite3.Connection = Depends(store_conn)) -> Tasks:
    tasks = reads.all_tasks(conn)
    needle = (q or "").lower()
    kept = [
        t for t in tasks
        if (status is None or t["status"] == status)
        and (milestone is None or (t.get("milestone") or "") == milestone)
        and (domain is None or (t.get("domain") or "") == domain)
        and (parent_id is None or (t.get("parent_id") or "") == parent_id)
        and (not needle or needle in t["id"].lower()
             or needle in (t.get("title") or "").lower())
    ]
    return Tasks(tasks=[_list_item(t) for t in kept], total=len(kept),
                 queue_stats=queue_stats(kept))


@router.get("/tasks/{task_id}", response_model=TaskDetail, responses=ROW_ERRORS)
def get_task(task_id: str,
             conn: sqlite3.Connection = Depends(store_conn)) -> TaskDetail:
    task = reads.get_task(conn, task_id)
    if task is None:
        raise HTTPException(404, f"unknown task: {task_id!r}")
    return TaskDetail(
        **_list_item(task).model_dump(),
        files_read=list(task.get("files_read") or []),
        files_write=list(task.get("files_write") or []),
        acceptance=list(task.get("acceptance") or []),
        cash_spend_usd=reads.task_cash_spend(conn, task_id),
        children=reads.child_task_ids(conn, task_id),
        spec=task,
        events=[Event(**e) for e in reads.task_events(conn, task_id)],
    )


def _candidate(raw: dict) -> Candidate:
    """Project a `TaskState` candidate (or an event-derived stand-in) for the UI.

    `diff` and `messages` are deliberately dropped: the diff can be hundreds of
    KB and the warm chat history is the single largest thing in a checkpoint.
    Neither belongs in a polled response — `has_diff` answers the only question
    the board asks.
    """
    log = raw.get("gate_log")
    return Candidate(
        cand_id=raw.get("cand_id") or "?", attempt=int(raw.get("attempt") or 0),
        status=raw.get("status"), model=raw.get("model"), branch=raw.get("branch"),
        worktree=raw.get("worktree"), no_patch=bool(raw.get("no_patch")),
        error=raw.get("error"),
        gate_log=log[-GATE_LOG_TAIL:] if isinstance(log, str) and log else None,
        notes=raw.get("notes"), has_diff=bool(raw.get("diff")),
        visual_facts=raw.get("visual_facts") if isinstance(
            raw.get("visual_facts"), dict) else None,
    )


@router.get("/tasks/{task_id}/candidates", response_model=Candidates,
            responses=ROW_ERRORS)
def get_candidates(task_id: str, run_id: str | None = None,
                   cfg: Config = Depends(resolve_project),
                   conn: sqlite3.Connection = Depends(store_conn)) -> Candidates:
    """Per-candidate detail: the checkpoint when there is one, else the event log.

    The checkpoint is authoritative and live (it holds the attempt currently
    running); the events fallback is history for a task whose checkpoint was
    pruned or which ran before the panel existed. `source` tells the UI which it
    got, because only the checkpoint can show an in-flight attempt.
    """
    if reads.get_task(conn, task_id) is None:
        raise HTTPException(404, f"unknown task: {task_id!r}")
    if run_id is None:
        recent = reads.events(conn, task_id=task_id, limit=2000)
        run_id = next((e["run_id"] for e in reversed(recent) if e["run_id"]), None)
        if run_id is None:
            latest = reads.latest_run(conn, ("running", "paused", "done", "aborted"))
            run_id = latest["id"] if latest else None

    if run_id:
        from_ckpt = reads.checkpoint_candidates(checkpoint_path(cfg), run_id, task_id)
        if from_ckpt:
            return Candidates(task_id=task_id, run_id=run_id, source="checkpoint",
                              candidates=[_candidate(c) for c in from_ckpt])
    from_events = reads.candidates_from_events(conn, task_id)
    return Candidates(task_id=task_id, run_id=run_id,
                      source="events" if from_events else "none",
                      candidates=[_candidate(c) for c in from_events])
