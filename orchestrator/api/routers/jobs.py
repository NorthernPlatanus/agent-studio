"""Jobs — the read side, plus the four spawners and stop.

Every endpoint here is a thin translation between HTTP and `api/jobs.py`; the
supervisor owns all the process logic and this module owns none of it. What does
live here is the set of preconditions a spawn must clear, in the order the UI
needs to distinguish them:

1. unknown project → 404 (`resolve_project`)
2. no `project.repo_path` → 409 (`require_repo_path`) — every one of the four
   commands ends up calling `cfg.repo_path()`, `import-backlog` included, since
   `make_backlog` resolves the backlog file against the checkout
3. a job already in flight → 409 (`JobSupervisor.spawn`)
4. `resume` with no paused run → 404, matching the CLI's own "nothing to resume"

Spending is gated in the request models, not here: a body without `confirm` is a
422 before this module is reached. See PLAN §3.1 rule 6.
"""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Body, Depends, Query, status

from ...core.config import Config
from .. import jobs as supervisor_mod
from .. import reads
from ..deps import require_repo_path, resolve_project, store_conn, store_path
from ..errors import JOB_READ_ERRORS, JOB_SPAWN_ERRORS, PROJECT_ERRORS
from ..jobs import JobError, JobRecord, JobSupervisor, get_supervisor
from ..schemas import (ImportBacklogRequest, Job, JobAccepted, JobLog, Jobs,
                       PlanRequest, ResumeRequest, RunRequest)

# No 409 here, unlike the read routers: jobs are supervised processes, not store
# rows, so a project with no state/<p>.sqlite3 still has an (empty) job list.
router = APIRouter(prefix="/api/projects/{project}", tags=["jobs"],
                   responses=PROJECT_ERRORS)


def _job(record: JobRecord) -> Job:
    return Job(job_id=record.job_id, project=record.project, command=record.command,
               status=record.status, argv=record.argv, pid=record.pid,
               run_id=record.run_id, started_at=record.started_at,
               ended_at=record.ended_at, exit_code=record.exit_code,
               log_path=str(record.log_path))


def _accepted(record: JobRecord) -> JobAccepted:
    return JobAccepted(job_id=record.job_id, project=record.project,
                       command=record.command, argv=record.argv, run_id=record.run_id)


def _spawn(sup: JobSupervisor, cfg: Config, command: str,
           argv: list[str]) -> JobAccepted:
    """The shared precondition + spawn path for all four commands."""
    require_repo_path(cfg)
    return _accepted(sup.spawn(cfg, command, argv))


# ---- read ----------------------------------------------------------------
@router.get("/jobs", response_model=Jobs)
def list_jobs(cfg: Config = Depends(resolve_project),
              sup: JobSupervisor = Depends(get_supervisor)) -> Jobs:
    # Resolved here too, not only on the detail route: the job console renders
    # from this list, and a job whose run_id only appeared once someone opened
    # its drawer could never be linked to its run. `resolve_run_id` is a no-op
    # once the answer is known or known-unknowable, so the poll stays cheap.
    store = store_path(cfg)
    return Jobs(jobs=[_job(sup.resolve_run_id(r, store)) for r in sup.list(cfg)])


@router.get("/jobs/{job_id}", response_model=Job, responses=JOB_READ_ERRORS)
def get_job(job_id: str, cfg: Config = Depends(resolve_project),
            sup: JobSupervisor = Depends(get_supervisor)) -> Job:
    # The run id is resolved on read, not at spawn: the child has not created its
    # run row yet when the 202 goes out, so this is the first place it can exist.
    return _job(sup.resolve_run_id(sup.get(cfg, job_id), store_path(cfg)))


@router.get("/jobs/{job_id}/log", response_model=JobLog, responses=JOB_READ_ERRORS)
def get_job_log(job_id: str, offset: int = Query(0, ge=0),
                max_bytes: int = Query(supervisor_mod.DEFAULT_LOG_CHUNK,
                                       ge=1, le=1024 * 1024),
                cfg: Config = Depends(resolve_project),
                sup: JobSupervisor = Depends(get_supervisor)) -> JobLog:
    record = sup.get(cfg, job_id)
    next_offset, text, eof = sup.read_log(record, offset, max_bytes)
    return JobLog(job_id=job_id,
                  # The CLAMPED start, not what was asked for. `read_log` pins a
                  # past-the-end offset back to the file size, so echoing the
                  # request would report offset > next_offset — a rewind, to a
                  # console that trusts the pair.
                  offset=min(offset, next_offset),
                  next_offset=next_offset, eof=eof, text=text)


# ---- spawn ---------------------------------------------------------------
@router.post("/jobs/run", response_model=JobAccepted, responses=JOB_SPAWN_ERRORS,
             status_code=status.HTTP_202_ACCEPTED)
def start_run(body: RunRequest = Body(...),
              cfg: Config = Depends(resolve_project),
              sup: JobSupervisor = Depends(get_supervisor)) -> JobAccepted:
    argv = supervisor_mod.run_argv(cfg.project_name, tasks=body.tasks, n=body.n,
                                   dry_run=body.dry_run)
    return _spawn(sup, cfg, "run", argv)


@router.post("/jobs/plan", response_model=JobAccepted, responses=JOB_SPAWN_ERRORS,
             status_code=status.HTTP_202_ACCEPTED)
def start_plan(body: PlanRequest = Body(...),
               cfg: Config = Depends(resolve_project),
               sup: JobSupervisor = Depends(get_supervisor)) -> JobAccepted:
    argv = supervisor_mod.plan_argv(cfg.project_name, tasks=body.tasks,
                                    all_needs_plan=body.all_needs_plan,
                                    limit=body.limit, note=body.note)
    return _spawn(sup, cfg, "plan", argv)


@router.post("/jobs/resume", response_model=JobAccepted, responses=JOB_SPAWN_ERRORS,
             status_code=status.HTTP_202_ACCEPTED)
def start_resume(body: ResumeRequest = Body(...),
                 cfg: Config = Depends(resolve_project),
                 conn: sqlite3.Connection = Depends(store_conn),
                 sup: JobSupervisor = Depends(get_supervisor)) -> JobAccepted:
    # Checked here as well as in the CLI so the UI can grey the button out: the
    # subprocess would print "no paused run to resume" and exit 1, which the
    # panel could only report as a failed job after the fact.
    paused = reads.latest_run(conn, statuses=("paused",))
    if paused is None:
        raise JobError(404, f"no paused run to resume in project "
                            f"{cfg.project_name!r}")
    return _spawn(sup, cfg, "resume", supervisor_mod.resume_argv(cfg.project_name))


@router.post("/jobs/import-backlog", response_model=JobAccepted,
             responses=JOB_SPAWN_ERRORS, status_code=status.HTTP_202_ACCEPTED)
def start_import_backlog(
        body: ImportBacklogRequest = Body(default_factory=ImportBacklogRequest),
        cfg: Config = Depends(resolve_project),
        sup: JobSupervisor = Depends(get_supervisor)) -> JobAccepted:
    argv = supervisor_mod.import_backlog_argv(cfg.project_name)
    return _spawn(sup, cfg, "import-backlog", argv)


# ---- stop ----------------------------------------------------------------
@router.post("/jobs/{job_id}/stop", response_model=Job, responses=JOB_READ_ERRORS)
async def stop_job(job_id: str, cfg: Config = Depends(resolve_project),
                   sup: JobSupervisor = Depends(get_supervisor)) -> Job:
    """SIGINT → grace → SIGTERM → grace → SIGKILL, then the final record.

    Async, and awaited rather than backgrounded, because the answer the UI needs
    is what the job actually did — a 202 "stopping" would leave the console
    guessing. Stopping an already-finished job is a no-op returning its record,
    not an error: the poll that raced the exit is the common case.
    """
    return _job(await sup.stop(sup.get(cfg, job_id)))
