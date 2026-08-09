"""Subprocess supervisor for `run` / `plan` / `resume` / `import-backlog`.

The API never executes the graph in-process (PLAN §3.1 rule 1). A job is a real
`python -m orchestrator <command> --project <p> …` child process, exactly what a
terminal run would be, with stdout+stderr tee'd into
`<state_dir>/jobs/<job_id>.log`. Four consequences that shaped this module:

- **The CLI stays the single execution path.** The panel cannot drift from a
  terminal run, because it *is* one. Nothing here imports `engine.runner`.
- **A crash cannot take the API down**, and the API exiting cannot kill a run:
  children get `start_new_session=True`, so they survive and finish (or stay
  resumable). Which is why the registry is rebuilt from disk — see `_rehydrate`.
- **Stop is a signal, not a kill.** SIGINT first: the runner checkpoints and the
  run stays resumable. SIGTERM, then SIGKILL, only if it ignores the escalation.
  Signals go to the process *group* so the provider CLI the run shelled out to
  gets them too, which is what actually makes Ctrl-C work in a terminal.
- **Jobs outlive their process.** Nothing is evicted on exit; the record keeps
  its exit code and its log so the console can still be read afterwards.

The registry is process-local by design: there is no second source of truth to
reconcile, and the sidecar `<job_id>.json` next to each log is enough to rebuild
it after a restart.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import HTTPException

from ..core.config import Config
from . import deps

# `import-backlog` is free; the other three spend subscription quota (or, for a
# dry run, nothing) and are confirm-gated in the request models.
JobCommand = str
COMMANDS: tuple[str, ...] = ("run", "plan", "resume", "import-backlog")

LIVE_STATUSES: tuple[str, ...] = ("starting", "running")

# Job ids become filenames, so they are minted (never accepted) and re-validated
# before any path join — the same defensive second line as `deps._SAFE_NAME`.
_SAFE_JOB_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[a-z-]+-[0-9a-f]{6}$")

DEFAULT_LOG_CHUNK = 64 * 1024
SIGINT_GRACE_S = 15.0
SIGTERM_GRACE_S = 5.0


class JobError(HTTPException):
    """Raised as the HTTP answer directly — same convention as `deps.py`."""


@dataclass
class JobRecord:
    """One supervised child. Mirrors `schemas.Job` plus the private handle."""

    job_id: str
    project: str
    command: JobCommand
    argv: list[str]
    started_at: float
    log_path: Path
    pid: int | None = None
    status: str = "starting"
    ended_at: float | None = None
    exit_code: int | None = None
    run_id: str | None = None
    # Not serialized: a stop that has been requested but not yet observed, so the
    # exit is reported as `stopped` rather than `failed` when the signal lands.
    stopping: bool = False
    # True when the record was rebuilt from a sidecar: the process is not our
    # child any more, so `poll()` is impossible and liveness is all we have.
    adopted: bool = False
    # Set once a terminal job has been looked up and found to have no run row —
    # the answer cannot change after that, and the job list is polled, so without
    # this every poll re-queries every runless job forever.
    run_id_settled: bool = False
    proc: subprocess.Popen | None = field(default=None, repr=False, compare=False)

    @property
    def sidecar_path(self) -> Path:
        return self.log_path.with_suffix(".json")

    def to_sidecar(self) -> dict:
        return {"job_id": self.job_id, "project": self.project,
                "command": self.command, "argv": self.argv, "pid": self.pid,
                "status": self.status, "started_at": self.started_at,
                "ended_at": self.ended_at, "exit_code": self.exit_code,
                "run_id": self.run_id}


# ---- argv construction ---------------------------------------------------
# Separate from spawning so it can be unit-tested without starting anything, and
# so the supervisor itself never builds an orchestrator command line: tests
# supervise `echo`/`sleep` through the very same code path.

def _base(command: str, project: str) -> list[str]:
    # sys.executable, not "python": the API may be running from a venv that is not
    # on PATH, and a job that resolves a different interpreter would import a
    # different orchestrator.
    return [sys.executable, "-m", "orchestrator", command, "--project", project]


def run_argv(project: str, *, tasks: list[str] | None = None,
             n: int | None = None, dry_run: bool = False) -> list[str]:
    argv = _base("run", project)
    if dry_run:
        argv.append("--dry-run")
    if tasks:
        argv += ["--tasks", ",".join(tasks)]
    if n is not None:
        argv += ["--n", str(n)]
    return argv


def plan_argv(project: str, *, tasks: list[str] | None = None,
              all_needs_plan: bool = False, limit: int | None = None,
              note: str = "") -> list[str]:
    argv = _base("plan", project)
    if all_needs_plan:
        argv.append("--all-needs-plan")
    if tasks:
        argv += ["--tasks", ",".join(tasks)]
    if limit is not None:
        argv += ["--limit", str(limit)]
    if note:
        # Positional, and last: the request model rejects a leading "-" so argparse
        # can never read the note as a flag.
        argv.append(note)
    return argv


def resume_argv(project: str) -> list[str]:
    return _base("resume", project)


def import_backlog_argv(project: str) -> list[str]:
    return _base("import-backlog", project)


# ---- the supervisor ------------------------------------------------------
class JobSupervisor:
    def __init__(self, *, sigint_grace_s: float = SIGINT_GRACE_S,
                 sigterm_grace_s: float = SIGTERM_GRACE_S):
        self._jobs: dict[str, JobRecord] = {}
        self._rehydrated: set[str] = set()
        # Serializes check-and-spawn. FastAPI runs `def` endpoints in a
        # threadpool, so two POSTs really do race: without this, eight concurrent
        # requests started seven runs, each spending quota and fighting the same
        # git worktrees. Reentrant because the critical section calls `in_flight`
        # → `list` → `_rehydrate`, and any of those may grow its own locking.
        self._lock = threading.RLock()
        self.sigint_grace_s = sigint_grace_s
        self.sigterm_grace_s = sigterm_grace_s

    # -- paths ------------------------------------------------------------
    def log_dir(self, cfg: Config) -> Path:
        state = deps._resolved_dir(cfg, "state_dir")
        if state is None:
            raise JobError(409, f"project {cfg.project_name!r} has no paths.state_dir, "
                                f"so a job has nowhere to write its log")
        return state / "jobs"

    # -- registry ---------------------------------------------------------
    def _rehydrate(self, cfg: Config) -> None:
        """Adopt the sidecars in this project's job dir, once per project.

        Called on first touch rather than from a startup hook: a project the API
        has not been asked about yet cannot have been missed, and the first
        request after a restart is exactly when the console needs the history.
        A job whose process is gone but which the sidecar recorded as live cannot
        have its exit code recovered (we never waited on it), so it reports
        `failed` with `exit_code: null` — an interrupted run is not a success,
        and null distinguishes "we never saw the code" from an observed non-zero.
        """
        if cfg.project_name in self._rehydrated:
            return
        self._rehydrated.add(cfg.project_name)
        try:
            sidecars = sorted(self.log_dir(cfg).glob("*.json"))
        except (JobError, OSError):
            return
        for path in sidecars:
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            job_id = str(data.get("job_id") or "")
            if not _SAFE_JOB_ID.match(job_id) or job_id in self._jobs:
                continue
            record = JobRecord(
                job_id=job_id, project=str(data.get("project") or cfg.project_name),
                command=str(data.get("command") or "run"),
                argv=[str(a) for a in (data.get("argv") or [])],
                started_at=float(data.get("started_at") or 0.0),
                log_path=path.with_suffix(".log"),
                pid=data.get("pid"), status=str(data.get("status") or "failed"),
                ended_at=data.get("ended_at"), exit_code=data.get("exit_code"),
                run_id=data.get("run_id"), adopted=True)
            if record.status in LIVE_STATUSES and not _alive(record.pid):
                record.status, record.ended_at = "failed", time.time()
            self._jobs[job_id] = record

    def _write_sidecar(self, record: JobRecord) -> None:
        with contextlib.suppress(OSError):
            record.sidecar_path.write_text(json.dumps(record.to_sidecar(), indent=1))

    def refresh(self, record: JobRecord) -> JobRecord:
        """Reap on demand. No background task: a poll of the registry is the only
        thing that ever needs the answer, and a reaper thread would be one more
        thing to shut down cleanly."""
        if record.status not in LIVE_STATUSES:
            return record
        if record.proc is not None:
            code = record.proc.poll()
            if code is None:
                record.status = "running"
                return record
            self._finish(record, code)
        elif not _alive(record.pid):
            self._finish(record, None)
        else:
            record.status = "running"
        return record

    def _finish(self, record: JobRecord, code: int | None) -> None:
        record.ended_at = time.time()
        record.exit_code = code
        if record.stopping:
            record.status = "stopped"
        elif code == 0:
            record.status = "exited"
        else:
            record.status = "failed"
        self._write_sidecar(record)

    def list(self, cfg: Config) -> list[JobRecord]:
        self._rehydrate(cfg)
        mine = [r for r in self._jobs.values() if r.project == cfg.project_name]
        return sorted((self.refresh(r) for r in mine),
                      key=lambda r: r.started_at, reverse=True)

    def get(self, cfg: Config, job_id: str) -> JobRecord:
        self._rehydrate(cfg)
        record = self._jobs.get(job_id)
        if record is None or record.project != cfg.project_name:
            raise JobError(404, f"unknown job: {job_id!r}")
        return self.refresh(record)

    def in_flight(self, cfg: Config) -> JobRecord | None:
        return next((r for r in self.list(cfg) if r.status in LIVE_STATUSES), None)

    # -- spawn ------------------------------------------------------------
    def spawn(self, cfg: Config, command: JobCommand, argv: list[str], *,
              run_id: str | None = None) -> JobRecord:
        """Start one job, or 409 if this project already has one in flight.

        `argv` is passed in rather than built here so that the tests supervise
        harmless commands (`echo`, `sleep`, `python -c`) through the exact code
        path a real run takes — no test may ever spawn an orchestrator command.

        The whole body holds `self._lock`, `Popen` included: the one-in-flight
        rule is only real if the new record is registered before another thread
        can run the `in_flight` check. Releasing the lock earlier and registering
        a placeholder would be the same critical section with more states.
        """
        with self._lock:
            busy = self.in_flight(cfg)
            if busy is not None:
                raise JobError(409, f"job {busy.job_id} ({busy.command}) is still "
                                    f"{busy.status} for project "
                                    f"{cfg.project_name!r} — one job at a time; "
                                    f"stop it first")
            log_dir = self.log_dir(cfg)
            log_dir.mkdir(parents=True, exist_ok=True)
            job_id = _mint_job_id(command)
            record = JobRecord(job_id=job_id, project=cfg.project_name,
                               command=command, argv=list(argv),
                               started_at=time.time(),
                               log_path=log_dir / f"{job_id}.log", run_id=run_id)
            # Append, not truncate: a job id is unique, so this only matters if a
            # log somehow survives its id — in which case losing the older bytes
            # is worse.
            log = open(record.log_path, "ab", buffering=0)
            try:
                proc = subprocess.Popen(
                    record.argv, cwd=deps.REPO_ROOT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT,
                    # Own process group: signals reach the provider CLI the run
                    # shells out to, and the child survives the API exiting
                    # rather than dying mid-merge.
                    start_new_session=True)
            except OSError as e:
                record.status, record.ended_at = "failed", time.time()
                self._jobs[job_id] = record
                self._write_sidecar(record)
                raise JobError(409, f"could not start {command}: {e}") from None
            finally:
                log.close()
            record.proc, record.pid, record.status = proc, proc.pid, "running"
            self._jobs[job_id] = record
            self._write_sidecar(record)
            return record

    # -- stop -------------------------------------------------------------
    async def stop(self, record: JobRecord) -> JobRecord:
        """SIGINT → grace → SIGTERM → grace → SIGKILL.

        SIGINT first because that is what the runner is written for: it
        checkpoints and the run stays resumable, so stopping is not losing work.
        Escalation exists only for a child that ignores it.
        """
        self.refresh(record)
        if record.status not in LIVE_STATUSES:
            return record
        record.stopping = True
        for sig, grace in ((signal.SIGINT, self.sigint_grace_s),
                           (signal.SIGTERM, self.sigterm_grace_s)):
            _signal_group(record.pid, sig)
            if await self._await_exit(record, grace):
                return record
        _signal_group(record.pid, signal.SIGKILL)
        await self._await_exit(record, self.sigterm_grace_s)
        return record

    async def _await_exit(self, record: JobRecord, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.refresh(record).status not in LIVE_STATUSES:
                return True
            await asyncio.sleep(0.05)
        return self.refresh(record).status not in LIVE_STATUSES

    # -- log --------------------------------------------------------------
    def read_log(self, record: JobRecord, offset: int = 0,
                 max_bytes: int = DEFAULT_LOG_CHUNK) -> tuple[int, str, bool]:
        """(next_offset, text, eof) from a byte offset.

        Byte offsets, not line numbers: the console polls this while the file is
        being appended to, and a partial last line must not be re-delivered as a
        different line next time. Decoding is lossy on purpose — a chunk boundary
        can split a UTF-8 sequence and a tail must never raise.
        """
        finished = record.status not in LIVE_STATUSES
        try:
            size = record.log_path.stat().st_size
        except OSError:
            return offset, "", finished
        start = min(max(offset, 0), size)
        with open(record.log_path, "rb") as fh:
            fh.seek(start)
            chunk = fh.read(max(max_bytes, 0))
        next_offset = start + len(chunk)
        return next_offset, chunk.decode("utf-8", "replace"), finished and next_offset >= size

    # -- run id -----------------------------------------------------------
    def resolve_run_id(self, record: JobRecord, store_path: Path | None) -> JobRecord:
        """Best-effort: attach the run row the subprocess minted.

        The job cannot know its own run id — the CLI creates it. Matching on
        "newest run that started after this job did" is what the runs table can
        answer; a couple of seconds of slack covers the child's startup. Purely
        cosmetic, so any failure leaves `run_id` null.

        Safe to call on every list poll: it returns immediately once the id is
        known, and `run_id_settled` stops it re-querying a finished job that
        never produced a run (a crash before the first write, or `import-backlog`).
        """
        if (record.run_id or record.command == "import-backlog"
                or record.run_id_settled):
            return record
        if store_path is None or not store_path.exists():
            return record
        try:
            conn = deps.open_read_only(store_path)
        except Exception:
            return record
        try:
            row = conn.execute(
                "SELECT id FROM runs WHERE started_at >= ? ORDER BY started_at DESC "
                "LIMIT 1", (record.started_at - 2.0,)).fetchone()
        except Exception:
            return record
        finally:
            conn.close()
        if row is not None:
            record.run_id = row["id"]
            self._write_sidecar(record)
        elif record.status not in LIVE_STATUSES:
            # Finished without ever creating a run row: stop asking.
            record.run_id_settled = True
        return record

    # -- digest for the live stream ---------------------------------------
    def cursor(self, cfg: Config) -> str:
        """A value that changes whenever this project's job list or a status does.

        The SSE loop compares strings; it must not have to diff records.
        """
        jobs = self.list(cfg)
        return "|".join(f"{r.job_id}:{r.status}:{r.exit_code}" for r in jobs) or "-"


def _mint_job_id(command: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    return f"{stamp}-{command}-{os.urandom(3).hex()}"


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True          # exists, owned by someone else
    return True


def _signal_group(pid: int | None, sig: int) -> None:
    """Signal the child's process group, falling back to the child alone.

    The group is the point: a run's provider CLI is a grandchild, and signalling
    only the parent leaves it holding the terminal and the quota.
    """
    if not pid:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(os.getpgid(pid), sig)
        return
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.kill(pid, sig)


_supervisor: JobSupervisor | None = None


def default_supervisor() -> JobSupervisor:
    global _supervisor
    if _supervisor is None:
        _supervisor = JobSupervisor()
    return _supervisor


def get_supervisor() -> JobSupervisor:
    """Overridable seam: `app.dependency_overrides[get_supervisor] = …`."""
    return default_supervisor()
