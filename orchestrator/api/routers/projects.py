"""Projects: discovery, the one-call dashboard summary, and the wave preview.

`/waves` is the zero-cost twin of `run --dry-run`: it re-runs the same pure
scheduler functions in-process (`next_batch` over a simulated queue) and never
touches git or a provider, so the panel can offer "what would run next" as a
free, always-safe affordance instead of a button that spends quota.
"""

from __future__ import annotations

import os
import sqlite3

from fastapi import APIRouter, Depends

from ...core.config import Config
from ...engine.graph import resolve_worker_pool
from ...engine.scheduler import (domain_stats, next_batch, queue_stats,
                                 seam_tasks_missing_deps)
from .. import reads
from ..deps import (ProjectRegistry, checkpoint_path, get_registry,
                    repo_path_provenance, resolve_project, store_conn, store_path)
from ..errors import READ_ERRORS
from ..schemas import (Project, Projects, Summary, TokenChannels, Wave, Waves,
                       WaveTask)
from .runs import run_item

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.get("", response_model=Projects)
def list_projects(registry: ProjectRegistry = Depends(get_registry)) -> Projects:
    active = os.environ.get("ORCH_PROJECT")
    out = []
    for name in registry.names():
        entry = registry.entries()[name]
        cfg = registry.config(name)
        store, ckpt = store_path(cfg), checkpoint_path(cfg)
        # repo_path never goes through cfg.repo_path(), which raises when it is
        # unset — "incomplete profile" is a fact to report here, not an error. The
        # provenance matters as much as the value: a global config/local.yaml
        # entry makes every project look runnable against one shared checkout.
        repo, source, detail = repo_path_provenance(registry, name, cfg)
        out.append(Project(
            name=name,
            profile_path=str(entry.profile_path) if entry.profile_path else None,
            has_store=bool(store and store.exists()),
            store_path=str(store) if store else None,
            has_checkpoints=bool(ckpt and ckpt.exists()),
            repo_path=repo,
            repo_path_source=source,
            runnable=repo is not None,
            runnable_detail=detail,
            is_active=name == active,
        ))
    return Projects(projects=out,
                    active=active if active and registry.has(active) else None)


@router.get("/{project}/summary", response_model=Summary, responses=READ_ERRORS)
def summary(cfg: Config = Depends(resolve_project),
            conn: sqlite3.Connection = Depends(store_conn)) -> Summary:
    tasks = reads.all_tasks(conn)
    active = reads.latest_run(conn, ("running", "paused"))
    recent = reads.runs(conn, limit=1)
    counts = reads.event_counts(conn)
    return Summary(
        project=cfg.project_name,
        task_count=len(tasks),
        queue_stats=queue_stats(tasks),
        domain_stats=domain_stats(tasks),
        active_run=run_item(conn, active) if active else None,
        last_run=run_item(conn, recent[0]) if recent else None,
        totals=TokenChannels(**reads.token_totals(conn)),
        cash_spend_usd=reads.cash_spend_total(conn),
        event_counts={k: counts.get(k, 0) for k in reads.METRIC_EVENT_KINDS},
        max_event_rowid=reads.max_event_rowid(conn),
    )


@router.get("/{project}/waves", response_model=Waves, responses=READ_ERRORS)
def waves(cfg: Config = Depends(resolve_project),
          conn: sqlite3.Connection = Depends(store_conn)) -> Waves:
    tasks = reads.all_tasks(conn)
    max_parallel = int(cfg.run.max_parallel_tasks)
    default_n = int(cfg.run.n_candidates)
    stats, dstats = queue_stats(tasks), domain_stats(tasks)

    # Same simulation as runner._plan_only: mark each scheduled task done in a
    # local copy so the next call sees satisfied deps. `all_tasks` already
    # returned fresh dicts, so mutating them cannot reach the store.
    out: list[Wave] = []
    scheduled: set[str] = set()
    index = 1
    while True:
        batch = next_batch(tasks, max_parallel)
        if not batch:
            break
        wave_tasks = []
        for t in batch:
            n = t.get("n_candidates") or default_n
            default, pool = resolve_worker_pool(cfg, t)
            cands = [default] if n <= 1 else (pool or [default])[:n]
            wave_tasks.append(WaveTask(
                id=t["id"], title=t["title"], domain=t.get("domain"),
                n_candidates=n, candidates=cands,
                files_write=list(t.get("files_write") or []),
                files_read=list(t.get("files_read") or []),
            ))
            t["status"] = "done"
            scheduled.add(t["id"])
        out.append(Wave(index=index, tasks=wave_tasks))
        index += 1

    unreachable = [t["id"] for t in tasks
                   if t["status"] == "ready" and t["id"] not in scheduled]
    return Waves(
        max_parallel_tasks=max_parallel, default_n_candidates=default_n,
        waves=out, unreachable=unreachable,
        seam_missing_deps=seam_tasks_missing_deps(tasks),
        queue_stats=stats, domain_stats=dstats,
    )
