"""The outer run loop: scheduler batches -> parallel task graphs.

Owns run lifecycle: budget/limit pause-and-checkpoint, resume, dry-run
planning. Each task runs as its own graph invocation with
thread_id = "{run_id}:{task_id}" against a SQLite checkpointer, so an
interrupted run resumes mid-task, mid-attempt.
"""

from __future__ import annotations

import asyncio
import logging

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ..ops.budget import Budget
from ..core.config import Config
from ..core.context import RunContext
from ..core.errors import BudgetExceeded, LimitExhausted
from ..ops.gitops import Git
from .graph import build_task_graph, resolve_worker_pool
from .scheduler import (domain_stats, next_batch, queue_stats,
                        seam_tasks_missing_deps)
from ..ops.store import Store

log = logging.getLogger("orchestrator.runner")


def make_context(cfg: Config, store: Store, run_id: str,
                 dry_run: bool = False, degraded: bool = False) -> RunContext:
    git = Git(cfg.repo_path(), cfg.work_dir(), cfg.project.feature_branch,
              cfg.project.base_branch, dry_run=dry_run)
    return RunContext(cfg=cfg, store=store, git=git,
                      budget=Budget(cfg, store, run_id),
                      run_id=run_id, dry_run=dry_run, degraded=degraded)


def _plan_only(ctx: RunContext, task_filter: set[str] | None) -> None:
    """--dry-run: print what would happen; zero tokens, zero git writes."""
    tasks = _filtered(ctx, task_filter)
    print(f"\n=== DRY RUN (project: {ctx.cfg.project_name}) ===")
    print(f"queue: {queue_stats(tasks)}")
    print(f"domains: {domain_stats(tasks)}")
    seam_missing = seam_tasks_missing_deps(tasks)
    if seam_missing:
        print(f"⚠  seam tasks with NO deps (should serialize after their domain "
              f"work): {seam_missing}")
    simulated_done: set[str] = set()
    wave = 1
    while True:
        batch = next_batch(tasks, int(ctx.cfg.run.max_parallel_tasks))
        if not batch:
            break
        print(f"\n-- wave {wave} (parallel, files_write-disjoint) --")
        for t in batch:
            n = t.get("n_candidates") or int(ctx.cfg.run.n_candidates)
            default, pool = resolve_worker_pool(ctx.cfg, t)
            cands = [default] if n <= 1 else (pool or [default])[:n]
            dom = f"   domain: {t['domain']}" if t.get("domain") else ""
            print(f"  {t['id']}: {t['title']}{dom}")
            print(f"    candidates: {cands}   writes: {t.get('files_write')}")
            print(f"    reads: {t.get('files_read')}")
            t["status"] = "done"  # simulate for wave computation
            simulated_done.add(t["id"])
        wave += 1
    remaining = [t["id"] for t in tasks
                 if t["status"] == "ready" and t["id"] not in simulated_done]
    if remaining:
        print(f"\nnot reachable this run (unmet deps): {remaining}")
    print("\nno tokens spent, no git mutations. Remove --dry-run to execute.\n")


def _filtered(ctx: RunContext, task_filter: set[str] | None) -> list[dict]:
    tasks = ctx.store.all_tasks()
    if task_filter:
        tasks = [t for t in tasks if t["id"] in task_filter]
    return tasks


async def _run_batch(batch: list[dict], run_task) -> None:
    """Run a batch, and when one task raises the run-level pause signal, STOP the
    others.

    A bare `asyncio.gather` propagates the first exception immediately but leaves
    the siblings running: `run_task` deliberately re-raises LimitExhausted /
    BudgetExceeded, so the pause handler would bookkeep a "paused" run while other
    tasks were still mid-LLM-call, mid-gate, or about to `integrate` — and a merge
    could land on the feature branch after the run was reported stopped, with the
    SQLite writes racing the pause. "Checkpoint and stop" has to mean stop.

    Cancellation lands inside `asyncio.to_thread` in some cases; a thread can't be
    interrupted, so an in-flight git/gate call finishes before the cancellation is
    observed. That is acceptable — those calls are short, and the LangGraph
    checkpoint makes whatever they completed resumable.
    """
    tasks = [asyncio.create_task(run_task(t)) for t in batch]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    for p in pending:
        p.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    for d in done:
        exc = d.exception()
        if exc is not None:
            raise exc


async def run(cfg: Config, *, dry_run: bool = False,
              task_filter: set[str] | None = None,
              n_candidates: int | None = None,
              resume_run_id: str | None = None,
              degraded: bool = False) -> None:
    store = Store(cfg.store_path())
    run_id = resume_run_id or store.create_run()
    ctx = make_context(cfg, store, run_id, dry_run=dry_run, degraded=degraded)

    if dry_run:
        _plan_only(ctx, task_filter)
        return

    ctx.git.ensure_feature_branch()
    if resume_run_id:
        store.set_run_status(run_id, "running", note="resumed")
        # Tasks stuck 'running' from the interrupted run resume via checkpoints.
        for t in store.all_tasks():
            if t["status"] == "running":
                store.set_task_status(t["id"], "ready")

    async with AsyncSqliteSaver.from_conn_string(str(cfg.checkpoint_path())) as saver:
        graph = build_task_graph(ctx).compile(checkpointer=saver)

        async def run_task(task: dict) -> None:
            thread = {"configurable": {"thread_id": f"{run_id}:{task['id']}"}}
            try:
                existing = await graph.aget_state(thread)
                if existing and existing.next:
                    log.info("%s: resuming from checkpoint at %s",
                             task["id"], existing.next)
                    await graph.ainvoke(None, thread)
                else:
                    await graph.ainvoke({
                        "run_id": run_id,
                        "task_id": task["id"],
                        "spec": task,
                        "n_candidates": n_candidates or task.get("n_candidates"),
                    }, thread)
            except (LimitExhausted, BudgetExceeded):
                raise  # run-level control flow: checkpoint & pause
            except Exception as e:
                # One task must not take the whole batch down (merge conflict,
                # git/env failure, ...): record, mark failed, keep the run alive.
                log.error("%s: task crashed: %s", task["id"], e)
                store.log_event(run_id, task["id"], "crashed", str(e)[:2000])
                store.set_task_status(task["id"], "failed")

        try:
            while True:
                batch = next_batch(_filtered(ctx, task_filter),
                                   int(cfg.run.max_parallel_tasks))
                if not batch:
                    break
                log.info("run %s: batch %s", run_id, [t["id"] for t in batch])
                await _run_batch(batch, run_task)
            store.set_run_status(run_id, "done")
            stats = queue_stats(store.all_tasks())
            print(f"\nrun {run_id} complete. queue: {stats}")
            print(f"cash spend this run: ${store.run_cash_spend(run_id):.2f}")

        except LimitExhausted as e:
            mode = cfg.run.on_limit_exhausted
            if mode == "degrade" and not ctx.degraded:
                log.warning("Claude limit exhausted -> degrading opus roles to %s",
                            cfg.run.degrade_model)
                store.log_event(run_id, None, "degraded", str(e)[:500])
                # Re-enter the loop with the degrade flag (the recursive call
                # builds a fresh context); interrupted tasks resume from
                # checkpoints. degraded=True guarantees no second recursion.
                await run(cfg, task_filter=task_filter,
                          n_candidates=n_candidates, resume_run_id=run_id,
                          degraded=True)
                return
            store.set_run_status(run_id, "paused", note=f"limit: {e}")
            print(f"\n⏸  Claude Code limit exhausted — run checkpointed.\n"
                  f"   {e}\n"
                  f"   When your limit resets:  python -m orchestrator resume "
                  f"--project {cfg.project_name}")

        except BudgetExceeded as e:
            store.set_run_status(run_id, "paused", note=f"budget: {e}")
            print(f"\n⏸  Budget cap hit — run checkpointed.\n   {e}\n"
                  f"   Raise budget.* in config and resume:  "
                  f"python -m orchestrator resume --project {cfg.project_name}")
