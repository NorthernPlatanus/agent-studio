"""Integrator + finalizer — pure Python + git, zero tokens.

Approve -> merge the winner's branch into the orchestrator's feature branch
(never main: feature -> main stays a human decision), write status back to
the backlog markdown, clean up worktrees. Fail/reject -> record and clean up.
"""

from __future__ import annotations

import asyncio
import logging

from ..ops.backlog import make_backlog
from ..core.context import RunContext
from ..core.state import TaskState, latest_candidates

log = logging.getLogger("orchestrator.integrator")


async def integrate(ctx: RunContext, state: TaskState) -> dict:
    spec = state["spec"]
    task_id = state["task_id"]
    winner_id = state["verdict"]["winner"]
    cand = latest_candidates(state)[winner_id]

    message = (f"feat({task_id.lower()}): {spec['title']} "
               f"[agent:{winner_id}]\n\nRefs {task_id}")
    commit = await asyncio.to_thread(
        ctx.git.merge_into_feature, cand["branch"], message)
    ctx.store.log_event(ctx.run_id, task_id, "merged",
                        f"{cand['branch']} -> {ctx.git.feature} @ {commit[:10]}")
    log.info("%s merged into %s (%s)", task_id, ctx.git.feature, commit[:10])
    return {"integration": {"merged_commit": commit}, "outcome": "done"}


async def finalize(ctx: RunContext, state: TaskState) -> dict:
    task_id = state["task_id"]
    outcome = state.get("outcome")
    if not outcome:
        verdict = state.get("verdict") or {}
        outcome = "rejected" if verdict.get("decision") == "reject" else "failed"

    status = {"done": "done", "failed": "failed", "rejected": "rejected"}[outcome]
    ctx.store.set_task_status(task_id, status,
                              retries=max(0, state.get("attempt", 1) - 1))

    # Conservative writeback into the human backlog.
    try:
        backlog = make_backlog(ctx.cfg)
        if outcome == "done":
            spend = ctx.store.task_cash_spend(task_id)
            winner = (state.get("verdict") or {}).get("winner", "?")
            commit = (state.get("integration") or {}).get("merged_commit", "")[:10]
            backlog.set_status(task_id, "done",
                               f"merged to `{ctx.git.feature}` @ {commit} "
                               f"(worker {winner}, ${spend:.2f})")
        else:
            backlog.set_status(task_id, "blocked", f"agent run {outcome} — needs human")
    except Exception as e:  # writeback must never kill a run
        log.warning("backlog writeback failed for %s: %s", task_id, e)

    # Worktree/branch cleanup for every candidate of this task.
    for cand in latest_candidates(state).values():
        wt_name = f"{task_id}-{cand['cand_id']}".lower()
        try:
            await asyncio.to_thread(ctx.git.remove_worktree, wt_name)
            await asyncio.to_thread(ctx.git.delete_branch, cand["branch"])
        except Exception as e:
            log.warning("cleanup failed for %s: %s", wt_name, e)

    ctx.store.log_event(ctx.run_id, task_id, "finalized", outcome)
    log.info("%s finalized: %s", task_id, outcome)
    return {"outcome": outcome}
