"""Integrator + finalizer — pure Python + git, zero tokens.

Approve -> merge the winner's branch into the orchestrator's feature branch
(never main: feature -> main stays a human decision), write status back to
the backlog markdown, clean up worktrees. Fail/reject -> record and clean up.
"""

from __future__ import annotations

import asyncio
import logging

from ..ops.backlog import make_backlog, parent_id
from ..ops import projectmap
from ..core.context import RunContext
from ..core.state import TaskState, latest_candidates

log = logging.getLogger("orchestrator.integrator")


def _sibling_ids(ctx: RunContext, parent: str) -> list[str]:
    """Every task in the store that decomposes `parent` (itself excluded)."""
    out = []
    for task in ctx.store.all_tasks():
        if task["id"] == parent:
            continue
        if (task.get("parent_id") or parent_id(task["id"])) == parent:
            out.append(task["id"])
    return sorted(out)


def write_back(ctx: RunContext, task_id: str, spec: dict, status: str,
               note: str) -> None:
    """Record a finished task's status on the human backlog.

    The direct path is an exact id match. When there is none the write used to be
    a silent no-op on the source of truth — the caller discarded `set_status`'s
    False — and the usual cause is a decomposition: backlog item **T-131** became
    specs **T-131a**/**T-131b**, so neither sub-id has a line. Observed: T-131a
    merged, and the board still showed `[ ] T-131` with no note anywhere.

    So fall back to the parent item (spec `parent_id`, else derived from the id):
      * flip it `done` only when EVERY sibling in the store is done — a parent is
        not finished because one of its children is;
      * flip it `blocked` when a child needs a human, which is what would have
        happened without the decomposition;
      * otherwise annotate it and leave the human's checkbox alone.
    A no-op that survives all of that is logged at WARNING: the board silently
    disagreeing with the store is the worse half of this bug.
    """
    backlog = make_backlog(ctx.cfg)
    if backlog.set_status(task_id, status, note):
        return

    parent = spec.get("parent_id") or parent_id(task_id)
    if not parent:
        log.warning("backlog has no line for %s and no parent to fall back to — "
                    "the board no longer agrees with the store (status=%s)",
                    task_id, status)
        return

    siblings = _sibling_ids(ctx, parent)
    unfinished = [s for s in siblings
                  if (ctx.store.get_task(s) or {}).get("status") != "done"]
    if status == "done" and siblings and not unfinished:
        written = backlog.set_status(
            parent, "done",
            f"all sub-tasks done ({', '.join(siblings)}) — last: {task_id} {note}")
    elif status != "done":
        written = backlog.set_status(parent, status, f"{task_id}: {note}")
    else:
        written = backlog.annotate(
            parent, f"{task_id} done — {note}. Still open: "
                    f"{', '.join(unfinished) or 'none'}")
    if not written:
        log.warning("backlog writeback found no line for %s or its parent %s "
                    "(status=%s) — the board is now out of date with the store",
                    task_id, parent, status)


async def integrate(ctx: RunContext, state: TaskState) -> dict:
    spec = state["spec"]
    task_id = state["task_id"]
    latest = latest_candidates(state)
    verdict = state.get("verdict") or {}
    winner_id = verdict.get("winner")
    if not winner_id:
        # Auto-integrate path (run.auto_integrate_low_risk): a green, low-risk,
        # single-candidate task reaches integrate without a reviewer verdict.
        # The router guarantees exactly one green candidate, so the winner is not
        # a choice — but record the event, because "merged without review" must
        # be visible in the ledger rather than inferred from an absent verdict.
        green = [cid for cid, c in latest.items() if c["status"] == "gate_passed"]
        winner_id = green[0]
        ctx.store.log_event(ctx.run_id, task_id, "auto_integrated",
                            f"{winner_id} merged without review "
                            f"(risk={spec.get('risk')}, single green candidate)")
        log.info("%s auto-integrating %s (low risk, no reviewer call)",
                 task_id, winner_id)
    cand = latest[winner_id]

    message = (f"feat({task_id.lower()}): {spec['title']} "
               f"[agent:{winner_id}]\n\nRefs {task_id}")
    commit = await ctx.git.amerge_into_feature(cand["branch"], message)
    ctx.store.log_event(ctx.run_id, task_id, "merged",
                        f"{cand['branch']} -> {ctx.git.feature} @ {commit[:10]}")
    log.info("%s merged into %s (%s)", task_id, ctx.git.feature, commit[:10])
    # `winner` travels with the integration record so the backlog writeback names
    # it on the auto-integrate path too (where there is no verdict to read it from).
    return {"integration": {"merged_commit": commit, "winner": winner_id},
            "outcome": "done"}


async def finalize(ctx: RunContext, state: TaskState) -> dict:
    task_id = state["task_id"]
    outcome = state.get("outcome")
    if not outcome:
        verdict = state.get("verdict") or {}
        outcome = "rejected" if verdict.get("decision") == "reject" else "failed"

    status = {"done": "done", "failed": "failed", "rejected": "rejected",
              # A structurally unverifiable acceptance criterion is not a failed
              # attempt — no retry can fix it, only a human (or a re-planned spec).
              "needs_human": "needs_human"}[outcome]
    ctx.store.set_task_status(task_id, status,
                              retries=max(0, state.get("attempt", 1) - 1))

    # Conservative writeback into the human backlog.
    try:
        spec = state.get("spec") or {}
        if outcome == "done":
            spend = ctx.store.task_cash_spend(task_id)
            integration = state.get("integration") or {}
            winner = (integration.get("winner")
                      or (state.get("verdict") or {}).get("winner", "?"))
            commit = integration.get("merged_commit", "")[:10]
            write_back(ctx, task_id, spec, "done",
                       f"merged to `{ctx.git.feature}` @ {commit} "
                       f"(worker {winner}, ${spend:.2f})")
        else:
            reason = state.get("blocked_reason") or f"agent run {outcome}"
            write_back(ctx, task_id, spec, "blocked", f"{reason} — needs human")
    except Exception as e:  # writeback must never kill a run
        log.warning("backlog writeback failed for %s: %s", task_id, e)

    # Regenerate the structural project-map from the (post-merge) integration
    # worktree. Serialized + guarded internally; a map failure never kills a run.
    if outcome == "done":
        try:
            await projectmap.regenerate_from_integration(ctx)
        except Exception as e:
            log.warning("projectmap regeneration failed for %s: %s", task_id, e)

    # Worktree/branch cleanup for every candidate of this task.
    for cand in latest_candidates(state).values():
        wt_name = f"{task_id}-{cand['cand_id']}".lower()
        try:
            await ctx.git.aremove_worktree(wt_name)
            await ctx.git.adelete_branch(cand["branch"])
        except Exception as e:
            log.warning("cleanup failed for %s: %s", wt_name, e)

    ctx.store.log_event(ctx.run_id, task_id, "finalized", outcome)
    log.info("%s finalized: %s", task_id, outcome)
    return {"outcome": outcome}
