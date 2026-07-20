"""The per-task LangGraph state machine.

    dispatch ──(Send fan-out)──> work_candidate ──> collect
       ^                                               │
       │              retries left & no green gate     │
       ├───────────────────────────────────────────────┤
       │                                               │ any gate_passed
       │            review says "revise"               v
       └─────────────────────────────────────────── review
                                                       │ approve         reject / out of retries
                                                       v                          v
                                                  integrate ──> finalize <────────┘

Notes on LangGraph 1.x correctness:
  - Send objects are returned ONLY from conditional edges (returning them
    from a node body is unsupported and breaks checkpoint serialization).
  - `candidates` uses an operator.add reducer; parallel Send branches append
    safely, and state.latest_candidates() collapses attempts.
  - State is plain data; runtime services live in RunContext closures, so
    checkpoints stay portable and `resume` just re-binds services.

One compiled graph invocation == one task, thread_id == f"{run_id}:{task_id}".
The outer loop over scheduler batches lives in runner.py.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ..core.context import RunContext
from ..nodes.integrator import finalize, integrate
from ..nodes.reviewer import review
from ..nodes.worker import run_candidate
from ..core.state import TaskState, latest_candidates

log = logging.getLogger("orchestrator.graph")


def build_task_graph(ctx: RunContext):
    async def dispatch(state: TaskState) -> dict:
        """Decide which candidates to (re)run and with what feedback."""
        attempt = state.get("attempt", 0) + 1
        latest = latest_candidates(state)
        verdict = state.get("verdict") or {}
        update: dict = {"attempt": attempt, "verdict": {}}

        if attempt == 1:
            n = state.get("n_candidates") or int(ctx.cfg.run.n_candidates)
            pool = list(ctx.cfg.roles.worker.candidates or [])
            default = ctx.cfg.roles.worker.default
            if n <= 1:
                to_run = [default]
            else:
                to_run = (pool or [default])[:n]
            update["to_run"] = to_run
            update["feedback"] = ""
        elif verdict.get("decision") == "revise":
            update["to_run"] = [verdict["winner"]]
            update["feedback"] = f"REVIEW NOTES:\n{verdict.get('notes', '')}"
        else:
            # Gate/patch failures: retry every failed candidate with its log.
            failed = {cid: c for cid, c in latest.items()
                      if c["status"] != "gate_passed"}
            update["to_run"] = list(failed)
            update["feedback"] = "\n\n".join(
                f"[{cid}] {c['status']}:\n{c.get('gate_log') or c.get('error')}"
                for cid, c in failed.items())[:int(ctx.cfg.gate.log_tail_chars)]
        ctx.store.set_task_status(state["task_id"], "running")
        log.info("%s dispatch attempt %d -> %s",
                 state["task_id"], attempt, update["to_run"])
        return update

    def fan_out(state: TaskState) -> list[Send]:
        return [
            Send("work_candidate", {
                "run_id": state["run_id"],
                "task_id": state["task_id"],
                "spec": state["spec"],
                "cand_id": cid,
                "attempt": state["attempt"],
                "feedback": state.get("feedback", ""),
            })
            for cid in state.get("to_run", [])
        ]

    async def work_candidate(payload: dict) -> dict:
        return await run_candidate(ctx, payload)

    async def collect(state: TaskState) -> dict:
        return {}  # barrier: all Send branches joined here

    def route_after_collect(state: TaskState) -> str:
        latest = latest_candidates(state)
        if any(c["status"] == "gate_passed" for c in latest.values()):
            return "review"
        if state["attempt"] < int(ctx.cfg.run.max_retries):
            return "dispatch"
        return "finalize"

    def route_after_review(state: TaskState) -> str:
        decision = state["verdict"]["decision"]
        if decision == "approve":
            return "integrate"
        if decision == "revise" and state["attempt"] < int(ctx.cfg.run.max_retries):
            return "dispatch"
        return "finalize"

    async def review_node(state: TaskState) -> dict:
        return await review(ctx, state)

    async def integrate_node(state: TaskState) -> dict:
        return await integrate(ctx, state)

    async def finalize_node(state: TaskState) -> dict:
        return await finalize(ctx, state)

    builder = StateGraph(TaskState)
    builder.add_node("dispatch", dispatch)
    builder.add_node("work_candidate", work_candidate)
    builder.add_node("collect", collect)
    builder.add_node("review", review_node)
    builder.add_node("integrate", integrate_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "dispatch")
    builder.add_conditional_edges("dispatch", fan_out, ["work_candidate"])
    builder.add_edge("work_candidate", "collect")
    builder.add_conditional_edges("collect", route_after_collect,
                                  ["review", "dispatch", "finalize"])
    builder.add_conditional_edges("review", route_after_review,
                                  ["integrate", "dispatch", "finalize"])
    builder.add_edge("integrate", "finalize")
    builder.add_edge("finalize", END)
    return builder
