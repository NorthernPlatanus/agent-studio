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

import asyncio
import logging
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ..core.context import RunContext
from ..nodes.integrator import finalize, integrate
from ..nodes.reviewer import review
from ..nodes.worker import run_candidate
from ..ops import visualgate
from ..core.state import TaskState, latest_candidates

log = logging.getLogger("orchestrator.graph")


def escalation_ready(cfg, state: TaskState, latest: dict, finished_attempt: int) -> bool:
    """True when the cheap warm loop has exhausted max_fix_rounds without a green
    gate and we haven't escalated yet. Escalation is a dispatch policy branch (not
    a new node), so Send objects still originate only in fan_out. Pure."""
    run = cfg.run
    if not bool(run.get("escalate_on_exhaustion", False)):
        return False
    if state.get("escalated"):
        return False
    if not latest:  # nothing has run yet
        return False
    green = any(c["status"] == "gate_passed" for c in latest.values())
    max_fix = int(run.get("max_fix_rounds", run.get("max_retries", 3)))
    return (not green) and finished_attempt >= max_fix


def resolve_worker_pool(cfg, spec: dict | None) -> tuple[str, list[str]]:
    """(default_key, candidate_pool) for a spec, honoring a `domain` override.

    Routing hooks HERE (dispatch's pool selection), not worker_target — the spec
    is in state at dispatch, whereas worker_target only maps a key to
    (provider, model) and never sees the spec. The scheduler still enforces
    files_write disjointness regardless of domain."""
    default = cfg.roles.worker.default
    pool = list(cfg.roles.worker.candidates or [])
    domain = (spec or {}).get("domain")
    domains = cfg.get("domains") or {}
    dcfg = domains.get(domain) if domain else None
    if dcfg is not None:
        dd = dcfg.get("worker_default")
        dcands = dcfg.get("candidates")
        if dd:
            default = dd
        if dcands:
            pool = list(dcands)
        elif dd:
            pool = [dd]
    return default, pool


def plan_dispatch(cfg, state: TaskState) -> dict:
    """Pure policy: given the current state, decide the next attempt's to_run +
    feedback (+ escalated once-guard). No side effects — the dispatch node adds
    store/logging around this."""
    attempt = state.get("attempt", 0) + 1
    latest = latest_candidates(state)
    verdict = state.get("verdict") or {}
    update: dict = {"attempt": attempt, "verdict": {}}

    def _logs(cands: dict) -> str:
        return "\n\n".join(
            f"[{cid}] {c['status']}:\n{c.get('gate_log') or c.get('error')}"
            for cid, c in cands.items())[:int(cfg.gate.log_tail_chars)]

    if attempt == 1:
        n = state.get("n_candidates") or int(cfg.run.n_candidates)
        default, pool = resolve_worker_pool(cfg, state.get("spec"))
        to_run = [default] if n <= 1 else (pool or [default])[:n]
        update["to_run"] = to_run
        update["feedback"] = ""
    elif escalation_ready(cfg, state, latest, attempt - 1):
        # Rung 3: hand the still-red task to the subscription senior. The senior
        # implements via the same patch->gate channel (no repo edits); writes stay
        # locked to that channel. Set the once-guard here (routers stay pure).
        update["to_run"] = ["senior"]
        update["escalated"] = True
        update["feedback"] = ("ESCALATED to senior — the cheaper workers could not "
                              "pass the gate. Prior failures:\n" + _logs(latest))
    elif verdict.get("decision") == "revise":
        update["to_run"] = [verdict["winner"]]
        update["feedback"] = f"REVIEW NOTES:\n{verdict.get('notes', '')}"
    else:
        # Gate/patch failures: retry every failed candidate with its log.
        failed = {cid: c for cid, c in latest.items()
                  if c["status"] != "gate_passed"}
        update["to_run"] = list(failed)
        update["feedback"] = _logs(failed)
    return update


def visual_needed(cfg, state: TaskState) -> bool:
    """A green gate needs the visual gate iff it's enabled AND the spec is
    flagged visual. Otherwise the visual node is skipped entirely."""
    vg = cfg.get("visual_gate") or {}
    enabled = vg.get("enabled", False) if hasattr(vg, "get") else False
    return bool(enabled) and bool((state.get("spec") or {}).get("visual"))


def _decide_when_not_green(cfg, state: TaskState, latest: dict) -> str:
    """Shared not-green tail. `visual_failed` candidates are not `gate_passed`,
    so they are treated as red here (and picked up by dispatch's retry branch —
    no empty-Send livelock)."""
    if state.get("escalated"):
        return "finalize"        # the senior already tried and failed -> human
    if escalation_ready(cfg, state, latest, state["attempt"]):
        return "dispatch"        # dispatch will take the escalation branch
    # Cheap-loop retry ceiling. With the escalation ladder ON, the warm cheap loop
    # is bounded by max_fix_rounds (the documented "fix depth before escalation");
    # escalation_ready fires at attempt >= max_fix_rounds just above. Bounding the
    # retry here by max_retries instead would finalize a red task at
    # attempt == max_retries BEFORE the escalation trigger could fire whenever
    # max_fix_rounds > max_retries (the shipped default is 4 > 3), silently
    # disabling the whole ladder. With the ladder OFF, the legacy max_retries
    # bound applies unchanged.
    run = cfg.run
    if bool(run.get("escalate_on_exhaustion", False)):
        ceiling = int(run.get("max_fix_rounds", run.get("max_retries", 3)))
    else:
        ceiling = int(run.max_retries)
    if state["attempt"] < ceiling:
        return "dispatch"
    return "finalize"


def decide_after_collect(cfg, state: TaskState) -> str:
    """Pure router for the collect barrier. A green + visual spec goes to the
    visual gate first; escalation is checked BEFORE max_retries -> finalize."""
    latest = latest_candidates(state)
    if any(c["status"] == "gate_passed" for c in latest.values()):
        return "visual_gate" if visual_needed(cfg, state) else "review"
    return _decide_when_not_green(cfg, state, latest)


def decide_after_visual(cfg, state: TaskState) -> str:
    """Router after the visual gate. Candidates that passed assertions are still
    `gate_passed` -> review; if all green candidates became `visual_failed`,
    retry/escalate/finalize like any red result (bounded by max_retries)."""
    latest = latest_candidates(state)
    if any(c["status"] == "gate_passed" for c in latest.values()):
        return "review"
    return _decide_when_not_green(cfg, state, latest)


def build_task_graph(ctx: RunContext):
    async def dispatch(state: TaskState) -> dict:
        """Decide which candidates to (re)run and with what feedback."""
        update = plan_dispatch(ctx.cfg, state)
        ctx.store.set_task_status(state["task_id"], "running")
        log.info("%s dispatch attempt %d -> %s",
                 state["task_id"], update["attempt"], update["to_run"])
        return update

    def fan_out(state: TaskState) -> list[Send]:
        latest = latest_candidates(state)
        return [
            Send("work_candidate", {
                "run_id": state["run_id"],
                "task_id": state["task_id"],
                "spec": state["spec"],
                "cand_id": cid,
                "attempt": state["attempt"],
                "feedback": state.get("feedback", ""),
                # Warm per-candidate history (empty for a fresh candidate/senior).
                "messages": latest.get(cid, {}).get("messages", []),
            })
            for cid in state.get("to_run", [])
        ]

    async def work_candidate(payload: dict) -> dict:
        return await run_candidate(ctx, payload)

    async def collect(state: TaskState) -> dict:
        return {}  # barrier: all Send branches joined here

    def route_after_collect(state: TaskState) -> str:
        return decide_after_collect(ctx.cfg, state)

    async def visual_gate_node(state: TaskState) -> dict:
        """Assert over each green candidate's running scene. On failure, append
        an updated Candidate with status `visual_failed` (via the reducer) so the
        existing dispatch retry branch picks it up — no empty-Send livelock."""
        latest = latest_candidates(state)
        green = {cid: c for cid, c in latest.items() if c["status"] == "gate_passed"}
        updated: list = []
        for cid, cand in green.items():
            try:
                res = await asyncio.to_thread(
                    visualgate.check, ctx.cfg, Path(cand["worktree"]))
            except Exception as e:   # a broken inspector must not wedge the run
                log.warning("%s visual_gate error for %s: %s",
                            state["task_id"], cid, e)
                continue
            # `visual_gate_skipped` (distinct kind, visible in status/events) makes
            # a blind pass-through auditable: the gate is enabled but no inspector
            # is wired, so the candidate passed WITHOUT any assertion running —
            # never let that masquerade as a real visual pass.
            kind = "visual_gate" if res.enforced else "visual_gate_skipped"
            ctx.store.log_event(ctx.run_id, state["task_id"], kind,
                                f"{cid} enforced={res.enforced} passed={res.passed} "
                                f"failures={res.failures}")
            if not res.passed:
                nc = dict(cand)
                nc["status"] = "visual_failed"
                nc["gate_log"] = "VISUAL GATE FAILED:\n" + "\n".join(res.failures)
                updated.append(nc)
        return {"candidates": updated}

    def route_after_visual(state: TaskState) -> str:
        return decide_after_visual(ctx.cfg, state)

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
    builder.add_node("visual_gate", visual_gate_node)

    builder.add_edge(START, "dispatch")
    builder.add_conditional_edges("dispatch", fan_out, ["work_candidate"])
    builder.add_edge("work_candidate", "collect")
    builder.add_conditional_edges("collect", route_after_collect,
                                  ["review", "dispatch", "finalize", "visual_gate"])
    builder.add_conditional_edges("visual_gate", route_after_visual,
                                  ["review", "dispatch", "finalize"])
    builder.add_conditional_edges("review", route_after_review,
                                  ["integrate", "dispatch", "finalize"])
    builder.add_edge("integrate", "finalize")
    builder.add_edge("finalize", END)
    return builder
