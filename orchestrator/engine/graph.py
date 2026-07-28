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
                                    verify ──> integrate ──> finalize <────────────┘
                              (visual specs only,       ^
                               serialized, inspects     │ scene verdict ok
                               the RUNNING app)         │
                                     └── rejected ──> dispatch / finalize

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
from ..ops import verifier, visualgate
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


def senior_first(cfg, spec: dict | None) -> bool:
    """True when a spec the planner already flagged as hard should go straight to
    the subscription senior instead of burning the cheap ladder first.

    The planner writes `risk` and `complexity` into every spec and, until now,
    nothing read either field. The escalation ladder only fires AFTER the cheap
    tier has failed max_fix_rounds times — for a task already marked high-risk
    that is several wasted rounds, each with a full gate run. Off by default
    (run.senior_first_for_high_risk); needs the ladder configured, since the
    senior is the ladder's target."""
    run = cfg.run
    if not bool(run.get("senior_first_for_high_risk", False)):
        return False
    if not bool(run.get("escalate_on_exhaustion", False)):
        return False              # no senior target configured
    spec = spec or {}
    return spec.get("risk") == "high" or spec.get("complexity") == "l"


def auto_integrate(cfg, spec: dict | None, latest: dict) -> bool:
    """True when a green single-candidate low-risk task may skip the reviewer.

    Saves one smart-tier call on the majority of tasks, which is the scarce
    resource. Deliberately narrow: exactly one candidate (nothing to select
    between), the gate is green, and the planner called the task low-risk. Off by
    default (run.auto_integrate_low_risk) — it trades a review for quota, and
    that is the user's call, not a default."""
    if not bool(cfg.run.get("auto_integrate_low_risk", False)):
        return False
    if (spec or {}).get("risk") != "low":
        return False
    green = [c for c in latest.values() if c["status"] == "gate_passed"]
    return len(green) == 1 and len(latest) == 1


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
        spec = state.get("spec")
        if senior_first(cfg, spec):
            # Straight to the senior — but WITHOUT the `escalated` once-guard, so
            # the task still gets the normal retry ladder rather than one shot.
            update["to_run"] = ["senior"]
            update["feedback"] = ""
        else:
            n = state.get("n_candidates") or int(cfg.run.n_candidates)
            default, pool = resolve_worker_pool(cfg, spec)
            update["to_run"] = [default] if n <= 1 else (pool or [default])[:n]
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


def retry_ceiling(cfg) -> int:
    """The single attempt ceiling for a task, used by EVERY retry router.

    With the escalation ladder ON, the warm cheap loop is bounded by
    max_fix_rounds (the documented "fix depth before escalation") — bounding it by
    the smaller max_retries would finalize a red task at attempt == max_retries
    BEFORE escalation_ready could fire whenever max_fix_rounds > max_retries (the
    shipped default is 4 > 3), silently disabling the whole ladder. With the
    ladder OFF, the legacy max_retries bound applies unchanged.

    This lives in one function because the gate path and the review path had
    drifted apart: a review-driven `revise` gave up one attempt earlier than an
    identical gate failure, and could never reach the senior."""
    run = cfg.run
    if bool(run.get("escalate_on_exhaustion", False)):
        return int(run.get("max_fix_rounds", run.get("max_retries", 3)))
    return int(run.max_retries)


def _decide_when_not_green(cfg, state: TaskState, latest: dict) -> str:
    """Shared not-green tail. `visual_failed` candidates are not `gate_passed`,
    so they are treated as red here (and picked up by dispatch's retry branch —
    no empty-Send livelock)."""
    if state.get("escalated"):
        return "finalize"        # the senior already tried and failed -> human
    if escalation_ready(cfg, state, latest, state["attempt"]):
        return "dispatch"        # dispatch will take the escalation branch
    if state["attempt"] < retry_ceiling(cfg):
        return "dispatch"
    return "finalize"


def decide_after_collect(cfg, state: TaskState) -> str:
    """Pure router for the collect barrier. A green + visual spec goes to the
    visual gate first; escalation is checked BEFORE max_retries -> finalize."""
    latest = latest_candidates(state)
    if any(c["status"] == "gate_passed" for c in latest.values()):
        if visual_needed(cfg, state):
            return "visual_gate"     # a visual spec is never auto-integrated
        return "integrate" if auto_integrate(cfg, state.get("spec"), latest) else "review"
    return _decide_when_not_green(cfg, state, latest)


def decide_after_visual(cfg, state: TaskState) -> str:
    """Router after the visual gate. Candidates that passed assertions are still
    `gate_passed` -> review; if all green candidates became `visual_failed`,
    retry/escalate/finalize like any red result (bounded by retry_ceiling)."""
    latest = latest_candidates(state)
    if any(c["status"] == "gate_passed" for c in latest.values()):
        return "review"
    return _decide_when_not_green(cfg, state, latest)


def decide_after_review(cfg, state: TaskState) -> str:
    """Pure router for the review verdict. A `revise` is a retry like any other,
    so it shares the one `retry_ceiling` with the gate path.

    An approved VISUAL spec goes to `verify` first: an approving reviewer has read
    a diff, which is not evidence about what the scene actually renders."""
    decision = state["verdict"]["decision"]
    if decision == "approve":
        return "verify" if verifier.verify_needed(cfg, state.get("spec")) \
            else "integrate"
    if decision == "revise" and state["attempt"] < retry_ceiling(cfg):
        return "dispatch"
    return "finalize"


def decide_after_verify(cfg, state: TaskState) -> str:
    """Router after the scene verdict. A rejected candidate is `visual_failed`,
    i.e. not green, so it rejoins the ordinary retry/escalate/finalize path — the
    same treatment a failed visual gate gets, and no new machinery."""
    latest = latest_candidates(state)
    if any(c["status"] == "gate_passed" for c in latest.values()):
        return "integrate"
    return _decide_when_not_green(cfg, state, latest)


def build_task_graph(ctx: RunContext):
    async def dispatch(state: TaskState) -> dict:
        """Decide which candidates to (re)run and with what feedback."""
        update = plan_dispatch(ctx.cfg, state)
        ctx.store.set_task_status(state["task_id"], "running")
        if update.get("escalated"):
            # Escalation frequency is one of the numbers that decides whether the
            # cheap tier earns its place, so it belongs in `events`, not only in
            # graph state where no query can reach it.
            ctx.store.log_event(ctx.run_id, state["task_id"], "escalated",
                                f"attempt={update['attempt']} after cheap-loop "
                                f"exhaustion")
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
            except Exception as e:
                # A broken inspector must not wedge the run — but it must not
                # look like a pass either. `continue` here left the candidate
                # `gate_passed`, making a crash indistinguishable from a green
                # scene. Fail the candidate and log a DISTINCT event kind.
                log.warning("%s visual_gate error for %s: %s",
                            state["task_id"], cid, e)
                ctx.store.log_event(ctx.run_id, state["task_id"],
                                    "visual_gate_error", f"{cid} {e}"[:2000])
                nc = dict(cand)
                nc["status"] = "visual_failed"
                nc["gate_log"] = f"VISUAL GATE ERROR (inspector crashed):\n{e}"
                updated.append(nc)
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
            elif res.enforced and res.facts:
                # A PASS used to discard res.facts entirely, so the measured
                # scene never reached the reviewer — which can look at nothing
                # itself and would otherwise grade a visual change on the diff
                # alone. Carry the facts on the candidate (same attempt, still
                # gate_passed, so latest_candidates just sees an enriched entry).
                nc = dict(cand)
                nc["visual_facts"] = res.facts
                updated.append(nc)
        return {"candidates": updated}

    def route_after_visual(state: TaskState) -> str:
        return decide_after_visual(ctx.cfg, state)

    def route_after_review(state: TaskState) -> str:
        return decide_after_review(ctx.cfg, state)

    async def verify_node(state: TaskState) -> dict:
        """Scene verdict on the WINNING candidate only, serialized.

        Runs once per task rather than per review round: the inspector exposes
        ~60 tools whose schemas are re-sent every turn, which measured ~6x the
        input tokens of a plain call. The lock is held across app startup, the
        agent call and teardown, because the dev server and bridge are bound to
        fixed ports — two concurrent verifications would inspect each other's app.
        """
        latest = latest_candidates(state)
        winner_id = (state.get("verdict") or {}).get("winner")
        cand = latest.get(winner_id) if winner_id else None
        if cand is None or cand.get("status") != "gate_passed":
            # Nothing green to look at; the ordinary not-green path handles it.
            return {}
        async with ctx.inspector_lock:
            try:
                verdict = await verifier.run_verification(ctx, state["spec"], cand)
            except Exception as e:
                # A broken inspector must never read as a clean scene — same rule
                # the visual gate learned. Fail the candidate, log distinctly.
                log.warning("%s verify error for %s: %s", state["task_id"],
                            winner_id, e)
                ctx.store.log_event(ctx.run_id, state["task_id"], "verify_error",
                                    f"{winner_id} {e}"[:2000])
                verdict = {"ok": False,
                           "findings": [f"verifier crashed: {e}"]}
        ctx.store.log_event(
            ctx.run_id, state["task_id"], "verify",
            f"{winner_id} ok={verdict['ok']} findings={verdict['findings']}"[:2000])
        if verdict["ok"]:
            nc = dict(cand)
            if verdict.get("facts"):
                nc["visual_facts"] = {**(cand.get("visual_facts") or {}),
                                      **verdict["facts"]}
            return {"candidates": [nc]}
        nc = dict(cand)
        nc["status"] = "visual_failed"
        nc["gate_log"] = ("SCENE VERDICT REJECTED (inspected the running app):\n"
                          + "\n".join(verdict["findings"]))
        return {"candidates": [nc]}

    def route_after_verify(state: TaskState) -> str:
        return decide_after_verify(ctx.cfg, state)

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
    builder.add_node("verify", verify_node)

    builder.add_edge(START, "dispatch")
    builder.add_conditional_edges("dispatch", fan_out, ["work_candidate"])
    builder.add_edge("work_candidate", "collect")
    builder.add_conditional_edges("collect", route_after_collect,
                                  ["review", "dispatch", "finalize", "visual_gate",
                                   "integrate"])
    builder.add_conditional_edges("visual_gate", route_after_visual,
                                  ["review", "dispatch", "finalize"])
    builder.add_conditional_edges("review", route_after_review,
                                  ["integrate", "dispatch", "finalize", "verify"])
    builder.add_conditional_edges("verify", route_after_verify,
                                  ["integrate", "dispatch", "finalize"])
    builder.add_edge("integrate", "finalize")
    builder.add_edge("finalize", END)
    return builder
