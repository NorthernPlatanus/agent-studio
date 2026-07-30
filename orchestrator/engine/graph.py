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


def is_stuck(state: TaskState, latest: dict) -> bool:
    """What "the cheap loop is getting nowhere" means — defined ONCE.

    Three shapes, each observed as a live failure before it was counted:

      * **nothing is green.** The original definition, and still the common case.
      * **green gate, repeated `revise`.** A candidate the reviewer keeps sending
        back is just as stuck; demanding a red gate here meant
        `escalate_on_exhaustion` could never fire on a review failure, and
        gate-green/spec-incomplete code finalized `failed` at max_fix_rounds with
        the ladder configured the whole time.
      * **green gate, scene verdict rejected the winner.** Only the winner is
        inspected — one verdict per task is what makes the phase affordable — so
        at best-of-N a never-inspected loser is still `gate_passed` and the
        `not any(green)` test read the task as healthy. Same stale green that let
        a rejected winner get merged; it also kept the senior out of exactly the
        tasks it exists for.

    Both callers (`escalation_ready`'s default and `plan_dispatch`'s escalation
    branch) go through here. They MUST agree: a router that says "dispatch,
    escalating" while dispatch quietly takes the ordinary retry branch is how the
    ladder went silent the first time."""
    verdict = state.get("verdict") or {}
    if verdict.get("decision") == "revise":
        return True
    winner = latest.get(verdict.get("winner"))
    if winner is not None and winner["status"] in ("visual_failed",
                                                   "visual_unverifiable"):
        return True
    return not any(c["status"] == "gate_passed" for c in latest.values())


def escalation_ready(cfg, state: TaskState, latest: dict, finished_attempt: int,
                     *, stuck: bool | None = None) -> bool:
    """True when the cheap warm loop has exhausted max_fix_rounds without getting
    anywhere and we haven't escalated yet. Escalation is a dispatch policy branch
    (not a new node), so Send objects still originate only in fan_out. Pure.

    `stuck` is what "getting nowhere" means; leave it None and `is_stuck` decides,
    which is what every caller in the graph does. An explicit override exists only
    for callers that already know the answer (and for tests)."""
    run = cfg.run
    if not bool(run.get("escalate_on_exhaustion", False)):
        return False
    if state.get("escalated"):
        return False
    if not latest:  # nothing has run yet
        return False
    if stuck is None:
        stuck = is_stuck(state, latest)
    max_fix = int(run.get("max_fix_rounds", run.get("max_retries", 3)))
    return bool(stuck) and finished_attempt >= max_fix


def senior_rounds_left(cfg, state: TaskState) -> bool:
    """True when the escalated senior may have another attempt.

    The senior used to get exactly one shot (`escalated` -> finalize on the next
    red result). Defensible in principle, but observed dying on a trivial missing
    import — the most expensive tier getting the fewest chances, on the task the
    cheap tier already failed. `run.senior_fix_rounds` (default 1) buys it that
    many RETRIES, each warmed with the gate log through the ordinary retry branch.

    Bounded independently of `retry_ceiling`: the senior's first attempt sits
    above that ceiling by construction (escalation fires AT the ceiling), so
    reusing it here would allow zero retries however this is configured.

    Rounds spent are counted the same way `attempts_used` counts them — an
    attempt in which the senior returned prose instead of `<file>`/`<edit>`
    blocks attempted nothing and is refunded. Measuring the RAW attempt here
    meant the most expensive tier could burn its whole allowance on two
    conversational replies without ever emitting a patch, while the cheap tier
    was forgiven exactly that (run.max_unproductive_attempts, which also caps
    this, so a permanently chatty senior still terminates)."""
    rounds = int(cfg.run.get("senior_fix_rounds", 1))
    if rounds <= 0:
        return False
    first = int(state.get("escalated_attempt") or 0)
    if not first:
        # No escalation attempt recorded (a checkpoint written before this field
        # existed, or senior_first, which deliberately skips the once-guard).
        # Fall back to the old one-shot behavior rather than guessing.
        return False
    spent = int(state.get("attempt", first)) - first
    return (spent - unproductive_attempts(cfg, state, since=first)) < rounds


def unproductive_attempts(cfg, state: TaskState, since: int = 0) -> int:
    """Attempts in which EVERY dispatched candidate produced no patch at all,
    capped by `run.max_unproductive_attempts`.

    A worker that answers a `revise` with prose instead of `<file>`/`<edit>`
    blocks has not attempted the fix (observed twice in a row on the review
    path), so charging it a fix round spends the ladder on nothing and forces a
    premature escalation. Those attempts are forgiven here — but only up to the
    cap, because a model that ALWAYS replies conversationally would otherwise
    retry forever, and an unbounded loop is worse than a wasted round.

    A mixed attempt (one candidate patched, another rambled) is productive: the
    ceiling counts rounds of work, and work happened.

    `since` restricts the count to attempts >= that number, for a ceiling that
    starts mid-task: `senior_rounds_left` must forgive the SENIOR's prose without
    also handing it the cheap tier's refunds, which `attempts_used` already
    spent. The cap applies to whatever window is asked for, so each ceiling is
    bounded on its own."""
    cap = int(cfg.run.get("max_unproductive_attempts", 2))
    if cap <= 0:
        return 0
    by_attempt: dict[int, list] = {}
    for cand in state.get("candidates", []):
        if cand["attempt"] < since:
            continue
        by_attempt.setdefault(cand["attempt"], []).append(cand)
    free = sum(1 for cands in by_attempt.values()
               if cands and all(c.get("no_patch") for c in cands))
    return min(free, cap)


def attempts_used(cfg, state: TaskState) -> int:
    """The attempt count every retry ceiling is measured against: attempts made,
    minus the (bounded) ones where nothing was actually attempted."""
    return max(0, int(state.get("attempt", 0)) - unproductive_attempts(cfg, state))


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
    # `blocked_reason` is cleared with the verdict: both describe a judgment about
    # the attempt that just finished, and a stale one would put the previous
    # round's scene findings on the backlog next to an unrelated later failure.
    update: dict = {"attempt": attempt, "verdict": {}, "blocked_reason": ""}

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
    elif escalation_ready(cfg, state, latest, attempts_used(cfg, state)):
        # Rung 3: hand the stuck task to the subscription senior. The senior
        # implements via the same patch->gate channel (no repo edits); writes stay
        # locked to that channel. Set the once-guard here (routers stay pure), plus
        # the attempt it started on so senior_rounds_left can bound its retries.
        update["to_run"] = ["senior"]
        update["escalated"] = True
        update["escalated_attempt"] = attempt
        if verdict.get("decision") == "revise":
            # Escalating out of a REVIEW deadlock: the gate is green, so a gate log
            # would say nothing. What the senior needs is the objection it has to
            # answer.
            update["feedback"] = (
                "ESCALATED to senior — the cheaper workers passed the gate but "
                "could not satisfy the reviewer. Latest review notes:\n"
                + str(verdict.get("notes", "")))
        else:
            update["feedback"] = ("ESCALATED to senior — the cheaper workers could "
                                  "not pass the gate. Prior failures:\n" + _logs(latest))
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


def apply_scene_verdict(cand: dict, verdict: dict) -> dict:
    """Pure: winning candidate + scene verdict -> the state update. Three outcomes.

    * **ok** — carry any measured facts onto the candidate (still `gate_passed`).
    * **unverifiable** — the verifier could not OBSERVE a criterion. Structural,
      not a defect: it attaches to an already-running app with read-only tools, so
      startup / transition / one-shot state is out of reach by construction and no
      retry can change that. Observed burning ~797k subscription tokens on two
      ~400k verdicts about code the reviewer had already approved twice. Ends the
      task as `needs_human` with the unmeasured criteria named.
    * **rejected** — measured and wrong. `visual_failed`, i.e. not green, so it
      rejoins the ordinary retry ladder like any red result.
    """
    nc = dict(cand)
    if verdict["ok"]:
        if verdict.get("facts"):
            nc["visual_facts"] = {**(cand.get("visual_facts") or {}),
                                  **verdict["facts"]}
        return {"candidates": [nc]}
    if verdict.get("unverifiable"):
        unmeasured = "; ".join(verdict["unverifiable"])
        nc["status"] = "visual_unverifiable"
        nc["gate_log"] = (
            "SCENE VERDICT UNVERIFIABLE (the running app cannot show this):\n"
            + "\n".join(verdict["unverifiable"])
            + "\n\nAlso reported:\n" + "\n".join(verdict.get("findings") or []))
        return {"candidates": [nc], "outcome": "needs_human",
                "blocked_reason": f"scene verdict could not observe: {unmeasured}"[:500]}
    nc["status"] = "visual_failed"
    findings = verdict.get("findings") or []
    nc["gate_log"] = ("SCENE VERDICT REJECTED (inspected the running app):\n"
                      + "\n".join(findings))
    # Also record WHY for the human backlog. Without it a task that exhausts its
    # retries after a scene rejection wrote `agent run failed — needs human` to the
    # board: the verifier's measured findings survived only in the candidate's
    # gate_log and the event log, i.e. nowhere the human actually looks. Safe to
    # set on a non-terminal outcome because `plan_dispatch` clears it when the next
    # attempt starts, so it always describes the judgment that ended the task.
    return {"candidates": [nc],
            "blocked_reason": ("scene verdict rejected: "
                               + "; ".join(findings))[:500]}


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
        # The senior tried and failed. It gets run.senior_fix_rounds retries (the
        # ordinary retry branch, warmed with the gate log) before this becomes a
        # human's problem.
        return "dispatch" if senior_rounds_left(cfg, state) else "finalize"
    used = attempts_used(cfg, state)
    if escalation_ready(cfg, state, latest, used):
        return "dispatch"        # dispatch will take the escalation branch
    if used < retry_ceiling(cfg):
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
    if decision == "revise":
        if state.get("escalated"):
            return "dispatch" if senior_rounds_left(cfg, state) else "finalize"
        used = attempts_used(cfg, state)
        if used < retry_ceiling(cfg):
            return "dispatch"
        # Out of fix rounds on a GREEN gate. Falling through to finalize here is
        # what made escalate_on_exhaustion unreachable for review failures: the
        # ladder is for a task the cheap tier cannot finish, and "passes the gate,
        # never satisfies the reviewer" is that task. `is_stuck` says so (this
        # decision is not repeated here — dispatch has to reach the same verdict
        # or the escalation branch is never taken).
        if escalation_ready(cfg, state, latest_candidates(state), used):
            return "dispatch"    # dispatch takes the escalation branch
    return "finalize"


def decide_after_verify(cfg, state: TaskState) -> str:
    """Router after the scene verdict. A rejected candidate is `visual_failed`,
    i.e. not green, so it rejoins the ordinary retry/escalate/finalize path — the
    same treatment a failed visual gate gets, and no new machinery.

    `visual_unverifiable` is the exception: the criterion cannot be observed by a
    read-only verifier that attaches to a running app, so retrying buys another
    ~400k-token verdict with the same answer. Straight to finalize (as
    needs_human), where the human gets the unmeasured criteria.

    Routed on the WINNER's status, not on `any(gate_passed)`. Only the winner is
    inspected (one verdict per task is what makes this affordable), and review
    does not demote the candidates it passed over — so at best-of-N a loser sits
    there still `gate_passed` and an `any` test read it as a pass. The task then
    went to `integrate`, which merges `latest[verdict["winner"]]`: the very
    candidate the scene verdict had just rejected or could not observe, with the
    `needs_human` outcome overwritten by `done` and a "merged" note on the
    backlog. Reachable with `--n` alone. The unverifiable case ends the task
    regardless of N — it is the CRITERION that cannot be observed, not the
    candidate, so no sibling can do better."""
    latest = latest_candidates(state)
    winner = latest.get((state.get("verdict") or {}).get("winner"))
    if winner is not None:
        if winner["status"] == "gate_passed":
            return "integrate"
        if winner["status"] == "visual_unverifiable":
            return "finalize"
    else:
        # No winner on record. `verify` is only reachable through a review
        # verdict, so this is a checkpoint predating one (or the auto-integrate
        # shape, which a visual spec never takes). Keep the old whole-set
        # behavior — there is no better signal — but still treat unverifiable as
        # terminal, since no retry can observe what no tool can reach.
        if any(c["status"] == "visual_unverifiable" for c in latest.values()):
            return "finalize"
        if any(c["status"] == "gate_passed" for c in latest.values()):
            return "integrate"
    # Measured and wrong (or nothing green to inspect): back onto the ordinary
    # ladder. dispatch's retry branch picks up exactly the non-green candidates,
    # i.e. this winner, and a green sibling stays available for review to select
    # next time round.
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
        update = apply_scene_verdict(cand, verdict)
        if update.get("outcome") == "needs_human":
            unmeasured = "; ".join(verdict["unverifiable"])
            log.warning("%s verify: %s reported unverifiable criteria — no retry "
                        "can observe them: %s", state["task_id"], winner_id,
                        unmeasured)
            ctx.store.log_event(ctx.run_id, state["task_id"], "verify_unverifiable",
                                f"{winner_id} {unmeasured}"[:2000])
        return update

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
