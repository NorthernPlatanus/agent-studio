"""LangGraph state schemas — plain, checkpoint-serializable data only.

Runtime services (config, store, git, providers) are NOT part of state; they
are bound into nodes via the RunContext closure (see graph.py). This keeps
checkpoints portable and resumable.
"""

from __future__ import annotations

from typing import Annotated, Any

from typing_extensions import TypedDict

Candidate = dict[str, Any]
# Candidate keys:
#   cand_id: str        worker_models key ("deepseek"), unique per candidate
#   model: str          resolved model id
#   attempt: int        1-based; retries produce new entries (latest wins)
#   status: str         gate_passed | gate_failed | patch_failed | llm_failed
#                       | visual_failed | visual_unverifiable | skipped
#                       (visual_failed = passed the deterministic gate but failed
#                       the visual/runtime assertions; treated as not-green so the
#                       dispatch retry branch reruns it — no empty-Send livelock.
#                       visual_unverifiable = the scene verdict could not OBSERVE a
#                       criterion (startup/transition/one-shot state a read-only
#                       verifier attaching to a running app cannot reach). Also not
#                       green, but retrying cannot help, so it routes straight to
#                       finalize as needs_human.)
#   worktree: str       path
#   branch: str         git branch
#   diff: str           unified diff vs feature branch (on gate_passed)
#   gate_log: str       failure tail (on gate_failed)
#   error: str          patch/llm error text
#   no_patch: bool      set when the response contained NO <file>/<edit> blocks at
#                       all (prose, a question, a summary). Nothing was attempted,
#                       so engine/graph.unproductive_attempts refunds the fix round
#                       — bounded by run.max_unproductive_attempts.
#   notes: str          worker's own plan/notes
#   visual_facts: dict  measured scene facts from an ENFORCED visual gate pass
#                       (absent when the gate is off, skipped, or produced none).
#                       Fed into the review payload: the reviewer cannot observe
#                       anything itself, so these are the only evidence it gets
#                       about the running scene rather than the diff's claims.
#   messages: list[dict] per-CANDIDATE warm chat history (plain role/content
#                        dicts only — never LangChain message objects, or the
#                        checkpoint breaks). Flows through the add_candidates
#                        reducer, which keeps the transcript ONLY on the newest
#                        attempt; latest_candidates() gives the warm chain per
#                        candidate. This is per-candidate on purpose: a single
#                        task-level add_messages channel would interleave all N
#                        candidates' turns into one history.


def add_candidates(left: list[Candidate] | None,
                   right: list[Candidate] | None) -> list[Candidate]:
    """Reducer for `candidates`: append, then keep `messages` only on the newest
    attempt per candidate id.

    Plain `operator.add` stored a COMPLETE copy of every attempt's chat — and the
    first turn of that chat is the frozen prefix with every files_read file
    inlined. A task with a 60k-token file block, 3 candidates and 4 attempts
    checkpointed ~12 copies of it, so the SQLite checkpoint (and every
    `aget_state` on resume) grew roughly quadratically in attempts.

    Nothing reads a superseded attempt's messages: `fan_out` warms a retry from
    `latest_candidates`, which already collapses to the newest per cand_id. The
    superseded entries keep status / gate_log / error / diff, so the audit trail
    of what each attempt did is intact — only the redundant transcript goes.

    Supersession matches `latest_candidates` exactly (`attempt >=`, so a
    same-attempt `visual_failed` supersedes the `gate_passed` it replaces); the
    LAST entry at the winning attempt is the one that keeps its messages.
    """
    merged = list(left or []) + list(right or [])
    winner: dict[str, int] = {}
    for i, cand in enumerate(merged):
        cid = cand["cand_id"]
        prev = winner.get(cid)
        if prev is None or cand["attempt"] >= merged[prev]["attempt"]:
            winner[cid] = i
    keep = set(winner.values())
    # Idempotent: an already-stripped entry has a falsy `messages` and is
    # returned as-is, so repeated reducer calls copy nothing.
    return [cand if i in keep or not cand.get("messages")
            else {**cand, "messages": []}
            for i, cand in enumerate(merged)]


class TaskState(TypedDict, total=False):
    run_id: str
    task_id: str
    spec: dict            # full planner spec
    n_candidates: int
    attempt: int          # current attempt number (1-based)
    to_run: list[str]     # candidate ids to (re)run this attempt
    feedback: str         # gate/review notes injected into the next attempt
    escalated: bool       # set once when a task hands off to the subscription senior
    escalated_attempt: int  # the attempt the senior started on; bounds its retries
                            # (run.senior_fix_rounds) without touching retry_ceiling
    candidates: Annotated[list[Candidate], add_candidates]
    verdict: dict         # {"decision", "winner", "notes"}
    integration: dict     # {"merged_commit"}
    outcome: str          # done | failed | rejected | needs_human
    blocked_reason: str   # why a non-done outcome needs a human, for the backlog note


def latest_candidates(state: TaskState) -> dict[str, Candidate]:
    """The reducer appends across attempts; collapse to newest per cand_id."""
    latest: dict[str, Candidate] = {}
    for cand in state.get("candidates", []):
        prev = latest.get(cand["cand_id"])
        if prev is None or cand["attempt"] >= prev["attempt"]:
            latest[cand["cand_id"]] = cand
    return latest
