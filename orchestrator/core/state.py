"""LangGraph state schemas — plain, checkpoint-serializable data only.

Runtime services (config, store, git, providers) are NOT part of state; they
are bound into nodes via the RunContext closure (see graph.py). This keeps
checkpoints portable and resumable.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict

Candidate = dict[str, Any]
# Candidate keys:
#   cand_id: str        worker_models key ("deepseek"), unique per candidate
#   model: str          resolved model id
#   attempt: int        1-based; retries produce new entries (latest wins)
#   status: str         gate_passed | gate_failed | patch_failed | llm_failed | skipped
#   worktree: str       path
#   branch: str         git branch
#   diff: str           unified diff vs feature branch (on gate_passed)
#   gate_log: str       failure tail (on gate_failed)
#   error: str          patch/llm error text
#   notes: str          worker's own plan/notes


class TaskState(TypedDict, total=False):
    run_id: str
    task_id: str
    spec: dict            # full planner spec
    n_candidates: int
    attempt: int          # current attempt number (1-based)
    to_run: list[str]     # candidate ids to (re)run this attempt
    feedback: str         # gate/review notes injected into the next attempt
    candidates: Annotated[list[Candidate], operator.add]
    verdict: dict         # {"decision", "winner", "notes"}
    integration: dict     # {"merged_commit"}
    outcome: str          # done | failed | rejected


def latest_candidates(state: TaskState) -> dict[str, Candidate]:
    """The reducer appends across attempts; collapse to newest per cand_id."""
    latest: dict[str, Candidate] = {}
    for cand in state.get("candidates", []):
        prev = latest.get(cand["cand_id"])
        if prev is None or cand["attempt"] >= prev["attempt"]:
            latest[cand["cand_id"]] = cand
    return latest
