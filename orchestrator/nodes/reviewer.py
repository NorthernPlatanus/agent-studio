"""Reviewer node — expensive judgment, invoked ONLY after a green gate.

Sees the task spec + diff(s). Does not see the queue, the worktrees, or other
tasks (narrow context applies to Opus too). For best-of-N it selects a winner
among gate-passing candidates.
"""

from __future__ import annotations

import json
import logging
import re

from ..core.context import RunContext
from ..providers import get_provider
from ..core.state import TaskState, latest_candidates

log = logging.getLogger("orchestrator.reviewer")

JSON_RE = re.compile(r"\{.*\}", re.S)


def _parse_verdict(text: str) -> dict:
    m = JSON_RE.search(text)
    if not m:
        return {"decision": "revise", "notes": f"Reviewer returned no JSON: {text[:500]}"}
    try:
        verdict = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"decision": "revise", "notes": f"Reviewer returned bad JSON: {text[:500]}"}
    if verdict.get("decision") not in ("approve", "revise", "reject"):
        verdict["decision"] = "revise"
    return verdict


async def review(ctx: RunContext, state: TaskState) -> dict:
    spec = state["spec"]
    passed = {cid: c for cid, c in latest_candidates(state).items()
              if c["status"] == "gate_passed"}
    provider_name, model = ctx.role_target("reviewer")
    provider = get_provider(ctx.cfg, provider_name)

    if len(passed) == 1:
        prompt_name = "reviewer"
    else:
        prompt_name = "reviewer_select"
    system = ctx.cfg.prompt(prompt_name)

    parts = [f"# TASK {spec['id']}: {spec['title']}", "", spec["description"], ""]
    if spec.get("acceptance"):
        parts += ["## Acceptance criteria", *[f"- {a}" for a in spec["acceptance"]], ""]
    for cid, cand in passed.items():
        parts += [f"## Candidate `{cid}` (model {cand['model']}) — gate GREEN",
                  "Worker notes: " + (cand.get("notes") or "-"),
                  "```diff", cand["diff"], "```", ""]

    result = await provider.complete(model=model, system=system,
                                     user="\n".join(parts),
                                     cwd=str(ctx.cfg.repo_path()))
    ctx.budget.record(
        task_id=state["task_id"], role="reviewer", provider=provider_name,
        provider_type=provider.type, model=model,
        input_tokens=result.input_tokens, output_tokens=result.output_tokens,
        cost_usd=result.cost_usd)

    verdict = _parse_verdict(result.text)
    if "winner" not in verdict or verdict["winner"] not in passed:
        verdict["winner"] = next(iter(passed))
    ctx.store.log_event(ctx.run_id, state["task_id"], "review",
                        json.dumps(verdict)[:2000])
    log.info("%s review: %s (winner=%s)", state["task_id"],
             verdict["decision"], verdict["winner"])
    return {"verdict": verdict}
