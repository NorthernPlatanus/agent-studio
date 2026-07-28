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

# Rubric dimensions and weights (acceptance dominates). Winner = max weighted.
RUBRIC_DIMS = ("acceptance", "tests", "minimality", "protocol_fit")
_WEIGHTS = {"acceptance": 2.0, "tests": 1.0, "minimality": 1.0, "protocol_fit": 1.0}


def weighted_score(scores: dict) -> float:
    return sum(_WEIGHTS[d] * float(scores.get(d, 0)) for d in RUBRIC_DIMS)


def passes_threshold(scores: dict, require_rubric: bool = True) -> bool:
    """Hard rule: approve only if acceptance >= 4 and no dimension < 2.

    An EMPTY scorecard fails closed. The rubric exists precisely to make an
    approve defensible; treating "no scores" as "no objection" hands a free
    merge to whichever reviewer is least able to produce a scorecard — which in
    degrade mode is the cheap model doing the reviewing, the case the guard was
    added for. `require_rubric=False` restores the legacy pass-through for
    reviewers that genuinely cannot emit one (config: review.require_rubric)."""
    if not scores:
        return not require_rubric
    if float(scores.get("acceptance", 0)) < 4:
        return False
    return all(float(scores.get(d, 0)) >= 2 for d in RUBRIC_DIMS)


def _parse_verdict(text: str, require_rubric: bool = True) -> dict:
    m = JSON_RE.search(text)
    if not m:
        return {"decision": "revise", "notes": f"Reviewer returned no JSON: {text[:500]}"}
    try:
        verdict = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"decision": "revise", "notes": f"Reviewer returned bad JSON: {text[:500]}"}
    if verdict.get("decision") not in ("approve", "revise", "reject"):
        verdict["decision"] = "revise"
    # Deterministic threshold guard: an "approve" with a failing or missing
    # scorecard is demoted to "revise" (repeatable, defensible verdicts).
    scores = verdict.get("scores") or {}
    if verdict["decision"] == "approve" and not passes_threshold(scores, require_rubric):
        verdict["decision"] = "revise"
        why = ("reviewer returned no scorecard — the prompt requires one, so an "
               "unscored approve cannot be verified"
               if not scores else
               "rubric threshold not met (need acceptance>=4 and no dimension<2): "
               + str(scores))
        verdict["notes"] = why + " | " + str(verdict.get("notes", ""))
    return verdict


def _select_winner(verdict: dict, passed: dict) -> str:
    """Rubric-driven selection: max weighted score among gate-passed candidates.
    Prefers per-candidate scores; falls back to the stated winner, then (last
    resort) the lowest candidate id.

    Every tiebreak here is deterministic on purpose. `max` already resolves ties
    by first-seen, so `scored` is built by iterating `passed` (stable) — and the
    last resort is `min(passed)`, not `next(iter(passed))`, whose answer depends
    on dict insertion order and so could pick a different winner for identical
    inputs across runs."""
    by_cand = verdict.get("scores_by_candidate") or {}
    scored = {cid: by_cand[cid] for cid in sorted(passed) if cid in by_cand}
    if scored:
        return max(scored, key=lambda cid: weighted_score(scored[cid]))
    if verdict.get("winner") in passed:
        return verdict["winner"]
    return min(passed)


def build_review_prompt(spec: dict, passed: dict) -> str:
    """Stable-first review payload, mirroring the worker's frozen prefix.

    Everything invariant across a task's review cycles (id, title, description,
    acceptance) comes FIRST, so the whole leading block is byte-identical when a
    `revise` sends the task back — only the diffs below it change, and every
    provider's prefix cache bills the head at the cached rate. Candidates are
    emitted in sorted id order for the same reason: the dict order here comes
    from the reducer's insertion order, which is not a contract, and a reshuffle
    would invalidate the prefix (and change the payload for identical inputs)."""
    parts = [f"# TASK {spec['id']}: {spec['title']}", "", spec["description"], ""]
    if spec.get("acceptance"):
        parts += ["## Acceptance criteria", *[f"- {a}" for a in spec["acceptance"]], ""]
    for cid in sorted(passed):
        cand = passed[cid]
        parts += [f"## Candidate `{cid}` (model {cand['model']}) — gate GREEN",
                  "Worker notes: " + (cand.get("notes") or "-"),
                  "```diff", cand["diff"], "```", ""]
        # Scene facts from the visual gate, when one ran and produced them. The
        # reviewer cannot look at anything itself, so measured facts about the
        # running scene are the only evidence it will ever have that the diff
        # does what it claims visually. Placed INSIDE the candidate block (the
        # already-volatile part) so the stable prefix above is untouched.
        facts = cand.get("visual_facts")
        if facts:
            parts += ["Measured scene facts (visual gate, not the model's claim):",
                      "```json", _facts_json(facts), "```", ""]
    return "\n".join(parts)


def _facts_json(facts: dict, max_chars: int = 4000) -> str:
    """Deterministic, size-capped rendering of the fact dict.

    `sort_keys` because an unstable key order would change the payload for
    identical inputs — the same reason candidates are emitted sorted."""
    try:
        text = json.dumps(facts, sort_keys=True, indent=2, default=str)
    except (TypeError, ValueError):
        return "(facts not serializable)"
    if len(text) > max_chars:
        return text[:max_chars] + "\n... (truncated)"
    return text


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

    result = await provider.complete(model=model, system=system,
                                     user=build_review_prompt(spec, passed),
                                     cwd=str(ctx.cfg.repo_path()),
                                     effort=ctx.role_effort("reviewer"))
    ctx.budget.record(
        task_id=state["task_id"], role="reviewer", provider=provider_name,
        provider_type=provider.type, model=model,
        input_tokens=result.input_tokens, output_tokens=result.output_tokens,
        cost_usd=result.cost_usd, cache_hit_tokens=result.cache_hit_tokens,
        cache_miss_tokens=result.cache_miss_tokens)

    verdict = _parse_verdict(
        result.text,
        require_rubric=bool((ctx.cfg.get("review") or {}).get("require_rubric", True)))
    # Rubric-driven winner (max weighted score), not first-in-dict.
    verdict["winner"] = _select_winner(verdict, passed)
    ctx.store.log_event(ctx.run_id, state["task_id"], "review",
                        json.dumps(verdict)[:2000])
    log.info("%s review: %s (winner=%s)", state["task_id"],
             verdict["decision"], verdict["winner"])
    return {"verdict": verdict}
