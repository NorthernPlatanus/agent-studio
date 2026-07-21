"""Planner — Opus via Claude Code CLI (subscription), invoked on demand by
`orchestrator plan`, not inside the per-task graph.

Turns backlog items (+ an optional human discussion note) into enriched,
machine-executable task specs: files_read / files_write / deps / acceptance /
agent_able / n_candidates. This file-list authorship is the single most
important act in the whole system — it defines each worker's entire world.
"""

from __future__ import annotations

import json
import logging
import re

from ..ops.backlog import make_backlog
from ..ops import projectmap
from ..core.context import RunContext
from ..core.errors import PlannerNeedsInput
from ..providers import get_provider

log = logging.getLogger("orchestrator.planner")

JSON_ARRAY_RE = re.compile(r"\[.*\]", re.S)
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.S)


def parse_planner_output(text: str) -> dict:
    """Accept BOTH the legacy bare `[specs]` array and the new tech-lead
    envelope `{questions, assumptions, specs}`. Returns a normalized envelope.
    A non-empty `questions` means the planner is asking, not planning."""
    # Prefer the object envelope; fall back to a bare array (back-compat).
    for rx in (JSON_OBJECT_RE, JSON_ARRAY_RE):
        m = rx.search(text)
        if not m:
            continue
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            return {"questions": [], "assumptions": [], "specs": data}
        if isinstance(data, dict):
            if "specs" in data or "questions" in data:
                return {"questions": data.get("questions") or [],
                        "assumptions": data.get("assumptions") or [],
                        "specs": data.get("specs") or []}
            # a bare single-spec object
            return {"questions": [], "assumptions": [], "specs": [data]}
    raise ValueError(f"Planner returned no JSON array/object:\n{text[:1000]}")


def _backlog_excerpt(ctx: RunContext, only_ids: list[str] | None) -> str:
    path = ctx.cfg.repo_path() / ctx.cfg.project.backlog_file
    text = path.read_text()
    if only_ids:
        # Keep headings + the requested items' lines (with their note lines).
        keep: list[str] = []
        current_match = False
        for line in text.splitlines():
            if line.startswith("#"):
                keep.append(line)
                current_match = False
            elif any(f"**{tid}**" in line for tid in only_ids):
                keep.append(line)
                current_match = True
            elif current_match and line.startswith((" ", "\t")):
                keep.append(line)
            else:
                current_match = False
        return "\n".join(keep)
    return text


async def plan_or_ask(ctx: RunContext, *, discussion: str = "", transcript: str = "",
                      only_ids: list[str] | None = None) -> dict:
    """One planner turn. Returns the normalized envelope
    {questions, assumptions, specs} WITHOUT persisting — so `discuss` can loop
    and `plan` can decide what to do with questions. `transcript` carries the
    running multi-turn conversation (the CLI planner is single-turn, so we pass
    history in the prompt rather than via complete_chat)."""
    provider_name, model = ctx.role_target("planner")
    provider = get_provider(ctx.cfg, provider_name)
    system = ctx.cfg.prompt("planner")

    existing = list(ctx.store.all_tasks())
    parts = ["# BACKLOG (source of truth)",
             _backlog_excerpt(ctx, only_ids), ""]
    # Insert the structural project-map (if present) between the backlog and the
    # current specs, so files_read authoring starts from a real skeleton.
    map_file = projectmap.map_path(ctx.cfg)
    if map_file.exists():
        parts += ["# PROJECT MAP (structure — verify against it before listing files)",
                  map_file.read_text()[:20000], ""]
    if existing:
        parts += ["# CURRENTLY PLANNED SPECS (update, don't duplicate)",
                  json.dumps(existing, indent=1)[:20000], ""]
    if transcript:
        parts += ["# CONVERSATION SO FAR (most recent last)", transcript, ""]
    if discussion:
        parts += ["# HUMAN NOTE — fold this into the plan", discussion, ""]
    proto = ctx.cfg.project.get("protocol_file")
    if proto:
        parts += [f"The repo protocol is in {proto} — read it with your tools "
                  f"before finalizing specs."]

    result = await provider.complete(model=model, system=system,
                                     user="\n".join(parts),
                                     cwd=str(ctx.cfg.repo_path()))
    ctx.budget.record(
        task_id=None, role="planner", provider=provider_name,
        provider_type=provider.type, model=model,
        input_tokens=result.input_tokens, output_tokens=result.output_tokens,
        cost_usd=result.cost_usd)
    return parse_planner_output(result.text)


def persist_specs(ctx: RunContext, specs: list[dict], note: str = "") -> list[dict]:
    """Validate + upsert planner specs. Shared by `plan` and `discuss`."""
    for spec in specs:
        for key in ("id", "title", "description"):
            if not spec.get(key):
                raise ValueError(f"Planner spec missing '{key}': {spec}")
        # The planner sees current specs (which carry queue status) and may
        # echo the field back — never let LLM output overwrite queue state.
        spec.pop("status", None)
        ctx.store.upsert_task(spec)
    ctx.store.log_event(ctx.run_id, None, "planned",
                        f"{len(specs)} specs" + (f" (note: {note[:100]})" if note else ""))
    log.info("planner produced %d task specs", len(specs))
    return specs


async def plan(ctx: RunContext, discussion: str = "",
               only_ids: list[str] | None = None) -> list[dict]:
    """One-shot planning. If the tech-lead planner asks clarifying questions,
    raise PlannerNeedsInput (the one-shot path must not silently guess —
    requirements elicitation belongs in `discuss`)."""
    env = await plan_or_ask(ctx, discussion=discussion, only_ids=only_ids)
    if env["questions"]:
        raise PlannerNeedsInput(env["questions"], env["assumptions"])
    return persist_specs(ctx, env["specs"], note=discussion)


def import_backlog_stubs(ctx: RunContext) -> int:
    """Cheap non-LLM import: register backlog items as stub tasks (status
    'needs_plan') so `status` shows the whole board before planning."""
    backlog = make_backlog(ctx.cfg)
    count = 0
    for item in backlog.parse():
        if ctx.store.get_task(item.id):
            continue
        status = {"todo": "needs_plan", "in_progress": "needs_plan",
                  "done": "done", "blocked": "needs_human"}[item.status]
        ctx.store.upsert_task({
            "id": item.id, "title": item.title, "milestone": item.milestone,
            "description": item.raw, "status": status, "agent_able": True,
        })
        count += 1
    return count
