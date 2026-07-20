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
from ..core.context import RunContext
from ..providers import get_provider

log = logging.getLogger("orchestrator.planner")

JSON_ARRAY_RE = re.compile(r"\[.*\]", re.S)


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


async def plan(ctx: RunContext, discussion: str = "",
               only_ids: list[str] | None = None) -> list[dict]:
    provider_name, model = ctx.role_target("planner")
    provider = get_provider(ctx.cfg, provider_name)
    system = ctx.cfg.prompt("planner")

    existing = [t for t in ctx.store.all_tasks()]
    parts = ["# BACKLOG (source of truth)",
             _backlog_excerpt(ctx, only_ids), ""]
    if existing:
        parts += ["# CURRENTLY PLANNED SPECS (update, don't duplicate)",
                  json.dumps(existing, indent=1)[:20000], ""]
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

    m = JSON_ARRAY_RE.search(result.text)
    if not m:
        raise ValueError(f"Planner returned no JSON array:\n{result.text[:1000]}")
    specs = json.loads(m.group(0))

    for spec in specs:
        for key in ("id", "title", "description"):
            if not spec.get(key):
                raise ValueError(f"Planner spec missing '{key}': {spec}")
        # The planner sees current specs (which carry queue status) and may
        # echo the field back — never let LLM output overwrite queue state.
        spec.pop("status", None)
        ctx.store.upsert_task(spec)
    ctx.store.log_event(ctx.run_id, None, "planned",
                        f"{len(specs)} specs" + (f" (note: {discussion[:100]})"
                                                 if discussion else ""))
    log.info("planner produced %d task specs", len(specs))
    return specs


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
