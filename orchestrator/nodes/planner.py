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

from ..ops.backlog import make_backlog, parent_id as derive_parent_id
from ..ops import assetops, projectmap
from ..core.context import RunContext
from ..core.errors import PlannerNeedsInput, SessionLost
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


def _full_prompt(ctx: RunContext, *, discussion: str, transcript: str,
                 only_ids: list[str] | None) -> str:
    """The complete, self-contained planner payload."""
    existing = list(ctx.store.all_tasks())
    parts = ["# BACKLOG (source of truth)",
             _backlog_excerpt(ctx, only_ids), ""]
    # Insert the structural project-map (if present) between the backlog and the
    # current specs, so files_read authoring starts from a real skeleton.
    map_file = projectmap.map_path(ctx.cfg)
    if map_file.exists():
        parts += ["# PROJECT MAP (structure — verify against it before listing files)",
                  map_file.read_text()[:20000], ""]
    # The available asset ops, verbatim. The planner may set `asset_op: <name>`
    # on a spec, and this list is the only place those names exist — it cannot
    # invent one (persist_specs rejects unknown names) and it cannot guess one
    # that was never shown. Omitted entirely when the project configures none, so
    # a profile without asset ops sends not one extra token.
    ops = assetops.available_ops(ctx.cfg)
    if ops:
        parts += ["# ASSET OPS AVAILABLE (fixed commands; reference BY NAME via "
                  "`asset_op`, never write a command yourself)"]
        for name in ops:
            cmd, _ = assetops.resolve_op(ctx.cfg, name)
            parts.append(f"- {name}: `{cmd}`")
        parts.append("")
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
    return "\n".join(parts)


def session_reuse_enabled(cfg) -> bool:
    return bool(cfg.run.get("session_reuse", False))


async def plan_or_ask(ctx: RunContext, *, discussion: str = "", transcript: str = "",
                      only_ids: list[str] | None = None,
                      session: str | None = None, delta: str = "",
                      effort: str | None = None, model: str | None = None,
                      session_reuse: bool | None = None) -> dict:
    """One planner turn. Returns the normalized envelope
    {questions, assumptions, specs} WITHOUT persisting — so `discuss` can loop
    and `plan` can decide what to do with questions. `transcript` carries the
    running multi-turn conversation (the CLI planner is single-turn, so we pass
    history in the prompt rather than via complete_chat).

    `session` + `delta` are the continuity path (run.session_reuse). The `discuss`
    loop otherwise re-sends backlog + project map + every current spec on EVERY
    turn — tens of thousands of tokens of unchanged context, re-read at full
    weight, against the subscription tier that is the binding constraint. When
    the provider reports the session is live, only `delta` (the human's new
    answer) is sent and the conversation supplies the rest. If the resume fails,
    SessionLost brings us back here and the full payload goes out once.

    `effort`/`model`/`session_reuse` override the configured planner target for
    this call only. They exist for `discuss`, where an operator adjusts the tier
    mid-conversation — a hard question deserves `high`, and the rest of the
    session should not pay for it. None means "leave the config alone", so every
    other caller is unaffected."""
    provider_name, configured_model = ctx.role_target("planner")
    model = model or configured_model
    provider = get_provider(ctx.cfg, provider_name)
    system = ctx.cfg.prompt("planner")

    reuse = (session_reuse_enabled(ctx.cfg) if session_reuse is None
             else session_reuse)
    use_session = session if reuse else None
    continuing = bool(use_session and delta and provider.session_active(use_session))
    user = (f"# HUMAN REPLY — continue the plan from our conversation\n{delta}"
            if continuing else
            _full_prompt(ctx, discussion=discussion, transcript=transcript,
                         only_ids=only_ids))

    effort = effort or ctx.role_effort("planner")
    try:
        result = await provider.complete(model=model, system=system, user=user,
                                         cwd=str(ctx.cfg.repo_path()),
                                         session=use_session, effort=effort)
    except SessionLost:
        # The abbreviated payload has no context without the session — resend it
        # whole exactly once (the provider already dropped the dead session id,
        # so this call opens a fresh one to continue from).
        log.warning("planner session lost; resending the full payload")
        result = await provider.complete(
            model=model, system=system,
            user=_full_prompt(ctx, discussion=discussion, transcript=transcript,
                              only_ids=only_ids),
            cwd=str(ctx.cfg.repo_path()), session=use_session, effort=effort)

    ctx.budget.record(
        task_id=None, role="planner", provider=provider_name,
        provider_type=provider.type, model=model,
        input_tokens=result.input_tokens, output_tokens=result.output_tokens,
        cost_usd=result.cost_usd, cache_hit_tokens=result.cache_hit_tokens,
        cache_miss_tokens=result.cache_miss_tokens)
    return parse_planner_output(result.text)


def validate_spec(spec: dict, asset_ops: list[str] | None = None) -> None:
    """Reject a spec the executor cannot run, at PLAN time, with a message the
    planner can act on. Raises ValueError.

    `files_write` is the worker's entire write allowlist (ops/patch), and both of
    its degenerate forms are silently bad much later:
      * key absent  -> apply_response gets None -> NO allowlist; the worker may
        write any path in the worktree, defeating the containment the spec exists
        to express.
      * empty list  -> every write is rejected, so the task can never go green;
        it burns the full retry ladder (and the scheduler runs it alone in a
        batch of one) before failing for a reason no log makes obvious.
    `asset_op` is checked against the profile's configured names when
    `asset_ops` is passed. The op is a fixed human-authored command and the name
    is the ONLY part the planner writes, so a name that doesn't exist is a typo
    or a hallucination — caught here, at plan time, rather than failing every
    candidate of that task later for a reason the retry feedback can't fix.

    Human-only specs (`agent_able: false`) never reach a worker, so they are
    exempt."""
    for key in ("id", "title", "description"):
        if not spec.get(key):
            raise ValueError(f"Planner spec missing '{key}': {spec}")
    if not spec.get("agent_able", True):
        return
    asset_op = spec.get("asset_op")
    if asset_op and asset_ops is not None and asset_op not in asset_ops:
        raise ValueError(
            f"Planner spec {spec['id']} names an unknown asset_op "
            f"{asset_op!r}. Asset ops are fixed commands defined by a human in "
            f"the project profile; the available names are: "
            f"{', '.join(asset_ops) or '(none configured)'}. Use one of those, "
            f"or drop the field.")
    files_write = spec.get("files_write")
    if files_write is None:
        raise ValueError(
            f"Planner spec {spec['id']} has no 'files_write': an agent_able task "
            f"must declare the exact files it may write (that list IS the worker's "
            f"write allowlist). Add files_write, or set agent_able: false.")
    if not isinstance(files_write, list) or not files_write:
        raise ValueError(
            f"Planner spec {spec['id']} has an empty/invalid 'files_write' "
            f"({files_write!r}): with nothing writable the task can never pass its "
            f"gate. List the files to change, or set agent_able: false.")


def persist_specs(ctx: RunContext, specs: list[dict], note: str = "") -> list[dict]:
    """Validate + upsert planner specs. Shared by `plan` and `discuss`."""
    asset_ops = assetops.available_ops(ctx.cfg)
    for spec in specs:
        validate_spec(spec, asset_ops=asset_ops)
        # The planner sees current specs (which carry queue status) and may
        # echo the field back — never let LLM output overwrite queue state.
        spec.pop("status", None)
        # Decomposition bookkeeping: backlog item T-131 planned as T-131a/T-131b.
        # Neither sub-id has a line on the board, so the integrator's writeback
        # needs to know which item they belong to or it writes nothing at all
        # (see ops/backlog.parent_id). Derived only when the planner didn't say.
        if not spec.get("parent_id"):
            derived = derive_parent_id(spec["id"])
            if derived:
                spec["parent_id"] = derived
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


def needs_plan_ids(store, limit: int | None = None) -> list[str]:
    """Every task still awaiting a spec — the batch for `plan --all-needs-plan`.

    Planner cost scales with the number of `plan` INVOCATIONS, not with the number
    of tasks planned: the static payload (prompt + project map + backlog + protocol)
    is only ~17k tokens, and the other ~400k is one agentic Read/Grep/Glob pass over
    the repo, amortised across everything in the call. Measured 385k for one task
    and 425k for one item decomposed into two, so planning one at a time is the most
    expensive way to use this harness.

    `limit` exists because the opposite extreme is also wrong: specs written far
    ahead of the work go stale as earlier tasks land (a reviewer was observed
    reasoning about a helper that a sibling task had since merged). Batch per
    milestone. Ordered by id (store.all_tasks is), so the batch is deterministic.
    """
    ids = [t["id"] for t in store.all_tasks() if t["status"] == "needs_plan"]
    return ids[:limit] if limit and limit > 0 else ids


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
