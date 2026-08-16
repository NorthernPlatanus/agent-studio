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
from collections.abc import Callable

from ..ops.backlog import make_backlog, parent_id as derive_parent_id
from ..ops import assetops, handoff, projectmap
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


#: How much of a COMPLETED item's line survives. Measured on a pre-alpha test
#: board of a few dozen items: the done ones account for roughly half the whole
#: backlog by volume, averaging ~800 chars each, because a finished item
#: accumulates its own completion write-up ("**Done** — see ADR-0031.
#: `features/ghost/model` (pure): records ..."), while the items actually
#: awaiting a plan are a fraction of that. So half the most expensive block in
#: the prompt is implementation archaeology about work that is finished,
#: re-sent on every planner turn.
#:
#: What the planner needs from a done item is that it EXISTS, so it can reason
#: about dependencies and not re-plan it. The detail is still in the file, and
#: the planner has Read.
DONE_TITLE_CHARS = 160

#: Cross-references rescued out of a collapsed tail. Truncation otherwise drops
#: exactly the pointers worth keeping — on this board 9 of 19 done items name an
#: ADR past the cutoff — and a bare `ADR-0031` costs ~10 chars to preserve
#: against ~700 saved. Deliberately generic (`ADR-0031`, `T-045`): a finished
#: item that names another task id is a dependency hint, not just prose.
REF_RE = re.compile(r"\b[A-Z]{2,}-\d+\b")


def _collapse_done_items(ctx: RunContext, text: str) -> str:
    """Abbreviate completed backlog items, keeping their ids and references.

    Structure-preserving by construction: headings, blank lines, open items and
    every line that is not a completed item pass through byte-identical, so the
    planner still sees the board's real shape and ordering.
    """
    try:
        backlog = make_backlog(ctx.cfg)
        item_re, done_char = backlog.item_re, backlog.status_chars["done"]
    except (AttributeError, KeyError, TypeError):
        return text                    # unusual backlog config — leave it alone

    out: list[str] = []
    collapsed = 0
    for line in text.splitlines():
        m = item_re.match(line)
        try:
            is_done = bool(m) and m.group("status") == done_char
            title = m.group("title") if m else ""
        except (IndexError, re.error):    # pattern without the named groups
            return text
        if not is_done or len(title) <= DONE_TITLE_CHARS:
            out.append(line)
            continue
        head = title[:DONE_TITLE_CHARS].rstrip()
        tail = [r for r in dict.fromkeys(REF_RE.findall(title[DONE_TITLE_CHARS:]))
                if r not in head]
        refs = f"  [also refs: {', '.join(tail)}]" if tail else ""
        out.append(line[:m.start("title")] + head + " …" + refs)
        collapsed += 1
    if collapsed:
        out += ["", f"> ({collapsed} completed item(s) above are abbreviated — "
                    f"their full write-ups are in "
                    f"{ctx.cfg.project.backlog_file}, which you can Read if a "
                    f"finished task's detail actually matters. Everything still "
                    f"open is shown in full.)"]
    return "\n".join(out)


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
        # No collapse here: this path already dropped every item the caller did
        # not ask for, so a done item is only present because it was requested.
        return "\n".join(keep)
    return _collapse_done_items(ctx, text)


#: Character budget for the CURRENTLY PLANNED SPECS block. Measured on a real
#: planner payload: 42 specs serialise to 80,677 chars, so this block is always
#: the one that overflows.
SPECS_BUDGET_CHARS = 20000

#: Same, for the structural project map. Applied to a markdown document, so a cut
#: here is merely lossy rather than malformed — but it is still announced.
MAP_BUDGET_CHARS = 20000

#: Fields kept when a spec is summarised rather than sent whole. The planner
#: needs enough to avoid duplicating or colliding with an existing task; it does
#: not need that task's acceptance criteria or worker notes.
SLIM_SPEC_FIELDS = ("id", "title", "status", "milestone", "deps", "files_write",
                    "parent_id", "agent_able")

#: Titles are clamped in the SLIM form only. `title` is nominally a short
#: imperative phrase, but nothing enforces that and the field drifts: on a
#: pre-alpha test board the longest observed is ~4,000 characters of prose that
#: belongs in `description`. Unclamped, three such entries consume the block's
#: budget and push every other id out of view — which is the one failure this
#: block exists to prevent. A promoted spec still carries its title in full.
SLIM_TITLE_CHARS = 120


def _slim(spec: dict) -> dict:
    row = {k: spec[k] for k in SLIM_SPEC_FIELDS if k in spec}
    title = row.get("title")
    if isinstance(title, str) and len(title) > SLIM_TITLE_CHARS:
        row["title"] = title[:SLIM_TITLE_CHARS].rstrip() + "…"
    return row


def _specs_block(existing: list[dict], only_ids: list[str] | None,
                 budget: int = SPECS_BUDGET_CHARS) -> str:
    """Serialise the current queue to fit `budget` WITHOUT slicing the JSON.

    The previous implementation was `json.dumps(existing)[:20000]`, which on a
    real board cut the array mid-token: the planner was handed a malformed
    fragment listing a quarter of the queue under a heading telling it not to
    duplicate the rest. It paid ~5k tokens for something it could not parse.

    The budget is spent deliberately instead, and the ORDER matters. Every spec
    is listed in its slim form first, because the block's primary job is to stop
    the planner reinventing or colliding with an id — and an id it cannot see is
    an id it will reuse. Only the leftover budget buys full detail, promoting the
    specs actually in play (the requested ids, plus anything still awaiting a
    spec). Dropping entries entirely is the last resort, and it is announced:
    a planner that knows 12 specs are hidden behaves very differently from one
    handed a truncated array it believes is complete.
    """
    if not existing:
        return "[]"                    # caller guards, but "all 0 omitted" is not a sentence
    focus_ids = set(only_ids or ())

    def is_focus(spec: dict) -> bool:
        return (spec.get("id") in focus_ids
                or spec.get("parent_id") in focus_ids
                or spec.get("status") == "needs_plan")

    def render(rows: list[dict], omitted: int) -> str:
        text = json.dumps(rows, indent=1)
        if omitted:
            text += (f"\n\n({omitted} further spec(s) omitted to fit the prompt "
                     f"— the queue holds {len(existing)} in total. Ask before "
                     f"assuming an id is free.)")
        return text

    # Floor: everyone visible, nobody detailed.
    entries = [_slim(spec) for spec in existing]
    omitted = 0
    while entries and len(render(entries, omitted)) > budget:
        entries.pop()
        omitted += 1
    if not entries:
        return (f"(all {len(existing)} specs omitted — they do not fit the "
                f"prompt budget. Ask before assuming an id is free.)")

    # Then spend what's left on detail, skipping (not stopping at) any single
    # spec too large to fit — one oversized entry must not starve the rest.
    for i, spec in enumerate(existing[:len(entries)]):
        if not is_focus(spec):
            continue
        candidate = entries[:i] + [spec] + entries[i + 1:]
        if len(render(candidate, omitted)) <= budget:
            entries = candidate
    return render(entries, omitted)


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
        text = map_file.read_text()
        if len(text) > MAP_BUDGET_CHARS:
            text = (text[:MAP_BUDGET_CHARS].rsplit("\n", 1)[0]
                    + f"\n\n(map truncated at {MAP_BUDGET_CHARS} chars — it is "
                      f"not the complete tree; use Glob/Grep for anything below.)")
        parts += ["# PROJECT MAP (structure — verify against it before listing files)",
                  text, ""]
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
                  _specs_block(existing, only_ids), ""]
    # The digest of an expired session, if one is on file. Placed after the
    # durable facts and before the live conversation, which is where it belongs
    # chronologically — it IS the older conversation, compressed.
    previous = handoff.prompt_block(ctx)
    if previous:
        parts += [previous, ""]
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
                      session_reuse: bool | None = None,
                      on_progress: Callable[[dict], None] | None = None) -> dict:
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
    other caller is unaffected.

    `on_progress` is forwarded to the provider for callers that can show
    in-flight events. A planner turn is minutes of silence otherwise (measured:
    334s and up), which the operator cannot tell apart from a hang."""
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
                                         session=use_session, effort=effort,
                                         on_progress=on_progress)
    except SessionLost:
        # The abbreviated payload has no context without the session — resend it
        # whole exactly once (the provider already dropped the dead session id,
        # so this call opens a fresh one to continue from).
        log.warning("planner session lost; resending the full payload")
        result = await provider.complete(
            model=model, system=system,
            user=_full_prompt(ctx, discussion=discussion, transcript=transcript,
                              only_ids=only_ids),
            cwd=str(ctx.cfg.repo_path()), session=use_session, effort=effort,
            on_progress=on_progress)

    ctx.budget.record(
        task_id=None, role="planner", provider=provider_name,
        provider_type=provider.type, model=model,
        input_tokens=result.input_tokens, output_tokens=result.output_tokens,
        cost_usd=result.cost_usd, cache_hit_tokens=result.cache_hit_tokens,
        cache_miss_tokens=result.cache_miss_tokens)
    env = parse_planner_output(result.text)
    # Write the handoff NOW, while the envelope is in hand and free. Deferring it
    # to the moment a session is found expired would mean reconstructing it from
    # a conversation that is, by then, expensive to read.
    handoff.record(ctx, env)
    return env


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
    specs = persist_specs(ctx, env["specs"], note=discussion)
    # Same conclusion rule as `discuss`: the specs landed, so there is no
    # interrupted conversation for a digest to bridge. Leaving it would inject
    # this run's questions into the next, unrelated one.
    handoff.clear(ctx)
    return specs


def needs_plan_ids(store, limit: int | None = None) -> list[str]:
    """Every task still awaiting a spec — the batch for `plan --all-needs-plan`.

    Planner cost scales with the number of `plan` INVOCATIONS, not with the number
    of tasks planned. Measured on a pre-alpha test run, from the CLI's own
    session log (5 API calls, one operator message):

        first-call payload   47,368 tok   backlog + map + specs + system prompt
        tool exploration     32,877 tok   9 tool calls
        prefix re-reads     238,596 tok   the same payload, re-sent per step
        ----------------------------------------------------------------------
        total input         325,721 tok   (output: 12,273)

    Two consequences. The payload dominates the genuinely-new tokens (47k vs 33k),
    NOT the repo sweep — an earlier version of this comment claimed a "~17k payload
    and a ~400k agentic pass", and both halves were wrong. And because every
    agentic step re-sends the whole prefix, cost is roughly payload x steps: the
    same work took 505k when it happened to need 7 steps instead of 5. So planning
    one task at a time is the most expensive way to use this harness.

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
