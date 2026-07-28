"""Interactive `discuss` — a multi-turn requirements loop with the tech-lead
planner.

Providers are single-turn (the planner is a CLI provider), so multi-turn is
carried as a TRANSCRIPT in the prompt rather than via complete_chat — this keeps
`discuss` identical on Claude and Codex. The loop:

  1. Call the planner with {backlog, current specs, transcript}.
  2. If it returns questions -> print them, read the human's answers, append to
     the transcript, repeat.
  3. If it returns specs -> preview them, ask apply?[y/edit/abort]. On y, upsert
     via the shared planner persistence path; on edit, capture a note as the next
     transcript turn; on abort, stop.

`read`/`write` are injected so the loop is unit-testable with scripted stdin.
"""

from __future__ import annotations

import json
import logging

from ..core.context import RunContext
from ..providers import get_provider
from .planner import persist_specs, plan_or_ask

log = logging.getLogger("orchestrator.discuss")


def _format(turns: list[tuple[str, str]]) -> str:
    return "\n".join(f"{role.upper()}: {text}" for role, text in turns)


def _preview(specs: list[dict], write) -> None:
    write(f"\nPROPOSED PLAN — {len(specs)} task(s):")
    for s in specs:
        deps = f" deps={s['deps']}" if s.get("deps") else ""
        flags = []
        if s.get("complexity"):
            flags.append(f"complexity={s['complexity']}")
        if s.get("risk"):
            flags.append(f"risk={s['risk']}")
        if not s.get("agent_able", True):
            flags.append("human-only")
        meta = ("  [" + ", ".join(flags) + "]") if flags else ""
        write(f"  {s.get('id', '?')}: {s.get('title', '')}{deps}{meta}")
        if s.get("files_write"):
            write(f"      writes: {s['files_write']}")


async def run_discuss(ctx: RunContext, initial: str, *,
                      read=input, write=print) -> list[dict]:
    """Run the clarify->approve->persist loop. Returns the applied specs (or []
    if aborted). Persists the transcript so a session can resume."""
    session = ctx.cfg.project_name
    turns: list[tuple[str, str]] = []
    prior = ctx.store.load_discussion(session)
    if prior:
        turns.append(("system", f"(resumed session)\n{prior}"))
    turns.append(("user", initial))

    # Provider-side continuity key for this loop (honored only when
    # run.session_reuse is on and the provider supports it). Scoped to one
    # discuss run: the planner may see the whole conversation, but never another
    # run's. `delta` is the newest human turn — the only genuinely new text on
    # turn 2+, since backlog/map/specs are unchanged and already in the session.
    llm_session = f"discuss:{ctx.run_id}"
    delta = ""

    while True:
        env = await plan_or_ask(ctx, transcript=_format(turns),
                                session=llm_session, delta=delta)
        ctx.store.save_discussion(session, _format(turns))

        if env["questions"]:
            for a in env.get("assumptions", []):
                write(f"assumption: {a}")
            for q in env["questions"]:
                write(f"[{q.get('id', '?')}] {q.get('q', '')}"
                      + (f"  (why: {q['why']})" if q.get("why") else ""))
            answer = read("your answer> ")
            turns.append(("planner", json.dumps(env)))
            turns.append(("user", answer))
            delta = answer
            continue

        specs = env["specs"]
        _preview(specs, write)
        choice = (read("apply? [y/edit/abort]> ") or "").strip().lower()
        if choice in ("y", "yes"):
            persist_specs(ctx, specs, note="discuss")
            ctx.store.save_discussion(session, _format(turns) + "\nAPPLIED")
            write(f"applied {len(specs)} spec(s).")
            _end_session(ctx, llm_session)
            return specs
        if choice in ("abort", "a", "n", "no"):
            write("aborted — nothing applied.")
            _end_session(ctx, llm_session)
            return []
        # edit: capture a free-text note as the next turn
        note = read("edit note> ")
        turns.append(("planner", json.dumps(env)))
        turns.append(("user", f"EDIT: {note}"))
        delta = f"EDIT: {note}"


def _end_session(ctx: RunContext, key: str) -> None:
    """Close the provider-side conversation when the loop ends, so the next
    discuss starts clean rather than inheriting this one's context."""
    try:
        provider_name, _ = ctx.role_target("planner")
        get_provider(ctx.cfg, provider_name).end_session(key)
    except Exception as e:      # never let cleanup break a completed discuss
        log.debug("could not end planner session %s: %s", key, e)
