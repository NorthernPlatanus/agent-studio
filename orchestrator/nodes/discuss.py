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

    while True:
        env = await plan_or_ask(ctx, transcript=_format(turns))
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
            continue

        specs = env["specs"]
        _preview(specs, write)
        choice = (read("apply? [y/edit/abort]> ") or "").strip().lower()
        if choice in ("y", "yes"):
            persist_specs(ctx, specs, note="discuss")
            ctx.store.save_discussion(session, _format(turns) + "\nAPPLIED")
            write(f"applied {len(specs)} spec(s).")
            return specs
        if choice in ("abort", "a", "n", "no"):
            write("aborted — nothing applied.")
            return []
        # edit: capture a free-text note as the next turn
        note = read("edit note> ")
        turns.append(("planner", json.dumps(env)))
        turns.append(("user", f"EDIT: {note}"))
