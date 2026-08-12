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

Two seams exist for the control panel, and both are strictly additive — with
neither supplied this file behaves exactly as the CLI has always behaved:

* **`emit`** — the same events the loop prints, as dicts. The API needs typed
  frames (`assumption`, `question`, `specs_preview`, …) and the alternative was
  regexing the printed prose back apart, which breaks the first time anyone
  rewords a line. So the loop emits the structure it already has, and `write`
  keeps rendering the terminal's version of it.
* **`settings`** — a *callable*, re-consulted at the top of every turn, not a
  value captured at the start. The panel lets an operator pin a file, raise the
  reasoning effort, or narrow the backlog scope in the middle of a conversation,
  and those have to reach the next planner call rather than the next session.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from ..core.context import RunContext
from ..core.errors import LimitExhausted, PlannerNeedsInput, SessionLost
from ..providers import get_provider
from .planner import persist_specs, plan_or_ask

log = logging.getLogger("orchestrator.discuss")


@dataclass
class PinnedFile:
    """A file the operator sent, to sit in front of the planner every turn.

    The planner has `Read`/`Grep`/`Glob` and finds most things itself, at ~400k
    tokens of exploration per invocation. Pinning is the cheap override for the
    cases where it looks in the wrong place, or where the thing it needs is not
    in the checkout at all: a crash log, a spec, notes pasted out of a tracker.
    The content is in the prompt, so it costs its own length once per turn.

    Content only — never a path into the checkout. `path` is a display name the
    API assigns under `uploaded/`, and the prompt says so, because the heading
    otherwise reads like a file the planner could go open and it would waste a
    turn looking for one that does not exist.
    """

    path: str
    text: str
    truncated: bool = False


@dataclass
class DiscussSettings:
    """What the operator can change about a session, including mid-session.

    Everything here maps to a lever that already exists — a `plan_or_ask`
    argument or a config value the planner call reads. Nothing here is a
    preference the loop merely stores.
    """

    #: Folded into every planner turn as the HUMAN NOTE block.
    note: str = ""
    #: Restrict the backlog excerpt to these ids (`plan_or_ask(only_ids=)`).
    only_ids: list[str] | None = None
    #: Overrides `roles.planner.effort` for this session only.
    effort: str | None = None
    #: Overrides `roles.planner.model` for this session only.
    model: str | None = None
    #: Overrides `run.session_reuse` for this session only.
    session_reuse: bool | None = None
    #: Files pinned into the prompt (see `PinnedFile`).
    pinned: list[PinnedFile] = field(default_factory=list)
    #: Force a proposal after this many question rounds. 0 = no limit.
    max_question_rounds: int = 0

    def context_block(self) -> str:
        if not self.pinned:
            return ""
        # The "not in the checkout" line is not decoration: each heading below
        # looks like a path, and without it the planner spends a turn trying to
        # open files that exist only in this prompt.
        parts = ["# ATTACHED FILES (the operator sent these — read them before "
                 "searching for anything else. They were uploaded, and are not "
                 "files in the checkout: do not try to open these paths)"]
        for pin in self.pinned:
            suffix = "  [truncated — only the head is here]" if pin.truncated else ""
            parts.append(f"\n## {pin.path}{suffix}\n{pin.text}")
        return "\n".join(parts)


def _default_settings() -> DiscussSettings:
    return DiscussSettings()


async def _ask(read, prompt: str) -> str:
    """Read one human turn, from a sync `read` or an async one.

    The CLI passes `input`, which returns a string. The API passes a coroutine
    function that awaits the operator's next message off a queue — and it has to
    be awaitable rather than blocking, because the discuss loop shares its event
    loop with every other request the API is serving. Accepting both keeps the
    loop single-threaded and the CLI untouched.
    """
    value = read(prompt)
    return await value if inspect.isawaitable(value) else value


async def _wait_or_interrupt(read, prompt: str, seconds: float) -> str | None:
    """Wait out `seconds`, unless the operator says something first.

    Returns their message, or None if the wait simply expired.

    This is how a freeze stays abortable. Plain `asyncio.sleep` would be simpler
    and wrong: the API delivers an abort by putting a sentinel on the very queue
    `read` awaits, so a loop that is sleeping instead of reading would leave a
    closed browser tab holding the project's write lock for the length of a
    five-hour window.

    The CLI's `read` is blocking `input()`, which cannot be polled without
    blocking the shared event loop — there the wait is just a wait, and the
    terminal operator interrupts with Ctrl-C as they always have.
    """
    value = read(prompt)
    if not inspect.isawaitable(value):
        await asyncio.sleep(seconds)
        return None
    try:
        return await asyncio.wait_for(value, seconds)
    except (asyncio.TimeoutError, TimeoutError):
        return None


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
                      read=input, write=print,
                      emit: Callable[[dict], None] | None = None,
                      settings: Callable[[], DiscussSettings] | None = None,
                      ) -> list[dict]:
    """Run the clarify->approve->persist loop. Returns the applied specs (or []
    if aborted). Persists the transcript so a session can resume."""
    session = ctx.cfg.project_name
    say = emit or (lambda _event: None)
    current = settings or _default_settings
    turns: list[tuple[str, str]] = []
    prior = ctx.store.load_discussion(session)
    if prior:
        turns.append(("system", f"(resumed session)\n{prior}"))
    turns.append(("user", initial))
    say({"kind": "you", "text": initial})

    # Provider-side continuity key for this loop (honored only when
    # run.session_reuse is on and the provider supports it). Scoped to one
    # discuss run: the planner may see the whole conversation, but never another
    # run's. `delta` is the newest human turn — the only genuinely new text on
    # turn 2+, since backlog/map/specs are unchanged and already in the session.
    llm_session = f"discuss:{ctx.run_id}"
    delta = ""
    question_rounds = 0

    while True:
        opts = current()
        # Pinned content rides in the transcript rather than as a separate
        # `plan_or_ask` argument: it is conversation state (it can change between
        # turns), and the transcript is the one payload that is already re-sent
        # whole whenever the provider session is not being reused.
        pinned = opts.context_block()
        transcript = _format(turns) + (f"\n{pinned}" if pinned else "")
        # Persist BEFORE the call, not after. A planner turn is minutes long and
        # can fail (a wedge, a 5xx, a limit hit); saving only on success meant a
        # failure on turn 1 persisted nothing at all, and the operator restarted
        # the conversation from their first message.
        ctx.store.save_discussion(session, _format(turns))
        say({"kind": "thinking"})
        try:
            env = await plan_or_ask(ctx, discussion=opts.note, transcript=transcript,
                                    only_ids=opts.only_ids, session=llm_session,
                                    delta=delta, effort=opts.effort, model=opts.model,
                                    session_reuse=opts.session_reuse,
                                    on_progress=lambda e: say({"kind": "progress", **e}))
        except (PlannerNeedsInput, SessionLost):
            raise                       # control flow, not failure — let it out
        except Exception as e:          # noqa: BLE001 — offered back to the operator
            # One failed turn is not a failed conversation. The transcript is
            # intact and the next attempt is cheap to ask for, so report it and
            # hand control back rather than tearing the session down — which is
            # what used to happen, discarding every answer the operator had
            # already typed.
            log.warning("planner turn failed: %s", e, exc_info=True)
            # Drop provider-side continuity. `session_reuse` sends only the newest
            # human turn once the provider reports a live session, which is a bet
            # that the session holds everything else — and after a failed turn
            # that bet is off (the call may have died before its payload ever
            # landed). Ending it costs one re-send of context we still have, and
            # buys a next attempt that is self-contained.
            _end_session(ctx, llm_session)
            delta = ""

            # A planning session can outlast the five-hour window it is spending:
            # one turn was measured at ~520k subscription tokens, so a real
            # conversation exhausts the quota mid-plan. That is not a failure to
            # report — it is a clock that has not rolled over — so the loop
            # FREEZES until the window reopens and retries the turn itself.
            wait_s = _freeze_seconds(e, ctx.cfg)
            if wait_s is not None:
                log.warning("subscription limit hit; freezing %.0fs until the %s "
                            "window resets", wait_s, e.limit_type or "usage")
                write(f"subscription limit reached — waiting "
                      f"{wait_s / 60:.0f} min for the "
                      f"{e.limit_type or 'usage'} window to reset, then "
                      f"retrying this turn.")
                say({"kind": "limit_paused", "text": str(e),
                     "resets_at": e.resets_at, "limit_type": e.limit_type,
                     "seconds": wait_s})
                say({"kind": "awaiting", "expects": "frozen"})
                typed = await _wait_or_interrupt(
                    read, "waiting for the limit to reset — "
                          "type 'abort' to give up> ", wait_s)
                if typed is not None and typed.strip().lower() in (
                        "abort", "a", "quit", "q"):
                    write("aborted while waiting for the limit to reset.")
                    say({"kind": "aborted",
                         "reason": "operator gave up on the freeze"})
                    return []
                if typed:
                    turns.append(("user", typed))
                    say({"kind": "you", "text": typed})
                    delta = typed
                say({"kind": "note", "text": "the usage window reset — retrying"})
                continue
            # Anything else — including a limit we cannot time — is offered back.
            detail = f"{type(e).__name__}: {e}"
            write(f"the planner turn failed: {detail}")
            say({"kind": "turn_failed", "text": detail})
            say({"kind": "awaiting", "expects": "retry"})
            answer = (await _ask(read, "retry? [enter to retry / or type more "
                                       "context / 'abort']> ") or "").strip()
            if answer.lower() in ("abort", "a", "quit", "q"):
                write("aborted.")
                say({"kind": "aborted", "reason": "operator abandoned a failed turn"})
                return []
            if answer:
                # Anything typed is context for the retry, not a new question —
                # the planner never got to answer the previous one.
                turns.append(("user", answer))
                say({"kind": "you", "text": answer})
                delta = answer
            continue

        capped = (opts.max_question_rounds > 0
                  and question_rounds >= opts.max_question_rounds)
        if env["questions"] and not capped:
            question_rounds += 1
            for a in env.get("assumptions", []):
                write(f"assumption: {a}")
                say({"kind": "assumption", "text": a})
            for q in env["questions"]:
                write(f"[{q.get('id', '?')}] {q.get('q', '')}"
                      + (f"  (why: {q['why']})" if q.get("why") else ""))
                say({"kind": "question", "id": q.get("id"), "q": q.get("q", ""),
                     "why": q.get("why")})
            say({"kind": "awaiting", "expects": "answer",
                 "round": question_rounds,
                 "max_rounds": opts.max_question_rounds or None})
            answer = await _ask(read, "your answer> ")
            turns.append(("planner", json.dumps(env)))
            turns.append(("user", answer))
            say({"kind": "you", "text": answer})
            delta = answer
            continue

        if env["questions"] and capped:
            # The operator asked for a bounded conversation and it ran out. Say so
            # rather than dropping the unanswered questions on the floor: they are
            # what the planner would have asked, and the specs below were written
            # without their answers.
            note = (f"reached max_question_rounds={opts.max_question_rounds}; "
                    f"the planner still had {len(env['questions'])} question(s)")
            write(f"note: {note}")
            say({"kind": "note", "text": note,
                 "questions": [{"id": q.get("id"), "q": q.get("q", "")}
                               for q in env["questions"]]})

        specs = env["specs"]
        for a in env.get("assumptions", []):
            say({"kind": "assumption", "text": a})
        _preview(specs, write)
        say({"kind": "specs_preview", "specs": specs})
        say({"kind": "awaiting", "expects": "decision"})
        raw = (await _ask(read, "apply? [y/edit/abort]> ") or "").strip()
        choice = raw.lower()
        if choice in ("y", "yes"):
            persist_specs(ctx, specs, note="discuss")
            ctx.store.save_discussion(session, _format(turns) + "\nAPPLIED")
            write(f"applied {len(specs)} spec(s).")
            say({"kind": "applied", "count": len(specs), "specs": specs})
            _end_session(ctx, llm_session)
            return specs
        if choice in ("abort", "a", "n", "no"):
            write("aborted — nothing applied.")
            say({"kind": "aborted"})
            _end_session(ctx, llm_session)
            return []
        # Anything else is the revision. Two forms, because two callers:
        #
        # The CLI's operator types the word `edit` at a `[y/edit/abort]` prompt
        # and expects to be asked for the note — so that read still happens, and
        # it announces itself first. Without the `awaiting` frame it is a read
        # nobody was told about: `Session.reply` 409s anything sent while the
        # status is `running`, so an API caller landing here could not answer and
        # the session sat wedged until the idle TTL closed it.
        #
        # A chat client has no `[y/edit/abort]` prompt to type at. It offers
        # buttons for approve and discard, and a box for the revision — one
        # string, the note itself. Reading that as a *choice* and then asking
        # again would throw the note away and block on a question the operator
        # already answered.
        if choice in ("edit", "e", ""):
            say({"kind": "awaiting", "expects": "answer"})
            note = await _ask(read, "edit note> ")
        else:
            note = raw
        turns.append(("planner", json.dumps(env)))
        turns.append(("user", f"EDIT: {note}"))
        say({"kind": "you", "text": f"EDIT: {note}"})
        delta = f"EDIT: {note}"


#: How long a session may sit frozen waiting for a usage window, when the config
#: says nothing. Six hours clears a five-hour window with room for clock skew;
#: anything longer is a wait the operator should be asked about instead.
DEFAULT_FREEZE_CAP_S = 6 * 3600


def _freeze_cap(cfg) -> float:
    """The longest freeze this project will sit through. 0 disables freezing
    entirely, which turns a limit back into an ordinary failed turn."""
    try:
        return float(cfg.run.get("limit_freeze_max_s", DEFAULT_FREEZE_CAP_S))
    except (AttributeError, TypeError, ValueError):
        return DEFAULT_FREEZE_CAP_S


def _freeze_seconds(exc: Exception, cfg) -> float | None:
    """How long to freeze for this failure, or None to treat it as an ordinary
    failed turn.

    Only a `LimitExhausted` that came with a reported reset time is waited on. A
    limit with no `resets_at` is NOT guessed at — a fabricated wait is how a
    session disappears for hours over what may have been a transient refusal —
    and neither is one further out than the operator's cap.
    """
    if not isinstance(exc, LimitExhausted):
        return None
    wait_s = exc.seconds_until_reset
    if wait_s is None:
        log.warning("limit hit with no reported reset time; not freezing")
        return None
    cap = _freeze_cap(cfg)
    if wait_s > cap or cap <= 0:
        log.warning("limit resets in %.0fs, past the %.0fs cap; not freezing",
                    wait_s, cap)
        return None
    return wait_s


def _end_session(ctx: RunContext, key: str) -> None:
    """Close the provider-side conversation when the loop ends, so the next
    discuss starts clean rather than inheriting this one's context."""
    try:
        provider_name, _ = ctx.role_target("planner")
        get_provider(ctx.cfg, provider_name).end_session(key)
    except Exception as e:      # never let cleanup break a completed discuss
        log.debug("could not end planner session %s: %s", key, e)
