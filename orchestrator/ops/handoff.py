"""Planner handoff — what survives when a planning session goes cold.

A `discuss` conversation reuses ONE Claude CLI session so that turn 2+ costs a
delta rather than the whole payload. That saving is real only while the
conversation's prefix is still in the prompt cache; past the TTL, resuming
replays everything at full weight (see providers.claude_cli.DEFAULT_SESSION_MAX_IDLE_S).

So an expired session is abandoned, and the next turn starts a fresh one. This
module is what stops that from being an amnesia event: after every planner turn
we write a small DIGEST (what was assumed, what was asked, what was proposed)
plus a full SNAPSHOT on disk. A cold start inlines the digest — a few hundred
tokens — and merely names the snapshot's path.

That split is the whole design. The planner reads the digest for free, decides
whether it already knows enough, and pays for the snapshot only if it doesn't
(it has `Read`). The alternative — replaying the dead conversation to recover
its contents — is exactly the full-price replay the expiry exists to avoid.

Both artifacts are written at the END of a turn, from the envelope already in
hand. Nothing here ever calls a model.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

log = logging.getLogger("orchestrator.handoff")

#: Ceiling for the inlined digest. It rides in front of every cold-start prompt,
#: so it is a recurring cost and must stay small — the snapshot carries detail.
MAX_DIGEST_CHARS = 4000

#: Specs listed by id+title in the digest. Beyond this the list stops earning its
#: tokens and the snapshot is the better answer.
MAX_DIGEST_SPECS = 40


def snapshot_path(ctx) -> Path:
    """Where the full previous proposal lives.

    Beside the project's sqlite state file rather than in the projects/ overlay:
    the store is already the per-project, gitignored, machine-local artifact, and
    tests point it at a tmp dir — which keeps this module from ever writing into
    a real checkout during a test run.
    """
    db = Path(ctx.store.path)
    return db.parent / f"{db.stem}.planner_handoff.json"


def _digest(env: dict, specs: list[dict], when: float,
            carried: bool = False) -> str:
    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(when))
    parts = [f"(written {stamp}, at the end of the previous planning turn)"]

    assumptions = [a for a in (env.get("assumptions") or []) if a]
    if assumptions:
        parts.append("\n## Assumptions the plan already rests on")
        parts += [f"- {a}" for a in assumptions]

    # Tolerant of shape: `questions` is whatever the model emitted, and a model
    # that answers with bare strings instead of {id, q} objects must degrade to a
    # thinner digest, not lose the handoff entirely.
    questions = env.get("questions") or []
    if questions:
        parts.append("\n## Questions last asked (the human's reply is in the "
                     "conversation below, if there is one)")
        for q in questions:
            if isinstance(q, dict):
                parts.append(f"- [{q.get('id', '?')}] {q.get('q', '')}")
            elif q:
                parts.append(f"- {q}")

    if specs:
        when_ = "from an earlier turn" if carried else "last proposed"
        parts.append(f"\n## Specs {when_} ({len(specs)}) — proposed, "
                     f"NOT necessarily approved")
        for spec in specs[:MAX_DIGEST_SPECS]:
            flags = []
            if spec.get("complexity"):
                flags.append(f"complexity={spec['complexity']}")
            if spec.get("risk"):
                flags.append(f"risk={spec['risk']}")
            if not spec.get("agent_able", True):
                flags.append("human-only")
            meta = ("  [" + ", ".join(flags) + "]") if flags else ""
            parts.append(f"- {spec.get('id', '?')}: {spec.get('title', '')}{meta}")
        if len(specs) > MAX_DIGEST_SPECS:
            parts.append(f"- ... and {len(specs) - MAX_DIGEST_SPECS} more "
                         f"(see the snapshot)")

    text = "\n".join(parts)
    if len(text) > MAX_DIGEST_CHARS:
        marker = "\n- ... (truncated)"
        head = text[:MAX_DIGEST_CHARS - len(marker)]
        # Prefer a line boundary, but never let a paragraph with no newline in it
        # push the result back over the cap.
        text = (head.rsplit("\n", 1)[0] if "\n" in head else head) + marker
    return text


def _previous(ctx) -> dict:
    """The last snapshot, or an empty one."""
    try:
        prior = json.loads(snapshot_path(ctx).read_text())
        return prior if isinstance(prior, dict) else {}
    except Exception:                           # noqa: BLE001 — absent/garbage
        return {}


def record(ctx, env: dict) -> None:
    """Persist the digest + snapshot for one completed planner turn.

    Best-effort by construction: this runs immediately after a turn the operator
    has already paid for, and a failure to write a convenience artifact must
    never turn that turn into an error the caller sees.
    """
    session = ctx.cfg.project_name
    specs = [s for s in (env.get("specs") or []) if isinstance(s, dict)]
    assumptions = [a for a in (env.get("assumptions") or []) if a]
    # A clarifying turn returns questions and NOTHING else. Overwriting the
    # record with those empty lists would erase the proposal and the premises an
    # earlier turn established — and those are the most valuable things the
    # digest carries, being exactly what an expired session takes with it. So
    # empty means "nothing new to say", not "there are none". Questions are the
    # exception: they are a live prompt to the operator, and a stale one is worse
    # than none, so they are always replaced.
    prior = _previous(ctx) if not (specs and assumptions) else {}
    carried = False
    if not specs:
        specs = [s for s in (prior.get("specs") or []) if isinstance(s, dict)]
        carried = bool(specs)
    if not assumptions:
        assumptions = [a for a in (prior.get("assumptions") or []) if a]
    now = time.time()
    path: str | None = None
    try:
        target = snapshot_path(ctx)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(
            {"written_at": now, "session": session,
             "questions": env.get("questions") or [],
             "assumptions": assumptions,
             "specs": specs}, indent=1))
        path = str(target)
    except Exception as e:                      # noqa: BLE001 — never fail a turn
        log.debug("could not write planner snapshot: %s", e)
    try:
        ctx.store.save_handoff(
            session, _digest({**env, "assumptions": assumptions}, specs, now,
                             carried), path)
    except Exception as e:                      # noqa: BLE001 — never fail a turn
        log.debug("could not save planner handoff: %s", e)


def clear(ctx) -> None:
    """Forget the handoff — the conversation it bridges has concluded.

    The digest exists to survive an INTERRUPTED session. Once specs are applied
    or the operator walks away, replaying its open questions into the next,
    unrelated conversation would have the planner re-litigating settled ground.
    """
    try:
        ctx.store.save_handoff(ctx.cfg.project_name, "", None)
    except Exception as e:                      # noqa: BLE001 — cleanup only
        log.debug("could not clear planner handoff: %s", e)
    try:
        snapshot_path(ctx).unlink(missing_ok=True)
    except Exception as e:                      # noqa: BLE001 — cleanup only
        log.debug("could not remove planner snapshot: %s", e)


def prompt_block(ctx) -> str:
    """The `# PREVIOUS PLANNING SESSION` block, or "" when there is nothing.

    The closing instruction is load-bearing: without it the planner treats the
    named path like any other file it ought to verify and opens it every turn,
    which reinstates the cost the digest exists to avoid.
    """
    try:
        row = ctx.store.load_handoff(ctx.cfg.project_name)
    except Exception as e:                      # noqa: BLE001 — optional context
        log.debug("could not load planner handoff: %s", e)
        return ""
    if not row or not (row.get("digest") or "").strip():
        return ""
    parts = ["# PREVIOUS PLANNING SESSION (digest)",
             "Our earlier conversation is no longer live, so this is what carries "
             "over from it.",
             "",
             row["digest"]]
    path = row.get("snapshot_path")
    if path and Path(path).exists():
        parts += ["",
                  f"The full previous proposal is at `{path}` (JSON: questions, "
                  f"assumptions, complete specs).",
                  "Read it ONLY if this digest leaves you unable to answer the "
                  "request — it is a fallback, not a required step, and the "
                  "specs it contains are also summarised under CURRENTLY PLANNED "
                  "SPECS below."]
    return "\n".join(parts)
