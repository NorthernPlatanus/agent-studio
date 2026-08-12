"""The planner chat: `nodes/discuss.run_discuss` driven over HTTP.

`run_discuss` is a terminal program — it prints and it blocks on `input()`. Two
injected seams turn it into a chat without changing what it does (PLAN §3.3):

    read  -> await the operator's next message off a queue
    emit  -> typed frames onto a fan-out, instead of parsing printed prose

Everything else here is the bookkeeping that a browser needs and a terminal does
not: a frame log so a reload replays the conversation instead of losing it, live
settings the operator can change between turns, pinned files, and an explicit
lifecycle so a session cannot be left running against a closed tab.

**This is the one place the API writes to the store.** `run_discuss` persists the
transcript, records planner usage, and on approval upserts specs — all through
the same `Store` the CLI uses. PLAN §3.1 rule 2 keeps *GET* read-only and says
writers are CLI subprocesses; §3.3 designs this session as an in-process task,
and the two are reconciled by making a session mutually exclusive with jobs (see
`DiscussManager.start`) so there is never a second writer at the same time.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from fastapi import HTTPException

from ..core.config import Config
from ..engine import runner
from ..nodes.discuss import DiscussSettings, PinnedFile, run_discuss
from ..ops.store import Store

log = logging.getLogger("orchestrator.api.discuss")

#: A session with no reply for this long is closed — a browser tab that goes away
#: mid-question would otherwise hold the project's write lock forever.
IDLE_TTL_S = 30 * 60

#: Per pinned file. The planner can still `Read` the whole thing; the pin is a
#: prompt-cost decision, and an unbounded one would blow the context on a lockfile.
MAX_PIN_BYTES = 64 * 1024

#: Frames kept for replay. A long clarify loop is tens of frames, not thousands.
MAX_FRAMES = 2000


class DiscussError(HTTPException):
    """HTTP-shaped failure, mirroring `api.jobs.JobError` — an `HTTPException`
    subclass so FastAPI's own handler renders it and no router needs a try."""


class _Aborted(Exception):
    """Raised inside the loop's `read` when the session is being torn down."""


@dataclass
class Frame:
    """One event, as the UI consumes it. `seq` is the replay cursor."""

    seq: int
    ts: float
    kind: str
    data: dict


@dataclass
class Session:
    session_id: str
    project: str
    request: str
    started_at: float
    status: str = "running"          # running | awaiting | done | aborted | failed
    expects: str | None = None       # answer | decision, while status == awaiting
    error: str | None = None
    applied: list[dict] = field(default_factory=list)
    frames: list[Frame] = field(default_factory=list)
    settings: DiscussSettings = field(default_factory=DiscussSettings)
    last_activity: float = field(default_factory=time.time)

    _seq: int = 0
    _inbox: asyncio.Queue = field(default_factory=asyncio.Queue, repr=False)
    _subscribers: list[asyncio.Queue] = field(default_factory=list, repr=False)
    _task: asyncio.Task | None = field(default=None, repr=False)
    _aborting: bool = False

    # -- frames -----------------------------------------------------------
    def push(self, event: dict) -> Frame:
        kind = str(event.get("kind", "note"))
        self._seq += 1
        frame = Frame(seq=self._seq, ts=time.time(), kind=kind,
                      data={k: v for k, v in event.items() if k != "kind"})
        self.frames.append(frame)
        if len(self.frames) > MAX_FRAMES:
            del self.frames[:-MAX_FRAMES]
        # `status` is derived from the frames rather than tracked separately, so
        # a reconnect that replays the log lands in the same state the live
        # stream would have shown.
        if kind == "awaiting":
            self.status, self.expects = "awaiting", event.get("expects")
        elif kind in ("applied", "aborted"):
            self.status = "done" if kind == "applied" else "aborted"
            self.expects = None
        elif kind == "thinking":
            self.status, self.expects = "running", None
        for queue in list(self._subscribers):
            queue.put_nowait(frame)
        return frame

    def since(self, cursor: int) -> list[Frame]:
        return [f for f in self.frames if f.seq > cursor]

    @property
    def live(self) -> bool:
        return self.status in ("running", "awaiting")

    # -- the operator's side ----------------------------------------------
    async def read(self, _prompt: str) -> str:
        """`run_discuss`'s `read`, awaiting the next posted reply."""
        text = await self._inbox.get()
        if text is _ABORT:
            raise _Aborted
        return str(text)

    def reply(self, text: str) -> None:
        if not self.live:
            raise DiscussError(409, f"session {self.session_id} is {self.status}")
        if self.status != "awaiting":
            # Not a queue: a reply nobody asked for would be consumed by whatever
            # question came next, answering it with text written before it existed.
            raise DiscussError(409, "the planner is still working — no question is "
                                    "pending yet")
        self.last_activity = time.time()
        self.status, self.expects = "running", None
        self._inbox.put_nowait(text)

    def attach(self) -> asyncio.Queue:
        """Start receiving frames, **now** — before the caller replays the log.

        Deliberately not an async generator. A generator's body does not run
        until its first `__anext__`, so `live = session.subscribe()` followed by
        a replay loop registered no subscriber at all until the replay was over:
        every frame the planner pushed while the backlog was being written to the
        socket went to an empty subscriber list and was lost. Nothing recovers
        it — `GET …/discuss` is fetched on load and on mutation, never on a
        timer, so a dropped `awaiting` frame leaves the composer disabled and the
        question unrendered until the operator reloads the page.

        Attaching first means the window can only ever *duplicate*, which the
        caller already handles with its `seq` cursor.
        """
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(queue)
        return queue

    def detach(self, queue: asyncio.Queue) -> None:
        with contextlib.suppress(ValueError):
            self._subscribers.remove(queue)

    async def subscribe(self, queue: asyncio.Queue) -> AsyncIterator[Frame]:
        """Frames from an already-attached queue, detaching when the caller stops."""
        try:
            while True:
                yield await queue.get()
        finally:
            self.detach(queue)


_ABORT = object()


class DiscussManager:
    """One session per project, in this process. Sessions are not durable: the
    transcript is (`store.save_discussion`), the pending question is not."""

    def __init__(self, *, idle_ttl_s: float = IDLE_TTL_S):
        self._sessions: dict[str, Session] = {}
        self.idle_ttl_s = idle_ttl_s

    # -- lookup -----------------------------------------------------------
    def get(self, session_id: str) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise DiscussError(404, f"unknown discuss session: {session_id!r}")
        return session

    def for_project(self, project: str) -> Session | None:
        """The session the panel should be showing: the live one, else the last
        one to finish.

        A finished session is not nothing — it holds the specs that were just
        applied and the conversation that produced them, and dropping it the
        instant the loop returns would blank the screen at the moment of the
        result. It is replaced when the next session starts.
        """
        self.reap()
        mine = [s for s in self._sessions.values() if s.project == project]
        return next((s for s in mine if s.live),
                    max(mine, key=lambda s: s.started_at, default=None))

    def active(self, project: str) -> Session | None:
        """Only a LIVE session — the mutual-exclusion question, which a finished
        one must not answer yes to."""
        session = self.for_project(project)
        return session if session is not None and session.live else None

    def reap(self) -> None:
        for session in list(self._sessions.values()):
            idle = time.time() - session.last_activity
            if session.live and idle > self.idle_ttl_s:
                log.info("discuss session %s idle for %.0fs — closing",
                         session.session_id, idle)
                self._tear_down(session, "closed after "
                                         f"{self.idle_ttl_s / 60:.0f} idle minutes")

    # -- lifecycle --------------------------------------------------------
    def start(self, cfg: Config, request: str, *,
              store_factory=Store, settings: DiscussSettings | None = None) -> Session:
        self.reap()
        existing = self.active(cfg.project_name)
        if existing is not None:
            raise DiscussError(409, f"a discuss session ({existing.session_id}) is "
                                    f"already open for {cfg.project_name!r}")
        # The previous, finished session is kept only until this one exists —
        # one conversation per project on screen, and no unbounded history in
        # process memory.
        for old in [s for s in self._sessions.values()
                    if s.project == cfg.project_name and not s.live]:
            del self._sessions[old.session_id]

        session = Session(session_id=uuid.uuid4().hex[:12], project=cfg.project_name,
                          request=request, started_at=time.time(),
                          settings=settings or DiscussSettings())
        self._sessions[session.session_id] = session

        store = store_factory(cfg.store_path())
        run_id = store.create_run(note="discuss")
        ctx = runner.make_context(cfg, store, run_id)
        session._task = asyncio.create_task(
            self._drive(session, ctx, store, run_id, request),
            name=f"discuss:{session.session_id}")
        return session

    async def _drive(self, session: Session, ctx, store: Store, run_id: str,
                     request: str) -> None:
        try:
            session.applied = await run_discuss(
                ctx, request,
                read=session.read,
                write=lambda _line: None,   # the frames carry it; nothing to print
                emit=session.push,
                settings=lambda: session.settings)
            store.set_run_status(run_id, "done")
        except _Aborted:
            session.status = "aborted"
            session.push({"kind": "aborted", "reason": "closed by the operator"})
            store.set_run_status(run_id, "aborted", note="discuss: closed")
        except asyncio.CancelledError:
            store.set_run_status(run_id, "aborted", note="discuss: cancelled")
            raise
        except Exception as e:                      # noqa: BLE001 — reported, not swallowed
            log.exception("discuss session %s failed", session.session_id)
            session.status, session.error = "failed", f"{type(e).__name__}: {e}"
            session.push({"kind": "error", "text": session.error})
            store.set_run_status(run_id, "aborted", note=f"discuss failed: {e}")
        finally:
            session.expects = None
            session.last_activity = time.time()
            session.push({"kind": "closed", "status": session.status})

    def close(self, session_id: str) -> Session:
        return self._tear_down(self.get(session_id), "closed by the operator")

    def _tear_down(self, session: Session, why: str) -> Session:
        if session.live:
            session._aborting = True
            # The sentinel rather than `task.cancel()`: cancelling mid-`await`
            # inside a provider call leaves the subprocess and the run row to be
            # cleaned up by whatever notices; the sentinel unwinds the loop
            # through its own `except`, which does both.
            session._inbox.put_nowait(_ABORT)
            session.error = session.error or why
        return session

    async def shutdown(self) -> None:
        for session in list(self._sessions.values()):
            self._tear_down(session, "the API is shutting down")
        tasks = [s._task for s in self._sessions.values() if s._task is not None]
        if tasks:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait(tasks, timeout=5)


_manager: DiscussManager | None = None


def get_manager() -> DiscussManager:
    """Overridable seam, matching `jobs.get_supervisor`."""
    global _manager
    if _manager is None:
        _manager = DiscussManager()
    return _manager


# ---- pinning -------------------------------------------------------------
# A pin is content the operator sent, never a path they named. Naming a path was
# the earlier design and it is gone: it made the operator hand-type a repo-
# relative path the planner could usually find on its own, and every such path
# had to be re-checked for containment before it could go into a provider
# prompt. Sending the file is what the operator expects from a chat, and it also
# covers the case a path never could — a log or a spec that is not in the repo.

#: Where an uploaded pin's display path lives. Not a real directory — it exists
#: so a pin can never be mistaken for, or collide with, a checkout path.
UPLOAD_PREFIX = "uploaded/"

#: Anything else in a filename is dropped. Uploads are named by the operator's
#: filesystem and the name ends up as a markdown heading in a provider prompt;
#: newlines and backticks in it would let the name break out of that heading.
_SAFE_UPLOAD_NAME = re.compile(r"[^A-Za-z0-9._-]+")

#: Above this share of U+FFFD, the "text" is a binary file the browser decoded
#: with replacement — a PNG comes through as almost nothing else.
_MAX_REPLACEMENT_RATIO = 0.1


def upload_name(name: str) -> str:
    """A filename reduced to something safe to put in a prompt and a dict key."""
    base = PurePosixPath(name.replace("\\", "/")).name
    cleaned = _SAFE_UPLOAD_NAME.sub("-", base).strip("-.")
    return cleaned[:120] or "upload.txt"


def upload_pin(name: str, text: str) -> PinnedFile:
    """Take content the operator sent from their own machine.

    Uploads arrive as text in JSON rather than as a multipart body, because the
    one request that most needs them — `POST …/discuss`, which both creates the
    session and starts the billable first turn — is JSON and cannot be split
    without racing the turn the pins are for. One mechanism for both the staged
    and the live case beats two that can drift.

    The planner prompt is text. An image cannot be sent to it at all, so one is
    refused here with the reason rather than pinned as a page of U+FFFD that
    silently spends context and tells the planner nothing.
    """
    if "\x00" in text:
        raise DiscussError(415, f"{name!r} is a binary file. The planner prompt is "
                                f"text — attach a text file, or paste the relevant "
                                f"part into the message.")
    if text and text.count("\ufffd") / len(text) > _MAX_REPLACEMENT_RATIO:
        raise DiscussError(415, f"{name!r} does not decode as text (it looks like an "
                                f"image or another binary format). The planner "
                                f"reads text only — describe it in the message, or "
                                f"attach a text file.")
    encoded = text.encode("utf-8")
    truncated = len(encoded) > MAX_PIN_BYTES
    if truncated:
        text = encoded[:MAX_PIN_BYTES].decode("utf-8", errors="ignore")
    return PinnedFile(path=f"{UPLOAD_PREFIX}{upload_name(name)}", text=text,
                      truncated=truncated)
