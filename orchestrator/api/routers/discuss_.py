"""The planner chat over HTTP.

Named `discuss_.py` for the same reason as `live_.py`: `api/discuss.py` holds the
session machinery and this module is only the HTTP shell.

Three things here are not obvious from the endpoint list:

* **Starting a session is confirm-gated.** The first planner turn is a real call
  against the subscription tier (measured: 385-425k input tokens), so it is
  treated as spend, like `plan`.
* **A session and a job are mutually exclusive.** `run_discuss` writes to the
  store — transcript, usage, and the specs themselves on approval — and a job
  subprocess is writing the same file. One at a time, enforced both ways.
* **The stream replays.** Every frame has a `seq`, so a reload reconnects with
  `?since=` and gets the conversation back instead of an empty panel with a
  question already asked.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Body, Depends, Query, Request, status
from sse_starlette.sse import EventSourceResponse

from ...core.config import Config
from ...nodes.discuss import DiscussSettings
from .. import discuss as discuss_mod
from ..deps import require_repo_path, resolve_project
from ..discuss import DiscussError, DiscussManager, Session, get_manager
from ..errors import DISCUSS_ERRORS
from ..jobs import JobSupervisor, get_supervisor
from ..schemas import (DiscussFrame, DiscussOptions, DiscussReplyRequest,
                       DiscussSessionModel, DiscussSettingsModel, DiscussState,
                       PinnedFileInfo, PinRequest, StartDiscussRequest,
                       UploadedPin)

router = APIRouter(prefix="/api/projects/{project}", tags=["discuss"],
                   responses=DISCUSS_ERRORS)

#: The efforts `providers.<name>.effort` accepts (config/default.yaml).
EFFORTS = ["low", "medium", "high", "xhigh", "max"]


# ---- projection ----------------------------------------------------------
def _settings_model(settings: DiscussSettings) -> DiscussSettingsModel:
    return DiscussSettingsModel(
        note=settings.note, only_ids=settings.only_ids, effort=settings.effort,
        model=settings.model, session_reuse=settings.session_reuse,
        max_question_rounds=settings.max_question_rounds)


def _session_model(session: Session, *, since: int = 0) -> DiscussSessionModel:
    return DiscussSessionModel(
        session_id=session.session_id, project=session.project,
        request=session.request, status=session.status, expects=session.expects,
        started_at=session.started_at, last_activity_at=session.last_activity,
        error=session.error, applied=session.applied,
        settings=_settings_model(session.settings),
        # Encoded length, not `len(text)`: the cap this is measured against is a
        # byte cap, and a transcript of non-ASCII would otherwise report a size
        # under the limit that the limit had already cut.
        pins=[PinnedFileInfo(path=p.path, bytes=len(p.text.encode("utf-8")),
                             truncated=p.truncated)
              for p in session.settings.pinned],
        frames=[DiscussFrame(seq=f.seq, ts=f.ts, kind=f.kind, data=f.data)
                for f in session.since(since)])


def _options(cfg: Config) -> DiscussOptions:
    roles = cfg.roles.get("planner") or {}
    provider = roles.get("provider") or cfg.roles.get("smart_provider", "claude_cli")
    provider_cfg = cfg.providers.get(provider) or {}
    # The planner runs on the smart tier, so the offered models are that
    # provider's — plus whatever the config already names, so a hand-edited model
    # id never disappears from its own dropdown.
    known = list(provider_cfg.get("models") or [])
    for extra in (roles.get("model"), "opus", "sonnet"):
        if extra and extra not in known:
            known.append(extra)
    return DiscussOptions(
        efforts=EFFORTS, models=known, configured_provider=provider,
        configured_model=roles.get("model"),
        configured_effort=roles.get("effort") or provider_cfg.get("effort"),
        configured_session_reuse=bool(cfg.run.get("session_reuse", False)),
        max_pin_bytes=discuss_mod.MAX_PIN_BYTES,
        idle_ttl_s=discuss_mod.IDLE_TTL_S)


def _apply_settings(session: Session, model: DiscussSettingsModel) -> None:
    """Overwrite the tunables, preserving pins (which have their own endpoints)."""
    session.settings = DiscussSettings(
        note=model.note, only_ids=model.only_ids, effort=model.effort,
        model=model.model, session_reuse=model.session_reuse,
        pinned=session.settings.pinned,
        max_question_rounds=model.max_question_rounds)


# ---- read ----------------------------------------------------------------
@router.get("/discuss", response_model=DiscussState)
async def get_discuss(since: int = Query(0, ge=0, description="replay cursor: return only "
                                                        "frames newer than this seq"),
                cfg: Config = Depends(resolve_project),
                manager: DiscussManager = Depends(get_manager),
                sup: JobSupervisor = Depends(get_supervisor)) -> DiscussState:
    session = manager.for_project(cfg.project_name)
    busy = sup.in_flight(cfg)
    persisted = _persisted(cfg)
    # In-memory first, the store second. A session this process is driving is the
    # live truth; the stored one is a snapshot of it that is at best equal and,
    # mid-turn, a frame or two behind.
    model = _session_model(session, since=since) if session else \
        _restored_session(persisted["log"], since=since)
    return DiscussState(
        project=cfg.project_name,
        session=model,
        # Suppressed once the frame log can speak for itself. The flattened
        # `ROLE: text` blob is the planner's payload, not a conversation, and
        # showing it *beside* the same conversation rendered properly is how the
        # page ended up looking like a data dump with a chat missing.
        transcript="" if model is not None else persisted["transcript"],
        options=_options(cfg),
        blocked_by_job=busy.command if busy is not None else None)


#: Statuses a restored session cannot honestly still claim. The asyncio task that
#: drove the loop died with the process, so nothing is running and nothing is
#: waiting for an answer that could ever be delivered.
_LIVE_STATUSES = ("running", "awaiting")


def _restored_session(log: dict | None, *, since: int) -> DiscussSessionModel | None:
    """The last conversation, rebuilt from the store, read-only.

    Sessions are process-local and always will be — `run_discuss` is an awaited
    coroutine holding a provider subprocess. What is recoverable is what the
    operator was *reading*, and that is worth recovering: restarting the API
    used to blank the planner screen down to a start form and a folded blob of
    `USER:` lines, which is not what the page is for.

    A session caught mid-flight comes back `aborted`, with a frame saying why.
    Reporting the stored `awaiting` instead would put the composer back on
    screen with a question above it and no loop behind it — the operator types
    an answer and the POST 404s on a session id that no longer exists.
    """
    if log is None:
        return None
    frames = [f for f in log.get("frames") or [] if isinstance(f, dict)]
    if not frames:
        return None

    status = str(log.get("status") or "aborted")
    interrupted = status in _LIVE_STATUSES
    if interrupted:
        status = "aborted"
        # Appended rather than replacing the last frame: the conversation ended
        # where it ended, and this is a note about the ending, not part of it.
        # `note` because that is what it is — a loop-level remark, rendered in
        # the transcript's quiet row rather than as a failure.
        frames = [*frames, {
            "seq": max(int(f.get("seq") or 0) for f in frames) + 1,
            "ts": float(log.get("last_activity") or log.get("started_at") or 0.0),
            "kind": "note",
            "data": {"text": "The API restarted while this conversation was open, "
                             "so the planner turn behind it is gone. Starting a new "
                             "session sends this conversation along with it."},
        }]

    return DiscussSessionModel(
        session_id=str(log.get("session_id") or "restored"),
        project=str(log.get("project") or ""),
        request=str(log.get("request") or ""),
        status=status,
        expects=None,
        started_at=float(log.get("started_at") or 0.0),
        last_activity_at=float(log.get("last_activity") or 0.0),
        error=log.get("error") if isinstance(log.get("error"), str) else None,
        applied=[s for s in log.get("applied") or [] if isinstance(s, dict)],
        settings=_settings_model(DiscussSettings()),
        # A pin belonged to a prompt loop that no longer exists, and listing
        # attachments the next session will not inherit would be a lie.
        pins=[],
        frames=[DiscussFrame(seq=int(f.get("seq") or 0), ts=float(f.get("ts") or 0.0),
                             kind=str(f.get("kind") or "note"),
                             data=f.get("data") if isinstance(f.get("data"), dict) else {})
               for f in frames if int(f.get("seq") or 0) > since])


def _persisted(cfg: Config) -> dict:
    """The stored conversation: the flattened transcript and the frame log.

    Reading needs no store write — but `Store()` writes on construction, so this
    goes through the read-only connection instead of instantiating one.
    """
    from ..deps import open_read_only, store_path
    blank: dict = {"transcript": "", "log": None}
    path = store_path(cfg)
    if path is None or not path.exists():
        return blank
    conn = open_read_only(path)
    try:
        row = conn.execute("SELECT * FROM discussions WHERE session=?",
                           (cfg.project_name,)).fetchone()
    except Exception:       # noqa: BLE001 — an old store may predate the table
        return blank
    finally:
        conn.close()
    if row is None:
        return blank
    keys = row.keys()
    raw = row["frames"] if "frames" in keys else None     # a store older than the column
    log = None
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        log = parsed if isinstance(parsed, dict) else None
    return {"transcript": row["transcript"] if "transcript" in keys else "", "log": log}


# ---- lifecycle -----------------------------------------------------------
@router.post("/discuss", response_model=DiscussSessionModel,
             status_code=status.HTTP_201_CREATED)
async def start_discuss(body: StartDiscussRequest = Body(...),
                  cfg: Config = Depends(resolve_project),
                  manager: DiscussManager = Depends(get_manager),
                  sup: JobSupervisor = Depends(get_supervisor)) -> DiscussSessionModel:
    # Demanded even though no pin reads from disk any more: the planner's own
    # turn is an agentic pass over the checkout, so a project without one has
    # nothing to plan against.
    require_repo_path(cfg)
    busy = sup.in_flight(cfg)
    if busy is not None:
        # Same rule as one-job-at-a-time, for the same reason: two writers on one
        # sqlite file. Stated as the job, because that is what the operator stops.
        raise DiscussError(409, f"job {busy.job_id} ({busy.command}) is running — "
                                f"a discuss session writes to the same store")

    settings = DiscussSettings(
        note=body.settings.note, only_ids=body.settings.only_ids,
        effort=body.settings.effort, model=body.settings.model,
        session_reuse=body.settings.session_reuse,
        max_question_rounds=body.settings.max_question_rounds,
        # Validated before the session exists: a binary upload must 4xx here,
        # not after the billable first turn has already started.
        pinned=[discuss_mod.upload_pin(u.name, u.text) for u in body.uploads])
    return _session_model(manager.start(cfg, body.request, settings=settings))


@router.delete("/discuss/{session_id}", response_model=DiscussSessionModel)
async def close_discuss(session_id: str, cfg: Config = Depends(resolve_project),
                  manager: DiscussManager = Depends(get_manager),
                  ) -> DiscussSessionModel:
    """Abort and close. The transcript is already persisted, so history survives."""
    return _session_model(manager.close(session_id))


# ---- the conversation ----------------------------------------------------
@router.post("/discuss/{session_id}/reply", response_model=DiscussSessionModel)
async def reply(session_id: str, body: DiscussReplyRequest = Body(...),
          cfg: Config = Depends(resolve_project),
          manager: DiscussManager = Depends(get_manager)) -> DiscussSessionModel:
    """Answer the pending question, or decide at the preview (`y`/`edit`/`abort`).

    409 rather than a queue when nothing is pending: a reply typed before the
    planner asked would silently become the answer to whatever it asks next.
    """
    session = manager.get(session_id)
    session.reply(body.text)
    return _session_model(session)


@router.get("/discuss/{session_id}/stream", responses={200: {
    "model": DiscussFrame,
    "description": "SSE. Each `message` frame is one DiscussFrame. Reconnect with "
                   "?since=<seq> to replay from a cursor rather than lose the "
                   "conversation. The stream ends after the `closed` frame.",
    "content": {"text/event-stream": {}}}})
async def stream_discuss(session_id: str, request: Request,
                         since: int = Query(0, ge=0),
                         cfg: Config = Depends(resolve_project),
                         manager: DiscussManager = Depends(get_manager),
                         ) -> EventSourceResponse:
    session = manager.get(session_id)

    async def frames():
        # Attach BEFORE the replay snapshot, and deduplicate on the way out.
        # Each `yield` below writes to the socket and suspends, so the planner
        # really can push a frame mid-replay: subscribing afterwards drops it
        # (nothing re-fetches, so a lost `awaiting` hangs the chat until a
        # reload), while subscribing first can only repeat one — and `cursor`
        # already filters repeats. `attach` is synchronous for exactly this
        # reason; an async generator's body would not have run yet here.
        queue = session.attach()
        try:
            pending = session.since(since)
            for frame in pending:
                yield _frame(frame)
            cursor = pending[-1].seq if pending else since
            async for frame in session.subscribe(queue):
                if frame.seq <= cursor:
                    continue
                cursor = frame.seq
                if await request.is_disconnected():
                    break
                yield _frame(frame)
                if frame.kind == "closed":
                    break
        finally:
            # `subscribe` detaches on its own, but only once it has been entered:
            # a client that disconnects during the replay never gets that far.
            session.detach(queue)

    return EventSourceResponse(frames())


def _frame(frame) -> dict:
    # json.dumps for the same reason as the live stream: sse_starlette str()s a
    # non-string payload into a Python repr that no JSON.parse accepts.
    return {"event": "message",
            "data": json.dumps({"seq": frame.seq, "ts": frame.ts,
                                "kind": frame.kind, "data": frame.data})}


# ---- settings & pins -----------------------------------------------------
@router.post("/discuss/{session_id}/settings", response_model=DiscussSessionModel)
async def update_settings(session_id: str, body: DiscussSettingsModel = Body(...),
                    cfg: Config = Depends(resolve_project),
                    manager: DiscussManager = Depends(get_manager),
                    ) -> DiscussSessionModel:
    """Change the session's tunables. Takes effect on the next planner turn —
    including one already being composed, since the loop re-reads at the top."""
    session = manager.get(session_id)
    _apply_settings(session, body)
    return _session_model(session)


@router.post("/discuss/{session_id}/pins", response_model=DiscussSessionModel)
async def upload_pin(session_id: str, body: UploadedPin = Body(...),
               cfg: Config = Depends(resolve_project),
               manager: DiscussManager = Depends(get_manager)) -> DiscussSessionModel:
    """Put a file the operator sent in front of the planner, from the next turn on.

    No `require_repo_path`: a pin is content, never a location, so there is no
    path to resolve and nothing is read from disk.
    """
    session = manager.get(session_id)
    pin = discuss_mod.upload_pin(body.name, body.text)
    session.settings.pinned = [p for p in session.settings.pinned
                               if p.path != pin.path] + [pin]
    return _session_model(session)


@router.post("/discuss/{session_id}/pins/remove", response_model=DiscussSessionModel)
async def remove_pin(session_id: str, body: PinRequest = Body(...),
               cfg: Config = Depends(resolve_project),
               manager: DiscussManager = Depends(get_manager)) -> DiscussSessionModel:
    """Unpin. A POST rather than a DELETE because the display path is a body
    field: a name with slashes in it cannot travel in a path segment."""
    session = manager.get(session_id)
    session.settings.pinned = [p for p in session.settings.pinned
                               if p.path != body.path]
    return _session_model(session)


