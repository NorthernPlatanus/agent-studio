"""`GET …/stream` — the SSE endpoint.

Named `live_.py` to match the plan's naming and to avoid shadowing `api/live.py`,
which holds all the logic; this module is only the HTTP wrapper.

The route is deliberately absent from the generated TypeScript's useful surface:
`EventSource` is not `fetch`, so `studio-web` hand-writes the subscription in
`shared/api/sse.ts`. What the contract still owes it is the *payload* shapes, so
the SSE frame models are declared in `schemas.py` and referenced here even though
no response body is ever serialized through them.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from sse_starlette.sse import EventSourceResponse

from ...core.config import Config
from .. import live
from ..deps import resolve_project, store_path
from ..errors import PROJECT_ERRORS
from ..jobs import JobSupervisor, get_supervisor
from ..schemas import StreamFrame

router = APIRouter(prefix="/api/projects/{project}", tags=["live"],
                   responses=PROJECT_ERRORS)


@router.get("/stream", responses={200: {
    "model": StreamFrame,
    "description": "SSE. Named events: hello, tasks, runs, usage, events, jobs, "
                   "heartbeat. Every one but `events` carries only a cursor — "
                   "refetch the entity through its normal GET. The declared model "
                   "is one frame's `data`; the frame name arrives in the SSE "
                   "`event:` line.",
    "content": {"text/event-stream": {}}}})
async def stream(request: Request, cfg: Config = Depends(resolve_project),
                 sup: JobSupervisor = Depends(get_supervisor)) -> EventSourceResponse:
    store: Path | None = store_path(cfg)

    async def frames():
        async for frame in live.stream(cfg, store, sup):
            # `sse_starlette` also drops the generator on disconnect, but only
            # after the next yield — which is up to a poll interval away. This
            # ends the loop at the top of the tick instead.
            if await request.is_disconnected():
                break
            # `json.dumps`, not the dict itself: `sse_starlette` writes a
            # non-string `data` through `str()`, which emits a Python repr —
            # `'single quotes'`, `None`, `False` — and every `JSON.parse` in the
            # browser fails on it. `live.stream` keeps yielding dicts because
            # that is what its own tests read; the wire format belongs here.
            yield {"event": frame["event"], "data": json.dumps(frame["data"])}

    # No 409 when the store is missing: a project that has never run still has a
    # meaningful stream (jobs, and the moment its store appears).
    return EventSourceResponse(frames())
