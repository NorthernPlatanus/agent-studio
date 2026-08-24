"""ASGI app factory.

Run it with `orchestrator serve` or
`uvicorn orchestrator.api.app:app --host 127.0.0.1 --port 8787`.

Bound to loopback with no auth on purpose: this is a single-user local tool, and
the endpoints that spend subscription quota (phase 3) are reachable by anyone who
can reach the port. Do not expose it.
"""

from __future__ import annotations

import logging
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from starlette.responses import JSONResponse

from .discuss import get_manager
from .routers import (config_, discuss_, events, jobs, live_, projects, runs,
                      tasks, usage)
from .schemas import Health

API_VERSION = "0.1.0"

log = logging.getLogger("orchestrator.api")

# The Vite dev server, both spellings — a browser treats 127.0.0.1 and localhost
# as different origins, and which one the user types is not predictable.
DEV_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")


def operation_id(route: APIRoute) -> str:
    """The handler's own name, as the OpenAPI operation id.

    FastAPI's default is `<name>_<mangled path>_<method>`, so
    `generated.ts`'s `operations[…]` keys churn whenever a path or a method
    changes — `studio-contract` had to tell the UI to avoid them entirely and use
    `components["schemas"]` instead. Handler names are the stable half, and
    `tests/api/test_openapi.py` asserts they stay unique, which is the one
    property this trades away.
    """
    return route.name


class JsonErrors:
    """Turn an unhandled exception into a JSON response, inside the CORS layer.

    Raw ASGI rather than `BaseHTTPMiddleware` on purpose: this app streams two
    SSE endpoints (`/live`, `/discuss/{id}/stream`), and `BaseHTTPMiddleware`
    pumps every response body through an anyio stream pair and holds the request
    open in a task group — the documented source of stalled long-lived streams.
    This wrapper touches nothing on the success path; it only substitutes a
    response when the app raises *before* sending its response start.

    Once a stream has begun there is no status line left to change, so a mid-body
    failure is re-raised for the server to log and drop the connection — which is
    what the EventSource `onerror` reconnect on the client already handles.

    `detail` carries the exception's type and message. This binds to loopback with
    no auth, so there is no untrusted reader to leak it to, and the alternative (a
    bare `Internal Server Error` in `text/plain`, with no CORS headers) is what
    made the `reconcile` bug take a browser session to find.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = False

        async def _send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, _send)
        except Exception as e:                                # noqa: BLE001
            log.error("unhandled error on %s %s\n%s", scope.get("method"),
                      scope.get("path"), traceback.format_exc())
            if started:
                raise
            response = JSONResponse(
                status_code=500,
                content={"detail": f"{type(e).__name__}: {e}",
                         "path": scope.get("path", "")})
            await response(scope, receive, send)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    # A discuss session outliving the process is a provider subprocess nobody
    # will reap and a run row left `running` forever — the exact zombie
    # `reconcile` exists to clean up. Unwind them through their own abort path.
    await get_manager().shutdown()


def create_app(*, cors_origins: tuple[str, ...] = DEV_ORIGINS) -> FastAPI:
    app = FastAPI(
        lifespan=lifespan,
        title="agent-studio control panel API",
        version=API_VERSION,
        description="Read/control layer over the orchestrator's store, scheduler "
                    "and CLI. GET endpoints are strictly read-only.",
        generate_unique_id_function=operation_id,
    )
    # Added FIRST so it ends up INSIDE the CORS layer: `add_middleware` prepends,
    # so the last one added is the outermost. Order is the whole point here — an
    # unhandled exception propagates past CORSMiddleware untouched, up to
    # Starlette's own error middleware, which answers `500 Internal Server Error`
    # with no `Access-Control-Allow-Origin`. The browser then reports a *CORS*
    # failure and the 500 never reaches devtools, so a server bug is misdiagnosed
    # as a config one. (Observed: `Job.command` had no `reconcile` member, and
    # every `GET /jobs` poll after the first reconcile read as "blocked by CORS
    # policy".) Catching inside the CORS layer means the error response is a
    # normal response by the time CORS sees it, and gets its headers.
    app.add_middleware(JsonErrors)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cors_origins),
        allow_credentials=False,      # no cookies, no auth — nothing to send
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/healthz", response_model=Health, tags=["meta"])
    def healthz() -> Health:
        return Health(status="ok", version=API_VERSION)

    for module in (projects, tasks, runs, usage, events, jobs, discuss_, config_,
                   live_):
        app.include_router(module.router)
    return app


app = create_app()
