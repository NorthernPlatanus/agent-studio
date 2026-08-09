"""ASGI app factory.

Run it with `orchestrator serve` or
`uvicorn orchestrator.api.app:app --host 127.0.0.1 --port 8787`.

Bound to loopback with no auth on purpose: this is a single-user local tool, and
the endpoints that spend subscription quota (phase 3) are reachable by anyone who
can reach the port. Do not expose it.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute

from .routers import events, jobs, live_, projects, runs, tasks, usage
from .schemas import Health

API_VERSION = "0.1.0"

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


def create_app(*, cors_origins: tuple[str, ...] = DEV_ORIGINS) -> FastAPI:
    app = FastAPI(
        title="agent-studio control panel API",
        version=API_VERSION,
        description="Read/control layer over the orchestrator's store, scheduler "
                    "and CLI. GET endpoints are strictly read-only.",
        generate_unique_id_function=operation_id,
    )
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

    for module in (projects, tasks, runs, usage, events, jobs, live_):
        app.include_router(module.router)
    return app


app = create_app()
