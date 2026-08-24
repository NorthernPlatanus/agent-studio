"""The declared error surface.

FastAPI documents `200` and (wherever a parameter is validated) `422` on its own,
but an `HTTPException` raised in a handler is invisible to the schema — so a
generated client sees no error type at all and the UI ends up hand-typing what
the API already promises. These `responses=` fragments put the 404/409 envelopes
that `deps.py` and the routers actually raise into `/openapi.json`.

`ApiError` mirrors FastAPI's own `HTTPException` envelope (`{"detail": "..."}`),
which is deliberately NOT the 422 shape: validation errors carry a list of
`ValidationError` objects instead, and pretending otherwise would generate a
client that unpacks the wrong field.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ApiError(BaseModel):
    """FastAPI's `HTTPException` envelope: a single human-readable string."""
    detail: str = Field(description="human-readable reason; safe to show verbatim")


def _r(description: str) -> dict[str, Any]:
    return {"model": ApiError, "description": description}


# Every route under /api/projects/{project} can 404 on the allowlist.
UNKNOWN_PROJECT = _r("project is not in the allowlist")

# Every route that opens the store can 409 before it reads anything.
NO_STORE = _r("the project is real but has no state/<project>.sqlite3 yet")

INCOMPLETE_PROFILE = _r("the profile has no project.repo_path, so nothing that "
                        "needs a checkout can run")
JOB_IN_FLIGHT = _r("another job for this project is still running")
UNKNOWN_JOB = _r("no such job in this API process's registry")
UNKNOWN_SESSION = _r("no such discuss session (never started, or expired)")

# Composed sets, one per router, so a new route cannot forget one.
PROJECT_ERRORS: dict[int | str, dict[str, Any]] = {404: UNKNOWN_PROJECT}
READ_ERRORS: dict[int | str, dict[str, Any]] = {404: UNKNOWN_PROJECT, 409: NO_STORE}
ROW_ERRORS: dict[int | str, dict[str, Any]] = {
    404: _r("unknown project, or no such row in this project"),
    409: NO_STORE,
}
JOB_READ_ERRORS: dict[int | str, dict[str, Any]] = {
    404: _r("unknown project, or unknown job id"),
}
JOB_SPAWN_ERRORS: dict[int | str, dict[str, Any]] = {
    404: _r("unknown project (and, for resume, no paused run to continue)"),
    409: _r("a job is already in flight for this project, the profile has no "
            "project.repo_path, or the store does not exist yet"),
}
CONFIG_WRITE_ERRORS: dict[int | str, dict[str, Any]] = {
    404: UNKNOWN_PROJECT,
    409: _r("a job or a discuss session is live for this project, so its "
            "assignment overlay is locked — the same one-writer rule as a spawn"),
}
DISCUSS_ERRORS: dict[int | str, dict[str, Any]] = {
    404: _r("unknown project, or unknown discuss session"),
    409: _r("a discuss session is already open for this project, or the profile "
            "has no project.repo_path"),
}
