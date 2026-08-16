"""App-level guarantees: health, schema, CORS, and the project allowlist."""

from __future__ import annotations

from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from starlette.responses import StreamingResponse

from orchestrator.api.app import DEV_ORIGINS, create_app


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_openapi_documents_every_read_endpoint(client):
    paths = client.get("/openapi.json").json()["paths"]
    for path in ("/api/projects",
                 "/api/projects/{project}/summary",
                 "/api/projects/{project}/tasks",
                 "/api/projects/{project}/tasks/{task_id}",
                 "/api/projects/{project}/tasks/{task_id}/candidates",
                 "/api/projects/{project}/waves",
                 "/api/projects/{project}/runs",
                 "/api/projects/{project}/runs/{run_id}",
                 "/api/projects/{project}/usage",
                 "/api/projects/{project}/metrics",
                 "/api/projects/{project}/events",
                 "/api/projects/{project}/jobs"):
        assert path in paths, path
        assert "get" in paths[path]


def test_cors_allows_the_vite_dev_server(client):
    for origin in DEV_ORIGINS:
        r = client.get("/healthz", headers={"Origin": origin})
        assert r.headers["access-control-allow-origin"] == origin


def test_a_crash_answers_json_with_cors_headers_not_a_bare_500():
    """The failure mode that hid the `reconcile` bug for a whole browser session.

    An unhandled exception propagates past `CORSMiddleware` to Starlette's own
    error handler, which answers `text/plain` with no `Access-Control-Allow-
    Origin`. The browser then reports "blocked by CORS policy" and the real
    status never reaches devtools — a server bug that reads as a config one.
    `JsonErrors` sits inside the CORS layer so the answer is a normal response
    by the time CORS sees it.
    """
    app = create_app()

    @app.get("/boom")
    def boom():
        raise RuntimeError("kaboom")

    origin = DEV_ORIGINS[0]
    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.get("/boom", headers={"Origin": origin})

    assert r.status_code == 500
    assert r.headers["access-control-allow-origin"] == origin
    assert r.headers["content-type"].startswith("application/json")
    # The message, not a traceback: this binds to loopback, but the detail is for
    # a human reading devtools, and the traceback goes to the server log.
    assert r.json() == {"detail": "RuntimeError: kaboom", "path": "/boom"}


def test_the_error_wrapper_stays_out_of_the_way_of_a_stream():
    """`JsonErrors` must stay raw ASGI, for the sake of the two SSE endpoints.

    `BaseHTTPMiddleware` — the obvious way to write this — pumps every response
    body through an anyio stream pair inside a task group, which is the
    documented way to stall a long-lived stream. Asserted structurally rather
    than by consuming `/stream`, because that endpoint never ends: a test that
    read it would hang exactly as long as the bug it was meant to catch.
    """
    from starlette.middleware.base import BaseHTTPMiddleware

    from orchestrator.api.app import JsonErrors

    app = create_app()
    classes = [m.cls for m in app.user_middleware]
    assert BaseHTTPMiddleware not in classes
    # Inside the CORS layer: `add_middleware` prepends, so later = further out,
    # and only an error caught *within* CORS gets its headers.
    assert classes.index(JsonErrors) > classes.index(CORSMiddleware)

    @app.get("/chunks")
    def chunks():
        return StreamingResponse((f"{i}\n".encode() for i in range(3)),
                                 media_type="text/plain")

    with TestClient(app) as client:
        r = client.get("/chunks")
    assert r.status_code == 200 and r.text == "0\n1\n2\n"


def test_unknown_project_is_404(client):
    assert client.get("/api/projects/nope/summary").status_code == 404
    assert client.get("/api/projects/nope/tasks").status_code == 404


def test_traversal_in_the_project_name_is_404(client):
    # %2F survives into the path parameter, so `project` really is "../etc" here.
    for name in ("%2E%2E%2Fetc", "%2E%2E", "not-in-the-allowlist"):
        r = client.get(f"/api/projects/{name}/summary")
        assert r.status_code == 404, (name, r.status_code)
