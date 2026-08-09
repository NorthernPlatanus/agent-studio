"""App-level guarantees: health, schema, CORS, and the project allowlist."""

from __future__ import annotations

from orchestrator.api.app import DEV_ORIGINS


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


def test_unknown_project_is_404(client):
    assert client.get("/api/projects/nope/summary").status_code == 404
    assert client.get("/api/projects/nope/tasks").status_code == 404


def test_traversal_in_the_project_name_is_404(client):
    # %2F survives into the path parameter, so `project` really is "../etc" here.
    for name in ("%2E%2E%2Fetc", "%2E%2E", "demo-project"):
        r = client.get(f"/api/projects/{name}/summary")
        assert r.status_code == 404, (name, r.status_code)
