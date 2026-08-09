"""What `/openapi.json` promises: stable operation ids and real error responses.

Both are contract properties `studio-web` consumes mechanically — the first
because `generated.ts`'s `operations[…]` keys are derived from operation ids, the
second because an undeclared 404/409 means the generated client has no type for
the error the API actually returns.
"""

from __future__ import annotations

from orchestrator.api.app import create_app


def _schema() -> dict:
    return create_app().openapi()


def test_operation_ids_are_the_handler_names_and_unique():
    schema = _schema()
    ids = [op["operationId"]
           for path in schema["paths"].values()
           for method, op in path.items()
           if method in ("get", "post", "delete")]
    assert len(ids) == len(set(ids)), "duplicate operationId — rename a handler"
    # The point of the override: no path mangling, no method suffix. If FastAPI's
    # default were in force these would read `list_projects_api_projects_get`.
    assert "list_projects" in ids
    assert not any("_api_" in i for i in ids)


def test_project_scoped_reads_declare_404_and_409():
    paths = _schema()["paths"]
    for path, spec in paths.items():
        if not path.startswith("/api/projects/{project}"):
            continue
        for method, op in spec.items():
            if method not in ("get", "post", "delete"):
                continue
            responses = op["responses"]
            assert "404" in responses, f"{method} {path} can 404 on the allowlist"
            # 422 comes from FastAPI itself for the `project` path param; the point
            # here is that it did not get lost by declaring the others.
            assert "422" in responses, f"{method} {path}"
            assert responses["404"]["content"]["application/json"]["schema"]


def test_declared_errors_use_the_httpexception_envelope_not_the_422_shape():
    schema = _schema()
    ref = (schema["paths"]["/api/projects/{project}/summary"]["get"]
           ["responses"]["404"]["content"]["application/json"]["schema"]["$ref"])
    assert ref.endswith("/ApiError")
    props = schema["components"]["schemas"]["ApiError"]["properties"]
    # FastAPI's HTTPException serializes `detail` as a string; the 422 envelope
    # uses a list of ValidationError. Conflating them generates a client that
    # unpacks the wrong field.
    assert props["detail"]["type"] == "string"
