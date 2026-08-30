"""The API app served against a PINNED config root — the capture's second phase.

`scripts/capture-fixtures.py` in the UI repo captures most fixtures from
`orchestrator.api.app:app` directly, which resolves `example` through the real
repo root and therefore through the gitignored `projects/example/profile.yaml`.
That is fine for every endpoint that reads the store (the store is seeded), and
wrong for the two that read CONFIG: their responses would be a photograph of one
machine's local profile.

This factory serves the same app with `deps.get_registry` overridden by a
registry rooted at `$ORCH_FIXTURE_ROOT` — the directory
`seed_store.seed_config_root()` just wrote. That root contains a copy of the
committed `config/default.yaml` (so `presets:`/`providers:` are the real ones)
and a pinned profile (so `worker_models:`/`roles:` are fixed constants), and no
`projects/` tree at all, so nothing local can leak in.

    ORCH_FIXTURE_ROOT=<root> uvicorn --factory \\
        tests.api.fixtures.fixture_app:create --port 8789

A factory rather than a module-level `app`, so importing this module (pytest
collects the package) never depends on the variable being set.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI

from orchestrator.api import deps
from orchestrator.api.app import create_app


def create() -> FastAPI:
    root = Path(os.environ["ORCH_FIXTURE_ROOT"]).expanduser().resolve()
    registry = deps.ProjectRegistry(root=root)
    app = create_app()
    app.dependency_overrides[deps.get_registry] = lambda: registry
    return app
