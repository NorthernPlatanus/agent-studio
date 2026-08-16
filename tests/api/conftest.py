"""Fixtures for the API tests.

Configuration is built EXPLICITLY here. `tests/conftest.py` deletes every `ORCH_*`
variable for the whole session, so the `ORCH_PATHS_STATE_DIR` trick that isolates
an out-of-process server silently does nothing inside pytest — a test that relied
on it would look isolated while reading the operator's real store.
So: copy the `example` profile's config, point `paths.state_dir` at a tmpdir, and
inject it through `deps.get_registry`.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from orchestrator.api import deps
from orchestrator.api.app import create_app
from orchestrator.core.config import Config, load_config
from tests.api.fixtures import seed_store

PROJECT = seed_store.PROJECT


@pytest.fixture(scope="session")
def state_dir(tmp_path_factory) -> Path:
    """A seeded fixture state dir — store plus checkpoints, session-scoped.

    Session scope is safe precisely because every endpoint under test is
    read-only; `test_read_only.py` asserts that rather than assuming it.
    """
    directory = tmp_path_factory.mktemp("as-fixture-state")
    seed_store.seed(directory / f"{PROJECT}.sqlite3")
    return directory


@pytest.fixture(scope="session")
def cfg(state_dir: Path) -> Config:
    base = load_config(PROJECT, root=deps.REPO_ROOT)
    data = copy.deepcopy(base.as_dict())
    data["paths"]["state_dir"] = str(state_dir)
    return Config(data, PROJECT, deps.REPO_ROOT)


@pytest.fixture(scope="session")
def registry(cfg: Config) -> deps.ProjectRegistry:
    # root=state_dir: an empty tree, so the allowlist is exactly the injected
    # project and no test can reach a real profile under `projects/`.
    return deps.ProjectRegistry(root=Path(cfg.paths.state_dir),
                                configs={PROJECT: cfg})


@pytest.fixture
def client(registry: deps.ProjectRegistry):
    app = create_app()
    app.dependency_overrides[deps.get_registry] = lambda: registry
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def store_path(state_dir: Path) -> Path:
    return state_dir / f"{PROJECT}.sqlite3"


@pytest.fixture
def conn(store_path: Path):
    connection = deps.open_read_only(store_path)
    try:
        yield connection
    finally:
        connection.close()
