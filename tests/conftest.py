"""Isolate the test suite from this machine's configuration.

`load_config()` layers `config/local.yaml` and then any `ORCH_*` environment
overrides on top of `config/default.yaml`. Both are per-machine, and neither is
something a unit test asked for. Concretely: a real `gate.install_cmd` in
local.yaml makes every worker test's `ensure_deps` shell out to an actual
`npm ci` inside a temp worktree with no package-lock, so ten tests fail with an
`IndexError` on an empty provider-call list — a suite whose result depends on
whose laptop it runs on, and a failure that says nothing about the real cause.

Tests that want a non-default config build it explicitly instead.
"""
import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolate_machine_config():
    saved = {k: v for k, v in os.environ.items() if k.startswith("ORCH_")}
    for key in saved:
        del os.environ[key]
    os.environ["ORCH_SKIP_LOCAL_CONFIG"] = "1"
    yield
    os.environ.pop("ORCH_SKIP_LOCAL_CONFIG", None)
    os.environ.update(saved)
