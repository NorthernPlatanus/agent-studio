"""The contract `agent-studio-ui/scripts/capture-fixtures.py` captures against.

The UI repo's config fixtures are captured from `fixtures.fixture_app` served
over a config root that `seed_store.seed_config_root()` writes, with the panel's
overlay beside the store. Nothing in this repo's own test suite exercised that
combination, and it is the combination that has to hold: the fixtures are
committed, so a capture that answers with the wrong layers produces mocks the
frontend is then built against.

That is not hypothetical. The first capture of these two fixtures was silently
served by an orphaned `fixture_app` server left listening on the capture port by
an earlier session — `uvicorn` exits at once when the port is bound, and the
capture's health loop accepted the 200 from the stranger. The committed artifact
showed every row as `source: "profile"`, which reads as "the overlay does not
work" rather than "you are talking to the wrong process". The capture script now
checks that its own child is alive; this file pins the half that lives here.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.api.fixtures import fixture_app, seed_store

PROJECT = seed_store.PROJECT


@pytest.fixture
def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """The capture's server, over a freshly seeded root — its exact arrangement."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    seed_store.seed(state_dir / f"{PROJECT}.sqlite3")
    seed_store.seed_assignments(state_dir)
    root = seed_store.seed_config_root(tmp_path / "config-root", state_dir)
    monkeypatch.setenv("ORCH_FIXTURE_ROOT", str(root))
    with TestClient(fixture_app.create()) as client:
        yield client


def test_config_root_is_the_only_project_the_capture_can_see(served: TestClient):
    # The whole point of the pinned root: no `projects/` tree, so a capture
    # cannot photograph this machine's profiles — or even learn their names.
    body = served.get("/api/projects").json()
    assert [p["name"] for p in body["projects"]] == [PROJECT]


def test_presets_come_from_the_committed_defaults(served: TestClient):
    presets = {p["key"]: p for p in served.get(
        f"/api/projects/{PROJECT}/config/presets").json()["presets"]}
    # Real, not invented: these are `config/default.yaml`'s own keys, which is
    # what makes "add a preset and re-capture" a working workflow.
    assert {"claude_cli_sonnet", "deepseek_flash"} <= set(presets)
    # A CLI preset rides a subscription and reports no prices; an API one spends
    # cash and reports both. The UI renders the two differently, so both shapes
    # have to be present in the captured catalogue.
    assert presets["claude_cli_sonnet"]["cash"] is False
    assert presets["deepseek_flash"]["cash"] is True
    # `efforts: []` is the "this backend has no reasoning dial" signal the panel
    # disables its effort control on — not an omission.
    assert presets["deepseek_flash"]["efforts"] == []
    assert presets["claude_cli_sonnet"]["efforts"]


def test_no_environment_value_reaches_the_captured_response(served: TestClient):
    # The same check the capture runs over the artifact before writing it. Here
    # because a fixture is the one place a leak would be permanent.
    text = served.get(f"/api/projects/{PROJECT}/config/presets").text
    for name, value in os.environ.items():
        if len(value) >= 8:
            assert value not in text, f"${name} leaked into the response"


def test_the_seeded_overlay_is_read_and_reported_as_the_overlay(served: TestClient):
    """The assertion the broken capture would have failed.

    `seed_store.ASSIGNMENT_OVERLAY` rebinds one worker and one role. Both must
    come back as `source: "overlay"` carrying the preset's own label, and every
    untouched row as `source: "profile"` — that pair is what lets the panel tell
    "leave this to the YAML" apart from "the operator chose this", which is the
    difference between omitting a row on save and pinning it.
    """
    body = served.get(f"/api/projects/{PROJECT}/config/assignments").json()
    workers = {row["key"]: row for row in body["workers"]}
    roles = {row["key"]: row for row in body["roles"]}

    assert workers["flash_hi"]["source"] == "overlay"
    assert workers["flash_hi"]["preset"] == "claude_cli_sonnet"
    assert roles["planner"]["source"] == "overlay"
    assert roles["planner"]["effort"] == "high"

    # `flash_mid` is bound to a preset BY THE PROFILE. A projection that read
    # `preset is not None` as "the overlay did this" would call it an overlay row
    # and the panel would pin it on the next save.
    assert workers["flash_mid"]["preset"] == "deepseek_flash"
    assert workers["flash_mid"]["source"] == "profile"
    # And an entry that names no preset at all — the pre-preset shape every old
    # profile still has — is a profile row with a null preset, not an error.
    assert workers["flash_lo"]["preset"] is None
    assert workers["flash_lo"]["source"] == "profile"
    assert roles["reviewer"]["source"] == "profile"

    assert body["default_worker"] == "flash_mid"
    assert body["locked"] is False


def test_seeding_is_reproducible(tmp_path: Path):
    """Two seeds of the same root differ in nothing — the capture is byte-stable.

    The committed fixtures are diffed by hand on every re-capture, so a seeder
    that varied per run would make that review worthless.
    """
    first = seed_store.seed_config_root(tmp_path / "a", tmp_path / "state")
    second = seed_store.seed_config_root(tmp_path / "b", tmp_path / "state")
    for name in ("default.yaml", f"projects/{PROJECT}.yaml"):
        assert (first / "config" / name).read_text() == (second / "config" / name).read_text()
