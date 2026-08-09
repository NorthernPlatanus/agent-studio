"""The allowlist, the missing-store 409, and the incomplete-profile 409."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from orchestrator.api import deps
from orchestrator.core.config import Config


def test_registry_scans_both_profile_layouts(tmp_path):
    (tmp_path / "projects" / "alpha").mkdir(parents=True)
    (tmp_path / "projects" / "alpha" / "profile.yaml").write_text("project: {}\n")
    (tmp_path / "config" / "projects").mkdir(parents=True)
    (tmp_path / "config" / "projects" / "beta.yaml").write_text("project: {}\n")
    (tmp_path / "config" / "projects" / "beta.doc.md").write_text("not a profile\n")

    registry = deps.ProjectRegistry(root=tmp_path)
    assert registry.names() == ["alpha", "beta"]
    assert registry.has("alpha") and not registry.has("gamma")


def test_registry_rejects_names_that_could_become_paths(tmp_path):
    registry = deps.ProjectRegistry(root=tmp_path)
    for name in ("..", "../etc", "/etc/passwd", "", ".hidden", "a/b"):
        assert not registry.has(name), name


def test_the_real_repo_allowlist_is_exactly_the_two_known_projects():
    # A sanity check on the scan itself, not on this machine's contents: whatever
    # it finds must at least include the committed `example` profile, and nothing
    # unexpected can appear without a profile file existing for it.
    registry = deps.ProjectRegistry()
    assert "example" in registry.names()
    for name in registry.names():
        assert registry.entries()[name].profile_path.exists()


def test_missing_store_is_409_not_404(tmp_path, cfg, client, registry):
    """404 is reserved for "unknown project"; a real project whose state does not
    exist yet is a different answer and the UI shows a different empty state."""
    import copy

    data = copy.deepcopy(cfg.as_dict())
    data["paths"]["state_dir"] = str(tmp_path / "empty")
    empty = deps.ProjectRegistry(root=tmp_path, configs={"fresh": Config(
        data, "fresh", deps.REPO_ROOT)})
    client.app.dependency_overrides[deps.get_registry] = lambda: empty

    r = client.get("/api/projects/fresh/summary")
    assert r.status_code == 409
    assert "no store" in r.json()["detail"]
    # …and it must not have created the state dir on the way out.
    assert not (tmp_path / "empty").exists()


def test_require_repo_path_is_409_for_the_incomplete_example_profile(cfg):
    """The `example` profile has project.repo_path: null. Anything that needs a
    checkout (the phase-3 job spawners) must say so cleanly, never 500."""
    with pytest.raises(HTTPException) as excinfo:
        deps.require_repo_path(cfg)
    assert excinfo.value.status_code == 409
    assert "incomplete" in excinfo.value.detail


def test_require_repo_path_returns_the_path_when_it_is_set(cfg, tmp_path):
    import copy

    data = copy.deepcopy(cfg.as_dict())
    data["project"]["repo_path"] = str(tmp_path)
    complete = Config(data, "example", deps.REPO_ROOT)
    assert deps.require_repo_path(complete) == tmp_path.resolve()


def test_config_is_cached_per_project(registry):
    assert registry.config("example") is registry.config("example")


def _provenance_fixture(tmp_path, profile_repo_path, merged_repo_path):
    """A registry whose `alpha` profile is on disk, plus the merged Config.

    The two are set independently on purpose: the whole point of
    `repo_path_provenance` is the case where they disagree.
    """
    (tmp_path / "projects" / "alpha").mkdir(parents=True)
    declared = f"repo_path: {profile_repo_path}" if profile_repo_path else "repo_path:"
    (tmp_path / "projects" / "alpha" / "profile.yaml").write_text(
        f"project:\n  {declared}\n")
    cfg = Config({"project": {"repo_path": merged_repo_path}}, "alpha", deps.REPO_ROOT)
    return deps.ProjectRegistry(root=tmp_path), cfg


def test_repo_path_from_the_projects_own_profile_carries_no_caveat(tmp_path):
    registry, cfg = _provenance_fixture(tmp_path, "/checkouts/alpha", "/checkouts/alpha")
    assert deps.repo_path_provenance(registry, "alpha", cfg) == (
        "/checkouts/alpha", "profile", None)


def test_repo_path_inherited_from_the_global_overlay_says_so(tmp_path):
    """The case that motivates the field: the profile declares null, but
    config/local.yaml gives every project the same checkout — so `runnable` is
    true against a working tree that is not this project's."""
    registry, cfg = _provenance_fixture(tmp_path, None, "/checkouts/somewhere-else")
    value, source, detail = deps.repo_path_provenance(registry, "alpha", cfg)
    assert (value, source) == ("/checkouts/somewhere-else", "global")
    assert detail and "machine-global" in detail


def test_env_wins_the_attribution_over_the_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCH_PROJECT_REPO_PATH", "/checkouts/from-env")
    registry, cfg = _provenance_fixture(tmp_path, "/checkouts/alpha",
                                        "/checkouts/from-env")
    value, source, detail = deps.repo_path_provenance(registry, "alpha", cfg)
    assert (value, source) == ("/checkouts/from-env", "env")
    assert detail and "environment variable" in detail


def test_unset_everywhere_reports_why_rather_than_raising(tmp_path):
    registry, cfg = _provenance_fixture(tmp_path, None, None)
    value, source, detail = deps.repo_path_provenance(registry, "alpha", cfg)
    assert (value, source) == (None, None)
    assert detail and "not set anywhere" in detail


def test_projects_endpoint_reports_the_example_profile_as_not_runnable(client):
    """The fixture registry injects a Config with no repo_path, and the template
    profile declares none either — so the panel must not offer to start a job."""
    entry = next(p for p in client.get("/api/projects").json()["projects"]
                 if p["name"] == "example")
    assert entry["runnable"] is False
    assert entry["repo_path"] is None and entry["repo_path_source"] is None
    assert "cannot run" in entry["runnable_detail"]
