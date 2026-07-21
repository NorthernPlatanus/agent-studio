"""Phase 0: two-layer prompt resolver + new project-profile path."""

import textwrap

from orchestrator.core.config import Config, load_config


def _cfg(root, data, project="proj"):
    return Config(data, project, root)


def test_project_override_wins_over_shared(tmp_path):
    shared = tmp_path / "config" / "prompts"
    shared.mkdir(parents=True)
    (shared / "worker_protocol.md").write_text("SHARED")
    proj = tmp_path / "projects" / "proj" / "prompts"
    proj.mkdir(parents=True)
    (proj / "worker_protocol.md").write_text("PROJECT")

    cfg = _cfg(tmp_path, {"prompts": {
        "shared_dir": "config/prompts",
        "project_dir": "projects/{project}/prompts",
    }})
    assert cfg.prompt("worker_protocol") == "PROJECT"


def test_falls_back_to_shared_when_project_absent(tmp_path):
    shared = tmp_path / "config" / "prompts"
    shared.mkdir(parents=True)
    (shared / "planner.md").write_text("SHARED-PLANNER")
    # project_dir configured but no file there
    (tmp_path / "projects" / "proj" / "prompts").mkdir(parents=True)

    cfg = _cfg(tmp_path, {"prompts": {
        "shared_dir": "config/prompts",
        "project_dir": "projects/{project}/prompts",
    }})
    assert cfg.prompt("planner") == "SHARED-PLANNER"


def test_project_dir_unconfigured_uses_shared(tmp_path):
    shared = tmp_path / "config" / "prompts"
    shared.mkdir(parents=True)
    (shared / "reviewer.md").write_text("R")
    cfg = _cfg(tmp_path, {"prompts": {"shared_dir": "config/prompts"}})
    assert cfg.project_prompts_dir() is None
    assert cfg.prompt("reviewer") == "R"


def test_legacy_prompts_dir_still_works(tmp_path):
    shared = tmp_path / "config" / "prompts"
    shared.mkdir(parents=True)
    (shared / "worker_system.md").write_text("LEGACY")
    # Only the old `dir:` key is present.
    cfg = _cfg(tmp_path, {"prompts": {"dir": "config/prompts"}})
    assert cfg.shared_prompts_dir() == shared
    assert cfg.prompt("worker_system") == "LEGACY"


def test_prompt_substitution_preserved(tmp_path):
    shared = tmp_path / "config" / "prompts"
    shared.mkdir(parents=True)
    (shared / "planner.md").write_text("hello {who}")
    cfg = _cfg(tmp_path, {"prompts": {"shared_dir": "config/prompts"}})
    assert cfg.prompt("planner", who="world") == "hello world"


def _write_default(root):
    cfgdir = root / "config"
    cfgdir.mkdir(parents=True, exist_ok=True)
    (cfgdir / "default.yaml").write_text(textwrap.dedent("""
        project: {repo_path: null}
        prompts: {shared_dir: config/prompts, project_dir: 'projects/{project}/prompts'}
    """))
    return cfgdir


def test_profile_loads_from_new_projects_path(tmp_path):
    _write_default(tmp_path)
    prof = tmp_path / "projects" / "acme" / "profile.yaml"
    prof.parent.mkdir(parents=True)
    prof.write_text("project: {repo_path: /tmp/acme}\n")

    cfg = load_config("acme", root=tmp_path)
    assert cfg.project.repo_path == "/tmp/acme"
    assert cfg.project_name == "acme"


def test_profile_falls_back_to_legacy_config_projects(tmp_path):
    cfgdir = _write_default(tmp_path)
    legacy = cfgdir / "projects" / "old.yaml"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("project: {repo_path: /tmp/old}\n")

    cfg = load_config("old", root=tmp_path)
    assert cfg.project.repo_path == "/tmp/old"


def test_new_profile_wins_over_legacy(tmp_path):
    cfgdir = _write_default(tmp_path)
    (cfgdir / "projects").mkdir(parents=True)
    (cfgdir / "projects" / "dup.yaml").write_text("project: {repo_path: /tmp/legacy}\n")
    newp = tmp_path / "projects" / "dup" / "profile.yaml"
    newp.parent.mkdir(parents=True)
    newp.write_text("project: {repo_path: /tmp/new}\n")

    cfg = load_config("dup", root=tmp_path)
    assert cfg.project.repo_path == "/tmp/new"


def test_missing_profile_raises(tmp_path):
    _write_default(tmp_path)
    try:
        load_config("nope", root=tmp_path)
    except FileNotFoundError as e:
        assert "nope" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError")
