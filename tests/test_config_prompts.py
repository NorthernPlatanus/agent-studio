"""Phase 0: two-layer prompt resolver + new project-profile path."""

import textwrap

from pathlib import Path

import pytest

from orchestrator.core.config import Config, _apply_env, load_config


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


# ---- env overrides reach underscored sections (item 16) ----------------------

def _data():
    return {"run": {"n_candidates": 1, "max_retries": 3},
            "visual_gate": {"enabled": False, "run_cmd": None},
            "worker_output": {"full_file_max_lines": 400},
            "roles": {"smart_provider": "claude_cli"}}


def test_env_override_simple_section(monkeypatch):
    monkeypatch.setenv("ORCH_RUN_N_CANDIDATES", "3")
    assert _apply_env(_data())["run"]["n_candidates"] == 3


def test_env_override_reaches_underscored_section(monkeypatch):
    # Splitting on the FIRST underscore looked for a section named "visual".
    monkeypatch.setenv("ORCH_VISUAL_GATE_ENABLED", "true")
    monkeypatch.setenv("ORCH_WORKER_OUTPUT_FULL_FILE_MAX_LINES", "120")
    out = _apply_env(_data())
    assert out["visual_gate"]["enabled"] is True
    assert out["worker_output"]["full_file_max_lines"] == 120


def test_env_override_prefers_the_longest_matching_section(monkeypatch):
    # Both "worker_output" and a hypothetical "worker" exist; the longer wins.
    data = _data()
    data["worker"] = {"output": "nope"}
    monkeypatch.setenv("ORCH_WORKER_OUTPUT_FULL_FILE_MAX_LINES", "77")
    out = _apply_env(data)
    assert out["worker_output"]["full_file_max_lines"] == 77
    assert out["worker"]["output"] == "nope"


def test_env_override_unknown_section_is_ignored(monkeypatch):
    monkeypatch.setenv("ORCH_NOPE_WHATEVER", "1")
    monkeypatch.setenv("ORCH_RUN", "1")          # no key half at all
    assert _apply_env(_data()) == _data()


def test_repo_path_error_names_the_current_layout_first():
    cfg = Config({"project": {"repo_path": None}}, "proj", Path("/tmp"))
    with pytest.raises(ValueError) as e:
        cfg.repo_path()
    msg = str(e.value)
    assert msg.index("projects/<name>/profile.yaml") < msg.index("config/projects")
