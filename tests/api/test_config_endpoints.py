"""The config read/write endpoints, and the overlay layer underneath them.

Three things here are load-bearing rather than incidental:

* **No API key value may leave this process.** `test_no_environment_value_ever_
  appears_in_a_config_response` sweeps the whole of `os.environ` against the raw
  response text of both GETs, not just the keys this repo happens to name today.
* **The overlay must never make the CLI unbootable.** A stale overlay pinning a
  deleted preset is dropped with a warning; `load_config` still returns.
* **The layer order is asserted, not assumed.** The overlay beats
  `config/local.yaml`; `ORCH_*` still beats the overlay.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from orchestrator.api import deps, jobs
from orchestrator.api.app import create_app
from orchestrator.core.assignments import overlay_path, write_overlay
from orchestrator.core.config import Config, load_config
from orchestrator.core.presets import resolve_entry
from tests.api.fixtures.seed_store import PROJECT

BASE = f"/api/projects/{PROJECT}"
ASSIGNMENTS = f"{BASE}/config/assignments"
PRESETS = f"{BASE}/config/presets"

SLEEPER = [sys.executable, "-c", "import time; time.sleep(30)"]

#: A value no config file, model id or provider name could ever equal, planted in
#: every API-key variable the shipped providers name.
SENTINEL = "SENTINEL-e3f1a-this-must-never-be-serialized"
KEY_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "COMETAPI_KEY")


# ---- fixtures ------------------------------------------------------------
# The worker tier these tests assert against, written out here rather than taken
# from the session `cfg` fixture. `projects/` is gitignored, so
# `projects/example/profile.yaml` is a per-machine file whose `worker_models` are
# whatever this operator last wrote — a committed test cannot assert against it.
# `presets:` and `providers:` still come from the committed `config/default.yaml`,
# which is the thing under test.
WORKERS = {
    "flash_lo": {"provider": "cometapi", "model": "deepseek-v4-flash",
                 "input_per_mtok": 0.12, "output_per_mtok": 0.24,
                 "params": {"temperature": 0.2}, "approach": "Simplest."},
    "flash_mid": {"provider": "cometapi", "model": "deepseek-v4-flash",
                  "input_per_mtok": 0.12, "output_per_mtok": 0.24},
    # Already preset-bound in the profile — a `source: "profile"` row that
    # nonetheless names a preset, which is the case a naive projection conflates
    # with "the panel set this".
    "flash_hi": {"preset": "deepseek_flash"},
}
ROLES = {
    "smart_provider": "claude_cli",
    "planner": {"provider": None, "model": "opus", "effort": None},
    "reviewer": {"provider": None, "model": "opus", "effort": None},
    "verifier": {"provider": None, "model": "opus", "effort": None,
                 "allowed_tools": None},
    "worker": {"default": "flash_mid",
               "candidates": ["flash_lo", "flash_mid", "flash_hi"]},
}


@pytest.fixture
def config_cfg(tmp_path: Path) -> Config:
    """The committed defaults, an explicit worker tier, and a writable state dir.

    Per test, not session-scoped, because these tests WRITE an overlay into the
    state dir and one of them spawns a job that writes a log beside it.
    """
    state, checkout = tmp_path / "state", tmp_path / "checkout"
    state.mkdir()
    checkout.mkdir()
    data = yaml.safe_load(
        (deps.REPO_ROOT / "config" / "default.yaml").read_text())
    data["paths"]["state_dir"] = str(state)
    data["paths"]["work_dir"] = str(state / "worktrees")
    # Only the 409 test needs a checkout, to spawn a job at all; harmless here.
    data.setdefault("project", {})["repo_path"] = str(checkout)
    data["worker_models"] = copy.deepcopy(WORKERS)
    data["roles"] = copy.deepcopy(ROLES)
    return Config(data, PROJECT, deps.REPO_ROOT)


@pytest.fixture
def sup() -> jobs.JobSupervisor:
    return jobs.JobSupervisor(sigint_grace_s=5.0, sigterm_grace_s=2.0)


@pytest.fixture
def config_client(config_cfg: Config, sup: jobs.JobSupervisor):
    registry = deps.ProjectRegistry(root=Path(config_cfg.paths.state_dir),
                                    configs={PROJECT: config_cfg})
    app = create_app()
    app.dependency_overrides[deps.get_registry] = lambda: registry
    app.dependency_overrides[jobs.get_supervisor] = lambda: sup
    with TestClient(app) as client:
        yield client


def _preset(body: dict, key: str) -> dict:
    return next(p for p in body["presets"] if p["key"] == key)


def _row(rows: list[dict], key: str) -> dict:
    return next(r for r in rows if r["key"] == key)


# ---- presets -------------------------------------------------------------
def test_presets_reports_every_shipped_backend(config_client):
    body = config_client.get(PRESETS).json()
    assert body["project"] == PROJECT
    assert body["efforts"] == ["low", "medium", "high", "xhigh", "max"]
    assert {"luna_high", "luna_med", "luna_xhigh", "claude_cli_sonnet",
            "claude_cli_opus", "codex_cli_default",
            "deepseek_flash"} <= {p["key"] for p in body["presets"]}

    luna = _preset(body, "luna_high")
    assert luna["label"] == "GPT-5.6 Luna · high"
    assert (luna["kind"], luna["cash"]) == ("api", True)
    assert (luna["provider"], luna["provider_type"]) == ("openai", "openai_responses")
    assert luna["model"] == "gpt-5.6-luna" and luna["model"] in luna["models"]
    assert luna["effort"] == "high"
    assert (luna["input_per_mtok"], luna["output_per_mtok"]) == (0.20, 1.20)
    # openai_responses has a reasoning dial, so every level is offerable.
    assert luna["efforts"] == body["efforts"]

    cli = _preset(body, "claude_cli_sonnet")
    assert (cli["kind"], cli["cash"]) == ("cli", False)
    # No prices, deliberately — a subscription tier spends no cash, and the panel
    # must not be able to render a dollar figure for it.
    assert cli["input_per_mtok"] is None and cli["output_per_mtok"] is None
    assert cli["efforts"] == body["efforts"]      # claude_cli.supports_effort

    # openai_compatible fronting DeepSeek drops effort; the UI needs to know so it
    # can disable the control instead of offering a setting that goes nowhere.
    assert _preset(body, "deepseek_flash")["efforts"] == []
    assert _preset(body, "codex_cli_default")["efforts"] == []

    assert all(isinstance(p["configured"], bool) for p in body["presets"])


def test_a_missing_key_is_reported_by_variable_name_only(config_client, monkeypatch):
    for var in KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    luna = _preset(config_client.get(PRESETS).json(), "luna_high")
    assert luna["configured"] is False
    assert "OPENAI_API_KEY" in luna["configured_detail"]


def test_a_present_key_makes_the_preset_configured(config_client, monkeypatch):
    for var in KEY_VARS:
        monkeypatch.setenv(var, SENTINEL)
    luna = _preset(config_client.get(PRESETS).json(), "luna_high")
    assert luna["configured"] is True and luna["configured_detail"] is None


def test_no_environment_value_ever_appears_in_a_config_response(config_client,
                                                                monkeypatch):
    """The one test this whole surface exists to satisfy.

    Sweeps `os.environ` rather than only the variables the shipped config names:
    the failure mode being guarded against is a future field that returns "the
    provider block" wholesale, which would leak whatever key a project profile
    invented. Values shorter than 8 characters are skipped — a two-letter locale
    or a `TERM` of `xterm` collides with ordinary English by accident, and the
    planted `SENTINEL` below is what actually pins the API-key case.
    """
    for var in KEY_VARS:
        monkeypatch.setenv(var, SENTINEL)
    for path in (PRESETS, ASSIGNMENTS):
        response = config_client.get(path)
        assert response.status_code == 200
        text = response.text
        assert SENTINEL not in text, f"{path} leaked a planted API key"
        assert "api_key" not in text, f"{path} names a key variable as a field"
        leaked = [name for name, value in os.environ.items()
                  if len(value) >= 8 and value in text]
        assert not leaked, f"{path} leaked the value of {leaked}"


# ---- assignments (read) --------------------------------------------------
def test_assignments_reports_the_profile_bindings(config_client):
    body = config_client.get(ASSIGNMENTS).json()
    assert body["project"] == PROJECT
    assert [w["key"] for w in body["workers"]] == ["flash_lo", "flash_mid", "flash_hi"]
    assert body["default_worker"] == "flash_mid"
    assert body["candidates"] == ["flash_lo", "flash_mid", "flash_hi"]
    assert body["locked"] is False and body["locked_reason"] is None

    row = _row(body["workers"], "flash_lo")
    # An entry that spells its backend out inline names no preset, and is
    # described by what it actually binds so the row is not blank in the UI.
    assert row["source"] == "profile" and row["preset"] is None
    assert row["label"] == "cometapi / deepseek-v4-flash"

    # A hand-written `preset:` is still `source: "profile"` — naming a preset and
    # being set from the panel are different facts, and the per-row "set from
    # panel" chip depends on not confusing them.
    preset_bound = _row(body["workers"], "flash_hi")
    assert preset_bound["source"] == "profile"
    assert preset_bound["preset"] == "deepseek_flash"
    assert preset_bound["label"] == "DeepSeek V4 Flash (CometAPI)"

    # The smart-tier roles, and only those: `smart_provider` is a string and
    # `worker` is a pool with its own two fields.
    assert {r["key"] for r in body["roles"]} == {"planner", "reviewer", "verifier"}
    assert all(r["source"] == "profile" for r in body["roles"])


# ---- assignments (write) -------------------------------------------------
def test_post_persists_the_overlay_and_the_next_get_reflects_it(config_client,
                                                                config_cfg):
    response = config_client.post(ASSIGNMENTS, json={
        "workers": {"flash_lo": {"preset": "claude_cli_sonnet", "effort": "high"}},
        "roles": {"planner": {"preset": "claude_cli_opus"}},
        "default_worker": "flash_hi",
        "candidates": ["flash_lo", "flash_hi"]})
    assert response.status_code == 200
    body = response.json()

    bound = _row(body["workers"], "flash_lo")
    assert bound["source"] == "overlay" and bound["preset"] == "claude_cli_sonnet"
    assert bound["effort"] == "high"
    assert bound["label"] == "Claude Code CLI · sonnet"
    # Untouched rows still report the profile as their source, which is what the
    # UI's "set from panel" chip keys off.
    assert _row(body["workers"], "flash_mid")["source"] == "profile"
    planner = _row(body["roles"], "planner")
    assert planner["source"] == "overlay" and planner["effort"] == "high"
    assert body["default_worker"] == "flash_hi"
    assert body["candidates"] == ["flash_lo", "flash_hi"]

    path = overlay_path(config_cfg.paths.state_dir, PROJECT)
    saved = json.loads(path.read_text())
    assert saved["version"] == 1
    assert saved["workers"] == {
        "flash_lo": {"preset": "claude_cli_sonnet", "effort": "high"}}
    # Exactly the fixed key list and nothing else — the overlay is not a place to
    # put arbitrary config, and the file on disk is the proof.
    assert set(saved) == {"version", "workers", "roles", "default_worker",
                          "candidates"}

    assert config_client.get(ASSIGNMENTS).json() == body


def test_post_rejects_an_unknown_preset(config_client):
    response = config_client.post(
        ASSIGNMENTS, json={"workers": {"flash_lo": {"preset": "no_such_preset"}}})
    assert response.status_code == 422
    assert "no_such_preset" in response.json()["detail"]


def test_post_rejects_an_unknown_worker_or_role(config_client):
    workers = config_client.post(
        ASSIGNMENTS, json={"workers": {"invented": {"preset": "luna_high"}}})
    assert workers.status_code == 422
    assert "invented" in workers.json()["detail"]
    # `worker` is a pool, not a bindable role — binding it would write a preset
    # into the object that holds `default` and `candidates`.
    roles = config_client.post(
        ASSIGNMENTS, json={"roles": {"worker": {"preset": "luna_high"}}})
    assert roles.status_code == 422


def test_post_rejects_an_unknown_effort(config_client):
    response = config_client.post(ASSIGNMENTS, json={
        "workers": {"flash_lo": {"preset": "luna_high", "effort": "turbo"}}})
    assert response.status_code == 422


def test_post_rejects_anything_outside_the_fixed_key_list(config_client):
    """`extra="forbid"` is the boundary that keeps this from being a config API.

    A `base_url` is the exact shape of the thing that must never be settable from
    a browser, so it is what the test sends.
    """
    nested = config_client.post(ASSIGNMENTS, json={
        "workers": {"flash_lo": {"preset": "luna_high",
                                 "base_url": "http://attacker.example"}}})
    assert nested.status_code == 422
    top_level = config_client.post(ASSIGNMENTS, json={
        "presets": {"mine": {"provider": "openai", "model": "x"}}})
    assert top_level.status_code == 422


def test_post_refuses_while_a_job_is_live(config_client, config_cfg, sup):
    record = sup.spawn(config_cfg, "run", SLEEPER)
    try:
        response = config_client.post(
            ASSIGNMENTS,
            json={"workers": {"flash_lo": {"preset": "claude_cli_sonnet"}}})
        assert response.status_code == 409
        assert record.job_id in response.json()["detail"]
        # And the read side says so in advance, so the form disables itself
        # rather than failing on submit.
        body = config_client.get(ASSIGNMENTS).json()
        assert body["locked"] is True
        assert record.job_id in body["locked_reason"]
        # Nothing was written.
        assert not overlay_path(config_cfg.paths.state_dir, PROJECT).exists()
    finally:
        record.proc.kill()
        record.proc.wait()


# ---- the config layer itself ---------------------------------------------
MERGE_PROJECT = "overlay-demo"

PROFILE = """
project: {repo_path: null}
worker_models:
  flash_lo: {provider: cometapi, model: deepseek-v4-flash,
             input_per_mtok: 0.12, output_per_mtok: 0.24,
             params: {temperature: 0.2}, approach: "Simplest correct solution."}
roles:
  worker: {default: flash_lo, candidates: [flash_lo]}
"""


@pytest.fixture
def merge_root(tmp_path: Path, monkeypatch) -> Path:
    """A throwaway repo root: the real defaults, one project, an empty state dir.

    Built rather than reusing this checkout because these tests exercise
    `load_config`'s real layering, which includes reading `config/local.yaml` and
    `state/<project>.assignments.json` — both of which are the operator's own
    files here. `ORCH_SKIP_LOCAL_CONFIG` (set session-wide by tests/conftest.py to
    keep exactly those files out of the suite) is cleared for the duration, which
    is why the root has to be a fake one.
    """
    root = tmp_path / "root"
    (root / "config").mkdir(parents=True)
    (root / "projects" / MERGE_PROJECT).mkdir(parents=True)
    (root / "state").mkdir()
    shutil.copy(deps.REPO_ROOT / "config" / "default.yaml",
                root / "config" / "default.yaml")
    (root / "projects" / MERGE_PROJECT / "profile.yaml").write_text(PROFILE)
    monkeypatch.delenv("ORCH_SKIP_LOCAL_CONFIG", raising=False)
    return root


def _write(root: Path, overlay: dict) -> None:
    write_overlay(overlay_path(root / "state", MERGE_PROJECT), overlay)


def test_the_overlay_rebinds_a_worker_and_beats_local_yaml(merge_root: Path):
    (merge_root / "config" / "local.yaml").write_text(
        "worker_models:\n  flash_lo: {model: set-in-local-yaml}\n")
    _write(merge_root, {"workers": {"flash_lo": {"preset": "claude_cli_sonnet"}}})

    cfg = load_config(MERGE_PROJECT, root=merge_root)
    entry = cfg.worker_models.get("flash_lo")
    assert entry.get("preset") == "claude_cli_sonnet"
    # The inline provider/model are STRIPPED, not merely shadowed. `resolve_entry`
    # is entry-wins, so leaving `model: set-in-local-yaml` in place would resolve
    # straight back to it and the rebinding would silently do nothing.
    assert entry.get("model") is None and entry.get("provider") is None
    resolved = resolve_entry(cfg, entry)
    assert (resolved["provider"], resolved["model"]) == ("claude_cli", "sonnet")
    # What the preset does NOT own survives: sampling knobs and the approach hint
    # are the entry's own business.
    assert entry.get("params").get("temperature") == 0.2
    assert entry.get("approach").startswith("Simplest")


def test_orch_env_overrides_still_beat_the_overlay(merge_root: Path, monkeypatch):
    _write(merge_root, {"workers": {"flash_lo": {"preset": "claude_cli_sonnet"}}})
    monkeypatch.setenv("ORCH_WORKER_MODELS_FLASH_LO", "{preset: deepseek_flash}")
    cfg = load_config(MERGE_PROJECT, root=merge_root)
    assert cfg.worker_models.get("flash_lo").get("preset") == "deepseek_flash"


def test_a_stale_overlay_is_dropped_with_a_warning_and_the_config_still_loads(
        merge_root: Path, caplog):
    _write(merge_root, {
        "workers": {"flash_lo": {"preset": "deleted_preset"},
                    "no_such_worker": {"preset": "luna_high"}},
        "roles": {"no_such_role": {"preset": "luna_high"}},
        "default_worker": "no_such_worker",
        "candidates": ["flash_lo", "no_such_worker"]})

    with caplog.at_level(logging.WARNING, logger="orchestrator.assignments"):
        cfg = load_config(MERGE_PROJECT, root=merge_root)

    # Every profile binding stands, unchanged.
    entry = cfg.worker_models.get("flash_lo")
    assert (entry.get("provider"), entry.get("model")) == ("cometapi",
                                                           "deepseek-v4-flash")
    assert entry.get("preset") is None
    assert cfg.roles.worker.get("default") == "flash_lo"
    # The whole pool is dropped rather than narrowed to its known member — a
    # best-of-N width nobody chose is harder to notice than no change at all.
    assert cfg.roles.worker.get("candidates") == ["flash_lo"]

    for named in ("deleted_preset", "no_such_worker", "no_such_role"):
        assert named in caplog.text


def test_an_unreadable_overlay_is_ignored_rather_than_fatal(merge_root: Path):
    overlay_path(merge_root / "state", MERGE_PROJECT).write_text("{not json")
    cfg = load_config(MERGE_PROJECT, root=merge_root)
    assert cfg.worker_models.get("flash_lo").get("model") == "deepseek-v4-flash"
