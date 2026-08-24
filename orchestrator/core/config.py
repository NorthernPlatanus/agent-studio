"""Layered configuration.

Merge order (later wins):
  1. config/default.yaml               — generic skeleton defaults
  2. projects/<name>/profile.yaml      — the project profile (--project / ORCH_PROJECT),
     falling back to the legacy config/projects/<name>.yaml when the new path is absent
  3. config/local.yaml                 — personal overrides (gitignored)
  4. state/<name>.assignments.json     — the Studio assignment overlay: preset and
     effort bindings written by the control panel (core/assignments.py). Placed
     here, above the profile and below the environment, so a panel binding beats
     the file a human hand-edited last week while ORCH_* stays the emergency
     override it has always been.
  5. ORCH_<SECTION>_<KEY> env vars     — e.g. ORCH_PROJECT_REPO_PATH, ORCH_RUN_N_CANDIDATES,
     ORCH_VISUAL_GATE_ENABLED. The section may itself contain underscores; it is
     resolved against the loaded config's real top-level keys (longest match
     wins). Scalars only, and exactly one level deep — providers.claude_cli.* and
     other nested values belong in config/local.yaml or the project profile.

Access is attribute-style: cfg.project.repo_path, cfg.gate.commands, ...

Prompts resolve through two layers (Phase 0): a per-project overlay
(prompts.project_dir, gitignored, {project}-substituted) wins over the shared
committed templates (prompts.shared_dir). The legacy prompts.dir key is treated
as shared_dir for back-compat.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .assignments import apply_overlay, overlay_path, read_overlay

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


class Section:
    """Read-only attribute access over a nested dict."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getattr__(self, key: str) -> Any:
        try:
            value = self._data[key]
        except KeyError:
            raise AttributeError(key) from None
        return Section(value) if isinstance(value, dict) else value

    def get(self, key: str, default: Any = None) -> Any:
        value = self._data.get(key, default)
        return Section(value) if isinstance(value, dict) else value

    def as_dict(self) -> dict[str, Any]:
        return self._data

    def keys(self):
        return self._data.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"Section({self._data!r})"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _split_section(rest: str, data: dict) -> tuple[str, str] | None:
    """Split `<section>_<key>` against the config's ACTUAL top-level keys.

    Splitting on the first underscore alone makes every underscored section
    unreachable — `ORCH_VISUAL_GATE_ENABLED` looks for a section named `visual`.
    So try successively longer prefixes and keep the longest that names a real
    dict section: `visual_gate` wins over `visual`, and `worker_output` /
    `worker_models` resolve to themselves rather than a nonexistent `worker`.
    Returns (section, key), or None when nothing matches."""
    parts = rest.split("_")
    for cut in range(len(parts) - 1, 0, -1):     # longest section prefix first
        section = "_".join(parts[:cut])
        if isinstance(data.get(section), dict):
            return section, "_".join(parts[cut:])
    return None


def _apply_env(data: dict) -> dict:
    """ORCH_<SECTION>_<KEY>=value overrides data[section][key] (scalars only).

    Both halves may contain underscores (`ORCH_VISUAL_GATE_ENABLED` ->
    visual_gate.enabled), resolved against the loaded config's own keys. Only
    ONE level deep: a nested value like providers.claude_cli.timeout_s is not
    reachable this way — put those in config/local.yaml or the project profile.
    """
    for name, raw in os.environ.items():
        if not name.startswith("ORCH_") or name == "ORCH_PROJECT":
            continue
        split = _split_section(name[len("ORCH_"):].lower(), data)
        if split is None:
            continue
        section, key = split
        # Find the matching key (env can't express underscored splits
        # unambiguously, so match against existing keys).
        for existing in data[section]:
            if existing.replace("_", "") == key.replace("_", ""):
                data[section][existing] = yaml.safe_load(raw)
                break
    return data


class Config(Section):
    def __init__(self, data: dict[str, Any], project_name: str, root: Path):
        super().__init__(data)
        self.project_name = project_name
        self.root = root  # orchestrator repo root

    # -- resolved path helpers -------------------------------------------
    def repo_path(self) -> Path:
        rp = self._data["project"]["repo_path"]
        if not rp:
            raise ValueError(
                "project.repo_path is not set — create a project profile at "
                "projects/<name>/profile.yaml (or the legacy "
                "config/projects/<name>.yaml) and pass --project <name>."
            )
        return Path(rp).expanduser().resolve()

    def state_dir(self) -> Path:
        p = Path(self._data["paths"]["state_dir"]).expanduser()
        p = p if p.is_absolute() else self.root / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    def work_dir(self) -> Path:
        p = Path(self._data["paths"]["work_dir"]).expanduser()
        p = p if p.is_absolute() else self.root / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _resolve_root(self, raw: str) -> Path:
        p = Path(raw).expanduser()
        return p if p.is_absolute() else self.root / p

    def shared_prompts_dir(self) -> Path:
        """Committed template prompts. Honors legacy `prompts.dir`."""
        prompts = self._data.get("prompts", {}) or {}
        raw = prompts.get("shared_dir") or prompts.get("dir") or "config/prompts"
        return self._resolve_root(raw)

    def project_prompts_dir(self) -> Path | None:
        """Per-project, gitignored prompt overlay (or None if unconfigured)."""
        prompts = self._data.get("prompts", {}) or {}
        raw = prompts.get("project_dir")
        if not raw:
            return None
        return self._resolve_root(raw.replace("{project}", self.project_name))

    # Back-compat alias: callers that want "the prompt dir" get the shared one.
    def prompts_dir(self) -> Path:
        return self.shared_prompts_dir()

    def prompt(self, name: str, **fmt: Any) -> str:
        filename = f"{name}.md"
        project_dir = self.project_prompts_dir()
        if project_dir is not None and (project_dir / filename).exists():
            path = project_dir / filename
        else:
            path = self.shared_prompts_dir() / filename
        text = path.read_text()
        if fmt:
            for key, value in fmt.items():
                text = text.replace("{" + key + "}", str(value))
        return text

    def store_path(self) -> Path:
        return self.state_dir() / f"{self.project_name}.sqlite3"

    def checkpoint_path(self) -> Path:
        return self.state_dir() / f"{self.project_name}.checkpoints.sqlite3"


def _assignments_path(data: dict, project: str, root: Path) -> Path:
    """Where this project's assignment overlay lives, mid-merge.

    `ORCH_PATHS_STATE_DIR` is read HERE rather than left to `_apply_env` below,
    which is the one place the documented layer order has to be bent: the
    overlay is a file whose *location* the environment can move, so the path has
    to be resolved before the environment layer is applied to the rest of the
    config. Nothing about the overlay's *contents* is decided early — the env
    layer still runs afterwards and still wins over everything it merges.
    """
    raw = os.environ.get("ORCH_PATHS_STATE_DIR") \
        or (data.get("paths") or {}).get("state_dir") or "state"
    directory = Path(str(raw)).expanduser()
    return overlay_path(directory if directory.is_absolute() else root / directory,
                        project)


def load_config(project: str | None = None, root: Path | None = None) -> Config:
    root = (root or CONFIG_DIR.parent).resolve()
    cfg_dir = root / "config"
    project = project or os.environ.get("ORCH_PROJECT")

    data = _load_yaml(cfg_dir / "default.yaml")
    if project:
        # New layout: gitignored projects/<name>/profile.yaml wins; fall back to
        # the legacy committed config/projects/<name>.yaml.
        new_profile = root / "projects" / project / "profile.yaml"
        legacy_profile = cfg_dir / "projects" / f"{project}.yaml"
        if new_profile.exists():
            profile = new_profile
        elif legacy_profile.exists():
            profile = legacy_profile
        else:
            raise FileNotFoundError(
                f"No project profile: looked for {new_profile} and {legacy_profile}"
            )
        data = _deep_merge(data, _load_yaml(profile))
    # ORCH_SKIP_LOCAL_CONFIG: drop the machine-local overlay. Set by the test
    # suite (tests/conftest.py) so unit tests read the committed defaults and
    # not whatever this developer happens to have in config/local.yaml — a real
    # `gate.install_cmd` there otherwise makes worker tests shell out to a real
    # `npm ci`, so the suite passes or fails depending on whose laptop it is.
    # Not a knob for normal runs; nothing but the tests should set it.
    #
    # The assignment overlay is gated by the same flag, and for the same reason:
    # it lives in `paths.state_dir`, which for the repo default is this checkout's
    # own `state/` — the directory holding the operator's real project stores. A
    # unit test that read it would pass or fail depending on what the operator
    # last clicked in the control panel, which is precisely the class of bug this
    # flag exists to prevent. Tests that want the overlay merge build their own
    # root and clear the flag for the duration.
    if not os.environ.get("ORCH_SKIP_LOCAL_CONFIG"):
        data = _deep_merge(data, _load_yaml(cfg_dir / "local.yaml"))
        if project:
            path = _assignments_path(data, project, root)
            data = apply_overlay(data, read_overlay(path), source=str(path))
    data = _apply_env(data)
    return Config(data, project or "default", root)
