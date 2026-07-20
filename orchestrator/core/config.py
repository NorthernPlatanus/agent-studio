"""Layered configuration.

Merge order (later wins):
  1. config/default.yaml           — generic skeleton defaults
  2. config/projects/<name>.yaml   — the project profile (--project / ORCH_PROJECT)
  3. config/local.yaml             — personal overrides (gitignored)
  4. ORCH_<SECTION>_<KEY> env vars — e.g. ORCH_PROJECT_REPO_PATH, ORCH_RUN_N_CANDIDATES

Access is attribute-style: cfg.project.repo_path, cfg.gate.commands, ...
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

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


def _apply_env(data: dict) -> dict:
    """ORCH_<SECTION>_<KEY>=value overrides data[section][key] (scalars only)."""
    for name, raw in os.environ.items():
        if not name.startswith("ORCH_") or name == "ORCH_PROJECT":
            continue
        parts = name[len("ORCH_"):].lower().split("_", 1)
        if len(parts) != 2:
            continue
        section, key = parts
        if section in data and isinstance(data[section], dict):
            # Find the matching key (env can't express nested/underscored splits
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
                "project.repo_path is not set — create a project profile in "
                "config/projects/<name>.yaml and pass --project <name>."
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

    def prompts_dir(self) -> Path:
        p = Path(self._data["prompts"]["dir"]).expanduser()
        return p if p.is_absolute() else self.root / p

    def prompt(self, name: str, **fmt: Any) -> str:
        text = (self.prompts_dir() / f"{name}.md").read_text()
        if fmt:
            for key, value in fmt.items():
                text = text.replace("{" + key + "}", str(value))
        return text

    def store_path(self) -> Path:
        return self.state_dir() / f"{self.project_name}.sqlite3"

    def checkpoint_path(self) -> Path:
        return self.state_dir() / f"{self.project_name}.checkpoints.sqlite3"


def load_config(project: str | None = None, root: Path | None = None) -> Config:
    root = (root or CONFIG_DIR.parent).resolve()
    cfg_dir = root / "config"
    project = project or os.environ.get("ORCH_PROJECT")

    data = _load_yaml(cfg_dir / "default.yaml")
    if project:
        profile = cfg_dir / "projects" / f"{project}.yaml"
        if not profile.exists():
            raise FileNotFoundError(f"No project profile: {profile}")
        data = _deep_merge(data, _load_yaml(profile))
    data = _deep_merge(data, _load_yaml(cfg_dir / "local.yaml"))
    data = _apply_env(data)
    return Config(data, project or "default", root)
