"""Request-scoped dependencies: project allowlist, config cache, read-only DB.

Three invariants live here, and every router inherits them:

1. **Project names are allowlisted.** The set is discovered by scanning
   `projects/*/profile.yaml` and the legacy `config/projects/*.yaml`; anything
   else is a 404 *before* the string can reach a path join or a subprocess
   argv. `load_config()` would itself refuse an unknown profile, but it does so
   only after building paths out of the caller's string — the allowlist keeps
   that string out of `Path()` entirely.
2. **Reads never write.** Endpoints get a `sqlite3.Connection` opened through a
   `file:…?mode=ro` URI, one per request. They must not construct `Store`: its
   `__init__` runs `executescript(SCHEMA)` plus migrations, i.e. it writes to
   the user's state on a GET.
3. **Nothing is configured through the environment at request time.** The
   registry is a single overridable FastAPI dependency (`get_registry`), so
   tests inject an explicit `Config` instead of setting `ORCH_*` — which
   `tests/conftest.py` deletes session-wide anyway.
"""

from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml
from fastapi import Depends, HTTPException

from ..core.config import Config, load_config
from .schemas import RepoPathSource

REPO_ROOT = Path(__file__).resolve().parents[2]

# Defensive second line after the allowlist: a name that cannot pass this can
# never become a path segment even if a future caller skips `resolve_project`.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class ProjectEntry:
    name: str
    profile_path: Path | None


class ProjectRegistry:
    """The allowlist plus a `Config` cache.

    Config loading reads several YAML files per call; the panel polls, so cache
    per project name for the process lifetime. `configs=` lets tests (and the
    fixture smoke server) hand over ready-made `Config` objects without any
    profile on disk.
    """

    def __init__(self, root: Path | None = None, *,
                 configs: Mapping[str, Config] | None = None):
        self.root = (root or REPO_ROOT).resolve()
        self._injected: dict[str, Config] = dict(configs or {})
        self._cache: dict[str, Config] = dict(self._injected)
        self._entries: dict[str, ProjectEntry] | None = None

    def entries(self) -> dict[str, ProjectEntry]:
        if self._entries is None:
            found: dict[str, ProjectEntry] = {}
            for profile in sorted((self.root / "projects").glob("*/profile.yaml")):
                found[profile.parent.name] = ProjectEntry(profile.parent.name, profile)
            for legacy in sorted((self.root / "config" / "projects").glob("*.yaml")):
                found.setdefault(legacy.stem, ProjectEntry(legacy.stem, legacy))
            for name in self._injected:
                found.setdefault(name, ProjectEntry(name, None))
            self._entries = found
        return self._entries

    def names(self) -> list[str]:
        return sorted(self.entries())

    def has(self, name: str) -> bool:
        return bool(_SAFE_NAME.match(name)) and name in self.entries()

    def config(self, name: str) -> Config:
        if name not in self._cache:
            self._cache[name] = load_config(name, root=self.root)
        return self._cache[name]

    def profile_repo_path(self, name: str) -> str | None:
        """`project.repo_path` as the project's OWN profile declares it.

        Read from the profile file alone, not the merged config, because the
        merge is exactly what hides the distinction this answers: `config/
        local.yaml` may set a single global `project.repo_path`, which then
        appears under every project — including one whose profile says `null` on
        purpose. Both real profiles on this machine do say `null`, so this is a
        fact to report, not an error (see `repo_path_provenance`).
        """
        entry = self.entries().get(name)
        if entry is None or entry.profile_path is None:
            return None
        try:
            with open(entry.profile_path) as f:
                data = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            return None
        section = data.get("project") if isinstance(data, dict) else None
        raw = section.get("repo_path") if isinstance(section, dict) else None
        return str(raw) if raw else None


_registry: ProjectRegistry | None = None


def default_registry() -> ProjectRegistry:
    global _registry
    if _registry is None:
        _registry = ProjectRegistry()
    return _registry


def get_registry() -> ProjectRegistry:
    """Overridable seam: `app.dependency_overrides[get_registry] = …` in tests."""
    return default_registry()


def resolve_project(project: str,
                    registry: ProjectRegistry = Depends(get_registry)) -> Config:
    if not registry.has(project):
        raise HTTPException(404, f"unknown project: {project!r}")
    return registry.config(project)


_GLOBAL_CAVEAT = (
    "project.repo_path comes from the machine-global config/local.yaml overlay, "
    "not from this project's own profile — every project whose profile leaves it "
    "null resolves to that same checkout, so a job started here would act on it")
_ENV_CAVEAT = (
    "project.repo_path comes from the ORCH_PROJECT_REPO_PATH environment variable "
    "of the serving process, not from this project's profile — it applies to every "
    "project alike")
_UNSET = ("project.repo_path is not set anywhere (profile, config/local.yaml, or "
          "ORCH_PROJECT_REPO_PATH), so jobs that need a checkout cannot run")


def effective_repo_path(cfg: Config) -> str | None:
    """The merged `project.repo_path`, unresolved and without raising.

    `Config.repo_path()` raises when it is unset; here "unset" is an answer.
    """
    section = cfg.get("project")
    raw = section.as_dict().get("repo_path") if section is not None else None
    return str(raw) if raw else None


def repo_path_provenance(
        registry: ProjectRegistry, name: str,
        cfg: Config) -> tuple[str | None, RepoPathSource | None, str | None]:
    """(effective repo_path, where it came from, the caveat to show the UI).

    Why the API reports this instead of just a boolean: with a global
    `project.repo_path` in `config/local.yaml`, *every* project answers "yes, I
    have a checkout" — including the template `example`, whose profile
    deliberately sets `null`. A panel that shows a Start-run button on that
    strength offers to run one project's queue against another project's working
    tree. The value is still reported as runnable (the CLI really would run, and
    both real profiles here rely on the overlay by design), but the source and
    the caveat travel with it so the UI can say which checkout it means.
    """
    effective = effective_repo_path(cfg)
    if effective is None:
        return None, None, _UNSET
    if os.environ.get("ORCH_PROJECT_REPO_PATH"):
        return effective, "env", _ENV_CAVEAT
    if registry.profile_repo_path(name):
        return effective, "profile", None
    return effective, "global", _GLOBAL_CAVEAT


def _resolved_dir(cfg: Config, key: str) -> Path | None:
    """`paths.<key>` resolved against the repo root — WITHOUT creating it.

    `Config.state_dir()` mkdirs, which is a filesystem write on the way to a
    GET. Harmless for an existing state dir, wrong for a project that has never
    run, so read endpoints resolve the path themselves.
    """
    paths = cfg.get("paths")
    raw = paths.as_dict().get(key) if paths is not None else None
    if not raw:
        return None
    p = Path(str(raw)).expanduser()
    return p if p.is_absolute() else cfg.root / p


def store_path(cfg: Config) -> Path | None:
    state = _resolved_dir(cfg, "state_dir")
    return None if state is None else state / f"{cfg.project_name}.sqlite3"


def checkpoint_path(cfg: Config) -> Path | None:
    state = _resolved_dir(cfg, "state_dir")
    return None if state is None else state / f"{cfg.project_name}.checkpoints.sqlite3"


def open_read_only(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def store_conn(cfg: Config = Depends(resolve_project)) -> Iterator[sqlite3.Connection]:
    path = store_path(cfg)
    if path is None or not path.exists():
        # 409, not 404: the project is real and allowlisted, its state just does
        # not exist yet (nothing has been imported or run). A 404 here would be
        # indistinguishable from an unknown project name.
        raise HTTPException(409, f"no store for project {cfg.project_name!r} yet "
                                 f"(expected {path}) — run import-backlog first")
    conn = open_read_only(path)
    try:
        yield conn
    finally:
        conn.close()


def require_repo_path(cfg: Config) -> Path:
    """`project.repo_path` or a clean 409.

    The template `example` profile has `repo_path: null`, so `Config.repo_path()`
    raises `ValueError` — which would surface as a 500. Anything that needs a
    checkout (the phase-3 job spawners) goes through here instead.
    """
    try:
        return cfg.repo_path()
    except ValueError as e:
        raise HTTPException(409, f"project {cfg.project_name!r} profile is "
                                 f"incomplete: {e}") from None
