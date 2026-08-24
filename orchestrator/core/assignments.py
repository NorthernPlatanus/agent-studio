"""The assignment overlay — the one configuration layer the control panel writes.

An overlay is a small JSON file at `state/<project>.assignments.json` that binds
existing `worker_models` keys and existing `roles` to existing **presets**. It is
merged by `load_config` as a layer after `config/local.yaml` and before the
`ORCH_*` environment overrides, so a machine-local YAML override still loses to a
panel binding, and the environment still wins over everything as the emergency
override it has always been.

**Why a JSON overlay and not a YAML writer against the project profile.** Those
profiles are roughly half comments, and the comments *are* the documentation —
`config/default.yaml` spends thirty lines explaining the idle clock alone. Every
round-trip YAML writer either loses those comments or mangles their placement, so
the first panel-driven edit would quietly destroy the file's real value. A
separate overlay keeps the hand-edited profile authoritative for preset
*definitions*, makes "undo everything the panel did" a single `rm`, and keeps the
panel out of the file a human is expected to read.

**What an overlay may say, and nothing else.** Per worker key and per role, a
preset name and (optionally) a reasoning effort; plus `roles.worker.default` and
`roles.worker.candidates`. It cannot define a preset, name a provider, set a
`base_url`, an argv, an API key or a price. That boundary is the whole reason
this layer is safe to expose over HTTP, and it is enforced twice: by the API's
`extra="forbid"` request model, and here by `resolve_overlay`, which builds the
merged config out of a fixed key list rather than out of whatever the file holds.

**Unresolvable keys are dropped, never raised.** A stale overlay pinning a preset
that a later profile edit deleted must not be able to stop `orchestrator run`
from booting — the panel would then be a way to brick the CLI. Every such key is
dropped with a warning naming it, and the underlying profile binding stands.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("orchestrator.assignments")

#: Written into every overlay. Nothing reads it yet; it exists so a future shape
#: change can recognize (and refuse, or migrate) a file written by an older
#: panel instead of silently misreading it.
OVERLAY_VERSION = 1

#: The reasoning levels an overlay may pin.
#:
#: One definition, shared with `api/routers/discuss_.EFFORTS` (which imports it),
#: because the list the settings UI offers and the list the writer accepts must
#: not be able to drift. `none` is deliberately absent even though
#: `openai_responses` accepts it: `claude_cli` does not, and an overlay is bound
#: to a preset whose provider can change under it, so the overlay only pins
#: levels that every effort-capable backend here understands.
EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

#: Keys a preset owns, and which are therefore STRIPPED from an entry the overlay
#: rebinds.
#:
#: This is the non-obvious half of applying an overlay. `core.presets.resolve_entry`
#: is deliberately entry-wins: a key written on the entry beats the preset it
#: names. The shipped worker entries write `provider` and `model` inline, so
#: merely adding `preset: luna_high` to one of them would resolve straight back to
#: DeepSeek and the panel's change would appear to do nothing at all. Rebinding
#: therefore clears the keys the preset is there to supply, and preserves the ones
#: it is not — `params`, `approach`, a role's `allowed_tools`.
PRESET_OWNED_KEYS: tuple[str, ...] = (
    "provider", "model", "effort", "label",
    "input_per_mtok", "output_per_mtok",
    "cache_read_per_mtok", "cache_write_per_mtok", "long_context",
)


# ---- file I/O ------------------------------------------------------------
def overlay_path(state_dir: Path | str, project: str) -> Path:
    """`state/<project>.assignments.json`.

    Beside the project's store and checkpoints, because it is per-project runtime
    state and not repository content — the same reason `state/` is gitignored.
    """
    return Path(state_dir) / f"{project}.assignments.json"


def read_overlay(path: Path | None) -> dict:
    """The overlay's raw contents, or `{}`.

    Every failure mode answers `{}`: no file, an unreadable one, a truncated one,
    a JSON document that is not an object. This is read on the boot path of every
    CLI command, and none of those are reasons to refuse to start — the profile
    binding underneath is still a complete, working configuration.
    """
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        log.warning("assignment overlay %s is unreadable (%s) — ignoring it and "
                    "using the profile's own bindings", path, e)
        return {}
    if not isinstance(data, dict):
        log.warning("assignment overlay %s is not a JSON object — ignoring it", path)
        return {}
    return data


def write_overlay(path: Path, overlay: dict) -> None:
    """Write the overlay atomically: temp file in the same directory, `os.replace`.

    Atomic because a reader is `load_config` on the boot path of every CLI
    command, and a half-written file there is a run that either dies at import or
    (worse) boots with half a binding. `os.replace` is atomic within a
    filesystem, which is why the temp file is a sibling rather than in `/tmp`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps({"version": OVERLAY_VERSION, **overlay},
                      indent=1, sort_keys=True) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(body)
    os.replace(tmp, path)


# ---- resolution ----------------------------------------------------------
@dataclass
class ResolvedOverlay:
    """An overlay checked against one config: only the parts that still resolve.

    Shared by the two callers that must agree — `apply_overlay` (the config
    layer) and the API's assignments projection (which reports `source:
    "overlay"` for exactly the keys that survived). A second copy of the
    "does this key still resolve?" rule is a second chance for the panel to show
    a binding the run does not actually use.
    """

    workers: dict[str, dict] = field(default_factory=dict)
    roles: dict[str, dict] = field(default_factory=dict)
    default_worker: str | None = None
    candidates: list[str] | None = None
    warnings: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.workers or self.roles or self.default_worker
                    or self.candidates is not None)


def _section(data: dict, name: str) -> dict:
    value = data.get(name)
    return value if isinstance(value, dict) else {}


def _binding(raw: Any) -> dict | None:
    """`{"preset": str, "effort": str?}` from one overlay entry, or None."""
    if not isinstance(raw, dict):
        return None
    preset = raw.get("preset")
    if not isinstance(preset, str) or not preset:
        return None
    out: dict = {"preset": preset}
    effort = raw.get("effort")
    if isinstance(effort, str) and effort:
        out["effort"] = effort
    return out


def role_keys(roles: dict) -> set[str]:
    """Role names an overlay may bind.

    `roles:` is not uniformly a mapping of role -> binding: it also holds
    `smart_provider` (a string) and `worker` (a candidate pool, whose default and
    candidates the overlay sets through their own fields). Both are excluded, so
    what remains is exactly the smart-tier roles — planner, reviewer, verifier.
    """
    return {k for k, v in roles.items() if isinstance(v, dict) and k != "worker"}


def resolve_overlay(data: dict, overlay: dict) -> ResolvedOverlay:
    """Check `overlay` against the config `data`, dropping what no longer binds.

    Four things must still be true of a key for it to survive: the target exists
    (a `worker_models` key, a smart-tier role), the preset it names is defined,
    the effort is one this repo's providers understand, and the whole thing is
    shaped like a binding. Anything else is a warning and a dropped key — see the
    module docstring for why this can never raise.
    """
    out = ResolvedOverlay()
    presets = _section(data, "presets")
    worker_models = _section(data, "worker_models")
    roles = role_keys(_section(data, "roles"))

    def _bindings(field_name: str, targets, kind: str) -> dict[str, dict]:
        kept: dict[str, dict] = {}
        for key, raw in _section(overlay, field_name).items():
            binding = _binding(raw)
            if binding is None:
                out.warnings.append(
                    f"{field_name}.{key} is not a {{preset, effort}} binding")
                continue
            if key not in targets:
                out.warnings.append(
                    f"{field_name}.{key} names no {kind} in this project's config "
                    f"— dropped (the panel binds existing entries, it never "
                    f"creates them)")
                continue
            if binding["preset"] not in presets:
                out.warnings.append(
                    f"{field_name}.{key} pins preset {binding['preset']!r}, which "
                    f"is not defined in presets: — dropped, so the profile's own "
                    f"binding stands")
                continue
            effort = binding.get("effort")
            if effort is not None and effort not in EFFORTS:
                out.warnings.append(
                    f"{field_name}.{key} pins effort {effort!r}, which is not one "
                    f"of {', '.join(EFFORTS)} — the preset's own effort is used")
                binding.pop("effort")
            kept[key] = binding
        return kept

    out.workers = _bindings("workers", worker_models, "worker_models entry")
    out.roles = _bindings("roles", roles, "bindable role")

    default_worker = overlay.get("default_worker")
    if default_worker is not None:
        if isinstance(default_worker, str) and default_worker in worker_models:
            out.default_worker = default_worker
        else:
            out.warnings.append(
                f"default_worker {default_worker!r} is not a worker_models key "
                f"— dropped, so roles.worker.default stands")

    candidates = overlay.get("candidates")
    if candidates is not None:
        unknown = [c for c in candidates
                   if not isinstance(c, str) or c not in worker_models] \
            if isinstance(candidates, list) else [candidates]
        if unknown:
            # The WHOLE list is dropped, not the unknown members. A pool silently
            # narrowed from three candidates to two is a best-of-N width the
            # operator never chose, and one that shows up only as a quieter,
            # cheaper, worse run — far harder to notice than the pool simply not
            # having changed.
            out.warnings.append(
                f"candidates names unknown worker_models keys ({unknown!r}) — the "
                f"whole pool is dropped rather than silently narrowed")
        else:
            out.candidates = list(candidates)

    return out


def bind_entry(entry: Any, binding: dict) -> dict:
    """One `worker_models` / `roles` entry, rebound to the overlay's preset.

    Returns a NEW dict: the entry's own keys minus everything the preset owns
    (see `PRESET_OWNED_KEYS`), plus the preset name and the pinned effort. The
    API's projection calls this too, so what the panel displays as bound is
    computed by the same function that binds it.
    """
    data = entry if isinstance(entry, dict) else {}
    merged = {k: v for k, v in data.items() if k not in PRESET_OWNED_KEYS}
    merged["preset"] = binding["preset"]
    if binding.get("effort"):
        merged["effort"] = binding["effort"]
    return merged


def apply_overlay(data: dict, overlay: dict, *, source: str = "") -> dict:
    """The merged config `data` with the overlay's surviving bindings applied.

    A shallow copy plus fresh copies of only the sections it touches — the caller
    (`load_config`) is mid-merge and the sections it has not reached must not be
    aliased into the result.
    """
    resolved = resolve_overlay(data, overlay)
    where = f" ({source})" if source else ""
    for warning in resolved.warnings:
        log.warning("assignment overlay%s: %s", where, warning)
    if resolved.is_empty():
        return data

    out = dict(data)
    if resolved.workers:
        worker_models = {k: (dict(v) if isinstance(v, dict) else v)
                         for k, v in _section(data, "worker_models").items()}
        for key, binding in resolved.workers.items():
            worker_models[key] = bind_entry(worker_models.get(key), binding)
        out["worker_models"] = worker_models

    pool_changed = resolved.default_worker is not None or resolved.candidates is not None
    if resolved.roles or pool_changed:
        roles = {k: (dict(v) if isinstance(v, dict) else v)
                 for k, v in _section(data, "roles").items()}
        for key, binding in resolved.roles.items():
            roles[key] = bind_entry(roles.get(key), binding)
        if pool_changed:
            worker = dict(roles.get("worker") or {})
            if resolved.default_worker is not None:
                worker["default"] = resolved.default_worker
            if resolved.candidates is not None:
                worker["candidates"] = list(resolved.candidates)
            roles["worker"] = worker
        out["roles"] = roles
    return out
