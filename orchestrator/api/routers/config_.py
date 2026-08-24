"""Backend presets, and the one config layer the panel is allowed to write.

Named `config_.py` for the same reason as `discuss_.py` and `live_.py`:
`core/config.py` is the configuration itself and this module is only the HTTP
shell over it.

Two GETs answer the question the settings UI has no other way to ask — *what
does this project's config actually permit?* — following the precedent
`DiscussOptions` set for the planner role. A hardcoded list in the frontend is a
list that drifts from the harness the first time a preset is added.

## The POST is a deliberate, narrow exception to PLAN §3.1 rule 2

The frontend repo's `DEVDOCS/PLAN.md` §3.1 rule 2 says *"GET is read-only,
literally… Writers stay exclusively the CLI subprocesses"*, and the settings page
says the panel must not become a second place configuration can be changed. This
endpoint reverses that, scoped as narrowly as the reversal can be made:

* The panel may **bind** an existing worker key or an existing role to an
  existing, server-defined **preset**, and pin a reasoning effort from a list the
  server supplies. That is the entire vocabulary — enforced by `extra="forbid"`
  on the request model and again by `core.assignments.resolve_overlay`.
* It may **not define** a preset, name a provider, or set a `base_url`, an argv,
  an API key or a price. **Nothing the browser sends becomes a subprocess
  argument or a URL.** Worker keys, role names, preset names and candidate ids
  are all checked for membership in the server's own config and rejected with a
  422 otherwise; none of them ever reaches a path join.
* Rule 2's actual subject — the SQLite store — is untouched. The overlay is a
  separate JSON file, and every GET in this app keeps reading through its
  existing `file:…?mode=ro` connection.

What the reversal buys is the thing the preset layer exists for: moving a worker
between a metered API and a CLI riding a subscription without hand-editing YAML.
What it costs is bounded by the list above.

The write refuses with 409 while a job or a discuss session is live, reusing the
same mutual-exclusion mechanism as `routers/jobs._spawn` rather than inventing a
second lock. A mid-run rebinding would apply only to tasks not yet dispatched,
producing a run whose recorded model silently varies by task — a result that is
neither the old configuration nor the new one. `Assignments.locked` /
`locked_reason` exist so the UI disables the form instead of failing on submit.
"""

from __future__ import annotations

import logging
import os
import shutil
import weakref
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException

from ...core.assignments import (bind_entry, read_overlay, resolve_overlay,
                                 role_keys, write_overlay)
from ...core.config import Config
from ...core.presets import as_dict, resolve_entry, section
from ...providers import PROVIDER_TYPES
from ..deps import ProjectRegistry, assignments_path, get_registry, resolve_project
from ..discuss import DiscussManager, get_manager
from ..errors import CONFIG_WRITE_ERRORS, PROJECT_ERRORS
from ..jobs import JobSupervisor, get_supervisor
from ..schemas import (Assignment, Assignments, AssignmentsRequest, Preset,
                       Presets)
from .discuss_ import EFFORTS

log = logging.getLogger("orchestrator.api.config")

router = APIRouter(prefix="/api/projects/{project}", tags=["config"],
                   responses=PROJECT_ERRORS)

#: Provider types that ride an existing subscription rather than spending cash.
#: Mirrors `core.validate._CLI_TYPES` and `Budget._CLI_TYPES`; kept as a literal
#: here rather than imported from either, because both are private to modules
#: this one has no other reason to depend on.
CLI_TYPES = ("claude_cli", "codex_cli")

#: `Config` -> {provider name: (configured, detail)}.
#:
#: Cached per Config instance, keyed weakly, exactly like the provider cache in
#: `providers/__init__`. The probe is `shutil.which` plus an `os.environ` lookup:
#: cheap, but this is a GET the panel polls, and `which` walks PATH on every
#: call. A subprocess (`claude --version`) would be the natural next step and is
#: deliberately not taken — a read endpoint must not execute anything, and the
#: answer "is it installed" does not need it.
#:
#: The consequence, stated rather than worked around: exporting a key after the
#: server started is not noticed until the Config is reloaded. The registry
#: caches Config for the process lifetime anyway, so this changes nothing.
_probes: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


# ---- provider probing ----------------------------------------------------
def _probe(cfg: Config, name: str) -> tuple[bool, str | None]:
    """Is provider `name` usable from this server? (configured, detail).

    **The detail may name an environment VARIABLE; it must never contain that
    variable's value.** Everything this function learns about a key is the
    boolean "is it non-empty".
    """
    pcfg = as_dict(section(cfg, "providers").get(name))
    if not pcfg:
        return False, (f"providers.{name} is not defined in this project's config, "
                       f"so a preset naming it cannot bind")
    ptype = str(pcfg.get("type") or "")
    if ptype in CLI_TYPES:
        binary = str(pcfg.get("binary") or "")
        if not binary:
            return False, f"providers.{name} names no binary to run"
        found = shutil.which(binary)
        if found is None:
            return False, (f"{binary!r} is not on this server's PATH — install the "
                           f"CLI, or set providers.{name}.binary to its full path")
        return True, None
    key_env = pcfg.get("api_key_env")
    if not key_env:
        # A local OpenAI-compatible endpoint (LM Studio, vLLM) legitimately needs
        # no key. "Configured" is the honest answer, with the reason attached so
        # the panel does not have to guess why there is no key to check.
        return True, (f"providers.{name} declares no api_key_env, so no key is "
                      f"required")
    if os.environ.get(str(key_env)):
        return True, None
    return False, (f"${key_env} is empty or unset in the environment this server "
                   f"was started from — export it and restart. Note that .env is "
                   f"not auto-loaded")


def _configured(cfg: Config, name: str) -> tuple[bool, str | None]:
    cache = _probes.get(cfg)
    if cache is None:
        cache = _probes[cfg] = {}
    if name not in cache:
        cache[name] = _probe(cfg, name)
    return cache[name]


# ---- projection ----------------------------------------------------------
def _effort(value: Any, where: str) -> str | None:
    """A configured effort, or None when it is not a level this repo knows.

    A GET that 500s because someone typo'd `effort: hgih` in a YAML file is worse
    than one that reports the binding as effort-unset: the panel is one of the
    few places that typo can be *seen*, and it cannot show anything if the
    response never validates. `core.validate` is where the typo gets named.
    """
    if value is None:
        return None
    text = str(value)
    if text in EFFORTS:
        return text
    log.warning("%s sets effort=%r, which is not one of %s — reporting it as "
                "unset", where, value, ", ".join(EFFORTS))
    return None


def _is_cash(ptype: str, pcfg: dict) -> bool:
    """Does a call through this provider spend money?

    `Budget._is_cash` minus the global `count_cli` / `count_claude_cli` toggles,
    on purpose: those change how subscription calls are BOOKED against the cash
    budget, not whether a dollar actually leaves. The panel's chip answers the
    second question.
    """
    if "count" in pcfg:                        # explicit per-provider override
        return bool(pcfg["count"])
    if ptype not in CLI_TYPES:
        return True
    return ptype == "codex_cli" and pcfg.get("auth", "subscription") == "api"


def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _presets(cfg: Config) -> list[Preset]:
    providers = section(cfg, "providers")
    out: list[Preset] = []
    for key, raw in section(cfg, "presets").items():
        entry = as_dict(raw)
        provider = str(entry.get("provider") or "")
        pcfg = as_dict(providers.get(provider))
        ptype = str(pcfg.get("type") or "")
        cls = PROVIDER_TYPES.get(ptype)
        model = entry.get("model")
        # The provider's known ids plus this preset's own, following `_options`
        # in discuss_: a model id written by hand must never disappear from the
        # dropdown that is supposed to show it.
        models = [str(m) for m in (pcfg.get("models") or [])]
        if model and str(model) not in models:
            models.append(str(model))
        configured, detail = _configured(cfg, provider)
        out.append(Preset(
            key=str(key),
            label=str(entry.get("label") or key),
            kind="cli" if ptype in CLI_TYPES else "api",
            provider=provider,
            provider_type=ptype,
            model=str(model) if model else None,
            models=models,
            # Empty when the backend has no reasoning dial at all, so the UI can
            # disable the effort control instead of offering a setting that the
            # provider drops with a warning nobody reading the panel will see.
            efforts=list(EFFORTS) if cls is not None and cls.supports_effort else [],
            effort=_effort(entry.get("effort"), f"presets.{key}"),
            cash=_is_cash(ptype, pcfg),
            input_per_mtok=_number(entry.get("input_per_mtok")),
            output_per_mtok=_number(entry.get("output_per_mtok")),
            configured=configured,
            configured_detail=detail))
    return out


def _describe(effective: dict) -> str | None:
    provider, model = effective.get("provider"), effective.get("model")
    if provider and model:
        return f"{provider} / {model}"
    return str(model or provider) if (model or provider) else None


def _row(cfg: Config, key: str, entry: Any, binding: dict | None) -> Assignment:
    """One assignment row, resolved exactly the way a run would resolve it.

    `bind_entry` is the same function `apply_overlay` uses, so a row the panel
    shows as bound to a preset is bound to that preset in the config a run
    loads — there is no second implementation of the rebinding rule to drift.
    """
    source = as_dict(entry)
    effective = resolve_entry(cfg, bind_entry(source, binding) if binding else source)
    preset = effective.get("preset")
    return Assignment(
        key=key,
        label=str(effective.get("label") or _describe(effective) or key),
        preset=str(preset) if preset else None,
        effort=_effort(effective.get("effort"), f"{key}"),
        source="overlay" if binding else "profile")


def _lock_reason(cfg: Config, sup: JobSupervisor,
                 manager: DiscussManager) -> str | None:
    """Why the overlay cannot be written right now, or None.

    The same two conditions `routers/jobs._spawn` checks, in the same order and
    against the same two objects — a config change and a job are mutually
    exclusive for the same reason two jobs are.
    """
    busy = sup.in_flight(cfg)
    if busy is not None:
        return (f"job {busy.job_id} ({busy.command}) is {busy.status} for "
                f"{cfg.project_name!r} — a rebinding now would reach only the "
                f"tasks it has not dispatched yet, so the run would record two "
                f"different models for no visible reason. Stop it first")
    session = manager.active(cfg.project_name)
    if session is not None:
        return (f"a discuss session ({session.session_id}) is open for "
                f"{cfg.project_name!r} and is mid-conversation on its planner "
                f"binding — close it first")
    return None


def _assignments(cfg: Config, overlay: dict, locked_reason: str | None) -> Assignments:
    """Project the merged config plus the overlay file into the response.

    The overlay is read HERE rather than trusted to have already been merged into
    `cfg`, because this projection has to report `source` per row — which layer
    bound this worker — and a merged config no longer remembers. Reading it also
    makes the response correct for a `Config` that was built before the overlay
    existed, which is exactly the state a POST leaves the registry in.
    """
    resolved = resolve_overlay(cfg.as_dict(), overlay)
    roles_section = section(cfg, "roles")
    pool = as_dict(roles_section.get("worker"))
    workers = [_row(cfg, key, raw, resolved.workers.get(key))
               for key, raw in section(cfg, "worker_models").items()]
    roles = [_row(cfg, key, roles_section.get(key), resolved.roles.get(key))
             for key in sorted(role_keys(roles_section))]
    candidates = resolved.candidates if resolved.candidates is not None \
        else [str(c) for c in (pool.get("candidates") or [])]
    default_worker = resolved.default_worker or pool.get("default")
    return Assignments(
        project=cfg.project_name, workers=workers, roles=roles,
        default_worker=str(default_worker) if default_worker else None,
        candidates=list(candidates),
        locked=locked_reason is not None, locked_reason=locked_reason)


# ---- read ----------------------------------------------------------------
@router.get("/config/presets", response_model=Presets)
def get_presets(cfg: Config = Depends(resolve_project)) -> Presets:
    return Presets(project=cfg.project_name, presets=_presets(cfg),
                   efforts=list(EFFORTS))


@router.get("/config/assignments", response_model=Assignments)
def get_assignments(cfg: Config = Depends(resolve_project),
                    sup: JobSupervisor = Depends(get_supervisor),
                    manager: DiscussManager = Depends(get_manager)) -> Assignments:
    return _assignments(cfg, read_overlay(assignments_path(cfg)),
                        _lock_reason(cfg, sup, manager))


# ---- write ---------------------------------------------------------------
def _validate(cfg: Config, body: AssignmentsRequest) -> dict:
    """The overlay to write, or a 422 naming every rejected value at once.

    Membership checks against the server's own config, never a pattern match:
    a worker key is a `worker_models` key, a role is a bindable role, a preset is
    a `presets:` key, a candidate is a `worker_models` key. Anything else is
    refused here rather than dropped later, because a write that silently
    discards half of what was sent leaves the panel showing a state the server
    never accepted.

    Reported together, not one at a time — the form submits every row at once, so
    fixing them one 422 at a time would be one round trip per bad row.
    """
    presets = section(cfg, "presets")
    worker_models = section(cfg, "worker_models")
    roles = role_keys(section(cfg, "roles"))
    bad: list[str] = []

    def _check(field: str, key: str, preset: str, targets, kind: str) -> None:
        if key not in targets:
            bad.append(f"{field}.{key}: no such {kind} (known: {_known(targets)})")
        elif preset not in presets:
            bad.append(f"{field}.{key}: unknown preset {preset!r} "
                       f"(known: {_known(presets)})")

    for key, update in body.workers.items():
        _check("workers", key, update.preset, worker_models, "worker_models key")
    for key, update in body.roles.items():
        _check("roles", key, update.preset, roles, "bindable role")
    if body.default_worker is not None and body.default_worker not in worker_models:
        bad.append(f"default_worker: {body.default_worker!r} is not a "
                   f"worker_models key (known: {_known(worker_models)})")
    for candidate in body.candidates or []:
        if candidate not in worker_models:
            bad.append(f"candidates: {candidate!r} is not a worker_models key "
                       f"(known: {_known(worker_models)})")
    if bad:
        raise HTTPException(422, "; ".join(bad))

    overlay: dict = {
        "workers": {k: v.model_dump(exclude_none=True)
                    for k, v in body.workers.items()},
        "roles": {k: v.model_dump(exclude_none=True)
                  for k, v in body.roles.items()},
    }
    # Absent keys, not nulls: "the panel did not set this" and "the panel set
    # this to nothing" are different, and only the first has a meaning here.
    if body.default_worker is not None:
        overlay["default_worker"] = body.default_worker
    if body.candidates is not None:
        overlay["candidates"] = list(body.candidates)
    return overlay


def _known(names) -> str:
    return ", ".join(sorted(str(n) for n in names)) or "none"


@router.post("/config/assignments", response_model=Assignments,
             responses=CONFIG_WRITE_ERRORS)
def set_assignments(body: AssignmentsRequest = Body(...),
                    cfg: Config = Depends(resolve_project),
                    registry: ProjectRegistry = Depends(get_registry),
                    sup: JobSupervisor = Depends(get_supervisor),
                    manager: DiscussManager = Depends(get_manager)) -> Assignments:
    # The lock first, like `_spawn`: "nothing will be written" is the more
    # decisive answer, and a form the UI should have disabled does not need its
    # field errors enumerated.
    locked = _lock_reason(cfg, sup, manager)
    if locked is not None:
        raise HTTPException(409, locked)
    overlay = _validate(cfg, body)
    path = assignments_path(cfg)
    if path is None:
        raise HTTPException(409, f"project {cfg.project_name!r} has no "
                                 f"paths.state_dir, so there is nowhere to keep "
                                 f"an assignment overlay")
    write_overlay(path, overlay)
    # The registry caches Config for the process lifetime, so without this every
    # later read in this process would answer from the layers as they were before
    # the write.
    registry.invalidate(cfg.project_name)
    fresh = registry.config(cfg.project_name)
    return _assignments(fresh, read_overlay(path), _lock_reason(fresh, sup, manager))
