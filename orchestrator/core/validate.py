"""Startup configuration validation — fail before the first call, not on task 7.

Every problem this module reports used to be discovered at DISPATCH: a typo'd
`preset:`, a provider name that no `providers:` block defines, a provider type
that is not registered, a `roles.worker.candidates` entry that is not a
worker_models key. Each of those raises somewhere deep in the graph, minutes
into a run, after real money and real subscription quota have been spent on the
tasks that happened to come first.

Two checks here are worth more than the rest:

* **A missing API key is an ERROR, not a warning.** `openai_compatible` logs a
  warning and substitutes the literal string "missing-key", so the symptom is a
  401 in the middle of a run rather than a refusal to start. Escalated only for
  providers an ACTIVE binding actually references — a project should not be
  blocked by a key for a backend it never calls.

* **The best-of-N collapse warning.** A candidate pool whose members resolve to
  the same (provider, model, effort) is not a pool; it is the same call issued N
  times at N times the price with none of the diversity the pool exists for.
  The live pools differentiate by `temperature`, and reasoning models reject
  `temperature` — the tolerant-400 retry strips it and the three "different"
  candidates become byte-identical requests. Nothing else in the system can
  notice that, because every layer below sees three legitimately distinct keys.

Severity follows REACHABILITY. A broken binding something actually uses is an
error; the same breakage on a preset or provider nothing references is a
warning, because the shipped presets deliberately include backends whose
provider type or API key may not exist on this machine yet. Problems are
collected and reported together: fixing config one exception at a time is the
slow way to find out you had four things wrong.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ..providers import PROVIDER_TYPES
from .presets import as_dict, resolve_entry, section

#: Provider types that ride an existing subscription and hold no API key. Used
#: only to skip the key check; the ledger's own cash/subscription split lives in
#: ops/budget.py and is not duplicated here.
_CLI_TYPES = ("claude_cli", "codex_cli")


@dataclass
class ValidationReport:
    """Everything wrong with a config, in one pass."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def format_errors(self) -> str:
        return "\n".join(f"  - {e}" for e in self.errors)


@dataclass
class _Binding:
    """One place the config names a backend: a role, or a worker_models entry.

    `active` means "this run can actually dispatch to it" — a role, or a worker
    key named by roles.worker.default / .candidates / run.degrade_model / a
    domain's worker_default. A worker_models entry defined but not wired to
    anything is inert, so its problems are warnings.
    """

    where: str
    raw: dict
    resolved: dict
    active: bool


def validate_config(cfg, *, will_spend: bool = True,
                    dispatches_workers: bool = True) -> ValidationReport:
    """Check a loaded Config for problems that would fail (or overspend) a run.

    Two checks are gated on what the CALLER is about to do, because a check that
    blocks a command it cannot possibly affect is just an outage:

    * `will_spend=False` (a `--dry-run`) downgrades the missing-API-key check to
      a warning. A dry run resolves the schedule and never opens a socket, so a
      401 it cannot provoke must not stop it — and a dry run is exactly how an
      operator inspects wiring BEFORE exporting a key.
    * `dispatches_workers=False` (`plan`, `discuss`) does the same for
      `roles.worker.default`. Those commands drive the smart tier and never
      reach the worker pool, so an unset worker default is a problem for a
      future `run`, not for this one.

    Both default to True, so anything that does not opt out gets the strict
    treatment.
    """
    report = ValidationReport()
    providers = section(cfg, "providers")
    presets = section(cfg, "presets")
    worker_models = section(cfg, "worker_models")
    roles = section(cfg, "roles")
    run = section(cfg, "run")
    domains = section(cfg, "domains")
    smart_provider = roles.get("smart_provider") or "claude_cli"

    worker_cfg = as_dict(roles.get("worker"))
    default_key = worker_cfg.get("default")
    candidates = [c for c in (worker_cfg.get("candidates") or []) if c]

    # ---- which worker_models keys can this run actually reach? --------------
    wired: set[str] = {c for c in candidates}
    for key in (default_key, run.get("degrade_model")):
        if key:
            wired.add(key)
    for dcfg in domains.values():
        dkey = as_dict(dcfg).get("worker_default")
        if dkey:
            wired.add(dkey)

    bindings: list[_Binding] = []
    for name, entry in roles.items():
        data = as_dict(entry)
        if not data.get("model") and not data.get("preset"):
            continue          # smart_provider (a string) and roles.worker (a pool)
        bindings.append(_Binding(f"roles.{name}", data, resolve_entry(cfg, entry),
                                 active=True))
    for name, entry in worker_models.items():
        bindings.append(_Binding(f"worker_models.{name}", as_dict(entry),
                                 resolve_entry(cfg, entry), active=name in wired))

    # ---- 1. every preset reference resolves --------------------------------
    for b in bindings:
        name = b.raw.get("preset")
        if name and name not in presets:
            _add(report, b.active,
                 f"{b.where}: preset {name!r} is not defined in presets: "
                 f"(known: {_known(presets)})")

    # ---- 2. every provider reference exists --------------------------------
    # Tracked as we go: a provider is "active" if an active binding routes to
    # it, which is what decides whether its own problems (unknown type, missing
    # key) are errors or warnings.
    active_providers: set[str] = set()
    for b in bindings:
        provider = b.resolved.get("provider")
        if provider is None and b.where.startswith("roles."):
            provider = smart_provider     # `provider: null` inherits the tier
        if not provider:
            _add(report, b.active, f"{b.where}: no provider (and no preset naming one)")
            continue
        if b.active:
            active_providers.add(provider)
        if provider not in providers:
            _add(report, b.active,
                 f"{b.where}: provider {provider!r} is not defined in providers: "
                 f"(known: {_known(providers)})")

    for name, preset in presets.items():
        provider = as_dict(preset).get("provider")
        # A preset nothing references is a menu entry, not a binding: it may name
        # a provider this project has not configured, and saying so is a warning.
        referenced = any(b.raw.get("preset") == name and b.active for b in bindings)
        if not provider:
            _add(report, referenced, f"presets.{name}: no provider")
        elif provider not in providers:
            _add(report, referenced,
                 f"presets.{name}: provider {provider!r} is not defined in "
                 f"providers: (known: {_known(providers)})")

    # ---- 3. every provider type is registered ------------------------------
    for name, pcfg in providers.items():
        ptype = as_dict(pcfg).get("type")
        if ptype not in PROVIDER_TYPES:
            _add(report, name in active_providers,
                 f"providers.{name}.type: {ptype!r} is not a registered provider "
                 f"type (known: {_known(PROVIDER_TYPES)})")

    # ---- 4. the worker pool names real worker_models keys ------------------
    if not default_key:
        _add(report, dispatches_workers,
             "roles.worker.default is not set — it must name a worker_models key "
             "(REQUIRED per project; see config/projects/example.yaml)")
    elif default_key not in worker_models:
        report.errors.append(
            f"roles.worker.default: {default_key!r} is not a worker_models key "
            f"(known: {_known(worker_models)})")
    for cand in candidates:
        if cand not in worker_models:
            report.errors.append(
                f"roles.worker.candidates: {cand!r} is not a worker_models key "
                f"(known: {_known(worker_models)})")

    # ---- 5. a referenced cash provider must have its API key ---------------
    for name in sorted(active_providers):
        pcfg = as_dict(providers.get(name))
        key_env = pcfg.get("api_key_env")
        if not key_env or pcfg.get("type") in _CLI_TYPES:
            continue          # subscription CLI: auth lives in the CLI itself
        if not os.environ.get(key_env):
            _add(report, will_spend,
                 f"providers.{name}: ${key_env} is empty or unset, and an active "
                 f"role/worker binding routes to this provider — the call would "
                 f"fail as a 401 mid-run. Export it, or bind that role/worker to "
                 f"a different preset.")

    # ---- 6. effort set on a backend that cannot use it ---------------------
    _check_effort_support(bindings, providers, smart_provider, report)

    # ---- 7. best-of-N collapse ---------------------------------------------
    _check_pool_diversity(cfg, candidates, worker_models, report)

    # ---- 8. a pool wider than the backend will run at once -----------------
    _check_pool_width(cfg, candidates, worker_models, providers, report)

    # ---- 9. anthropic_api needs an explicit max_tokens ---------------------
    _check_anthropic_max_tokens(bindings, presets, providers, smart_provider,
                                report)
    return report


def _check_effort_support(bindings: list[_Binding], providers: dict,
                          smart_provider: str, report: ValidationReport) -> None:
    """WARN when a binding sets `effort` on a provider type that drops it.

    The config can express "this worker runs at high reasoning" against a
    backend with no reasoning dial at all (`openai_compatible` fronting
    DeepSeek, `codex_cli`), and until the provider classes declared
    `supports_effort` nothing anywhere said so: the preset resolved, the level
    reached the call, and the last hop discarded it. The call still works — it
    just runs at the model's default depth while the config, the UI and the
    operator all believe otherwise, which is why this is the cheapest possible
    thing to check and the most expensive thing to miss.

    A WARNING, not an error: the binding is functional, and a run must not be
    blocked by a key that is merely inert. `validate_config` prints warnings; the
    fix is either a different preset or deleting the key.
    """
    for b in bindings:
        effort = b.resolved.get("effort")
        if not effort:
            continue
        provider = b.resolved.get("provider")
        if provider is None and b.where.startswith("roles."):
            provider = smart_provider     # `provider: null` inherits the tier
        ptype = as_dict(providers.get(provider)).get("type")
        cls = PROVIDER_TYPES.get(ptype)
        if cls is None or cls.supports_effort:
            continue                      # unknown provider/type: reported above
        report.warnings.append(
            f"{b.where}: effort={effort!r} but provider {provider!r} is a "
            f"{ptype} backend, which has no reasoning-effort dial — the level is "
            f"dropped at the call and the run silently uses the model's default "
            f"depth. Bind it to a preset on an effort-capable provider "
            f"(claude_cli, openai_responses) or remove the effort key.")


def _check_pool_diversity(cfg, candidates: list[str], worker_models: dict,
                          report: ValidationReport) -> None:
    """WARN when two candidates would issue the byte-identical call.

    The grouping key is (provider, model, effort, approach) — the four things
    that actually reach the model. `approach` is in there because it IS the
    diversity axis on a reasoning backend: `_build_stable_prompt` appends it
    LAST, after the cache prefix, so three candidates on one model at one effort
    with three different approaches share a single cached prefix and still ask
    three different questions. That is cheaper than diverging on effort, and it
    is the configuration this warning must not cry wolf about.

    `params` is deliberately NOT in the key, and that is the whole point. A pool
    differentiated only by `temperature` looks diverse in the profile and is not:
    a reasoning model rejects the parameter, the tolerant-400 retry strips it,
    and best-of-3 quietly becomes the same call three times at 3x the price.
    Nothing else in the system can detect that, which is why this exists.

    Deliberately NOT an error: a genuinely duplicated pool is a legitimate (if
    expensive) sampling strategy on a model that honors `temperature`.
    """
    groups: dict[tuple, list[str]] = {}
    for cand in candidates:
        entry = worker_models.get(cand)
        if entry is None:
            continue                      # already reported as an unknown key
        r = resolve_entry(cfg, entry)
        groups.setdefault(
            (r.get("provider"), r.get("model"), r.get("effort"),
             (r.get("approach") or "").strip()), []).append(cand)
    for (provider, model, effort, approach), keys in groups.items():
        if len(keys) > 1:
            report.warnings.append(
                f"roles.worker.candidates: {', '.join(keys)} all resolve to the "
                f"same backend ({provider}/{model}, effort={effort or 'default'}) "
                f"with {'the same' if approach else 'no'} `approach:` — best-of-N "
                f"will issue the same call {len(keys)} times at {len(keys)}x the "
                f"cost. Sampling params are not diversity: a reasoning model "
                f"rejects `temperature` and the retry drops it. Give each "
                f"candidate its own `approach:`, or differentiate by effort.")


def _check_pool_width(cfg, candidates: list[str], worker_models: dict,
                      providers: dict, report: ValidationReport) -> None:
    """WARN when more candidates route to one provider than it will run at once.

    The pool still runs — the semaphore queues the excess rather than dropping
    it — so this is a warning, not an error. What it costs is the thing the pool
    was bought for: with a 3-wide pool on a backend that admits 2, the third
    candidate does not start until one of the others finishes, so the task's
    wall clock is roughly two serial calls instead of one, on every attempt of
    every task. That is invisible from the inside (nothing errors, nothing is
    slower per call) and it is the single most likely reason a run bound to a
    CLI backend feels inexplicably sluggish.

    Counted per PROVIDER, not per preset: two presets on the same claude_cli
    provider share one semaphore, because they share one subscription.

    Deliberately does NOT multiply by `run.max_parallel_tasks`. Concurrent tasks
    contend for the same ceiling too, so the real queue is worse than this — but
    that is the ceiling doing its job (it exists to stop nine simultaneous CLI
    processes), whereas a pool that cannot fit inside it even once is a config
    the operator can simply fix.
    """
    routed: dict[str, list[str]] = {}
    for cand in candidates:
        entry = worker_models.get(cand)
        if entry is None:
            continue                      # already reported as an unknown key
        provider = resolve_entry(cfg, entry).get("provider")
        if provider:
            routed.setdefault(provider, []).append(cand)

    for provider, keys in sorted(routed.items()):
        pcfg = providers.get(provider)
        ptype = as_dict(pcfg).get("type")
        cls = PROVIDER_TYPES.get(ptype)
        if cls is None:
            continue                      # unknown provider/type: reported above
        limit = cls.configured_max_concurrency(pcfg)
        if limit is None or len(keys) <= limit:
            continue
        report.warnings.append(
            f"roles.worker.candidates: {len(keys)} candidates "
            f"({', '.join(keys)}) route to provider {provider!r}, a {ptype} "
            f"backend that runs at most {limit} call(s) at a time "
            f"(providers.{provider}.max_concurrency) — the rest queue, so the "
            f"pool costs best-of-{len(keys)} money at best-of-{limit} speed. "
            f"Narrow the pool, raise max_concurrency if the plan tolerates it, "
            f"or bind some candidates to a preset on a different provider.")


#: Provider type whose API makes `max_tokens` a REQUIRED request parameter.
#: Isolated here rather than inlined so the check below reads as a rule about
#: one backend's contract instead of a magic string comparison.
_MAX_TOKENS_REQUIRED = "anthropic_api"


def _declares_max_tokens(entry: dict, pcfg: dict) -> bool:
    """Is an output cap written ANYWHERE the call can actually see it?

    Mirrors `AnthropicApiProvider._max_tokens` exactly — `params.max_tokens` on
    the entry, `max_tokens` on the entry or the preset it names, then
    `providers.<name>.max_tokens`. Deliberately not a looser check: a value the
    provider cannot reach is config that promises something the code drops, and
    accepting one here would make this check worse than not having it.
    """
    if as_dict(entry.get("params")).get("max_tokens") is not None:
        return True
    return (entry.get("max_tokens") is not None
            or pcfg.get("max_tokens") is not None)


def _check_anthropic_max_tokens(bindings: list[_Binding], presets: dict,
                                providers: dict, smart_provider: str,
                                report: ValidationReport) -> None:
    """Require an explicit output cap on every binding that routes to the
    Messages API.

    `max_tokens` is not optional on that API the way it is on the OpenAI shapes:
    there is no server-side default and a request without one is a 400. The
    provider therefore falls back to `budget.assumed_max_output_tokens` so no
    call can hard-fail on a missing key — but 8000 tokens is a guess about a
    model the operator knows better than we do, and a worker silently truncated
    mid-`<file>` block looks exactly like a model that cannot follow the block
    protocol. Say it at startup instead.

    Severity follows reachability, like everything else here: an error on a
    binding this run will dispatch to, a warning on an inert entry or on a menu
    preset nothing references (the shipped presets deliberately include backends
    this machine may not be set up for yet).
    """
    def ptype(provider: str | None) -> str | None:
        return as_dict(providers.get(provider)).get("type")

    for b in bindings:
        provider = b.resolved.get("provider")
        if provider is None and b.where.startswith("roles."):
            provider = smart_provider     # `provider: null` inherits the tier
        if ptype(provider) != _MAX_TOKENS_REQUIRED:
            continue
        if _declares_max_tokens(b.resolved, as_dict(providers.get(provider))):
            continue
        _add(report, b.active,
             f"{b.where}: provider {provider!r} is an {_MAX_TOKENS_REQUIRED} "
             f"backend, whose API REQUIRES max_tokens on every call, and none "
             f"is set. Add `max_tokens:` to the preset or the entry (or "
             f"`params: {{max_tokens: N}}`, or providers.{provider}.max_tokens) "
             f"— otherwise every call silently uses "
             f"budget.assumed_max_output_tokens, and a worker truncated "
             f"mid-block is indistinguishable from one that cannot follow the "
             f"block protocol.")

    # A preset no binding names at all is a menu entry: still worth flagging,
    # because it is offered to the UI as a selectable backend, but only as a
    # warning. Presets a binding DOES name were covered above, through that
    # binding's resolved view (which already has the preset merged in).
    named = {b.raw.get("preset") for b in bindings}
    for name, preset in presets.items():
        if name in named:
            continue
        data = as_dict(preset)
        provider = data.get("provider")
        if ptype(provider) != _MAX_TOKENS_REQUIRED:
            continue
        if _declares_max_tokens(data, as_dict(providers.get(provider))):
            continue
        report.warnings.append(
            f"presets.{name}: provider {provider!r} is an "
            f"{_MAX_TOKENS_REQUIRED} backend, whose API requires max_tokens on "
            f"every call, and this preset sets none — anything bound to it "
            f"falls back to budget.assumed_max_output_tokens.")


def _add(report: ValidationReport, fatal: bool, message: str) -> None:
    (report.errors if fatal else report.warnings).append(message)


def _known(mapping) -> str:
    names = sorted(mapping.keys())
    return ", ".join(names) if names else "none defined"
