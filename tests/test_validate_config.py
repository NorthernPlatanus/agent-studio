"""core/validate — every problem the config can have, found before the run.

The two checks that carry their weight: a missing API key on a provider a live
binding routes to (today a warning and a literal "missing-key", i.e. a 401
discovered mid-run), and the best-of-N collapse warning, which is the only
place in the system that can notice three "diverse" candidates are the same
call three times.

Severity follows reachability throughout: the same breakage is an error on a
binding the run will dispatch to and a warning on one nothing references.
"""

from pathlib import Path

from orchestrator.core.config import Config
from orchestrator.core.validate import validate_config


def _cfg(**overrides) -> Config:
    """A config with nothing wrong with it, plus whatever the test breaks.

    claude_cli only, so the clean case does not depend on this machine's
    environment; the API-key tests add a cash provider explicitly.
    """
    data = {
        "presets": {
            "claude_cli_opus": {"label": "Claude Code CLI · opus",
                                "provider": "claude_cli", "model": "opus",
                                "effort": "high"},
        },
        "providers": {"claude_cli": {"type": "claude_cli", "binary": "claude"}},
        "worker_models": {
            "w1": {"provider": "claude_cli", "model": "sonnet"},
            "w2": {"provider": "claude_cli", "model": "sonnet", "effort": "high"},
        },
        "roles": {
            "smart_provider": "claude_cli",
            "planner": {"provider": None, "model": "opus", "effort": None},
            "reviewer": {"preset": "claude_cli_opus"},
            "worker": {"default": "w1", "candidates": ["w1", "w2"]},
        },
        "run": {"degrade_model": "w1"},
        "domains": {},
    }
    data.update(overrides)
    return Config(data, "proj", Path("/tmp"))


# ---- the clean case ---------------------------------------------------------

def test_a_clean_config_reports_nothing():
    report = validate_config(_cfg())
    assert report.errors == []
    assert report.warnings == []
    assert report.ok


# ---- preset references ------------------------------------------------------

def test_an_unknown_preset_on_a_live_worker_is_an_error():
    report = validate_config(_cfg(worker_models={
        "w1": {"preset": "luna_hi"}, "w2": {"provider": "claude_cli", "model": "sonnet"}}))
    assert any("preset 'luna_hi'" in e and "worker_models.w1" in e
               for e in report.errors), report.errors


def test_an_unknown_preset_on_a_role_is_an_error():
    report = validate_config(_cfg(roles={
        "smart_provider": "claude_cli",
        "planner": {"preset": "typo"},
        "worker": {"default": "w1", "candidates": ["w1"]}}))
    assert any("roles.planner" in e and "typo" in e for e in report.errors)


def test_an_unknown_preset_on_an_unwired_worker_is_only_a_warning():
    """A worker_models entry nothing references cannot be dispatched to, so it
    must not block a run that never touches it."""
    report = validate_config(_cfg(worker_models={
        "w1": {"provider": "claude_cli", "model": "sonnet"},
        "w2": {"provider": "claude_cli", "model": "sonnet"},
        "spare": {"preset": "gone"}}))
    assert report.errors == []
    assert any("spare" in w and "gone" in w for w in report.warnings)


# ---- provider references ----------------------------------------------------

def test_an_undefined_provider_on_a_worker_is_an_error():
    report = validate_config(_cfg(worker_models={
        "w1": {"provider": "nope", "model": "m"},
        "w2": {"provider": "claude_cli", "model": "sonnet"}}))
    assert any("worker_models.w1" in e and "'nope'" in e for e in report.errors)


def test_an_undefined_provider_on_a_referenced_preset_is_an_error():
    report = validate_config(_cfg(presets={
        "claude_cli_opus": {"provider": "ghost", "model": "opus"}}))
    assert any("presets.claude_cli_opus" in e and "'ghost'" in e
               for e in report.errors)


def test_an_undefined_provider_on_an_unreferenced_preset_is_only_a_warning():
    """The shipped preset list deliberately covers backends a given project has
    not configured — a menu entry is not a binding."""
    report = validate_config(_cfg(presets={
        "claude_cli_opus": {"provider": "claude_cli", "model": "opus"},
        "deepseek_flash": {"provider": "cometapi", "model": "deepseek-v4-flash"}}))
    assert report.errors == []
    assert any("presets.deepseek_flash" in w and "cometapi" in w
               for w in report.warnings)


# ---- provider types ---------------------------------------------------------

def test_an_unregistered_type_on_a_referenced_provider_is_an_error():
    report = validate_config(_cfg(providers={
        "claude_cli": {"type": "quantum_cli", "binary": "claude"}}))
    assert any("providers.claude_cli.type" in e and "quantum_cli" in e
               for e in report.errors)


def test_an_unregistered_type_on_an_unused_provider_is_only_a_warning():
    """A provider block nothing references must not hard-fail, whatever its type.

    Written against a MADE-UP type on purpose. It used to name `openai_responses`
    and then `anthropic_api` — the two types default.yaml shipped before their
    classes did — and each time the class landed and registered itself, the
    warning correctly stopped being emitted and the case had to move. Both are
    registered now, so there is no real unregistered type left to point at; the
    case is about severity following reachability, not about any one backend, so
    it keeps a type that will never be implemented rather than being deleted."""
    report = validate_config(_cfg(providers={
        "claude_cli": {"type": "claude_cli", "binary": "claude"},
        "hypothetical": {"type": "telepathy_api",
                         "api_key_env": "TELEPATHY_API_KEY"}}))
    assert report.errors == []
    assert any("providers.hypothetical.type" in w for w in report.warnings)


# ---- anthropic_api needs an output cap --------------------------------------

def _anthropic_cfg(preset: dict, *, wired: bool = True, pcfg: dict | None = None):
    """A config whose worker pool is bound to one anthropic_api preset."""
    return _cfg(
        presets={"claude_api": preset},
        providers={"claude_cli": {"type": "claude_cli", "binary": "claude"},
                   "anthropic": {"type": "anthropic_api", **(pcfg or {})}},
        worker_models={"w1": {"preset": "claude_api"}},
        roles={"smart_provider": "claude_cli",
               "worker": {"default": "w1",
                          "candidates": ["w1"] if wired else []}})


def test_an_active_anthropic_binding_without_max_tokens_is_an_error():
    """The Messages API has no server-side default, so the provider falls back
    to budget.assumed_max_output_tokens — but 8000 is a guess, and a worker
    truncated mid-block looks exactly like one that cannot follow the protocol."""
    report = validate_config(
        _anthropic_cfg({"provider": "anthropic", "model": "claude-opus-5"}))
    assert any("max_tokens" in e and "worker_models.w1" in e
               for e in report.errors)


def test_max_tokens_on_the_preset_satisfies_the_check():
    report = validate_config(_anthropic_cfg(
        {"provider": "anthropic", "model": "claude-opus-5", "max_tokens": 16000}))
    assert not any("max_tokens" in e for e in report.errors)
    assert not any("max_tokens" in w for w in report.warnings)


def test_a_tier_wide_provider_max_tokens_satisfies_the_check():
    """A role binding never passes `params`, so providers.<name>.max_tokens is
    the only per-tier place to set one — it has to count."""
    report = validate_config(_anthropic_cfg(
        {"provider": "anthropic", "model": "claude-opus-5"},
        pcfg={"max_tokens": 16000}))
    assert not any("max_tokens" in e for e in report.errors)


def test_an_unreached_anthropic_preset_without_max_tokens_is_only_a_warning():
    """Severity follows reachability here too: a menu entry nothing dispatches
    to must not block a run."""
    report = validate_config(_cfg(
        presets={"claude_api": {"provider": "anthropic", "model": "claude-opus-5"},
                 # kept because the clean config's reviewer names it
                 "claude_cli_opus": {"provider": "claude_cli", "model": "opus"}},
        providers={"claude_cli": {"type": "claude_cli", "binary": "claude"},
                   "anthropic": {"type": "anthropic_api"}}))
    assert report.errors == []
    assert any("presets.claude_api" in w and "max_tokens" in w
               for w in report.warnings)


# ---- the worker pool --------------------------------------------------------

def test_an_unset_worker_default_is_an_error():
    report = validate_config(_cfg(roles={
        "smart_provider": "claude_cli",
        "worker": {"default": None, "candidates": []}}))
    assert any("roles.worker.default is not set" in e for e in report.errors)


def test_a_worker_default_that_is_not_a_worker_models_key_is_an_error():
    report = validate_config(_cfg(roles={
        "smart_provider": "claude_cli",
        "worker": {"default": "w9", "candidates": ["w1"]}}))
    assert any("roles.worker.default" in e and "'w9'" in e for e in report.errors)


def test_a_candidate_that_is_not_a_worker_models_key_is_an_error():
    report = validate_config(_cfg(roles={
        "smart_provider": "claude_cli",
        "worker": {"default": "w1", "candidates": ["w1", "flash_hi"]}}))
    assert any("roles.worker.candidates" in e and "'flash_hi'" in e
               for e in report.errors)


# ---- API keys ---------------------------------------------------------------

def _cash_cfg() -> dict:
    return {
        "providers": {"claude_cli": {"type": "claude_cli"},
                      "comet": {"type": "openai_compatible",
                                "base_url": "http://x", "api_key_env": "TEST_KEY"}},
        "worker_models": {"w1": {"provider": "comet", "model": "m"}},
        "roles": {"smart_provider": "claude_cli",
                  "planner": {"provider": None, "model": "opus"},
                  "worker": {"default": "w1", "candidates": ["w1"]}},
    }


def test_a_missing_key_on_a_live_cash_provider_is_an_error(monkeypatch):
    """Today this is a log.warning and the literal string "missing-key", so it
    surfaces as a 401 on task 7 — after the earlier tasks were paid for."""
    monkeypatch.delenv("TEST_KEY", raising=False)
    report = validate_config(_cfg(**_cash_cfg()))
    assert any("providers.comet" in e and "TEST_KEY" in e for e in report.errors)


def test_a_present_key_is_accepted(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-something")
    assert validate_config(_cfg(**_cash_cfg())).errors == []


def test_a_missing_key_on_a_provider_nothing_routes_to_is_not_an_error(monkeypatch):
    monkeypatch.delenv("TEST_KEY", raising=False)
    data = _cash_cfg()
    data["worker_models"] = {"w1": {"provider": "claude_cli", "model": "sonnet"}}
    assert validate_config(_cfg(**data)).errors == []


def test_a_subscription_cli_needs_no_key():
    assert validate_config(_cfg()).errors == []


# ---- best-of-N collapse -----------------------------------------------------

def test_identical_candidates_warn_and_name_the_colliding_keys():
    """Three candidates that differ only by `temperature` are one call three
    times on a reasoning model, which rejects `temperature` outright."""
    report = validate_config(_cfg(
        worker_models={
            "luna_a": {"preset": "luna_high", "params": {"temperature": 0.2}},
            "luna_b": {"preset": "luna_high", "params": {"temperature": 0.5}},
            "luna_c": {"preset": "luna_high", "params": {"temperature": 0.8}},
        },
        presets={"luna_high": {"provider": "claude_cli", "model": "opus",
                               "effort": "high"}},
        roles={"smart_provider": "claude_cli",
               "worker": {"default": "luna_a",
                          "candidates": ["luna_a", "luna_b", "luna_c"]}},
        run={"degrade_model": "luna_a"}))
    collapse = [w for w in report.warnings if "resolve to the same backend" in w]
    assert len(collapse) == 1
    assert all(key in collapse[0] for key in ("luna_a", "luna_b", "luna_c"))
    assert "3 times" in collapse[0]
    assert report.errors == []          # a warning, never a block


def test_candidates_diversified_by_effort_do_not_warn():
    report = validate_config(_cfg(
        worker_models={"lo": {"preset": "p", "effort": "medium"},
                       "hi": {"preset": "p", "effort": "xhigh"}},
        presets={"p": {"provider": "claude_cli", "model": "opus"}},
        roles={"smart_provider": "claude_cli",
               "worker": {"default": "lo", "candidates": ["lo", "hi"]}},
        run={"degrade_model": "lo"}))
    assert report.warnings == []


# ---- reporting --------------------------------------------------------------

def test_every_problem_is_collected_in_one_pass():
    """Fixing config one exception at a time is the slow way to find out you had
    four things wrong."""
    report = validate_config(_cfg(
        worker_models={"w1": {"provider": "nope", "model": "m"}},
        roles={"smart_provider": "claude_cli",
               "planner": {"preset": "typo", "model": "opus"},
               "worker": {"default": "gone", "candidates": ["also-gone"]}}))
    assert len(report.errors) >= 4
    assert not report.ok
    assert report.format_errors().count("\n") == len(report.errors) - 1


# ---- checks gated on what the command actually does --------------------------
# A check that blocks a command it cannot possibly affect is an outage, not a
# guard. `--dry-run` opens no socket, and plan/discuss never reach the worker
# pool, so those two checks report rather than block for them.

def test_a_dry_run_is_not_blocked_by_a_missing_key(monkeypatch):
    """A dry run resolves the schedule and never calls out, so the 401 this
    check exists to pre-empt cannot happen — and a dry run is exactly how an
    operator inspects wiring BEFORE exporting a key."""
    monkeypatch.delenv("TEST_KEY", raising=False)
    report = validate_config(_cfg(**_cash_cfg()), will_spend=False)
    assert report.ok
    assert any("providers.comet" in w and "TEST_KEY" in w for w in report.warnings)


def test_a_spending_run_is_still_blocked_by_a_missing_key(monkeypatch):
    monkeypatch.delenv("TEST_KEY", raising=False)
    assert not validate_config(_cfg(**_cash_cfg()), will_spend=True).ok


def test_plan_is_not_blocked_by_an_unset_worker_default():
    """`plan` and `discuss` drive the smart tier and never dispatch a candidate,
    so an unset worker default is a problem for a future `run`, not this one."""
    cfg = _cfg(providers={"claude_cli": {"type": "claude_cli"}},
               roles={"smart_provider": "claude_cli",
                      "planner": {"provider": None, "model": "opus"},
                      "worker": {"default": None, "candidates": []}})
    report = validate_config(cfg, dispatches_workers=False)
    assert report.ok
    assert any("roles.worker.default" in w for w in report.warnings)


def test_run_is_still_blocked_by_an_unset_worker_default():
    cfg = _cfg(providers={"claude_cli": {"type": "claude_cli"}},
               roles={"smart_provider": "claude_cli",
                      "planner": {"provider": None, "model": "opus"},
                      "worker": {"default": None, "candidates": []}})
    assert not validate_config(cfg, dispatches_workers=True).ok


def test_candidates_diversified_by_approach_do_not_warn():
    """The shipped worker tier: one model, one effort, three `approach:` texts.

    This is the configuration the collapse warning must NOT fire on. `approach`
    is appended after the cache prefix, so these three share one cached prompt
    and still ask three different questions — cheaper than diverging on effort,
    and the reason `params` (which a reasoning model discards) is the only axis
    this check refuses to count.
    """
    report = validate_config(_cfg(
        worker_models={
            "luna_a": {"preset": "luna_high", "approach": "Simplest correct solution."},
            "luna_b": {"preset": "luna_high", "approach": "Refactor toward the seam."},
            "luna_c": {"preset": "luna_high", "approach": "Hunt the edge cases first."},
        },
        presets={"luna_high": {"provider": "claude_cli", "model": "opus",
                               "effort": "high"}},
        roles={"smart_provider": "claude_cli",
               "worker": {"default": "luna_a",
                          "candidates": ["luna_a", "luna_b", "luna_c"]}},
        run={"degrade_model": "luna_a"}))
    assert [w for w in report.warnings if "resolve to the same backend" in w] == []
    assert report.errors == []


def test_candidates_sharing_one_approach_still_warn():
    """Copy-pasting the same `approach:` onto every candidate is the same
    collapse as copy-pasting the same temperature, and must still be named."""
    same = "Simplest correct solution."
    report = validate_config(_cfg(
        worker_models={
            "luna_a": {"preset": "luna_high", "approach": same},
            "luna_b": {"preset": "luna_high", "approach": same},
        },
        presets={"luna_high": {"provider": "claude_cli", "model": "opus",
                               "effort": "high"}},
        roles={"smart_provider": "claude_cli",
               "worker": {"default": "luna_a", "candidates": ["luna_a", "luna_b"]}},
        run={"degrade_model": "luna_a"}))
    collapse = [w for w in report.warnings if "resolve to the same backend" in w]
    assert len(collapse) == 1
    assert "the same" in collapse[0]
