"""`preset:` resolution on workers and roles, via RunContext.

The contract in one line: the preset fills in, the entry wins, and an entry
with no `preset:` resolves to exactly what it always did. That last clause is
the reason every pre-existing config test in this suite still passes unchanged,
so it is asserted here rather than left implied.
"""

from pathlib import Path

from orchestrator.core.config import Config
from orchestrator.core.context import RunContext

PRESETS = {
    "luna_high": {"label": "GPT-5.6 Luna · high", "provider": "openai",
                  "model": "gpt-5.6-luna", "effort": "high",
                  "input_per_mtok": 0.20, "output_per_mtok": 1.20},
    "claude_cli_sonnet": {"label": "Claude Code CLI · sonnet",
                          "provider": "claude_cli", "model": "sonnet",
                          "effort": "medium"},
}


def _ctx(*, worker_models=None, roles=None, run=None, degraded=False) -> RunContext:
    data = {
        "presets": PRESETS,
        "providers": {"claude_cli": {"type": "claude_cli"},
                      "openai": {"type": "openai_responses"},
                      "comet": {"type": "openai_compatible"}},
        "worker_models": worker_models or {},
        "roles": roles or {"smart_provider": "claude_cli"},
        "run": run or {"degrade_model": None},
    }
    return RunContext(cfg=Config(data, "proj", Path("/tmp")), store=None,
                      git=None, budget=None, run_id="r", degraded=degraded)


# ---- workers ----------------------------------------------------------------

def test_a_worker_resolves_provider_model_and_effort_from_its_preset():
    ctx = _ctx(worker_models={"w": {"preset": "luna_high"}})
    assert ctx.worker_target("w") == ("openai", "gpt-5.6-luna")
    assert ctx.worker_effort("w") == "high"


def test_an_explicit_key_on_the_worker_beats_the_preset():
    ctx = _ctx(worker_models={"w": {"preset": "luna_high", "effort": "low",
                                    "model": "gpt-5.6-terra"}})
    assert ctx.worker_target("w") == ("openai", "gpt-5.6-terra")
    assert ctx.worker_effort("w") == "low"


def test_a_worker_without_a_preset_behaves_exactly_as_before():
    ctx = _ctx(worker_models={"w": {"provider": "comet", "model": "deepseek-v4-flash"}})
    assert ctx.worker_target("w") == ("comet", "deepseek-v4-flash")
    assert ctx.worker_effort("w") is None


def test_a_cli_preset_makes_a_worker_a_subscription_worker():
    """The whole point of the layer: moving a worker from a metered API to a
    CLI riding a subscription is one line and no code."""
    ctx = _ctx(worker_models={"w": {"preset": "claude_cli_sonnet"}})
    assert ctx.worker_target("w") == ("claude_cli", "sonnet")
    assert ctx.worker_effort("w") == "medium"


def test_per_worker_effort_is_no_longer_discarded():
    """Regression: worker_effort() hard-returned None for every candidate but
    `senior`, so `effort: high` on a worker was silently dropped — the exact
    'I set high reasoning and nothing happened' failure."""
    ctx = _ctx(worker_models={"w": {"provider": "openai", "model": "m",
                                    "effort": "xhigh"}})
    assert ctx.worker_effort("w") == "xhigh"


def test_a_worker_does_not_inherit_the_tier_wide_provider_effort():
    """providers.<name>.effort is the SMART tier's default. Leaking it onto the
    worker pool would re-price every cheap call the first time an operator set
    a tier default for the planner."""
    ctx = _ctx(worker_models={"w": {"provider": "claude_cli", "model": "sonnet"}})
    ctx.cfg._data["providers"]["claude_cli"]["effort"] = "max"
    assert ctx.worker_effort("w") is None


def test_senior_still_resolves_to_smart_provider_and_escalate_model():
    """`senior` is a pseudo-candidate, NOT a worker_models key: it is the
    subscription smart tier acting as an implementer."""
    ctx = _ctx(roles={"smart_provider": "codex_cli"},
               run={"escalate_model": "gpt-5.6-terra", "escalate_effort": "high",
                    "degrade_model": None})
    assert ctx.worker_target("senior") == ("codex_cli", "gpt-5.6-terra")
    assert ctx.worker_effort("senior") == "high"


def test_senior_is_unaffected_by_a_worker_models_key_of_the_same_name():
    ctx = _ctx(worker_models={"senior": {"preset": "luna_high"}},
               roles={"smart_provider": "claude_cli"},
               run={"escalate_model": "opus", "degrade_model": None})
    assert ctx.worker_target("senior") == ("claude_cli", "opus")


# ---- roles ------------------------------------------------------------------

def test_a_role_resolves_provider_model_and_effort_from_its_preset():
    ctx = _ctx(roles={"smart_provider": "claude_cli",
                      "planner": {"preset": "luna_high"}})
    assert ctx.role_target("planner") == ("openai", "gpt-5.6-luna")
    assert ctx.role_effort("planner") == "high"


def test_a_null_key_on_a_role_does_not_blank_the_preset():
    """Every shipped role entry is written `{provider: null, effort: null}`, so
    null has to mean 'unset', not 'override the preset with nothing'."""
    ctx = _ctx(roles={"smart_provider": "claude_cli",
                      "planner": {"preset": "luna_high", "provider": None,
                                  "effort": None}})
    assert ctx.role_target("planner") == ("openai", "gpt-5.6-luna")
    assert ctx.role_effort("planner") == "high"


def test_an_explicit_key_on_the_role_beats_the_preset():
    ctx = _ctx(roles={"smart_provider": "claude_cli",
                      "planner": {"preset": "luna_high", "effort": "max"}})
    assert ctx.role_effort("planner") == "max"


def test_a_role_without_a_preset_behaves_exactly_as_before():
    ctx = _ctx(roles={"smart_provider": "codex_cli",
                      "planner": {"provider": None, "model": "opus",
                                  "effort": None}})
    assert ctx.role_target("planner") == ("codex_cli", "opus")
    assert ctx.role_effort("planner") is None


def test_degrade_still_targets_a_worker_models_key_and_resolves_its_preset():
    ctx = _ctx(worker_models={"cheap": {"preset": "luna_high"}},
               roles={"smart_provider": "codex_cli",
                      "planner": {"provider": None, "model": "opus"}},
               run={"degrade_model": "cheap"}, degraded=True)
    assert ctx.role_target("planner") == ("openai", "gpt-5.6-luna")
    assert ctx.role_effort("planner") is None      # degrade drops effort


# ---- an unresolvable preset -------------------------------------------------

def test_an_unknown_preset_warns_and_leaves_the_entry_as_written(caplog):
    """Dispatch must never crash on a config typo three tasks into a run —
    core.validate is where that typo becomes a startup error instead."""
    ctx = _ctx(worker_models={"w": {"preset": "lunahigh", "provider": "comet",
                                    "model": "m"}})
    with caplog.at_level("WARNING"):
        assert ctx.worker_target("w") == ("comet", "m")
    assert "lunahigh" in caplog.text
