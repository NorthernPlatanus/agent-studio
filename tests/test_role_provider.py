"""Phase 1: smart_provider switch + role_target provider precedence.

Precedence: explicit role.provider > roles.smart_provider > claude_cli fallback.
Workers are resolved independently and never move with smart_provider.
"""

from pathlib import Path

from orchestrator.core.config import Config
from orchestrator.core.context import RunContext


def _ctx(roles, worker_models=None, run=None, degraded=False):
    data = {
        "roles": roles,
        "worker_models": worker_models or {},
        "run": run or {"degrade_model": None},
    }
    cfg = Config(data, "proj", Path("/tmp"))
    return RunContext(cfg=cfg, store=None, git=None, budget=None,
                      run_id="r", degraded=degraded)


def test_explicit_provider_wins():
    ctx = _ctx({
        "smart_provider": "codex_cli",
        "planner": {"provider": "claude_cli", "model": "opus"},
        "reviewer": {"provider": "claude_cli", "model": "opus"},
    })
    assert ctx.role_target("planner") == ("claude_cli", "opus")
    assert ctx.role_target("reviewer") == ("claude_cli", "opus")


def test_null_provider_inherits_smart_provider():
    ctx = _ctx({
        "smart_provider": "codex_cli",
        "planner": {"provider": None, "model": "opus"},
        "reviewer": {"provider": None, "model": "opus"},
    })
    assert ctx.role_target("planner") == ("codex_cli", "opus")
    assert ctx.role_target("reviewer") == ("codex_cli", "opus")


def test_missing_provider_key_inherits():
    ctx = _ctx({
        "smart_provider": "codex_cli",
        "planner": {"model": "opus"},
    })
    assert ctx.role_target("planner") == ("codex_cli", "opus")


def test_smart_provider_flip_moves_both_roles():
    roles = {
        "smart_provider": "claude_cli",
        "planner": {"provider": None, "model": "opus"},
        "reviewer": {"provider": None, "model": "opus"},
    }
    ctx = _ctx(roles)
    assert ctx.role_target("planner")[0] == "claude_cli"
    assert ctx.role_target("reviewer")[0] == "claude_cli"

    roles["smart_provider"] = "codex_cli"
    ctx = _ctx(roles)
    assert ctx.role_target("planner")[0] == "codex_cli"
    assert ctx.role_target("reviewer")[0] == "codex_cli"


def test_fallback_to_claude_cli_when_smart_provider_absent():
    ctx = _ctx({"planner": {"provider": None, "model": "opus"}})
    assert ctx.role_target("planner") == ("claude_cli", "opus")


def test_workers_unaffected_by_smart_provider():
    wm = {"ds": {"provider": "cometapi", "model": "deepseek", }}
    ctx = _ctx(
        {"smart_provider": "codex_cli",
         "planner": {"provider": None, "model": "opus"}},
        worker_models=wm,
    )
    # worker resolution is independent of the smart tier
    assert ctx.worker_target("ds") == ("cometapi", "deepseek")


def test_degrade_reroutes_to_worker_even_with_codex_planner():
    wm = {"ds": {"provider": "cometapi", "model": "deepseek"}}
    ctx = _ctx(
        {"smart_provider": "codex_cli",
         "planner": {"provider": None, "model": "opus"}},
        worker_models=wm,
        run={"degrade_model": "ds"},
        degraded=True,
    )
    assert ctx.role_target("planner") == ("cometapi", "deepseek")


def test_provider_cache_is_per_config():
    # Regression (F4): the provider cache must isolate distinct Config objects.
    # A process-global name->instance cache would hand cfg2 the provider bound to
    # cfg1 (stale base_url / price table / env key).
    from orchestrator.providers import get_provider

    data = {"providers": {"p": {"type": "openai_compatible",
                                "base_url": "http://a", "api_key_env": "X"}},
            "worker_models": {}}
    cfg1 = Config(dict(data), "proj", Path("/tmp"))
    cfg2 = Config(dict(data), "proj", Path("/tmp"))
    p1, p1_again = get_provider(cfg1, "p"), get_provider(cfg1, "p")
    p2 = get_provider(cfg2, "p")
    assert p1 is p1_again          # cached within a single Config
    assert p1 is not p2            # a different Config gets its own instance
    assert p2.cfg is cfg2          # and it is bound to the right Config
