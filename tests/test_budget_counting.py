"""Phase 1: generalized CLI cost counting (_is_cash), with back-compat."""

from pathlib import Path

from orchestrator.core.config import Config
from orchestrator.ops.budget import Budget


def _budget(providers, count_claude_cli=False, count_cli=False):
    data = {
        "budget": {"per_task_usd": 1.0, "per_run_usd": 10.0,
                   "count_claude_cli": count_claude_cli, "count_cli": count_cli},
        "providers": providers,
    }
    cfg = Config(data, "proj", Path("/tmp"))
    return Budget(cfg, store=None, run_id="r")


def test_openai_compatible_always_cash():
    b = _budget({"cometapi": {"type": "openai_compatible"}})
    assert b._is_cash("cometapi", "openai_compatible") is True


def test_claude_cli_subscription_not_counted_by_default():
    b = _budget({"claude_cli": {"type": "claude_cli"}})
    assert b._is_cash("claude_cli", "claude_cli") is False


def test_claude_cli_counted_with_backcompat_toggle():
    b = _budget({"claude_cli": {"type": "claude_cli"}}, count_claude_cli=True)
    assert b._is_cash("claude_cli", "claude_cli") is True


def test_codex_subscription_not_counted():
    b = _budget({"codex_cli": {"type": "codex_cli", "auth": "subscription"}})
    assert b._is_cash("codex_cli", "codex_cli") is False


def test_codex_api_auth_counted():
    b = _budget({"codex_cli": {"type": "codex_cli", "auth": "api"}})
    assert b._is_cash("codex_cli", "codex_cli") is True


def test_count_cli_counts_all_cli_providers():
    b = _budget({"codex_cli": {"type": "codex_cli", "auth": "subscription"},
                 "claude_cli": {"type": "claude_cli"}}, count_cli=True)
    assert b._is_cash("codex_cli", "codex_cli") is True
    assert b._is_cash("claude_cli", "claude_cli") is True


def test_per_provider_count_override_wins():
    # explicit count:false beats auth:api
    b = _budget({"codex_cli": {"type": "codex_cli", "auth": "api", "count": False}})
    assert b._is_cash("codex_cli", "codex_cli") is False
