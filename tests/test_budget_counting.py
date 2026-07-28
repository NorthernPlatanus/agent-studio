"""Phase 1: generalized CLI cost counting (_is_cash), with back-compat."""

import pytest
from pathlib import Path

from orchestrator.core.config import Config
from orchestrator.core.errors import BudgetExceeded
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


# ---- per-task cap is scoped to the current run (item 11) ---------------------

def test_task_cash_spend_scopes_to_run(tmp_path):
    from orchestrator.ops.store import Store
    store = Store(tmp_path / "s.sqlite3")
    store.record_usage("run-1", "T-1", "worker", "comet", "m", 1, 1, 0.90, True)
    store.record_usage("run-2", "T-1", "worker", "comet", "m", 1, 1, 0.30, True)
    assert store.task_cash_spend("T-1") == pytest.approx(1.20)          # lifetime
    assert store.task_cash_spend("T-1", run_id="run-2") == pytest.approx(0.30)
    store.close()


def test_per_task_cap_ignores_a_previous_runs_spend(tmp_path):
    """Re-running an already-expensive task must not pause instantly on spend the
    current run never made."""
    from orchestrator.ops.store import Store
    store = Store(tmp_path / "s.sqlite3")
    store.record_usage("run-1", "T-1", "worker", "comet", "m", 1, 1, 5.00, True)

    cfg = Config({"budget": {"per_task_usd": 1.0, "per_run_usd": 10.0},
                  "providers": {}}, "proj", Path("/tmp"))
    b = Budget(cfg, store, run_id="run-2")
    b.check("T-1")                       # nothing spent in run-2 yet -> fine

    store.record_usage("run-2", "T-1", "worker", "comet", "m", 1, 1, 1.50, True)
    with pytest.raises(BudgetExceeded, match="run-2"):
        b.check("T-1")
    store.close()


# ---- pre-flight estimate (item 10) ------------------------------------------

def _priced_budget(tmp_path, per_task=1.0, per_run=10.0):
    from orchestrator.ops.store import Store
    cfg = Config({
        "budget": {"per_task_usd": per_task, "per_run_usd": per_run,
                   "assumed_max_output_tokens": 1000},
        "providers": {"comet": {"type": "openai_compatible"},
                      "claude_cli": {"type": "claude_cli"}},
        "worker_models": {"w": {"provider": "comet", "model": "big",
                                "input_per_mtok": 100.0, "output_per_mtok": 200.0}},
    }, "proj", Path("/tmp"))
    return Budget(cfg, Store(tmp_path / "s.sqlite3"), run_id="r")


def test_estimate_blocks_a_call_that_would_breach_the_task_cap(tmp_path):
    b = _priced_budget(tmp_path, per_task=1.0)
    # 30k chars ~ 10k tokens at $100/Mtok = $1.00, plus 1k out at $200/Mtok = $0.20
    with pytest.raises(BudgetExceeded, match="would be exceeded"):
        b.estimate_and_check(task_id="T-1", provider="comet",
                             provider_type="openai_compatible", model="big",
                             prompt_chars=30000)


def test_estimate_allows_a_call_that_fits(tmp_path):
    b = _priced_budget(tmp_path, per_task=1.0)
    est = b.estimate_and_check(task_id="T-1", provider="comet",
                               provider_type="openai_compatible", model="big",
                               prompt_chars=300)
    assert 0 < est < 1.0


def test_estimate_accounts_for_what_the_run_already_spent(tmp_path):
    b = _priced_budget(tmp_path, per_task=10.0, per_run=1.0)
    b.store.record_usage("r", "T-1", "worker", "comet", "big", 1, 1, 0.95, True)
    with pytest.raises(BudgetExceeded, match="run budget"):
        b.estimate_and_check(task_id="T-1", provider="comet",
                             provider_type="openai_compatible", model="big",
                             prompt_chars=3000)


def test_subscription_calls_are_never_blocked(tmp_path):
    b = _priced_budget(tmp_path, per_task=0.0, per_run=0.0)
    assert b.estimate_and_check(task_id="T-1", provider="claude_cli",
                                provider_type="claude_cli", model="opus",
                                prompt_chars=10_000_000) == 0.0


def test_unpriced_model_does_not_block(tmp_path):
    # No price-table entry => nothing to estimate from; the post-hoc check in
    # record() remains the backstop rather than guessing a number here.
    b = _priced_budget(tmp_path, per_task=0.01)
    assert b.estimate_and_check(task_id="T-1", provider="comet",
                                provider_type="openai_compatible",
                                model="unknown-model", prompt_chars=999999) == 0.0


def test_explicit_max_tokens_beats_the_assumed_default(tmp_path):
    b = _priced_budget(tmp_path, per_task=10.0)
    small = b.estimate_and_check(task_id="T-1", provider="comet",
                                 provider_type="openai_compatible", model="big",
                                 prompt_chars=300, max_output_tokens=10)
    big = b.estimate_and_check(task_id="T-1", provider="comet",
                               provider_type="openai_compatible", model="big",
                               prompt_chars=300)
    assert small < big
