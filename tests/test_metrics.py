"""O3: the solve-rate / quota telemetry queries.

Everything needed was already being logged; this makes it one command instead of
hand-written SQL. The measurement itself needs real runs — these tests only check
that the aggregation reads the events and usage rows correctly.
"""

from orchestrator.ops.store import Store


def _store(tmp_path):
    s = Store(tmp_path / "m.sqlite3")
    # attempt 1: deepseek green, glm red. attempt 2: glm green.
    s.log_event("r", "T-1", "gate", "deepseek attempt=1 passed=True")
    s.log_event("r", "T-1", "gate", "glm attempt=1 passed=False")
    s.log_event("r", "T-1", "gate", "glm attempt=2 passed=True")
    s.log_event("r", "T-2", "gate", "deepseek attempt=1 passed=False")
    return s


def test_gate_outcomes_split_first_attempt_from_retries(tmp_path):
    rows = {(r["cand_id"], r["first_attempt"]): r for r in _store(tmp_path).gate_outcomes()}
    assert rows[("deepseek", True)]["passed"] == 1
    assert rows[("deepseek", True)]["failed"] == 1      # 50% first-try solve rate
    assert rows[("glm", True)]["failed"] == 1
    assert rows[("glm", False)]["passed"] == 1          # only landed on a retry


def test_gate_outcomes_ignores_malformed_details(tmp_path):
    s = _store(tmp_path)
    s.log_event("r", "T-3", "gate", "garbage")
    s.log_event("r", "T-3", "gate", "")
    assert sum(r["passed"] + r["failed"] for r in s.gate_outcomes()) == 4


def test_event_counts(tmp_path):
    s = _store(tmp_path)
    s.log_event("r", "T-1", "escalated", "x")
    s.log_event("r", "T-2", "escalated", "y")
    s.log_event("r", "T-2", "auto_integrated", "c0")
    counts = s.event_counts(("escalated", "auto_integrated", "crashed"))
    assert counts == {"escalated": 2, "auto_integrated": 1}   # absent kinds omitted


# ---- the run summary must show the 99% it used to hide (item 7) -------------

def test_run_token_totals_split_the_two_channels(tmp_path):
    s = Store(tmp_path / "m.sqlite3")
    s.record_usage("r", "T-1", "planner", "claude_cli", "opus",
                   385_261, 10_315, 0.6495, False,
                   cache_hit_tokens=316_748, cache_miss_tokens=68_513)
    s.record_usage("r", "T-1", "worker:cheap", "comet", "ds",
                   12_816, 8_193, 0.0035, True, cache_miss_tokens=12_816)
    s.record_usage("other-run", "T-2", "planner", "claude_cli", "opus",
                   1_000, 10, 0.01, False)
    totals = Store(tmp_path / "m.sqlite3").run_token_totals("r")
    assert totals["subscription"]["in_tok"] == 385_261     # the other run excluded
    assert totals["subscription"]["calls"] == 1
    assert totals["cash"]["in_tok"] == 12_816
    assert round(totals["cash"]["cost"], 4) == 0.0035


def test_run_token_totals_is_empty_for_an_untouched_run(tmp_path):
    totals = Store(tmp_path / "m.sqlite3").run_token_totals("nope")
    assert totals == {"cash": None, "subscription": None}


def test_summary_lines_report_tokens_and_cache_rate(tmp_path):
    """`cash spend this run: $0.02` described 0.8% of what a measured run
    consumed; the subscription input total is the number that runs out."""
    from orchestrator.engine.runner import format_run_tokens

    s = Store(tmp_path / "m.sqlite3")
    s.record_usage("r", "T-1", "reviewer", "claude_cli", "opus",
                   100, 20, 0.5, False, cache_hit_tokens=76, cache_miss_tokens=24)
    lines = format_run_tokens(s.run_token_totals("r"))
    assert len(lines) == 1
    assert lines[0].startswith("subscription: 100 in (76% cached) / 20 out "
                               "across 1 calls")
    assert "notional $0.50" in lines[0]


def test_sub_cent_cash_is_not_rounded_away(tmp_path):
    """$0.0035 for a worker call is the number that makes the two-tier split worth
    having; `$0.00` would hide it."""
    from orchestrator.engine.runner import format_run_tokens

    s = Store(tmp_path / "m.sqlite3")
    s.record_usage("r", "T-1", "worker:cheap", "comet", "ds",
                   12_816, 8_193, 0.0035, True, cache_miss_tokens=12_816)
    assert "billed $0.0035" in format_run_tokens(s.run_token_totals("r"))[0]


def test_summary_never_reports_an_unknown_cache_as_a_cold_one(tmp_path):
    from orchestrator.engine.runner import format_run_tokens

    s = Store(tmp_path / "m.sqlite3")
    s.record_usage("r", "T-1", "worker:cheap", "comet", "ds", 500, 60, 0.01, True)
    line = format_run_tokens(s.run_token_totals("r"))[0]
    assert "cache unreported" in line and "0% cached" not in line


def test_subscription_tokens_exclude_cash_rows(tmp_path):
    s = Store(tmp_path / "m.sqlite3")
    s.record_usage("r", "T-1", "planner", "claude_cli", "opus", 1000, 50, 0.0, False,
                   cache_hit_tokens=800, cache_miss_tokens=200)
    s.record_usage("r", "T-1", "worker:cheap", "comet", "ds", 5000, 900, 0.02, True)
    rows = s.subscription_tokens_by_role()
    assert [r["role"] for r in rows] == ["planner"]     # the cash row is not quota
    assert rows[0]["in_tok"] == 1000 and rows[0]["cache_hit"] == 800
