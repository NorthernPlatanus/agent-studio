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


def test_subscription_tokens_exclude_cash_rows(tmp_path):
    s = Store(tmp_path / "m.sqlite3")
    s.record_usage("r", "T-1", "planner", "claude_cli", "opus", 1000, 50, 0.0, False,
                   cache_hit_tokens=800, cache_miss_tokens=200)
    s.record_usage("r", "T-1", "worker:cheap", "comet", "ds", 5000, 900, 0.02, True)
    rows = s.subscription_tokens_by_role()
    assert [r["role"] for r in rows] == ["planner"]     # the cash row is not quota
    assert rows[0]["in_tok"] == 1000 and rows[0]["cache_hit"] == 800
