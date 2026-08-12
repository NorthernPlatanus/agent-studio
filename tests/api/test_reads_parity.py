"""`api/reads.py` must agree with `ops/store.py`, query for query.

The API cannot use `Store` (its constructor writes), so the aggregate SQL exists
twice. This test is what keeps the copy honest: it runs both against the same
fixture file and demands identical results. If someone fixes a query in one place
only, this fails instead of the dashboard quietly disagreeing with
`orchestrator status`.

Opening a real `Store` here is fine — the target is a tmpdir fixture, never the
live project's state.
"""

from __future__ import annotations

import pytest

from orchestrator.api import reads
from orchestrator.ops.store import Store
from tests.api.fixtures.seed_store import RUN_DONE, RUN_PAUSED


@pytest.fixture
def store(tmp_path):
    """A private copy, so the Store's schema/migration writes cannot perturb the
    session-scoped fixture the read-only tests assert byte-equality on."""
    from tests.api.fixtures import seed_store
    path = tmp_path / "example.sqlite3"
    seed_store.seed(path)
    s = Store(path)
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def ro_conn(store):
    from orchestrator.api.deps import open_read_only
    conn = open_read_only(store.path)
    try:
        yield conn
    finally:
        conn.close()


def test_all_tasks_parity(store, ro_conn):
    mine = reads.all_tasks(ro_conn)
    theirs = store.all_tasks()
    # reads.py additionally promotes `updated_at`, which the panel shows.
    for row in mine:
        row.pop("updated_at")
    assert mine == theirs


def test_get_task_parity(store, ro_conn):
    mine = reads.get_task(ro_conn, "T-102")
    mine.pop("updated_at")
    assert mine == store.get_task("T-102")
    assert reads.get_task(ro_conn, "T-nope") is None


def test_usage_summary_parity(store, ro_conn):
    assert reads.usage_summary(ro_conn) == store.usage_summary()


def test_gate_outcomes_parity(store, ro_conn):
    assert reads.gate_outcomes(ro_conn) == store.gate_outcomes()


def test_event_counts_parity(store, ro_conn):
    kinds = reads.METRIC_EVENT_KINDS
    assert reads.event_counts(ro_conn, kinds) == store.event_counts(kinds)


def test_subscription_tokens_by_role_parity(store, ro_conn):
    assert (reads.subscription_tokens_by_role(ro_conn)
            == store.subscription_tokens_by_role())


@pytest.mark.parametrize("run_id", [RUN_DONE, RUN_PAUSED])
def test_run_token_totals_parity(store, ro_conn, run_id):
    assert reads.run_token_totals(ro_conn, run_id) == store.run_token_totals(run_id)


def test_task_cash_spend_parity(store, ro_conn):
    for task_id in ("T-101", "T-102", "T-120", "T-130"):
        assert reads.task_cash_spend(ro_conn, task_id) == store.task_cash_spend(task_id)


def test_latest_run_parity(store, ro_conn):
    assert reads.latest_run(ro_conn) == store.latest_run()
    assert (reads.latest_run(ro_conn, ("paused",))
            == store.latest_run(statuses=("paused",)))


@pytest.mark.parametrize("run_id", [RUN_DONE, RUN_PAUSED, "no-such-run"])
def test_run_last_activity_parity(store, ro_conn, run_id):
    """The liveness signal is read on both sides — the API annotates `stale`,
    `reconcile` acts on it — so the two queries must not drift apart."""
    assert reads.run_last_activity(ro_conn, run_id) == store.run_last_activity(run_id)
