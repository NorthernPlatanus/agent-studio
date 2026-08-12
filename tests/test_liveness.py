"""Abandoned runs: detection, and the `reconcile` remedy.

The defect this covers, observed live: two runs in the user's store had said
`running` since 2026-07-28, eleven days after their processes died. Every
consumer that trusts the status column — `latest_run`, the panel's run pill, the
dashboard's "Active run" — reported a dead run as live, indefinitely.
"""

from __future__ import annotations

import time

import pytest

from orchestrator.ops import liveness
from orchestrator.ops.store import Store


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(tmp_path / "s.sqlite3")


def test_paused_is_never_stale():
    """A pause is a status somebody wrote on purpose; silence is its normal state."""
    assert liveness.is_stale("paused", time.time() - 10_000_000) is False
    assert liveness.is_stale("done", None) is False


def test_running_with_no_footprint_at_all_is_stale():
    assert liveness.is_stale("running", None) is True


def test_running_and_recently_active_is_live():
    # A slow planner call is minutes, not the 15-minute window: a run that is
    # merely thinking must never be reconciled out from under itself.
    assert liveness.is_stale("running", time.time() - 8 * 60) is False


def test_abandoned_runs_measures_footprint_not_start_time(store: Store):
    """A long run that is still working must not be reconciled.

    Both runs here *started* long ago, so `started_at` alone would condemn both.
    What separates them is that one of them is still writing rows.
    """
    long_dead = store.create_run(note="zombie")
    long_but_busy = store.create_run(note="working")
    ancient = time.time() - 86_400
    store._conn.execute("UPDATE runs SET started_at=?", (ancient,))
    store._conn.commit()
    store.log_event(long_but_busy, "T-1", "gate", "passed=True")

    stale = {r["id"] for r in store.abandoned_runs()}
    assert stale == {long_dead}


def test_reconcile_writes_a_terminal_status_and_explains_itself(store: Store):
    run_id = store.create_run()
    closed = store.abort_abandoned_runs(after_s=0)

    assert [r["id"] for r in closed] == [run_id]
    row = store.latest_run(statuses=("aborted",))
    assert row["id"] == run_id
    # `aborted`, not `failed`: nothing is known about how the work was going.
    assert row["status"] == "aborted"
    assert "reconciled" in row["note"]
    # And the run is gone from the query every "is something running" check uses.
    assert store.latest_run() is None


def test_reconcile_leaves_a_live_run_alone(store: Store):
    live = store.create_run()
    store.log_event(live, None, "planned", "")
    assert store.abort_abandoned_runs() == []
    assert store.latest_run()["id"] == live


def test_cli_reconcile_dry_run_changes_nothing(store: Store, capsys, monkeypatch):
    from orchestrator import cli

    run_id = store.create_run()
    store_path = store.path

    class _Cfg:
        def store_path(self):
            return store_path

    monkeypatch.setattr(cli, "load_config", lambda *_a, **_k: _Cfg())
    monkeypatch.setattr(cli, "Store", lambda _p: Store(store_path))

    assert cli.main(["reconcile", "--dry-run", "--after-minutes", "0"]) == 0
    assert "would be closed" in capsys.readouterr().out
    assert Store(store_path).latest_run()["id"] == run_id      # still running

    assert cli.main(["reconcile", "--after-minutes", "0"]) == 0
    assert Store(store_path).latest_run() is None
