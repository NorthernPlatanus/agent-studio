"""Item 5: a run-level pause must actually stop the batch.

`asyncio.gather` propagates the first exception but leaves siblings running, so a
"paused" run could still merge work onto the feature branch afterwards.
"""

import asyncio

import pytest

from orchestrator.core.errors import BudgetExceeded, LimitExhausted
from orchestrator.engine.runner import _run_batch


async def test_sibling_is_cancelled_when_one_task_pauses():
    reached_integrate = []

    async def run_task(task):
        if task["id"] == "T-fast":
            await asyncio.sleep(0)
            raise LimitExhausted("weekly limit reached")
        await asyncio.sleep(0.2)              # still mid-work when the pause fires
        reached_integrate.append(task["id"])  # stand-in for integrate/merge

    with pytest.raises(LimitExhausted):
        await _run_batch([{"id": "T-fast"}, {"id": "T-slow"}], run_task)
    assert reached_integrate == []            # the sibling never merged


async def test_budget_pause_also_cancels_and_propagates():
    started = []

    async def run_task(task):
        started.append(task["id"])
        if task["id"] == "T-1":
            raise BudgetExceeded("run budget exceeded")
        await asyncio.sleep(0.2)
        raise AssertionError("sibling should have been cancelled")

    with pytest.raises(BudgetExceeded):
        await _run_batch([{"id": "T-1"}, {"id": "T-2"}, {"id": "T-3"}], run_task)
    assert set(started) == {"T-1", "T-2", "T-3"}   # all started, two cancelled


async def test_clean_batch_runs_every_task_to_completion():
    done = []

    async def run_task(task):
        await asyncio.sleep(0.01)
        done.append(task["id"])

    await _run_batch([{"id": "A"}, {"id": "B"}, {"id": "C"}], run_task)
    assert sorted(done) == ["A", "B", "C"]


async def test_cancellation_is_awaited_not_left_dangling():
    """Cancelled siblings are awaited before the pause propagates, so their
    cleanup runs before the caller writes the paused bookkeeping."""
    cleaned = []

    async def run_task(task):
        if task["id"] == "T-boom":
            raise LimitExhausted("limit")
        try:
            await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            cleaned.append(task["id"])
            raise

    with pytest.raises(LimitExhausted):
        await _run_batch([{"id": "T-boom"}, {"id": "T-other"}], run_task)
    assert cleaned == ["T-other"]
