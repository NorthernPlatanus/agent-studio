"""Deterministic scheduler — pure functions, zero tokens.

Picks the next batch of runnable tasks:
  1. status == 'ready'
  2. all deps done
  3. pairwise-disjoint files_write within the batch (parallel-safe)
  4. batch width capped by run.max_parallel_tasks
Order: milestone, then task id (stable, matches backlog top-to-bottom flow).
"""

from __future__ import annotations

from typing import Any

Task = dict[str, Any]


def deps_satisfied(task: Task, by_id: dict[str, Task]) -> bool:
    for dep in task.get("deps") or []:
        dep_task = by_id.get(dep)
        if dep_task is None:
            # Unknown dep: treat as external/human-tracked -> don't block.
            continue
        if dep_task["status"] != "done":
            return False
    return True


def _sort_key(task: Task) -> tuple:
    return (task.get("milestone") or "", task["id"])


def next_batch(tasks: list[Task], max_parallel: int) -> list[Task]:
    by_id = {t["id"]: t for t in tasks}
    runnable = sorted(
        (t for t in tasks if t["status"] == "ready" and deps_satisfied(t, by_id)),
        key=_sort_key,
    )
    batch: list[Task] = []
    claimed: set[str] = set()
    for task in runnable:
        writes = set(task.get("files_write") or [])
        if not writes:
            # A task that declares no writable files can't be verified for
            # conflicts — run it alone (only as the first of a batch).
            if not batch:
                return [task]
            continue
        if writes & claimed:
            continue
        batch.append(task)
        claimed |= writes
        if len(batch) >= max_parallel:
            break
    return batch


def queue_stats(tasks: list[Task]) -> dict[str, int]:
    stats: dict[str, int] = {}
    for t in tasks:
        stats[t["status"]] = stats.get(t["status"], 0) + 1
    return stats


def domain_stats(tasks: list[Task]) -> dict[str, int]:
    """Task counts per domain (observability only — the scheduler invariant is
    unchanged; domains add specialization, not a new safety rule)."""
    stats: dict[str, int] = {}
    for t in tasks:
        d = t.get("domain") or "-"
        stats[d] = stats.get(d, 0) + 1
    return stats


def seam_tasks_missing_deps(tasks: list[Task]) -> list[str]:
    """`seam` tasks touch shared files and must serialize after the domain work
    they depend on — via planner-authored deps (no scheduler change). A seam task
    with no deps is almost certainly a planner mistake; surface it in dry-run."""
    return [t["id"] for t in tasks
            if t.get("domain") == "seam" and not (t.get("deps") or [])]
