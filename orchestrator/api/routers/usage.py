"""Usage rollups and the metrics view (`orchestrator metrics`, as JSON).

Derived rates are `null`, never 0, when their inputs are missing: a provider that
reports no cache telemetry must not read as a measured cold cache, and per-task
figures are meaningless before a task completes. Both rules are copied from the
CLI's reporting, which prints "-" for exactly these cases.
"""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, Query

from ...engine.scheduler import queue_stats
from .. import reads
from ..deps import store_conn
from ..errors import READ_ERRORS
from ..schemas import (GateOutcome, Metrics, RoleTokens, TokenChannels, Usage,
                       UsageGroupBy, UsageRow)

router = APIRouter(prefix="/api/projects/{project}", tags=["usage"],
                   responses=READ_ERRORS)


def _hit_rate(hit: int, miss: int) -> float | None:
    total = (hit or 0) + (miss or 0)
    return (hit or 0) / total if total else None


@router.get("/usage", response_model=Usage)
def usage(group_by: UsageGroupBy = Query("role"),
          conn: sqlite3.Connection = Depends(store_conn)) -> Usage:
    rows = [UsageRow(**row, cache_hit_rate=_hit_rate(row["cache_hit"],
                                                     row["cache_miss"]))
            for row in reads.usage_rollup(conn, group_by)]
    return Usage(group_by=group_by, rows=rows,
                 totals=TokenChannels(**reads.token_totals(conn)))


@router.get("/metrics", response_model=Metrics)
def metrics(conn: sqlite3.Connection = Depends(store_conn)) -> Metrics:
    tasks = reads.all_tasks(conn)
    done = sum(1 for t in tasks if t["status"] == "done")
    counts = reads.event_counts(conn)
    by_role = reads.subscription_tokens_by_role(conn)
    sub_in = sum(r["in_tok"] or 0 for r in by_role)
    return Metrics(
        completed_tasks=done,
        gate_outcomes=[
            GateOutcome(**row,
                        pass_rate=row["passed"] / (row["passed"] + row["failed"]))
            for row in reads.gate_outcomes(conn)
        ],
        event_counts={k: counts.get(k, 0) for k in reads.METRIC_EVENT_KINDS},
        subscription_tokens_by_role=[
            RoleTokens(role=r["role"], calls=r["calls"], in_tok=r["in_tok"] or 0,
                       out_tok=r["out_tok"] or 0, cache_hit=r["cache_hit"] or 0,
                       in_tok_per_completed_task=((r["in_tok"] or 0) / done
                                                  if done else None))
            for r in by_role
        ],
        subscription_in_tok_per_completed_task=sub_in / done if done else None,
        cash_usd_per_completed_task=(reads.cash_spend_total(conn) / done
                                     if done else None),
        queue_stats=queue_stats(tasks),
    )
