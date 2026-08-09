"""Read queries against a read-only `state/<project>.sqlite3` connection.

Why this file duplicates SQL that `ops/store.py` already has: `Store.__init__`
writes (schema + migrations), so a GET may not construct one. These functions
take a bare connection instead. The duplication is deliberate and pinned —
`tests/api/test_reads_parity.py` asserts every aggregate here returns exactly
what the matching `Store` method returns for the same fixture file, so a change
to one that isn't mirrored in the other fails the suite rather than drifting.

Everything returns plain dicts/lists; the routers own the Pydantic projection.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

# Kinds `orchestrator metrics` counts. Kept in one place so /metrics and the CLI
# report the same set.
METRIC_EVENT_KINDS: tuple[str, ...] = (
    "escalated", "auto_integrated", "crashed", "retrieval_exhausted",
    "visual_gate_error", "visual_gate_skipped", "no_patch",
    "verify_unverifiable",
)

_CHANNEL_FIELDS = ("calls", "in_tok", "out_tok", "cache_hit", "cache_miss", "cost")


# ---- tasks ---------------------------------------------------------------
def _row_to_task(row: sqlite3.Row) -> dict[str, Any]:
    """Same projection as `Store._row_to_task`: the spec blob is the base, the
    live columns win."""
    task = json.loads(row["spec_json"])
    task.update(status=row["status"], retries=row["retries"],
                cost_usd=row["cost_usd"], milestone=row["milestone"],
                updated_at=row["updated_at"])
    return task


def all_tasks(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM tasks ORDER BY id").fetchall()
    return [_row_to_task(r) for r in rows]


def get_task(conn: sqlite3.Connection, task_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    return _row_to_task(row) if row else None


def task_cash_spend(conn: sqlite3.Connection, task_id: str) -> float:
    row = conn.execute(
        "SELECT COALESCE(SUM(cost_usd),0) s FROM usage WHERE task_id=? AND cash=1",
        (task_id,)).fetchone()
    return row["s"]


def child_task_ids(conn: sqlite3.Connection, parent_id: str) -> list[str]:
    """Sub-tasks of a decomposed parent. `parent_id` lives inside the spec blob,
    so this is a scan — the tasks table is small (hundreds of rows at most)."""
    return [t["id"] for t in all_tasks(conn) if t.get("parent_id") == parent_id]


# ---- runs ----------------------------------------------------------------
def runs(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_run(conn: sqlite3.Connection, run_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    return dict(row) if row else None


def latest_run(conn: sqlite3.Connection,
               statuses: tuple[str, ...] = ("paused", "running")) -> dict | None:
    q = ",".join("?" for _ in statuses)
    row = conn.execute(
        f"SELECT * FROM runs WHERE status IN ({q}) ORDER BY started_at DESC LIMIT 1",
        statuses).fetchone()
    return dict(row) if row else None


def run_task_ids(conn: sqlite3.Connection, run_id: str) -> list[str]:
    """Tasks the run touched — from events and usage, since `runs` has no join
    table. A task can appear in usage without an event (planning) and vice
    versa (a crash before the first call), so it is the union."""
    rows = conn.execute(
        """SELECT DISTINCT task_id FROM events WHERE run_id=? AND task_id IS NOT NULL
           UNION
           SELECT DISTINCT task_id FROM usage WHERE run_id=? AND task_id IS NOT NULL""",
        (run_id, run_id)).fetchall()
    return sorted(r["task_id"] for r in rows)


# ---- usage ---------------------------------------------------------------
def _channels(rows: list[sqlite3.Row]) -> dict[str, dict | None]:
    out: dict[str, dict | None] = {"cash": None, "subscription": None}
    for r in rows:
        out["cash" if r["cash"] else "subscription"] = {k: r[k] for k in _CHANNEL_FIELDS}
    return out


_CHANNEL_SELECT = """SELECT cash, COUNT(*) calls,
          COALESCE(SUM(input_tokens),0) in_tok,
          COALESCE(SUM(output_tokens),0) out_tok,
          COALESCE(SUM(cache_hit_tokens),0) cache_hit,
          COALESCE(SUM(cache_miss_tokens),0) cache_miss,
          COALESCE(SUM(cost_usd),0) cost
   FROM usage"""


def run_token_totals(conn: sqlite3.Connection, run_id: str) -> dict[str, dict | None]:
    rows = conn.execute(f"{_CHANNEL_SELECT} WHERE run_id=? GROUP BY cash",
                        (run_id,)).fetchall()
    return _channels(rows)


def token_totals(conn: sqlite3.Connection) -> dict[str, dict | None]:
    """The same split across every run — the dashboard's headline figure."""
    return _channels(conn.execute(f"{_CHANNEL_SELECT} GROUP BY cash").fetchall())


def usage_summary(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """SELECT role, provider, model, COUNT(*) calls,
                  SUM(input_tokens) in_tok, SUM(output_tokens) out_tok,
                  SUM(cache_hit_tokens) cache_hit, SUM(cache_miss_tokens) cache_miss,
                  SUM(cost_usd) cost, MAX(cash) cash
           FROM usage GROUP BY role, provider, model ORDER BY cost DESC""").fetchall()
    return [dict(r) for r in rows]


# `day` buckets by local date so the chart matches the user's sense of "today";
# ts is a unix float, hence the modifier.
_GROUP_DIMS: dict[str, tuple[str, ...]] = {
    "role": ("role",),
    "model": ("provider", "model"),
    "provider": ("provider",),
    "day": ("day",),
}
_GROUP_EXPR = {"day": "date(ts, 'unixepoch', 'localtime')"}


def usage_rollup(conn: sqlite3.Connection, group_by: str) -> list[dict]:
    """Usage grouped by one dimension, ALWAYS also split by `cash`.

    Mixing the two billing channels in one row would make the cost column
    meaningless (subscription cost is notional quota, cash is money) and the
    cache-hit rate unattributable. So every rollup row belongs to exactly one
    channel and the UI can stack or filter.
    """
    dims = _GROUP_DIMS[group_by]
    select = ", ".join(f"{_GROUP_EXPR.get(d, d)} AS {d}" for d in dims)
    group = ", ".join(_GROUP_EXPR.get(d, d) for d in dims)
    rows = conn.execute(
        f"""SELECT {select}, cash, COUNT(*) calls,
                   COALESCE(SUM(input_tokens),0) in_tok,
                   COALESCE(SUM(output_tokens),0) out_tok,
                   COALESCE(SUM(cache_hit_tokens),0) cache_hit,
                   COALESCE(SUM(cache_miss_tokens),0) cache_miss,
                   COALESCE(SUM(cost_usd),0) cost
            FROM usage GROUP BY {group}, cash ORDER BY {group}, cash""").fetchall()
    out = []
    for r in rows:
        row = {d: r[d] for d in dims}
        row.update(key=" / ".join(str(r[d]) for d in dims),
                   cash=bool(r["cash"]),
                   **{k: r[k] for k in _CHANNEL_FIELDS})
        out.append(row)
    return out


def subscription_tokens_by_role(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """SELECT role, COUNT(*) calls, SUM(input_tokens) in_tok,
                  SUM(output_tokens) out_tok, SUM(cache_hit_tokens) cache_hit
           FROM usage WHERE cash=0 GROUP BY role ORDER BY in_tok DESC""").fetchall()
    return [dict(r) for r in rows]


def cash_spend_total(conn: sqlite3.Connection) -> float:
    row = conn.execute(
        "SELECT COALESCE(SUM(cost_usd),0) s FROM usage WHERE cash=1").fetchone()
    return row["s"]


# ---- events --------------------------------------------------------------
def events(conn: sqlite3.Connection, *, since_rowid: int | None = None,
           kind: str | None = None, task_id: str | None = None,
           run_id: str | None = None, limit: int = 200,
           order: str = "asc") -> list[dict]:
    """A page of the event log — `asc` = oldest-first, `desc` = newest-first.

    The cursor is sqlite's `rowid`: `events` has no id column, but rowid is
    monotonic for an append-only table and survives an API restart, which an
    in-process counter would not.

    `order` exists because the two consumers want opposite ends of the log: a
    poller pages forward from a cursor (`asc`), while the dashboard's "recent
    events" panel wants the newest N. Doing the latter client-side means asking
    for `since_rowid = max_rowid - N`, which is wrong under `kind=`/`task_id=`
    filters — it counts rows the filter excludes and silently returns a short
    page. `LIMIT` after `ORDER BY rowid DESC` applies to the *matching* rows, so
    the descending path is exact for any filter combination.
    """
    if order not in ("asc", "desc"):
        raise ValueError(f"order must be 'asc' or 'desc', got {order!r}")
    where, params = [], []
    if since_rowid is not None:
        where.append("rowid > ?")
        params.append(since_rowid)
    for col, val in (("kind", kind), ("task_id", task_id), ("run_id", run_id)):
        if val is not None:
            where.append(f"{col}=?")
            params.append(val)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    params.append(limit)
    direction = "DESC" if order == "desc" else "ASC"
    rows = conn.execute(
        f"SELECT rowid, * FROM events {clause} ORDER BY rowid {direction} LIMIT ?",
        params).fetchall()
    return [dict(r) for r in rows]


def task_events(conn: sqlite3.Connection, task_id: str, limit: int = 500) -> list[dict]:
    return events(conn, task_id=task_id, limit=limit)


def event_counts(conn: sqlite3.Connection,
                 kinds: tuple[str, ...] = METRIC_EVENT_KINDS) -> dict[str, int]:
    q = ",".join("?" for _ in kinds)
    rows = conn.execute(
        f"SELECT kind, COUNT(*) n FROM events WHERE kind IN ({q}) GROUP BY kind",
        kinds).fetchall()
    return {r["kind"]: r["n"] for r in rows}


def max_event_rowid(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(rowid),0) m FROM events").fetchone()
    return row["m"]


def gate_outcomes(conn: sqlite3.Connection) -> list[dict]:
    """Mirror of `Store.gate_outcomes` — see this module's docstring for why the
    query lives twice; `test_reads_parity` keeps the two honest."""
    rows = conn.execute("SELECT detail FROM events WHERE kind='gate'").fetchall()
    agg: dict[tuple[str, bool], dict] = {}
    for r in rows:
        parts = (r["detail"] or "").split()
        if len(parts) < 3:
            continue
        cand, attempt, passed = parts[0], parts[1], parts[2]
        first = attempt == "attempt=1"
        a = agg.setdefault((cand, first), {"cand_id": cand, "first_attempt": first,
                                           "passed": 0, "failed": 0})
        a["passed" if passed == "passed=True" else "failed"] += 1
    return sorted(agg.values(), key=lambda a: (a["cand_id"], not a["first_attempt"]))


# ---- candidates (LangGraph checkpoint, else the event log) ----------------
# Per-candidate detail is graph state, not store state: it lives in
# `state/<project>.checkpoints.sqlite3` under thread_id "{run_id}:{task_id}".
# We read that file with the SAME read-only sqlite URI and the library's own
# serializer rather than instantiating a saver — `SqliteSaver`/`AsyncSqliteSaver`
# run `executescript(CREATE TABLE …)` on setup, which is a write, and would also
# create the file for a project that never ran.

def checkpoint_candidates(path, run_id: str, task_id: str) -> list[dict] | None:
    """`TaskState.candidates` at the newest checkpoint for a task, or None.

    None means "no checkpoint for this thread" (never ran, or checkpoints were
    pruned) — distinct from `[]`, which means the task's graph has run but no
    candidate was recorded yet. Pending writes on the newest checkpoint are
    folded in through the real `add_candidates` reducer, so a task caught
    mid-attempt shows the attempt that is running, not the previous one.
    """
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    from ..core.state import add_candidates
    from .deps import open_read_only

    if path is None or not path.exists():
        return None
    thread_id = f"{run_id}:{task_id}"
    conn = open_read_only(path)
    try:
        try:
            row = conn.execute(
                """SELECT checkpoint_id, type, checkpoint FROM checkpoints
                   WHERE thread_id=? AND checkpoint_ns=''
                   ORDER BY checkpoint_id DESC LIMIT 1""", (thread_id,)).fetchone()
        except sqlite3.DatabaseError:
            # A checkpoint file from a different library version (or a partially
            # written one) must degrade to the event-log fallback, not 500.
            return None
        if row is None:
            return None
        serde = JsonPlusSerializer()
        try:
            ckpt = serde.loads_typed((row["type"], row["checkpoint"]))
        except Exception:
            return None
        values = (ckpt or {}).get("channel_values") or {}
        cands = values.get("candidates")
        cands = list(cands) if isinstance(cands, list) else []
        pending: list[dict] = []
        wrote = conn.execute(
            """SELECT type, value FROM writes
               WHERE thread_id=? AND checkpoint_ns='' AND checkpoint_id=?
                 AND channel='candidates' ORDER BY task_id, idx""",
            (thread_id, row["checkpoint_id"])).fetchall()
        for w in wrote:
            try:
                val = serde.loads_typed((w["type"], w["value"]))
            except Exception:
                continue
            if isinstance(val, list):
                pending.extend(c for c in val if isinstance(c, dict))
            elif isinstance(val, dict):
                pending.append(val)
        return add_candidates(cands, pending) if pending else cands
    finally:
        conn.close()


def candidates_from_events(conn: sqlite3.Connection, task_id: str) -> list[dict]:
    """History reconstructed from `gate` / `candidate_failed` rows.

    The formats are the ones `nodes/worker.py` writes:
      gate             `<cand> attempt=<n> passed=<bool>[ failed_step=… \\n<tail>]`
      candidate_failed `<cand> attempt=<n> <status>: <reason>`
    One entry per (cand_id, attempt) — the latest row for a pair wins, matching
    the reducer's supersession rule.
    """
    rows = events(conn, task_id=task_id, limit=2000)
    by_key: dict[tuple[str, int], dict] = {}
    for row in rows:
        if row["kind"] not in ("gate", "candidate_failed", "no_patch"):
            continue
        detail = row["detail"] or ""
        parts = detail.split(maxsplit=2)
        if len(parts) < 2 or not parts[1].startswith("attempt="):
            continue
        cand_id = parts[0]
        try:
            attempt = int(parts[1].split("=", 1)[1])
        except ValueError:
            continue
        rest = parts[2] if len(parts) > 2 else ""
        cand = by_key.setdefault((cand_id, attempt),
                                 {"cand_id": cand_id, "attempt": attempt,
                                  "status": None, "gate_log": None, "error": None,
                                  "no_patch": False})
        if row["kind"] == "gate":
            passed = rest.startswith("passed=True")
            cand["status"] = "gate_passed" if passed else "gate_failed"
            if not passed:
                cand["gate_log"] = rest
        elif row["kind"] == "candidate_failed":
            status, _, reason = rest.partition(":")
            cand["status"] = status.strip() or "failed"
            cand["error"] = reason.strip() or None
        else:                                    # no_patch
            cand["no_patch"] = True
    return [by_key[k] for k in sorted(by_key)]
