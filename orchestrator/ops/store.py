"""SQLite operational store: task queue, runs, usage/cost ledger, events.

The machine-facing task queue. The human-facing backlog markdown remains the
editorial source of truth; the planner imports it here (enriched with
files_read/files_write/deps) and the integrator writes statuses back
(see backlog.py).
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    milestone TEXT,
    title TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ready',
    -- ready | running | needs_human | done | failed | rejected | human_only
    retries INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    spec_json TEXT NOT NULL,          -- full planner spec (deps, files, acceptance...)
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',  -- running | paused | done | aborted
    note TEXT,
    cost_usd REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS usage (
    ts REAL NOT NULL,
    run_id TEXT,
    task_id TEXT,
    role TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    cash INTEGER NOT NULL DEFAULT 1,   -- 0 = subscription-covered (logged only)
    cache_hit_tokens INTEGER NOT NULL DEFAULT 0,
    cache_miss_tokens INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS events (
    ts REAL NOT NULL,
    run_id TEXT,
    task_id TEXT,
    kind TEXT NOT NULL,
    detail TEXT
);
CREATE TABLE IF NOT EXISTS discussions (
    session TEXT PRIMARY KEY,       -- usually the project name
    transcript TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""


class Store:
    def __init__(self, path: Path | str):
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Tolerant, additive migrations for pre-existing state files. CREATE
        TABLE IF NOT EXISTS won't add columns to an old `usage` table, so add
        them guarded by a column-exists check."""
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(usage)")}
        for col in ("cache_hit_tokens", "cache_miss_tokens"):
            if col not in cols:
                self._conn.execute(
                    f"ALTER TABLE usage ADD COLUMN {col} INTEGER NOT NULL DEFAULT 0")

    # ---- tasks ----------------------------------------------------------
    def upsert_task(self, spec: dict[str, Any]) -> None:
        """Insert or update a planner task spec. Preserves status/retries of
        already-finished tasks; re-planning an unfinished task resets it."""
        existing = self.get_task(spec["id"])
        status = spec.get("status")
        if status is None:
            if existing and existing["status"] in ("done", "failed", "rejected"):
                status = existing["status"]
            elif not spec.get("agent_able", True):
                status = "human_only"
            else:
                status = "ready"
        self._conn.execute(
            """INSERT INTO tasks(id, milestone, title, status, spec_json, updated_at)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 milestone=excluded.milestone, title=excluded.title,
                 status=excluded.status, spec_json=excluded.spec_json,
                 updated_at=excluded.updated_at""",
            (spec["id"], spec.get("milestone"), spec["title"], status,
             json.dumps(spec), time.time()),
        )
        self._conn.commit()

    def get_task(self, task_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return self._row_to_task(row) if row else None

    def all_tasks(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM tasks ORDER BY id").fetchall()
        return [self._row_to_task(r) for r in rows]

    def set_task_status(self, task_id: str, status: str, retries: int | None = None) -> None:
        if retries is None:
            self._conn.execute(
                "UPDATE tasks SET status=?, updated_at=? WHERE id=?",
                (status, time.time(), task_id))
        else:
            self._conn.execute(
                "UPDATE tasks SET status=?, retries=?, updated_at=? WHERE id=?",
                (status, retries, time.time(), task_id))
        self._conn.commit()

    def add_task_cost(self, task_id: str, usd: float) -> None:
        self._conn.execute(
            "UPDATE tasks SET cost_usd = cost_usd + ?, updated_at=? WHERE id=?",
            (usd, time.time(), task_id))
        self._conn.commit()

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> dict:
        task = json.loads(row["spec_json"])
        task.update(status=row["status"], retries=row["retries"],
                    cost_usd=row["cost_usd"], milestone=row["milestone"])
        return task

    # ---- runs -----------------------------------------------------------
    def create_run(self, note: str = "") -> str:
        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self._conn.execute(
            "INSERT INTO runs(id, started_at, status, note) VALUES(?,?,?,?)",
            (run_id, time.time(), "running", note))
        self._conn.commit()
        return run_id

    def set_run_status(self, run_id: str, status: str, note: str | None = None) -> None:
        if note is None:
            self._conn.execute("UPDATE runs SET status=? WHERE id=?", (status, run_id))
        else:
            self._conn.execute("UPDATE runs SET status=?, note=? WHERE id=?",
                               (status, note, run_id))
        self._conn.commit()

    def latest_run(self, statuses: tuple[str, ...] = ("paused", "running")) -> dict | None:
        q = ",".join("?" for _ in statuses)
        row = self._conn.execute(
            f"SELECT * FROM runs WHERE status IN ({q}) ORDER BY started_at DESC LIMIT 1",
            statuses).fetchone()
        return dict(row) if row else None

    # ---- usage / budget --------------------------------------------------
    def record_usage(self, run_id: str | None, task_id: str | None, role: str,
                     provider: str, model: str, input_tokens: int,
                     output_tokens: int, cost_usd: float, cash: bool,
                     cache_hit_tokens: int = 0, cache_miss_tokens: int = 0) -> None:
        self._conn.execute(
            """INSERT INTO usage(ts, run_id, task_id, role, provider, model,
                                 input_tokens, output_tokens, cost_usd, cash,
                                 cache_hit_tokens, cache_miss_tokens)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (time.time(), run_id, task_id, role, provider, model,
             input_tokens, output_tokens, cost_usd, 1 if cash else 0,
             cache_hit_tokens, cache_miss_tokens))
        if cash:
            if task_id:
                self.add_task_cost(task_id, cost_usd)
            if run_id:
                self._conn.execute(
                    "UPDATE runs SET cost_usd = cost_usd + ? WHERE id=?",
                    (cost_usd, run_id))
        self._conn.commit()

    def run_cash_spend(self, run_id: str) -> float:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(cost_usd),0) s FROM usage WHERE run_id=? AND cash=1",
            (run_id,)).fetchone()
        return row["s"]

    def task_cash_spend(self, task_id: str) -> float:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(cost_usd),0) s FROM usage WHERE task_id=? AND cash=1",
            (task_id,)).fetchone()
        return row["s"]

    def usage_summary(self) -> list[dict]:
        rows = self._conn.execute(
            """SELECT role, provider, model, COUNT(*) calls,
                      SUM(input_tokens) in_tok, SUM(output_tokens) out_tok,
                      SUM(cache_hit_tokens) cache_hit, SUM(cache_miss_tokens) cache_miss,
                      SUM(cost_usd) cost, MAX(cash) cash
               FROM usage GROUP BY role, provider, model ORDER BY cost DESC"""
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- discussions (interactive `discuss` transcript) ------------------
    def save_discussion(self, session: str, transcript: str) -> None:
        self._conn.execute(
            """INSERT INTO discussions(session, transcript, updated_at)
               VALUES(?,?,?)
               ON CONFLICT(session) DO UPDATE SET
                 transcript=excluded.transcript, updated_at=excluded.updated_at""",
            (session, transcript, time.time()))
        self._conn.commit()

    def load_discussion(self, session: str) -> str:
        row = self._conn.execute(
            "SELECT transcript FROM discussions WHERE session=?", (session,)).fetchone()
        return row["transcript"] if row else ""

    # ---- events ----------------------------------------------------------
    def log_event(self, run_id: str | None, task_id: str | None,
                  kind: str, detail: str = "") -> None:
        # Named columns (not positional VALUES) so a future events-schema change
        # can't silently misalign — the same hardening applied to `usage` (N12).
        self._conn.execute(
            "INSERT INTO events(ts, run_id, task_id, kind, detail) VALUES(?,?,?,?,?)",
            (time.time(), run_id, task_id, kind, detail))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
