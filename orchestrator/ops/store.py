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

from . import liveness

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
    frames TEXT,                    -- JSON frame log, for the panel (see save_discussion_log)
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS planner_handoff (
    session TEXT PRIMARY KEY,       -- usually the project name
    digest TEXT NOT NULL,           -- short TLDR inlined into a cold-start prompt
    snapshot_path TEXT,             -- full proposal on disk, read only on demand
    updated_at REAL NOT NULL
);
"""


class Store:
    def __init__(self, path: Path | str):
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        # WAL + a busy timeout, because this file has never had exactly one
        # writer: a CLI run, a `reconcile`, and the panel's discuss session are
        # separate processes that can overlap. In the default rollback journal a
        # reader blocks a writer outright; in WAL they don't block each other, and
        # the timeout turns the remaining writer-writer overlap into a wait rather
        # than an immediate `database is locked`. Both pragmas are no-ops when
        # already set — journal_mode is a persistent property of the file.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
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
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(discussions)")}
        if "frames" not in cols:
            # Nullable with no default: an old row has no frame log, and "" would
            # be indistinguishable from a conversation that genuinely had none.
            self._conn.execute("ALTER TABLE discussions ADD COLUMN frames TEXT")

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

    def run_last_activity(self, run_id: str) -> float | None:
        """Newest timestamp the run left anywhere, or None if it left nothing.

        `runs` records only `started_at`, so a run's own footprint is the union of
        the two tables it writes as it works. See `ops/liveness.py` for why this,
        and not a pid, is what liveness is measured from.
        """
        row = self._conn.execute(
            """SELECT MAX(ts) t FROM (
                   SELECT MAX(ts) ts FROM events WHERE run_id=?
                   UNION ALL
                   SELECT MAX(ts) ts FROM usage  WHERE run_id=?)""",
            (run_id, run_id)).fetchone()
        return row["t"]

    def abandoned_runs(self, *, after_s: float = liveness.STALE_AFTER_S) -> list[dict]:
        """`running` rows whose process is long gone (`ops/liveness.py`)."""
        rows = self._conn.execute(
            f"SELECT * FROM runs WHERE status IN ({','.join('?' for _ in liveness.LIVE_STATUSES)})"
            " ORDER BY started_at",
            liveness.LIVE_STATUSES).fetchall()
        now = time.time()
        out = []
        for row in rows:
            last = self.run_last_activity(row["id"]) or row["started_at"]
            if liveness.is_stale(row["status"], last, now=now, after_s=after_s):
                run = dict(row)
                run["last_activity_at"] = last
                out.append(run)
        return out

    def abort_abandoned_runs(self, *, after_s: float = liveness.STALE_AFTER_S) -> list[dict]:
        """Give every abandoned run the terminal status its process never wrote.

        `aborted`, not `failed`: nothing is known about whether the work was going
        well when the process died, only that it stopped. The note records how
        long the silence was so the row still explains itself later.
        """
        abandoned = self.abandoned_runs(after_s=after_s)
        now = time.time()
        for run in abandoned:
            silent_h = (now - run["last_activity_at"]) / 3600
            self.set_run_status(
                run["id"], "aborted",
                note=f"reconciled: no store activity for {silent_h:.1f}h "
                     f"— the process is gone")
        return abandoned

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

    def task_cash_spend(self, task_id: str, run_id: str | None = None) -> float:
        """Cash spent on a task. `run_id` scopes it to one run — which is what the
        BUDGET path wants: an unscoped sum makes a re-planned or re-run task start
        already over its per-task cap and pause immediately, blaming the current
        run for a previous one's spend. Reporting (`status`, backlog writeback)
        wants the unscoped lifetime figure, so that stays the default."""
        if run_id is None:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(cost_usd),0) s FROM usage "
                "WHERE task_id=? AND cash=1", (task_id,)).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(cost_usd),0) s FROM usage "
                "WHERE task_id=? AND run_id=? AND cash=1", (task_id, run_id)).fetchone()
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

    def gate_outcomes(self) -> list[dict]:
        """Per-candidate gate pass/fail counts, split by attempt 1 vs later.

        The `events` rows look like `<cand_id> attempt=<n> passed=<bool>`; parsing
        them here keeps the solve-rate question ("which worker actually lands a
        task on the first try?") a single query instead of an ad-hoc script."""
        rows = self._conn.execute(
            "SELECT detail FROM events WHERE kind='gate'").fetchall()
        agg: dict[tuple[str, bool], dict] = {}
        for r in rows:
            parts = (r["detail"] or "").split()
            if len(parts) < 3:
                continue
            cand, attempt, passed = parts[0], parts[1], parts[2]
            first = attempt == "attempt=1"
            key = (cand, first)
            a = agg.setdefault(key, {"cand_id": cand, "first_attempt": first,
                                     "passed": 0, "failed": 0})
            a["passed" if passed == "passed=True" else "failed"] += 1
        return sorted(agg.values(), key=lambda a: (a["cand_id"], not a["first_attempt"]))

    def event_counts(self, kinds: tuple[str, ...]) -> dict[str, int]:
        q = ",".join("?" for _ in kinds)
        rows = self._conn.execute(
            f"SELECT kind, COUNT(*) n FROM events WHERE kind IN ({q}) GROUP BY kind",
            kinds).fetchall()
        return {r["kind"]: r["n"] for r in rows}

    def run_token_totals(self, run_id: str) -> dict:
        """Token totals for ONE run, split cash vs subscription.

        `cash spend this run: $0.02` is true and nearly useless: with
        budget.count_cli false (the default) the subscription tiers are logged but
        not counted, and on a measured run they were 99.2 % of the notional cost
        and 92 % of the input tokens. Reporting needs the token figures
        INDEPENDENTLY of whether the budget counts them — on a subscription plan
        the input-token total is the resource that actually runs out.
        """
        rows = self._conn.execute(
            """SELECT cash, COUNT(*) calls, COALESCE(SUM(input_tokens),0) in_tok,
                      COALESCE(SUM(output_tokens),0) out_tok,
                      COALESCE(SUM(cache_hit_tokens),0) cache_hit,
                      COALESCE(SUM(cache_miss_tokens),0) cache_miss,
                      COALESCE(SUM(cost_usd),0) cost
               FROM usage WHERE run_id=? GROUP BY cash""", (run_id,)).fetchall()
        out = {"cash": None, "subscription": None}
        for r in rows:
            out["cash" if r["cash"] else "subscription"] = {
                k: r[k] for k in ("calls", "in_tok", "out_tok", "cache_hit",
                                  "cache_miss", "cost")}
        return out

    def subscription_tokens_by_role(self) -> list[dict]:
        """Subscription-tier (cash=0) token totals per role — the quota proxy.
        Divided by completed tasks, this is the number that decides whether the
        cheap worker tier earns its place."""
        rows = self._conn.execute(
            """SELECT role, COUNT(*) calls, SUM(input_tokens) in_tok,
                      SUM(output_tokens) out_tok, SUM(cache_hit_tokens) cache_hit
               FROM usage WHERE cash=0 GROUP BY role ORDER BY in_tok DESC"""
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

    def save_discussion_log(self, session: str, log: dict[str, Any]) -> None:
        """Persist the panel's frame log for a session.

        Separate from `save_discussion` because the two have different owners and
        different shapes. The transcript is what the *planner* is re-sent on a
        resumed conversation — `ROLE: text`, flattened, lossy on purpose. This is
        what the *operator* reads: the typed frames, with their kinds, sequence
        and timestamps intact.

        Without it the panel had nothing to show after an API restart but that
        flattened blob, so a conversation came back as a wall of `USER:` /
        `PLANNER:` lines in a folded `<pre>` — the chat replaced by a dump of
        itself. Sessions live in process memory (`api.discuss.DiscussManager`)
        and always will; this is what makes losing them survivable.

        Writes the whole log each time rather than appending: it is tens of
        frames, it is rewritten only on a real frame (never on `progress`), and
        one row that is always complete cannot half-restore.
        """
        self._conn.execute(
            """INSERT INTO discussions(session, transcript, frames, updated_at)
               VALUES(?,'',?,?)
               ON CONFLICT(session) DO UPDATE SET
                 frames=excluded.frames, updated_at=excluded.updated_at""",
            (session, json.dumps(log), time.time()))
        self._conn.commit()

    def load_discussion_log(self, session: str) -> dict[str, Any] | None:
        """The persisted frame log, or None if this store predates it."""
        try:
            row = self._conn.execute(
                "SELECT frames FROM discussions WHERE session=?", (session,)).fetchone()
        except sqlite3.OperationalError:        # a store older than the column
            return None
        if row is None or not row["frames"]:
            return None
        try:
            log = json.loads(row["frames"])
        except json.JSONDecodeError:
            return None
        return log if isinstance(log, dict) else None

    # ---- planner handoff (TLDR carried across an expired session) ---------
    def save_handoff(self, session: str, digest: str,
                     snapshot_path: str | None = None) -> None:
        self._conn.execute(
            """INSERT INTO planner_handoff(session, digest, snapshot_path, updated_at)
               VALUES(?,?,?,?)
               ON CONFLICT(session) DO UPDATE SET
                 digest=excluded.digest,
                 snapshot_path=excluded.snapshot_path,
                 updated_at=excluded.updated_at""",
            (session, digest, snapshot_path, time.time()))
        self._conn.commit()

    def load_handoff(self, session: str) -> dict | None:
        row = self._conn.execute(
            """SELECT digest, snapshot_path, updated_at
               FROM planner_handoff WHERE session=?""", (session,)).fetchone()
        return dict(row) if row else None

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
