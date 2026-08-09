"""The one definition of the fixture store, shared by every lane.

`tests/api` uses `seed()` in-process; `studio-verify`'s smoke server and
`studio-contract`'s OpenAPI/fixture capture use the module entry point:

    python -m tests.api.fixtures.seed_store /tmp/as-fixture
    ORCH_PROJECT=example ORCH_PATHS_STATE_DIR=/tmp/as-fixture \\
        uvicorn orchestrator.api.app:app --host 127.0.0.1 --port 8788

Two things are deliberate:

- **Ids and timestamps are fixed constants.** `Store.create_run()` mints a
  time-based id, which would make every captured fixture and every assertion
  drift by the second. Runs are therefore inserted directly (the only place this
  file goes around `Store`), so `RUN_DONE` / `RUN_PAUSED` are stable strings a
  test or a UI fixture can name.
- **The dataset covers the awkward cases, not the happy path.** Every task status
  including `needs_plan`/`human_only`/`needs_human`, a decomposed parent, a
  subscription row with no cache telemetry at all (so the UI's "unreported"
  branch is exercised, not just 0 %), and the event kinds that mean something
  went sideways: `escalated`, `no_patch`, `visual_gate_skipped`,
  `verify_unverifiable`, `crashed`.

It writes a real store through `ops.store.Store`, so if the schema changes the
fixture follows it automatically. Never point it at `state/demo-project.*`.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from orchestrator.ops.store import Store

PROJECT = "example"
# The id prefix is the run's start time in UTC — keep it in step with T0/T1.
RUN_DONE = "20260701-101500-aaaaaa"
RUN_PAUSED = "20260702-141500-bbbbbb"

# The two runs' wall-clock starts as unix seconds — fixed so `group_by=day`
# rollups and any "last N days" chart are reproducible. These are derived from,
# and must keep agreeing with, the timestamps encoded in the run ids above:
# T0 = 2026-07-01T10:15:00Z, T1 = 2026-07-02T14:15:00Z. (An earlier version had
# T0 pointing at 2026-06-21 while the ids said July, which read as a UI bug the
# first time a runs table showed both.)
T0 = 1782900900.0
T1 = 1783001700.0

PAUSE_NOTE = "limit exhausted: subscription tier hit its 5h cap (resume to continue)"


def _spec(task_id: str, title: str, **over: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "id": task_id, "title": title, "milestone": "M1",
        "deps": [], "files_read": ["src/app.ts"], "files_write": [f"src/{task_id}.ts"],
        "acceptance": ["typecheck passes", "unit tests pass"],
        "domain": "frontend", "risk": "low", "complexity": "s",
        "visual": False, "agent_able": True, "n_candidates": 1,
        "parent_id": None,
    }
    spec.update(over)
    return spec


# 12 tasks: every status, a decomposed parent (T-131 -> T-131a/T-131b), a seam
# task with no deps (the planner mistake /waves warns about), a human-only task,
# and a visual one.
TASKS: list[dict[str, Any]] = [
    _spec("T-101", "Extract the settings panel into its own route",
          status="done", files_write=["src/routes/settings.tsx"]),
    _spec("T-102", "Add optimistic updates to the task table",
          status="done", n_candidates=3, risk="medium",
          files_write=["src/widgets/task-table.tsx"], deps=["T-101"]),
    _spec("T-110", "Wire the live stream provider", status="ready",
          files_write=["src/app/live.tsx"], deps=["T-102"]),
    _spec("T-111", "Chart the token split by channel", status="ready",
          domain="charts", visual=True, complexity="m",
          files_write=["src/widgets/token-panel.tsx"]),
    _spec("T-112", "Unify the shared fetch wrapper", status="ready",
          domain="seam", files_write=["src/shared/api/client.ts"]),
    _spec("T-120", "Migrate the queue board to the new API", status="running",
          n_candidates=2, risk="medium", files_write=["src/widgets/queue-board.tsx"]),
    _spec("T-121", "Rework the planner chat state machine", status="needs_human",
          risk="high", complexity="l", files_write=["src/widgets/planner-chat.tsx"]),
    _spec("T-122", "Port the legacy stats page", status="failed", retries=2,
          files_write=["src/pages/stats.tsx"]),
    _spec("T-123", "Drop the abandoned websocket client", status="rejected",
          files_write=["src/shared/api/ws.ts"]),
    _spec("T-130", "Decide the deployment story for the panel",
          status="human_only", agent_able=False, files_write=[]),
    _spec("T-131", "Rebuild the run timeline", status="needs_plan",
          files_write=["src/widgets/run-timeline.tsx"]),
    _spec("T-131a", "Run timeline: event grouping", status="ready",
          parent_id="T-131", files_write=["src/widgets/run-timeline/group.ts"],
          deps=["T-110"]),
    _spec("T-131b", "Run timeline: wave lanes", status="ready",
          parent_id="T-131", files_write=["src/widgets/run-timeline/lanes.tsx"],
          deps=["T-131a"]),
]

# (run_id, task_id, role, provider, model, in, out, cost, cash, hit, miss)
USAGE: list[tuple] = [
    (RUN_DONE, None, "planner", "claude_cli", "opus", 402_000, 8_400, 0.0, False,
     361_000, 41_000),
    (RUN_DONE, "T-101", "worker", "deepseek", "deepseek-chat", 48_000, 6_200,
     0.0121, True, 12_000, 36_000),
    (RUN_DONE, "T-101", "reviewer", "claude_cli", "opus", 62_000, 1_900, 0.0, False,
     55_000, 7_000),
    (RUN_DONE, "T-102", "worker", "deepseek", "deepseek-chat", 91_000, 11_400,
     0.0243, True, 20_000, 71_000),
    (RUN_DONE, "T-102", "worker", "kimi", "kimi-k2", 88_000, 9_800, 0.0198, True,
     0, 88_000),
    (RUN_DONE, "T-102", "reviewer", "claude_cli", "opus", 71_000, 2_400, 0.0, False,
     60_000, 11_000),
    # No cache telemetry at all (hit and miss both 0): the UI must show "-" here,
    # not a measured 0 % hit rate.
    (RUN_DONE, "T-102", "verifier", "claude_cli", "sonnet", 24_000, 900, 0.0, False,
     0, 0),
    (RUN_PAUSED, "T-120", "worker", "deepseek", "deepseek-chat", 52_000, 7_100,
     0.0139, True, 9_000, 43_000),
    (RUN_PAUSED, "T-120", "worker", "kimi", "kimi-k2", 50_000, 6_800, 0.0131, True,
     0, 50_000),
    (RUN_PAUSED, "T-121", "senior", "claude_cli", "opus", 145_000, 12_600, 0.0,
     False, 96_000, 49_000),
    (RUN_PAUSED, "T-122", "worker", "deepseek", "deepseek-chat", 44_000, 5_200,
     0.0104, True, 7_000, 37_000),
]

# (run_id, task_id, kind, detail) — detail formats match what the nodes write, so
# `gate_outcomes()` and the candidate-from-events fallback parse them for real.
EVENTS: list[tuple] = [
    (RUN_DONE, None, "planned", "13 specs written (1 human-only)"),
    (RUN_DONE, "T-101", "gate", "deepseek attempt=1 passed=True"),
    (RUN_DONE, "T-101", "review", json.dumps(
        {"decision": "approve", "winner": "deepseek", "notes": "clean extraction"})),
    (RUN_DONE, "T-101", "merged", "agents/T-101-deepseek -> agents/feature @ 4f2c1ab9de"),
    (RUN_DONE, "T-101", "finalized", "done"),
    (RUN_DONE, "T-102", "gate", "deepseek attempt=1 passed=False failed_step="
                                "npm run typecheck\nsrc/widgets/task-table.tsx(42,7): "
                                "error TS2345: Argument of type 'string' is not "
                                "assignable to parameter of type 'TaskId'."),
    (RUN_DONE, "T-102", "gate", "kimi attempt=1 passed=True"),
    (RUN_DONE, "T-102", "visual_gate_skipped",
     "kimi enforced=False passed=True failures=[]"),
    (RUN_DONE, "T-102", "review", json.dumps(
        {"decision": "approve", "winner": "kimi", "notes": "kimi handled the "
                                                           "TaskId branding"})),
    (RUN_DONE, "T-102", "merged", "agents/T-102-kimi -> agents/feature @ 91ab77c0de"),
    (RUN_DONE, "T-102", "finalized", "done"),
    (RUN_PAUSED, "T-120", "gate", "deepseek attempt=1 passed=False failed_step="
                                  "npm test\n 2 failing"),
    (RUN_PAUSED, "T-120", "candidate_failed",
     "kimi attempt=1 patch_failed: hunk #2 did not apply to src/widgets/queue-board.tsx"),
    (RUN_PAUSED, "T-120", "no_patch",
     "kimi attempt=2 (no blocks; fix round refunded)"),
    (RUN_PAUSED, "T-120", "gate", "deepseek attempt=2 passed=True"),
    (RUN_PAUSED, "T-121", "escalated", "attempt=3 after cheap-loop exhaustion"),
    (RUN_PAUSED, "T-121", "verify_unverifiable",
     "senior startup toast is a one-shot state a read-only verifier cannot observe"),
    (RUN_PAUSED, "T-121", "finalized", "needs_human"),
    (RUN_PAUSED, "T-122", "crashed",
     "GateError: install step failed: npm ci exited 1 (no package-lock.json)"),
    (RUN_PAUSED, "T-122", "finalized", "failed"),
]

# `TaskState.candidates` for the running task, so the candidate board has a live
# checkpoint to read rather than only the event-log fallback.
CHECKPOINT_TASK = "T-120"
CHECKPOINT_CANDIDATES: list[dict[str, Any]] = [
    {"cand_id": "deepseek", "model": "deepseek-chat", "attempt": 2,
     "status": "gate_passed", "worktree": "/tmp/worktrees/T-120-deepseek",
     "branch": "agents/T-120-deepseek", "diff": "--- a/x\n+++ b/x\n@@\n+ok\n",
     "notes": "reworked the selector memoization", "messages": []},
    {"cand_id": "kimi", "model": "kimi-k2", "attempt": 2, "status": "llm_failed",
     "worktree": "/tmp/worktrees/T-120-kimi", "branch": "agents/T-120-kimi",
     "error": "provider returned prose, no <file> blocks", "no_patch": True,
     "messages": []},
]


def _insert_runs(path: Path) -> None:
    """Runs, with fixed ids. Direct SQL because `Store.create_run` derives the id
    from the clock; everything else in this file goes through `Store`."""
    conn = sqlite3.connect(path)
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO runs(id, started_at, status, note, cost_usd) "
            "VALUES(?,?,?,?,0)",
            [(RUN_DONE, T0, "done", "nightly queue drain"),
             (RUN_PAUSED, T1, "paused", PAUSE_NOTE)])
        conn.commit()
    finally:
        conn.close()


def _freeze_timestamps(path: Path) -> None:
    """Pin `usage.ts`, `events.ts` and `tasks.updated_at` to the fixture clock.

    `Store.record_usage`/`log_event` stamp `time.time()` — correct for the real
    thing, useless for a fixture: a `group_by=day` rollup would collapse to one
    bucket ("today") and every captured UI fixture would differ from the last by
    the second. Rows keep their insertion order (rowid), so only the displayed
    clock changes: each row gets its run's base time plus its ordinal.
    """
    conn = sqlite3.connect(path)
    try:
        for table in ("usage", "events"):
            rows = conn.execute(f"SELECT rowid, run_id FROM {table} "
                                "ORDER BY rowid").fetchall()
            for offset, (rowid, run_id) in enumerate(rows):
                base = T1 if run_id == RUN_PAUSED else T0
                conn.execute(f"UPDATE {table} SET ts=? WHERE rowid=?",
                             (base + offset * 37.0, rowid))
        conn.execute("UPDATE tasks SET updated_at=?", (T1 + 3600.0,))
        conn.commit()
    finally:
        conn.close()


def seed(store_path: Path) -> None:
    """Write the fixture dataset to `store_path` (creating parents).

    Also seeds `<project>.checkpoints.sqlite3` next to it, because the candidate
    board's primary source is the LangGraph checkpoint and a fixture without one
    would only ever exercise the fallback path.
    """
    store_path = Path(store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    store = Store(store_path)
    try:
        _insert_runs(store_path)
        for spec in TASKS:
            body = {k: v for k, v in spec.items() if k not in ("retries",)}
            store.upsert_task(body)
        # retries is a column, not a spec field, and only set_task_status writes it.
        for spec in TASKS:
            if spec.get("retries"):
                store.set_task_status(spec["id"], spec["status"], spec["retries"])
        for (run_id, task_id, role, provider, model, in_tok, out_tok, cost, cash,
             hit, miss) in USAGE:
            store.record_usage(run_id, task_id, role, provider, model, in_tok,
                               out_tok, cost, cash, hit, miss)
        for run_id, task_id, kind, detail in EVENTS:
            store.log_event(run_id, task_id, kind, detail)
        store.save_discussion(
            PROJECT,
            "you: rebuild the run timeline\nplanner: [q-1] should waves be lanes "
            "or nested rows?\nyou: lanes\nplanner: applied 2 spec(s).")
    finally:
        store.close()

    _freeze_timestamps(store_path)
    seed_checkpoint(store_path.with_name(f"{store_path.stem}.checkpoints.sqlite3"))


def seed_checkpoint(checkpoint_path: Path) -> None:
    """One thread ("{RUN_PAUSED}:T-120") holding two candidates mid-attempt."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        config = {"configurable": {"thread_id": f"{RUN_PAUSED}:{CHECKPOINT_TASK}",
                                   "checkpoint_ns": ""}}
        checkpoint = {
            "v": 1,
            "ts": "2026-07-02T14:22:03.000000+00:00",
            # A uuid6-shaped id: the saver orders threads by this string.
            "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
            "channel_values": {
                "run_id": RUN_PAUSED, "task_id": CHECKPOINT_TASK,
                "attempt": 2, "n_candidates": 2,
                "candidates": CHECKPOINT_CANDIDATES,
            },
            "channel_versions": {}, "versions_seen": {},
        }
        saver.put(config, checkpoint, {"source": "loop", "step": 4}, {})


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m tests.api.fixtures.seed_store <state_dir>",
              file=sys.stderr)
        return 2
    state_dir = Path(args[0]).expanduser().resolve()
    if "demo-project" in str(state_dir) or (state_dir / "demo-project.sqlite3").exists():
        # The live project is never a fixture target; refuse rather than trust the
        # caller to have read the runbook.
        print(f"refusing to seed into {state_dir}: it holds the live project's state",
              file=sys.stderr)
        return 2
    path = state_dir / f"{PROJECT}.sqlite3"
    seed(path)
    print(f"seeded {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
