"""CLI: python -m orchestrator <command> [--project <name>] ...

Commands:
  plan            enrich backlog items into machine-executable specs (Opus)
  discuss         interactive multi-turn requirements loop (tech-lead planner)
  run             execute the queue (--dry-run to plan without spending)
  resume          continue a paused/interrupted run from checkpoints
  status          queue, budgets, and cost breakdown
  metrics         solve rate, escalation frequency, subscription tokens/task
  import-backlog  register backlog items as stubs (no LLM)
  serve           control-panel HTTP API (read layer + job control), localhost only
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from .engine import runner
from .core.config import load_config
from .core.errors import PlannerNeedsInput
from .nodes.discuss import run_discuss
from .nodes.planner import (import_backlog_stubs, needs_plan_ids as plan_batch_ids,
                            plan)
from .engine.scheduler import queue_stats
from .ops.store import Store


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--project", help="project profile name (projects/<name>/profile.yaml, "
                                     "falling back to config/projects/<name>.yaml); "
                                     "or set ORCH_PROJECT")
    p.add_argument("-v", "--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orchestrator", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan", help="turn backlog items into task specs (Opus)")
    _add_common(p)
    p.add_argument("note", nargs="?", default="",
                   help="optional discussion note to fold into the plan")
    p.add_argument("--tasks", help="comma-separated ids to (re)plan, e.g. T-120,T-121")
    # Batching is the single largest saving available with no code change: the
    # planner's cost is ~400k tokens of repo exploration per INVOCATION, amortised
    # across everything in the call (measured: 385k for one task, 425k for one item
    # split into two). Planning one item at a time is the most expensive way to use
    # this harness, so make the cheap path the easy one.
    p.add_argument("--all-needs-plan", action="store_true",
                   help="plan every needs_plan task in ONE call (~8-9x cheaper per "
                        "task than planning them one at a time)")
    p.add_argument("--limit", type=int, default=None,
                   help="with --all-needs-plan: cap how many items go into the call "
                        "(specs written far ahead of the work go stale — batch per "
                        "milestone, not the whole backlog). 0 or omitted = no cap")

    p = sub.add_parser("discuss", help="interactive multi-turn requirements loop (tech-lead planner)")
    _add_common(p)
    p.add_argument("request", help="your initial request / feature description")

    p = sub.add_parser("run", help="execute the task queue")
    _add_common(p)
    p.add_argument("--dry-run", action="store_true",
                   help="print waves/candidates/files; no tokens, no git")
    p.add_argument("--tasks", help="limit to comma-separated task ids")
    p.add_argument("--n", type=int, default=None,
                   help="override n_candidates (best-of-N) for this run")

    p = sub.add_parser("resume", help="continue the latest paused run")
    _add_common(p)

    p = sub.add_parser("status", help="queue + cost report")
    _add_common(p)

    p = sub.add_parser("metrics", help="solve rate, escalation frequency, "
                                       "subscription tokens per completed task")
    _add_common(p)

    p = sub.add_parser("import-backlog", help="register backlog items as stubs (no LLM)")
    _add_common(p)

    p = sub.add_parser("serve", help="run the control-panel HTTP API (localhost only)")
    _add_common(p)
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address; keep it loopback — the API has no auth")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--reload", action="store_true", help="uvicorn autoreload (dev)")
    return parser


def _print_metrics(store) -> None:
    """The three numbers that decide the worker-tier question, straight off the
    telemetry the harness already records. Meaningful after ~20 real tasks."""
    tasks = store.all_tasks()
    done = sum(1 for t in tasks if t["status"] == "done")

    print("solve rate by candidate (gate pass/fail):")
    for row in store.gate_outcomes() or []:
        total = row["passed"] + row["failed"]
        when = "attempt 1" if row["first_attempt"] else "retries  "
        print(f"  {row['cand_id']:<16} {when}  passed={row['passed']:<4} "
              f"failed={row['failed']:<4} rate={row['passed'] / total * 100:.0f}%")

    counts = store.event_counts(("escalated", "auto_integrated", "crashed",
                                 "retrieval_exhausted", "visual_gate_error",
                                 "visual_gate_skipped", "no_patch",
                                 "verify_unverifiable"))
    print(f"\nqueue: {queue_stats(tasks)}")
    for kind in sorted(counts):
        print(f"  {kind:<22} {counts[kind]}")

    print("\nsubscription tokens by role (the quota proxy):")
    for row in store.subscription_tokens_by_role():
        in_tok = row["in_tok"] or 0
        per_task = f"{in_tok / done:,.0f}" if done else "-"
        print(f"  {row['role']:<16} calls={row['calls']:<4} in={in_tok:<10} "
              f"out={row['out_tok'] or 0:<9} cache_hit={row['cache_hit'] or 0:<10} "
              f"in/completed_task={per_task}")
    if not done:
        print("  (no completed tasks yet — per-task figures need a finished run)")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    cfg = load_config(args.project)

    if args.command == "plan":
        store = Store(cfg.store_path())
        run_id = store.create_run(note="plan")
        ctx = runner.make_context(cfg, store, run_id)
        only = set(args.tasks.split(",")) if args.tasks else None
        if args.all_needs_plan:
            pending = plan_batch_ids(store, args.limit)
            if not pending:
                print("nothing to plan: no tasks with status needs_plan "
                      "(run `import-backlog` first?)")
                store.set_run_status(run_id, "done")
                return 0
            only = (only or set()) | set(pending)
            print(f"planning {len(only)} item{'s' if len(only) != 1 else ''} in "
                  f"one call: {', '.join(sorted(only))}")
        try:
            specs = asyncio.run(plan(ctx, args.note, sorted(only) if only else None))
        except PlannerNeedsInput as e:
            # One-shot mode must not silently guess — elicitation belongs in discuss.
            store.set_run_status(run_id, "done")
            print("planner needs input before it can plan — run `discuss` to answer:")
            for q in e.questions:
                print(f"  [{q.get('id', '?')}] {q.get('q', '')}"
                      + (f"  (why: {q['why']})" if q.get("why") else ""))
            return 2
        store.set_run_status(run_id, "done")
        print(f"planned {len(specs)} tasks:")
        for s in specs:
            flag = "" if s.get("agent_able", True) else "  [human-only]"
            print(f"  {s['id']}: {s['title']}{flag}")
        return 0

    if args.command == "discuss":
        store = Store(cfg.store_path())
        run_id = store.create_run(note="discuss")
        ctx = runner.make_context(cfg, store, run_id)
        specs = asyncio.run(run_discuss(ctx, args.request))
        store.set_run_status(run_id, "done")
        return 0 if specs else 1

    if args.command == "run":
        task_filter = set(args.tasks.split(",")) if args.tasks else None
        asyncio.run(runner.run(cfg, dry_run=args.dry_run,
                               task_filter=task_filter, n_candidates=args.n))
        return 0

    if args.command == "resume":
        store = Store(cfg.store_path())
        paused = store.latest_run(statuses=("paused",))
        if not paused:
            print("no paused run to resume")
            return 1
        print(f"resuming run {paused['id']} (paused: {paused.get('note')})")
        asyncio.run(runner.run(cfg, resume_run_id=paused["id"]))
        return 0

    if args.command == "status":
        store = Store(cfg.store_path())
        tasks = store.all_tasks()
        print(f"project: {cfg.project_name}   tasks: {len(tasks)}   "
              f"queue: {queue_stats(tasks)}\n")
        for t in tasks:
            cost = f"  ${t['cost_usd']:.2f}" if t.get("cost_usd") else ""
            retries = f"  retries={t['retries']}" if t.get("retries") else ""
            print(f"  [{t['status']:>11}] {t['id']}: {t['title'][:80]}{cost}{retries}")
        print("\nusage by role/model:")
        for row in store.usage_summary():
            tag = "" if row["cash"] else "  (subscription)"
            hit, miss = row["cache_hit"] or 0, row["cache_miss"] or 0
            # Hit rate over the cached+uncached input the provider actually
            # reported — "-" (not 0%) when a provider reports no cache telemetry
            # at all, so an unknown never reads as a measured cold cache.
            rate = f"{hit / (hit + miss) * 100:.0f}%" if (hit + miss) else "-"
            print(f"  {row['role']:<16} {row['model']:<28} calls={row['calls']:<4} "
                  f"in={row['in_tok'] or 0:<9} out={row['out_tok'] or 0:<8} "
                  f"cache_hit={hit:<9} cache_miss={miss:<9} hit_rate={rate:<5} "
                  f"${row['cost'] or 0:.3f}{tag}")
        return 0

    if args.command == "metrics":
        _print_metrics(Store(cfg.store_path()))
        return 0

    if args.command == "import-backlog":
        store = Store(cfg.store_path())
        ctx = runner.make_context(cfg, store, run_id="import")
        n = import_backlog_stubs(ctx)
        print(f"imported {n} new backlog items as stubs (status: needs_plan). "
              f"Run `plan` to enrich them into executable specs.")
        return 0

    if args.command == "serve":
        # Imported here, not at module scope: [api] is an optional extra and a
        # machine without fastapi must still be able to `plan`/`run`.
        try:
            import uvicorn
        except ModuleNotFoundError:
            print('serve needs the [api] extra: pip install -e ".[dev,api]"')
            return 2
        if args.project:
            # The API resolves the project per request; this only marks which one
            # the panel preselects, and it survives into the --reload child.
            os.environ["ORCH_PROJECT"] = args.project
        print(f"control panel API: http://{args.host}:{args.port}"
              f"   (schema: /openapi.json)")
        uvicorn.run("orchestrator.api.app:app", host=args.host, port=args.port,
                    reload=args.reload)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
