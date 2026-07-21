"""CLI: python -m orchestrator <command> [--project <name>] ...

Commands:
  plan            enrich backlog items into machine-executable specs (Opus)
  discuss         interactive multi-turn requirements loop (tech-lead planner)
  run             execute the queue (--dry-run to plan without spending)
  resume          continue a paused/interrupted run from checkpoints
  status          queue, budgets, and cost breakdown
  import-backlog  register backlog items as stubs (no LLM)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .engine import runner
from .core.config import load_config
from .core.errors import PlannerNeedsInput
from .nodes.discuss import run_discuss
from .nodes.planner import import_backlog_stubs, plan
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

    p = sub.add_parser("import-backlog", help="register backlog items as stubs (no LLM)")
    _add_common(p)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    cfg = load_config(args.project)

    if args.command == "plan":
        store = Store(cfg.store_path())
        run_id = store.create_run(note="plan")
        ctx = runner.make_context(cfg, store, run_id)
        only = set(args.tasks.split(",")) if args.tasks else None
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
            print(f"  {row['role']:<16} {row['model']:<28} calls={row['calls']:<4} "
                  f"in={row['in_tok'] or 0:<9} out={row['out_tok'] or 0:<8} "
                  f"${row['cost'] or 0:.3f}{tag}")
        return 0

    if args.command == "import-backlog":
        store = Store(cfg.store_path())
        ctx = runner.make_context(cfg, store, run_id="import")
        n = import_backlog_stubs(ctx)
        print(f"imported {n} new backlog items as stubs (status: needs_plan). "
              f"Run `plan` to enrich them into executable specs.")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
