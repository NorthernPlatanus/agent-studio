"""LangGraph Studio entrypoint — `langgraph dev` (see langgraph.json).

Gives the n8n-style visual UI over the per-task state machine: the graph
topology, live runs, and state inspection at every step.

With ORCH_PROJECT set (and a valid project profile) the graph is fully
wired — you can invoke a single task from Studio against the real store and
repo. Without it we fall back to a topology-only context: the graph renders
and routes, but nodes that need git/providers will fail if invoked (by
design — viewing costs nothing).
"""

from __future__ import annotations

import logging

# Absolute imports: `langgraph dev` may load this file by path (standalone,
# no parent package), where relative imports fail.
from orchestrator.core.config import load_config
from orchestrator.core.context import RunContext
from orchestrator.engine.graph import build_task_graph
from orchestrator.ops.store import Store

log = logging.getLogger("orchestrator.studio")


def make_graph():
    cfg = load_config()  # honors ORCH_PROJECT
    try:
        from orchestrator.engine.runner import make_context
        ctx = make_context(cfg, Store(cfg.store_path()), run_id="studio")
    except Exception as e:  # no project profile / repo_path -> view-only
        log.warning("studio: topology-only context (%s)", e)
        ctx = RunContext(cfg=cfg, store=Store(":memory:"), git=None,
                         budget=None, run_id="studio", dry_run=True)
    # No explicit checkpointer: `langgraph dev` provides its own persistence.
    return build_task_graph(ctx).compile()


graph = make_graph()
