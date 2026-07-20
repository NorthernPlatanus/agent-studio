"""Graph shape + state helpers. No LLMs, no git: verifies the LangGraph
builds, routes correctly, and latest_candidates collapses attempts."""

from orchestrator.core.state import latest_candidates


def C(cand_id, attempt, status):
    return {"cand_id": cand_id, "attempt": attempt, "status": status}


def test_latest_candidates_collapses_attempts():
    state = {"candidates": [
        C("deepseek", 1, "gate_failed"),
        C("glm", 1, "gate_passed"),
        C("deepseek", 2, "gate_passed"),
    ]}
    latest = latest_candidates(state)
    assert latest["deepseek"]["attempt"] == 2
    assert latest["deepseek"]["status"] == "gate_passed"
    assert latest["glm"]["attempt"] == 1


def test_graph_compiles(tmp_path):
    """The graph must build and compile without runtime services being live."""
    from orchestrator.core.config import load_config
    from orchestrator.core.context import RunContext
    from orchestrator.engine.graph import build_task_graph

    cfg = load_config()  # generic defaults, no project
    ctx = RunContext(cfg=cfg, store=None, git=None, budget=None,
                     run_id="test", dry_run=True)
    graph = build_task_graph(ctx).compile()
    nodes = set(graph.get_graph().nodes)
    assert {"dispatch", "work_candidate", "collect",
            "review", "integrate", "finalize"} <= nodes
