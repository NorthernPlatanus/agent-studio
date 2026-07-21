"""Phase 5: domain-aware worker routing, domain protocol injection, and
scheduler visibility (safety invariant unchanged)."""

from pathlib import Path

from orchestrator.core.config import Config
from orchestrator.core.context import RunContext
from orchestrator.engine.graph import resolve_worker_pool
from orchestrator.engine.scheduler import (domain_stats, next_batch,
                                           seam_tasks_missing_deps)
from orchestrator.nodes.worker import _build_stable_prompt


def _cfg(domains=None, prompts=None):
    data = {"roles": {"worker": {"default": "w", "candidates": ["w"]}},
            "domains": domains or {}}
    if prompts:
        data["prompts"] = prompts
    return Config(data, "p", Path("/tmp"))


def test_pool_uses_domain_override():
    cfg = _cfg({"physics": {"worker_default": "phys_w", "protocol": "physics"}})
    assert resolve_worker_pool(cfg, {"domain": "physics"}) == ("phys_w", ["phys_w"])


def test_pool_falls_back_without_domain():
    cfg = _cfg({"physics": {"worker_default": "phys_w"}})
    assert resolve_worker_pool(cfg, {}) == ("w", ["w"])
    assert resolve_worker_pool(cfg, {"domain": "unknown"}) == ("w", ["w"])


def test_domain_candidates_override():
    cfg = _cfg({"render": {"candidates": ["r1", "r2"]}})
    default, pool = resolve_worker_pool(cfg, {"domain": "render"})
    assert pool == ["r1", "r2"]


def test_domain_protocol_injected(tmp_path):
    pd = tmp_path / "prompts"
    pd.mkdir()
    (pd / "worker_protocol.md").write_text("GENERIC PROTO")
    (pd / "physics.md").write_text("PHYSICS RULES")
    cfg = _cfg({"physics": {"protocol": "physics"}},
               prompts={"shared_dir": str(pd)})
    ctx = RunContext(cfg=cfg, store=None, git=None, budget=None, run_id="r")
    spec = {"id": "T", "title": "t", "description": "d", "domain": "physics",
            "files_write": ["a.py"]}
    prompt = _build_stable_prompt(ctx, spec, {})
    assert "GENERIC PROTO" in prompt
    assert "DOMAIN PROTOCOL (physics)" in prompt and "PHYSICS RULES" in prompt


def test_no_domain_protocol_when_absent(tmp_path):
    pd = tmp_path / "prompts"
    pd.mkdir()
    (pd / "worker_protocol.md").write_text("GENERIC PROTO")
    cfg = _cfg({}, prompts={"shared_dir": str(pd)})
    ctx = RunContext(cfg=cfg, store=None, git=None, budget=None, run_id="r")
    spec = {"id": "T", "title": "t", "description": "d", "files_write": ["a.py"]}
    prompt = _build_stable_prompt(ctx, spec, {})
    assert "DOMAIN PROTOCOL" not in prompt


def test_scheduler_still_refuses_write_overlap_across_domains():
    tasks = [
        {"id": "A", "status": "ready", "files_write": ["x.py"], "domain": "physics",
         "milestone": ""},
        {"id": "B", "status": "ready", "files_write": ["x.py"], "domain": "render",
         "milestone": ""},
    ]
    batch = next_batch(tasks, 5)
    assert len(batch) == 1        # disjoint-write invariant holds regardless of domain


def test_seam_without_deps_flagged():
    tasks = [
        {"id": "S", "domain": "seam", "deps": []},
        {"id": "S2", "domain": "seam", "deps": ["A"]},
        {"id": "X", "domain": "physics"},
    ]
    assert seam_tasks_missing_deps(tasks) == ["S"]
    assert domain_stats(tasks) == {"seam": 2, "physics": 1}
