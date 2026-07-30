"""Defect-plan #2 item 1: the reviewer must inspect the code it is reviewing.

The reviewer holds read-only Read/Grep/Glob so it can check surrounding context,
and its cwd used to be `repo_path()` — the user's primary checkout, sitting on the
base branch. So it browsed a tree that contains neither the diff under review nor
the waves already merged into the feature branch, then spent 123k-167k input
tokens looking for symbols that were never going to be there (vs ~43k when it
answers from the diff).
"""

from pathlib import Path
from types import SimpleNamespace

from orchestrator.core.config import Config
from orchestrator.nodes.reviewer import review_cwd


def _ctx(tmp_path, *, git=True):
    (tmp_path / "repo").mkdir(exist_ok=True)
    (tmp_path / "wt").mkdir(exist_ok=True)
    cfg = Config({"project": {"repo_path": str(tmp_path / "repo")}}, "p", tmp_path)
    return SimpleNamespace(
        cfg=cfg,
        git=SimpleNamespace(work_dir=tmp_path / "wt") if git else None)


def _cand(worktree=None):
    return {"cand_id": "c0", "status": "gate_passed",
            **({"worktree": str(worktree)} if worktree else {})}


def test_single_candidate_is_reviewed_in_its_own_worktree(tmp_path):
    """The load-bearing case: one green candidate, so the tree that produced the
    diff is unambiguous and a lookup either confirms or refutes it."""
    ctx = _ctx(tmp_path)
    wt = tmp_path / "wt" / "t-1-c0"
    wt.mkdir()
    assert review_cwd(ctx, {"c0": _cand(wt)}) == str(wt)
    # explicitly NOT the primary checkout any more
    assert review_cwd(ctx, {"c0": _cand(wt)}) != str(ctx.cfg.repo_path())


def test_best_of_n_uses_the_shared_integration_checkout(tmp_path):
    """Browsing candidate A's tree while judging B would mix them. `_integration`
    is on the feature branch — the true base every candidate was built from, and
    strictly better than a primary checkout that lacks merged waves."""
    ctx = _ctx(tmp_path)
    integration = tmp_path / "wt" / "_integration"
    integration.mkdir()
    passed = {"a": _cand(tmp_path / "wt" / "t-1-a"),
              "b": _cand(tmp_path / "wt" / "t-1-b")}
    assert review_cwd(ctx, passed) == str(integration)


def test_falls_back_to_the_repo_when_integration_does_not_exist_yet(tmp_path):
    """`_integration` is created lazily on the first merge. A cwd that does not
    exist would make the provider subprocess fail outright, so the stale-but-real
    checkout is the last resort."""
    ctx = _ctx(tmp_path)
    passed = {"a": _cand(tmp_path / "wt" / "t-1-a"),
              "b": _cand(tmp_path / "wt" / "t-1-b")}
    assert review_cwd(ctx, passed) == str(ctx.cfg.repo_path())


def test_candidate_without_a_recorded_worktree_does_not_break_the_call(tmp_path):
    ctx = _ctx(tmp_path)
    assert review_cwd(ctx, {"c0": _cand()}) == str(ctx.cfg.repo_path())


def test_no_git_service_still_resolves(tmp_path):
    """`review` is also exercised in dry-run/test contexts where git is absent."""
    ctx = _ctx(tmp_path, git=False)
    assert review_cwd(ctx, {}) == str(ctx.cfg.repo_path())


def test_a_recorded_worktree_that_is_gone_falls_back(tmp_path):
    """A path in a checkpoint is not proof of a directory on disk — finalize
    removes worktrees, and a resumed run replays state that outlived them. A cwd
    that does not exist kills the provider subprocess outright, which is the
    whole reason this function has a last resort."""
    ctx = _ctx(tmp_path)
    gone = tmp_path / "wt" / "t-1-c0"                      # never created
    assert review_cwd(ctx, {"c0": _cand(gone)}) == str(ctx.cfg.repo_path())


def test_the_integration_fallback_is_preferred_when_it_exists(tmp_path):
    """Same case, but with `_integration` present: the shared base beats a stale
    primary checkout."""
    ctx = _ctx(tmp_path)
    (tmp_path / "wt" / "_integration").mkdir()
    gone = tmp_path / "wt" / "t-1-c0"
    assert review_cwd(ctx, {"c0": _cand(gone)}) == str(tmp_path / "wt" / "_integration")
