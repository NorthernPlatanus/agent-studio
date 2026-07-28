"""Item 9: concurrent git worktree/branch mutations must be serialized.

`git` fails fast on `.git/index.lock` rather than waiting, so with
max_parallel_tasks x n_candidates in flight, unserialized `worktree add` calls
error out and can leave `*.lock` debris behind (this repo carried two broken
`*.lock.bak` refs from exactly that). Marked slow: it shells out to real git.
"""

import asyncio
import subprocess

import pytest

from orchestrator.ops.gitops import Git

pytestmark = pytest.mark.slow


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    run = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True,
                                    capture_output=True)
    run("init", "-q", "-b", "main")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    (repo / "README.md").write_text("x\n")
    run("add", "-A")
    run("commit", "-qm", "init")
    return repo


async def test_eight_concurrent_worktrees_all_succeed(tmp_path):
    repo = _repo(tmp_path)
    git = Git(repo, tmp_path / "wt", "agents/feature", "main")
    git.ensure_feature_branch()

    paths = await asyncio.gather(
        *(git.acreate_worktree(f"t-{i}") for i in range(8)))

    assert len(set(paths)) == 8
    assert all(p.exists() for p in paths)
    listed = subprocess.run(["git", "worktree", "list"], cwd=repo,
                            capture_output=True, text=True).stdout
    for i in range(8):
        assert f"t-{i}" in listed

    # No lock debris, and every ref is readable (a broken ref makes `git branch`
    # print "warning: ignoring broken ref").
    branches = subprocess.run(["git", "branch"], cwd=repo,
                              capture_output=True, text=True)
    assert "broken ref" not in branches.stderr
    assert not list((repo / ".git" / "refs").rglob("*.lock"))

    await asyncio.gather(*(git.aremove_worktree(f"t-{i}") for i in range(8)))


async def test_repeated_recreate_rounds_do_not_race(tmp_path):
    """The shape that actually breaks: retries re-create an existing worktree, so
    `create_worktree` runs remove + `branch -D` + `worktree add` while siblings do
    the same. Unserialized, this fails within a few rounds with errors like
    `fatal: failed to read .git/worktrees/<other>/commondir` — one candidate's
    prune racing another's add."""
    repo = _repo(tmp_path)
    git = Git(repo, tmp_path / "wt", "agents/feature", "main")
    git.ensure_feature_branch()

    for rnd in range(6):
        await asyncio.gather(
            *(git.acreate_worktree(f"t-{i}") for i in range(12)),
            *(git.adelete_branch(git.wt_branch(f"gone-{rnd}-{i}")) for i in range(12)))

    branches = subprocess.run(["git", "branch"], cwd=repo,
                              capture_output=True, text=True)
    assert "broken ref" not in branches.stderr
    assert all((tmp_path / "wt" / f"t-{i}").exists() for i in range(12))
    assert not list((repo / ".git").rglob("*.lock"))


async def test_concurrent_commits_land_every_ref(tmp_path):
    """Every candidate reaches its commit at roughly the same moment (they all
    finish patching together), so each branch must end up with its own commit and
    no ref debris.

    Unlike the worktree cases above, this one passes with or without the lock —
    per-worktree indexes and distinct per-ref locks are enough. It guards the
    invariant rather than reproducing a break, so that routing commits through
    `acommit_all` cannot silently regress into lost or shared commits."""
    repo = _repo(tmp_path)
    git = Git(repo, tmp_path / "wt", "agents/feature", "main")
    git.ensure_feature_branch()
    n = 8
    paths = await asyncio.gather(*(git.acreate_worktree(f"c-{i}") for i in range(n)))
    for i, p in enumerate(paths):
        (p / f"f{i}.txt").write_text(f"{i}\n")

    shas = await asyncio.gather(
        *(git.acommit_all(p, f"wip {i}") for i, p in enumerate(paths)))

    assert len(set(shas)) == n            # every commit distinct and created
    for i in range(n):
        head = subprocess.run(
            ["git", "rev-parse", git.wt_branch(f"c-{i}")], cwd=repo,
            capture_output=True, text=True)
        assert head.returncode == 0 and head.stdout.strip() == shas[i]

    branches = subprocess.run(["git", "branch"], cwd=repo,
                              capture_output=True, text=True)
    assert "broken ref" not in branches.stderr
    assert not list((repo / ".git" / "refs").rglob("*.lock"))
