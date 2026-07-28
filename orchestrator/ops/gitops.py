"""Git operations: feature branch, per-candidate worktrees, diffs, merges.

The orchestrator owns git. Workers never see the filesystem; each candidate
gets an isolated `git worktree` on its own branch, and the integrator merges
winning branches into the feature branch through a dedicated integration
worktree — the user's primary checkout is never touched.

In dry-run mode every mutating call raises DryRunViolation.

Concurrency: git serializes on `.git/index.lock`, and it does NOT wait — a second
mutating command against the same `.git` fails outright. With max_parallel_tasks
× n_candidates worktree creations plus the shared `_integration` worktree, that
is up to a dozen concurrent mutations, so the ref-touching entry points are
guarded by REPO_LOCK (see `Git.repo_lock`). Callers hop through
`asyncio.to_thread`, so the lock must be an asyncio lock held by the async
caller — a threading lock inside these sync methods would block the event loop.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from ..core.errors import DryRunViolation, OrchestratorError


class Git:
    def __init__(self, repo: Path, work_dir: Path, feature_branch: str,
                 base_branch: str, dry_run: bool = False):
        self.repo = repo
        self.work_dir = work_dir
        self.feature = feature_branch
        self.base = base_branch
        self.dry_run = dry_run
        self._lock: asyncio.Lock | None = None

    # ---- concurrency -------------------------------------------------------
    @property
    def repo_lock(self) -> asyncio.Lock:
        """Serializes ref-mutating git commands against this repo. Created lazily
        so it binds to the running loop (one Git instance == one run == one loop),
        never at import time."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def _locked(self, fn, *args):
        """Run a mutating git method under the repo lock, off the event loop.

        Prefer these async wrappers over `asyncio.to_thread(git.<method>)` at call
        sites: git's own locking (`.git/index.lock`) fails fast rather than
        waiting, so unserialized concurrent worktree/branch commands don't queue —
        they error, and can leave `*.lock` debris behind."""
        async with self.repo_lock:
            return await asyncio.to_thread(fn, *args)

    async def acreate_worktree(self, name: str, from_branch: str | None = None) -> Path:
        return await self._locked(self.create_worktree, name, from_branch)

    async def aremove_worktree(self, name: str) -> None:
        await self._locked(self.remove_worktree, name)

    async def amerge_into_feature(self, branch: str, message: str) -> str:
        return await self._locked(self.merge_into_feature, branch, message)

    async def adelete_branch(self, branch: str) -> None:
        await self._locked(self.delete_branch, branch)

    async def acommit_all(self, worktree: Path, message: str) -> str:
        """Commit a candidate worktree under the repo lock.

        HARDENING, not a fix for a demonstrated race: committing runs in a
        worktree, so the index is private and each candidate updates a DIFFERENT
        ref (`refs/heads/agents/wt/<name>`), which git locks per-ref. Attempts to
        reproduce a failure — 12 concurrent commits, and commits interleaved with
        `worktree prune`/`add` from sibling tasks — came back clean.

        It is still routed through the lock so that EVERY repo-mutating entry
        point has the same shape: `create_worktree` does race (see
        tests/test_gitops_concurrency.py), and the value of the `a*` wrappers is
        that no call site has to know which commands are the dangerous ones.
        Serializing three short commits costs nothing."""
        return await self._locked(self.commit_all, worktree, message)

    # ---- plumbing --------------------------------------------------------
    def _git(self, *args: str, cwd: Path | None = None, check: bool = True,
             mutating: bool = True) -> subprocess.CompletedProcess:
        if self.dry_run and mutating:
            raise DryRunViolation(f"git {' '.join(args)} attempted during --dry-run")
        proc = subprocess.run(
            ["git", *args], cwd=str(cwd or self.repo),
            capture_output=True, text=True)
        if check and proc.returncode != 0:
            raise OrchestratorError(
                f"git {' '.join(args)} failed:\n{proc.stderr.strip()}")
        return proc

    def _read(self, *args: str, cwd: Path | None = None) -> str:
        return self._git(*args, cwd=cwd, mutating=False).stdout.strip()

    # ---- branches ---------------------------------------------------------
    def branch_exists(self, name: str) -> bool:
        proc = self._git("rev-parse", "--verify", "--quiet", f"refs/heads/{name}",
                         check=False, mutating=False)
        return proc.returncode == 0

    def ensure_feature_branch(self) -> None:
        if not self.branch_exists(self.feature):
            self._git("branch", self.feature, self.base)

    # ---- worktrees ---------------------------------------------------------
    def create_worktree(self, name: str, from_branch: str | None = None) -> Path:
        """Create work_dir/<name> on a new branch agents/wt/<name>."""
        path = self.work_dir / name
        branch = f"agents/wt/{name}"
        if path.exists():
            self.remove_worktree(name)
        if self.branch_exists(branch):
            self._git("branch", "-D", branch)
        self._git("worktree", "add", "-b", branch, str(path),
                  from_branch or self.feature)
        return path

    def remove_worktree(self, name: str) -> None:
        path = self.work_dir / name
        self._git("worktree", "remove", "--force", str(path), check=False)
        self._git("worktree", "prune", check=False)

    def wt_branch(self, name: str) -> str:
        return f"agents/wt/{name}"

    # ---- diff / commit / merge ----------------------------------------------
    def commit_all(self, worktree: Path, message: str) -> str:
        self._git("add", "-A", cwd=worktree)
        proc = self._git("commit", "-m", message, cwd=worktree, check=False)
        if proc.returncode != 0 and "nothing to commit" not in proc.stdout + proc.stderr:
            raise OrchestratorError(f"commit failed: {proc.stderr.strip()}")
        return self._read("rev-parse", "HEAD", cwd=worktree)

    def diff_against_feature(self, worktree: Path) -> str:
        base = self._read("merge-base", self.feature, "HEAD", cwd=worktree)
        return self._read("diff", base, "HEAD", cwd=worktree)

    def uncommitted_diff(self, worktree: Path) -> str:
        self._git("add", "-N", ".", cwd=worktree)  # include untracked in diff
        return self._read("diff", cwd=worktree)

    def merge_into_feature(self, branch: str, message: str) -> str:
        """Merge a candidate branch into the feature branch via a dedicated
        integration worktree (never touching the user's checkout)."""
        integ = self.work_dir / "_integration"
        if not integ.exists():
            self._git("worktree", "add", str(integ), self.feature)
        self._git("checkout", self.feature, cwd=integ)
        proc = self._git("merge", "--no-ff", "-m", message, branch,
                         cwd=integ, check=False)
        if proc.returncode != 0:
            self._git("merge", "--abort", cwd=integ, check=False)
            raise OrchestratorError(
                f"merge of {branch} into {self.feature} conflicted:\n{proc.stdout}\n{proc.stderr}")
        return self._read("rev-parse", "HEAD", cwd=integ)

    def delete_branch(self, branch: str) -> None:
        self._git("branch", "-D", branch, check=False)
