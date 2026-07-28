"""Deterministic, read-only retrieval executors for the worker loop.

Workers stay context-starved and have NO tools/shell/network. Instead they may
emit plain-text read-only request blocks (<grep>/<read>/<ls>) which the
orchestrator executes here, in the candidate worktree, and pastes back into the
volatile suffix of the next prompt. Everything is:

  * read-only — never writes, never runs worker-supplied shell. ripgrep args are
    built BY US; the worker supplies only the pattern (length-capped, passed as a
    literal -e argument so a leading '-' can't become a flag).
  * path-safe — reads reuse the same rule as worker._read_task_file (relative,
    no '..', scoped to the worktree, with project.untracked_doc_prefixes read
    from the primary checkout). This is the READ allowlist, deliberately NOT
    ops/patch._safe_path (that is the WRITE allowlist and would reject legit
    reads).
  * bounded — match/snippet caps here; the round budget lives in the worker loop.

Pure and side-effect-free except for spawning ripgrep as a read-only subprocess.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

MAX_PATTERN_LEN = 200
_GREP_TIMEOUT_S = 20


def resolve_read_path(repo_path: Path, worktree: Path,
                      untracked_doc_prefixes, rel: str) -> Path | None:
    """The shared read-safety rule. Returns a resolved path inside the tree, or
    None if the request is unsafe/escaping. Code comes from the worktree; paths
    under untracked_doc_prefixes come from the primary checkout."""
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        return None
    prefixes = untracked_doc_prefixes or []
    root = repo_path if any(rel.startswith(pre) for pre in prefixes) else worktree
    return root / rel


def read_file(path: Path | None, max_bytes: int) -> tuple[str | None, str]:
    """Return (contents, status). status ∈ {ok, missing, too_large, unreadable}."""
    if path is None:
        return None, "missing"
    if not path.exists() or not path.is_file():
        return None, "missing"
    try:
        text = path.read_text()
    except (UnicodeDecodeError, OSError):
        return None, "unreadable"
    if len(text) > max_bytes:
        return None, "too_large"
    return text, "ok"


def ls_dir(path: Path | None) -> list[str] | None:
    """Directory listing, names only (dirs suffixed with '/'). None if invalid."""
    if path is None or not path.exists() or not path.is_dir():
        return None
    out = []
    for child in sorted(path.iterdir()):
        out.append(child.name + ("/" if child.is_dir() else ""))
    return out


def grep(worktree: Path, pattern: str, *, max_matches: int,
         snippet_lines: int) -> tuple[list[str], str]:
    """Read-only ripgrep in the worktree. Returns (matches, status).

    matches are 'path:line: text' strings, capped at max_matches. The pattern is
    validated (length cap) and passed as a literal -e argument — never through a
    shell — so the worker cannot inject flags or commands. status ∈
    {ok, empty, pattern_too_long, no_backend}.

    `empty` means "searched, found nothing" and `no_backend` means "could not
    search". Keeping those apart matters: a worker told `empty` concludes the
    symbol does not exist and plans around that, so a backend failure reported as
    `empty` is a lie that costs an attempt. Both backends distinguish the two by
    exit code (0 = matches, 1 = none, >1 = real error), so we do too.
    """
    if len(pattern) > MAX_PATTERN_LEN:
        return [], "pattern_too_long"

    rg = shutil.which("rg")
    if rg:
        args = [rg, "--line-number", "--no-heading", "--color", "never",
                "-e", pattern, "."]
    else:
        # Fallback: git grep (also read-only, args built by us). `rg` is often
        # absent — note a shell function/alias does NOT satisfy shutil.which — so
        # this path is load-bearing, not theoretical.
        git = shutil.which("git")
        if not git:
            return [], "no_backend"
        # --untracked: git grep defaults to TRACKED files only, which would hide
        # the files the worker itself just wrote in this attempt (nothing is
        # committed until after the patch applies). Ignored paths stay excluded,
        # so node_modules/ and build output don't drown the results.
        args = [git, "grep", "-n", "--no-color", "--untracked", "-e", pattern]

    try:
        proc = subprocess.run(args, cwd=str(worktree), capture_output=True,
                              text=True, timeout=_GREP_TIMEOUT_S)
    except (subprocess.TimeoutExpired, OSError):
        return [], "no_backend"
    if proc.returncode > 1:
        # Not "no matches" (that is exit 1) — the backend itself failed: git grep
        # outside a repo, an unreadable tree, a bad regex. Say so.
        return [], "no_backend"

    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    matches: list[str] = []
    for ln in lines[:max_matches]:
        # ripgrep: path:line:text ; git grep: path:line:text — normalize snippet
        parts = ln.split(":", 2)
        if len(parts) == 3:
            path, line, text = parts
            snippet = text.strip()[: snippet_lines * 200]
            matches.append(f"{path}:{line}: {snippet}")
        else:
            matches.append(ln[: snippet_lines * 200])
    return matches, ("ok" if matches else "empty")
