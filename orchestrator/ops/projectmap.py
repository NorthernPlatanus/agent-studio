"""Project-map — a cheap, always-fresh structural index for the planner.

Two layers:
  * Structural (free, deterministic): the tracked-file tree (respecting
    .gitignore via `git ls-files`) plus a symbol/export index built with
    ripgrep-style regexes. Zero tokens.
  * Prose digest (optional, cheap): a short architecture summary regenerated
    only when the symbol index materially changes (detected via a hash). Gated
    behind projectmap.digest and an injected digest_fn — the integrator does NOT
    pass one, so the default path spends nothing.

The map is written into the gitignored projects/<name>/ overlay, never the
target repo. The integrator regenerates it after a successful merge, scanning
the `_integration` worktree (already pinned to the feature branch).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import subprocess
from pathlib import Path

log = logging.getLogger("orchestrator.projectmap")

# Language-agnostic definition/export patterns (TS/JS/Py/Rust/Go...). Each must
# expose the symbol name as group 1.
DEFAULT_SYMBOL_PATTERNS = [
    r"^\s*export\s+(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_]\w*)",
    r"^\s*export\s+(?:abstract\s+)?class\s+([A-Za-z_]\w*)",
    r"^\s*export\s+(?:interface|type|enum|const)\s+([A-Za-z_]\w*)",
    r"^\s*(?:async\s+)?function\s+([A-Za-z_]\w*)",
    r"^\s*class\s+([A-Za-z_]\w*)",
    r"^\s*def\s+([A-Za-z_]\w*)",
    r"^\s*(?:pub\s+)?fn\s+([A-Za-z_]\w*)",
    r"^\s*type\s+([A-Za-z_]\w*)\s+struct",
]

_TEXT_EXT = {".ts", ".tsx", ".js", ".jsx", ".py", ".rs", ".go", ".java",
             ".c", ".h", ".cpp", ".hpp", ".rb", ".mjs", ".cjs", ".svelte", ".vue"}
_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist",
              "build", ".pytest_cache", ".mypy_cache", "target"}

_lock = asyncio.Lock()   # serialize concurrent finalizes racing on the map file


def list_files(root: Path) -> list[str]:
    """Tracked files (relative, sorted). Prefer `git ls-files` so .gitignore is
    respected; fall back to a walk that skips common junk dirs."""
    try:
        proc = subprocess.run(["git", "ls-files"], cwd=str(root),
                              capture_output=True, text=True, timeout=30)
        if proc.returncode == 0 and proc.stdout.strip():
            return sorted(ln for ln in proc.stdout.splitlines() if ln.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            rel = os.path.relpath(os.path.join(dirpath, fn), root)
            out.append(rel.replace(os.sep, "/"))
    return sorted(out)


def symbol_index(root: Path, files: list[str],
                 patterns: list[str] | None = None) -> dict[str, list[str]]:
    """module path -> sorted unique symbol names. Deterministic."""
    compiled = [re.compile(p) for p in (patterns or DEFAULT_SYMBOL_PATTERNS)]
    index: dict[str, list[str]] = {}
    for rel in files:
        if Path(rel).suffix not in _TEXT_EXT:
            continue
        path = root / rel
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        found: set[str] = set()
        for line in text.splitlines():
            for rx in compiled:
                m = rx.match(line)
                if m:
                    found.add(m.group(1))
        if found:
            index[rel] = sorted(found)
    return index


def _tree_lines(files: list[str]) -> list[str]:
    """Render a compact indented tree from a sorted file list."""
    lines: list[str] = []
    prev: list[str] = []
    for rel in files:
        parts = rel.split("/")
        for depth, part in enumerate(parts):
            if depth < len(prev) and prev[depth] == part:
                continue
            lines.append("  " * depth + part + ("/" if depth < len(parts) - 1 else ""))
        prev = parts
    return lines


def render_map(files: list[str], symbols: dict[str, list[str]]) -> str:
    out = ["# Project map (auto-generated — do not edit)", "",
           "## Tree", "```"]
    out += _tree_lines(files)
    out += ["```", "", "## Module → exported symbols", ""]
    for rel in sorted(symbols):
        out.append(f"- `{rel}`: {', '.join(symbols[rel])}")
    return "\n".join(out) + "\n"


def structure_hash(symbols: dict[str, list[str]]) -> str:
    canon = ";".join(f"{k}:{','.join(symbols[k])}" for k in sorted(symbols))
    return hashlib.sha256(canon.encode()).hexdigest()


def map_path(cfg) -> Path:
    pm = cfg.get("projectmap") or {}
    name = pm.get("path", "projectmap.md") if hasattr(pm, "get") else "projectmap.md"
    return cfg.root / "projects" / cfg.project_name / name


def regenerate(cfg, root: Path, out_path: Path,
               patterns: list[str] | None = None, digest_fn=None) -> str:
    """Write the structural map to out_path. Returns the structure hash. If the
    hash is unchanged since last time, the map is still rewritten (cheap) but the
    optional prose digest is skipped."""
    files = list_files(root)
    symbols = symbol_index(root, files, patterns)
    new_hash = structure_hash(symbols)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    hash_file = out_path.with_suffix(out_path.suffix + ".hash")
    old_hash = hash_file.read_text().strip() if hash_file.exists() else ""

    md = render_map(files, symbols)
    pm = cfg.get("projectmap") or {}
    want_digest = bool(pm.get("digest", False)) if hasattr(pm, "get") else False
    if want_digest and digest_fn is not None and new_hash != old_hash:
        try:
            digest = digest_fn(files, symbols)
            if digest:
                md += f"\n## Architecture digest\n\n{digest}\n"
        except Exception as e:  # digest must never break map generation
            log.warning("projectmap digest failed: %s", e)

    out_path.write_text(md)
    hash_file.write_text(new_hash)
    return new_hash


async def regenerate_from_integration(ctx) -> None:
    """Integrator hook: regenerate from the `_integration` worktree (pinned to
    the feature branch) after a successful merge. Serialized so concurrent
    finalizes don't race on the map file. Best-effort — never raises."""
    pm = ctx.cfg.get("projectmap") or {}
    if hasattr(pm, "get") and not pm.get("enabled", True):
        return
    integ = ctx.git.work_dir / "_integration"
    if not integ.exists():
        return
    patterns = None
    async with _lock:
        try:
            await asyncio.to_thread(regenerate, ctx.cfg, integ, map_path(ctx.cfg), patterns)
        except Exception as e:
            log.warning("projectmap regeneration failed: %s", e)
