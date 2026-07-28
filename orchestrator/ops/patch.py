"""Worker output parsing + patch application.

The worker protocol (see prompts/worker_system.md) allows three block types:

  <file path="...">whole file content</file>
  <edit path="...">
  <<<<<<< SEARCH / ======= / >>>>>>> REPLACE  (aider-style, exact match)
  </edit>
  <need_files> one path per line </need_files>

Pure functions; no git, no network. Path safety: every path must be relative,
inside the tree, and within the task's files_write allowlist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ..core.errors import PatchError

FILE_RE = re.compile(r'<file\s+path="(?P<path>[^"]+)"\s*>\n?(?P<body>.*?)</file>', re.S)
EDIT_RE = re.compile(r'<edit\s+path="(?P<path>[^"]+)"\s*>\n?(?P<body>.*?)</edit>', re.S)
NEED_RE = re.compile(r"<need_files>\n?(?P<body>.*?)</need_files>", re.S)
# Phase 2 read-only retrieval blocks (executed by the orchestrator, never applied).
READ_RE = re.compile(r"<read>\n?(?P<body>.*?)</read>", re.S)
GREP_RE = re.compile(r"<grep>\n?(?P<body>.*?)</grep>", re.S)
LS_RE = re.compile(r"<ls>\n?(?P<body>.*?)</ls>", re.S)
SR_RE = re.compile(
    r"<{7} SEARCH\n(?P<search>.*?)\n?={7}\n(?P<replace>.*?)\n?>{7} REPLACE", re.S
)


@dataclass
class ParsedResponse:
    files: dict[str, str] = field(default_factory=dict)          # path -> full content
    edits: dict[str, list[tuple[str, str]]] = field(default_factory=dict)  # path -> [(search, replace)]
    need_files: list[str] = field(default_factory=list)          # <read> + <need_files> (file-content requests)
    grep: list[str] = field(default_factory=list)                # <grep> patterns
    ls: list[str] = field(default_factory=list)                  # <ls> dirs
    plan: str = ""

    @property
    def touched_paths(self) -> set[str]:
        return set(self.files) | set(self.edits)

    @property
    def has_retrieval(self) -> bool:
        return bool(self.need_files or self.grep or self.ls)

    @property
    def is_empty(self) -> bool:
        return not self.files and not self.edits and not self.has_retrieval


def _lines(body: str) -> list[str]:
    return [ln.strip() for ln in body.splitlines() if ln.strip()]


def parse_worker_response(text: str) -> ParsedResponse:
    out = ParsedResponse()
    # <read> generalizes <need_files>; both feed the file-content request list.
    for m in READ_RE.finditer(text):
        out.need_files.extend(_lines(m.group("body")))
    need = NEED_RE.search(text)
    if need:
        out.need_files.extend(_lines(need.group("body")))
    for m in GREP_RE.finditer(text):
        out.grep.extend(_lines(m.group("body")))
    for m in LS_RE.finditer(text):
        out.ls.extend(_lines(m.group("body")))
    for m in FILE_RE.finditer(text):
        out.files[m.group("path").strip()] = m.group("body")
    for m in EDIT_RE.finditer(text):
        path = m.group("path").strip()
        pairs = [(s.group("search"), s.group("replace")) for s in SR_RE.finditer(m.group("body"))]
        if not pairs:
            raise PatchError(f"<edit> block for {path} contains no valid SEARCH/REPLACE pair")
        out.edits.setdefault(path, []).extend(pairs)
    out.plan = text[: text.find("<")].strip() if "<" in text else text.strip()
    return out


def _safe_path(root: Path, rel: str, allowed: set[str] | None) -> Path:
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        raise PatchError(f"Unsafe path: {rel}")
    if allowed is not None and rel not in allowed:
        raise PatchError(f"Path not in files_write allowlist: {rel}")
    resolved = (root / p).resolve()
    # The lexical checks above are not sufficient: `resolve()` follows symlinks,
    # so a symlinked directory anywhere in the worktree (`link -> /tmp/x`) turns
    # the perfectly relative, ..-free `link/pwned.txt` into a write outside the
    # tree. Compare the RESOLVED path against the resolved root — the worktree is
    # the sandbox boundary, and it holds whatever the repo happens to contain.
    if not resolved.is_relative_to(root.resolve()):
        raise PatchError(f"Path escapes the worktree (symlink?): {rel}")
    return resolved


def apply_response(root: Path, parsed: ParsedResponse,
                   files_write: list[str] | None = None) -> list[str]:
    """Apply parsed blocks under root. Returns list of written paths.
    Raises PatchError on any mismatch — nothing is partially applied first
    (all edits are computed in memory before any write)."""
    allowed = set(files_write) if files_write is not None else None
    pending: dict[Path, str] = {}

    for rel, content in parsed.files.items():
        pending[_safe_path(root, rel, allowed)] = content

    for rel, pairs in parsed.edits.items():
        target = _safe_path(root, rel, allowed)
        if target in pending:
            text = pending[target]
        elif target.exists():
            text = target.read_text()
        else:
            raise PatchError(f"<edit> targets missing file: {rel}")
        for search, replace in pairs:
            count = text.count(search)
            if count == 0:
                raise PatchError(
                    f"SEARCH block not found in {rel} (must match exactly):\n"
                    f"---\n{search[:500]}\n---")
            if count > 1:
                raise PatchError(f"SEARCH block is not unique in {rel} ({count} matches)")
            text = text.replace(search, replace, 1)
        pending[target] = text

    # Compute every relative path BEFORE the first write: `relative_to` raising
    # after a write leaves the file on disk and the bare ValueError escapes as an
    # unhandled crash (it is not an OrchestratorError, so the worker's PatchError
    # handler never sees it). Anything that can fail must fail before we touch
    # the filesystem, and it must fail as a PatchError.
    root_resolved = root.resolve()
    writes: list[tuple[Path, str, str]] = []
    for target, content in pending.items():
        try:
            rel = str(target.relative_to(root_resolved))
        except ValueError as e:
            raise PatchError(f"Path escapes the worktree: {target}") from e
        writes.append((target, content, rel))

    for target, content, _rel in writes:
        target.parent.mkdir(parents=True, exist_ok=True)
        if content and not content.endswith("\n"):
            content += "\n"
        target.write_text(content)
    return [rel for _t, _c, rel in writes]
