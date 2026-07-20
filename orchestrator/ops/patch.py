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
SR_RE = re.compile(
    r"<{7} SEARCH\n(?P<search>.*?)\n?={7}\n(?P<replace>.*?)\n?>{7} REPLACE", re.S
)


@dataclass
class ParsedResponse:
    files: dict[str, str] = field(default_factory=dict)          # path -> full content
    edits: dict[str, list[tuple[str, str]]] = field(default_factory=dict)  # path -> [(search, replace)]
    need_files: list[str] = field(default_factory=list)
    plan: str = ""

    @property
    def touched_paths(self) -> set[str]:
        return set(self.files) | set(self.edits)

    @property
    def is_empty(self) -> bool:
        return not self.files and not self.edits and not self.need_files


def parse_worker_response(text: str) -> ParsedResponse:
    out = ParsedResponse()
    need = NEED_RE.search(text)
    if need:
        out.need_files = [ln.strip() for ln in need.group("body").splitlines() if ln.strip()]
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
    return (root / p).resolve()


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

    written = []
    for target, content in pending.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        if content and not content.endswith("\n"):
            content += "\n"
        target.write_text(content)
        written.append(str(target.relative_to(root)))
    return written
