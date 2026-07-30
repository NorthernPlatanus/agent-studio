"""Backlog adapter: the bridge between the human-readable backlog markdown
and the SQLite task store.

Default adapter: a markdown checklist (`- [ ] **T-101** title ...`). The item
regex and status characters are config-driven, so any checklist-style backlog
works without code changes. Other formats (GitHub issues, Linear, ...) can be
added as new adapters implementing the same three methods.

Writeback is deliberately conservative: only the status character inside
`[ ]` is flipped; the human's text is never rewritten.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


# A sub-task id: a backlog-shaped id plus a suffix that starts with a letter
# (`T-131a`, `T-131-b`) or a separated digit (`T-131.1`). Digits alone never
# split, so `T-1311` is its own item and not a child of `T-131`.
SUB_ID_RE = re.compile(r"^(?P<parent>[A-Za-z]+-\d+)(?:[.\-_]?[A-Za-z]|[.\-_]\d)"
                       r"[A-Za-z0-9]*$")


def parent_id(task_id: str) -> str | None:
    """The backlog item a decomposed sub-task belongs to, or None.

    When the planner splits backlog item **T-131** into **T-131a** and **T-131b**,
    neither sub-id has a line on the board: `set_status` matches ids exactly, finds
    nothing, and returns False — which the caller used to discard, so a merged
    task left no trace anywhere in the human's source of truth. This is the
    fallback target. Derivation is purely syntactic on purpose: it needs no LLM
    cooperation, and a spec's explicit `parent_id` (which the planner may write)
    takes precedence over it at the call site."""
    m = SUB_ID_RE.match((task_id or "").strip())
    return m.group("parent") if m else None


@dataclass
class BacklogItem:
    id: str
    title: str
    status: str          # todo | in_progress | done | blocked
    milestone: str | None
    line_no: int
    raw: str


class MarkdownChecklistBacklog:
    def __init__(self, path: Path, item_pattern: str, status_chars: dict[str, str]):
        self.path = path
        self.item_re = re.compile(item_pattern)
        self.status_chars = status_chars
        self.char_to_status = {v: k for k, v in status_chars.items()}
        self.heading_re = re.compile(r"^#{2,3}\s+(.*)$")

    def parse(self) -> list[BacklogItem]:
        items: list[BacklogItem] = []
        milestone = None
        for i, line in enumerate(self.path.read_text().splitlines()):
            heading = self.heading_re.match(line)
            if heading:
                m = re.search(r"\b(M\d+)\b", heading.group(1))
                if m:
                    milestone = m.group(1)
            m = self.item_re.match(line)
            if m:
                items.append(BacklogItem(
                    id=m.group("id"),
                    title=m.group("title").strip(),
                    status=self.char_to_status.get(m.group("status"), "todo"),
                    milestone=milestone,
                    line_no=i,
                    raw=line,
                ))
        return items

    def set_status(self, task_id: str, status: str, note: str | None = None) -> bool:
        """Flip the checkbox char for task_id; optionally append an agent note
        line right below the item. Returns False if the item wasn't found."""
        return self._update(task_id, self.status_chars[status], note)

    def annotate(self, task_id: str, note: str) -> bool:
        """Append an agent note under an item WITHOUT touching its checkbox.

        For the decomposition case: when one of several sub-tasks of a backlog item
        finishes, the parent has news worth recording but is NOT done — and the
        checkbox is the human's, so it stays as they left it until every child
        lands."""
        return self._update(task_id, None, note)

    def _update(self, task_id: str, char: str | None, note: str | None) -> bool:
        lines = self.path.read_text().splitlines(keepends=False)
        for i, line in enumerate(lines):
            m = self.item_re.match(line)
            if m and m.group("id") == task_id:
                if char is not None:
                    prefix = line[: line.index("[") + 1]
                    suffix = line[line.index("[") + 2:]
                    lines[i] = prefix + char + suffix
                if note:
                    indent = " " * (len(line) - len(line.lstrip()))
                    lines.insert(i + 1, f"{indent}  - **Agent:** {note}")
                self.path.write_text("\n".join(lines) + "\n")
                return True
        return False


def make_backlog(cfg) -> MarkdownChecklistBacklog:
    adapter = cfg.backlog.adapter
    if adapter != "markdown_checklist":
        raise ValueError(f"Unknown backlog adapter: {adapter}")
    return MarkdownChecklistBacklog(
        path=cfg.repo_path() / cfg.project.backlog_file,
        item_pattern=cfg.backlog.item_pattern,
        status_chars=cfg.backlog.status_chars.as_dict(),
    )
