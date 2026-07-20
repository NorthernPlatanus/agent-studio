from pathlib import Path

from orchestrator.ops.backlog import MarkdownChecklistBacklog

PATTERN = r'^\s*-\s*\[(?P<status>[ x~!])\]\s*\*\*(?P<id>[A-Z]+-\d+)\*\*\s*(?P<title>.+)$'
CHARS = {"todo": " ", "in_progress": "~", "done": "x", "blocked": "!"}

SAMPLE = """# Backlog

### M3 — Data Layer
- [x] **T-101** Load the config. — *Acceptance:* parses.
- [~] **T-102** Cache extractor.
- [!] **T-103** Blocked thing.

### M4 — Core Features
- [ ] **T-111** Run structure.
"""


def make(tmp_path: Path) -> MarkdownChecklistBacklog:
    p = tmp_path / "BACKLOG.md"
    p.write_text(SAMPLE)
    return MarkdownChecklistBacklog(p, PATTERN, CHARS)


def test_parse(tmp_path):
    items = make(tmp_path).parse()
    assert [(i.id, i.status, i.milestone) for i in items] == [
        ("T-101", "done", "M3"),
        ("T-102", "in_progress", "M3"),
        ("T-103", "blocked", "M3"),
        ("T-111", "todo", "M4"),
    ]


def test_set_status_flips_only_checkbox(tmp_path):
    backlog = make(tmp_path)
    assert backlog.set_status("T-111", "done", note="merged @ abc123")
    text = backlog.path.read_text()
    assert "- [x] **T-111** Run structure." in text
    assert "**Agent:** merged @ abc123" in text
    # everything else untouched
    assert "- [~] **T-102** Cache extractor." in text


def test_set_status_missing_id(tmp_path):
    assert make(tmp_path).set_status("T-999", "done") is False
