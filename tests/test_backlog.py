from pathlib import Path

from orchestrator.ops.backlog import MarkdownChecklistBacklog, parent_id

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


# ---- decomposed sub-tasks (defect-plan #2 item 5) ---------------------------

def test_parent_id_derivation():
    """The planner splits backlog item T-131 into T-131a/T-131b; neither sub-id has
    a line on the board, so writeback needs a fallback target."""
    assert parent_id("T-131a") == "T-131"
    assert parent_id("T-131-b") == "T-131"
    assert parent_id("T-131.1") == "T-131"
    assert parent_id("ABC-12x") == "ABC-12"


def test_parent_id_is_none_for_ordinary_ids():
    """Digits alone must never split, or T-1311 becomes a phantom child of T-131
    and its writeback lands on the wrong item."""
    assert parent_id("T-131") is None
    assert parent_id("T-1311") is None
    assert parent_id("") is None
    assert parent_id("nonsense") is None


def test_annotate_leaves_the_checkbox_alone(tmp_path):
    """A parent with one child done has news worth recording but is not done, and
    the checkbox is the human's."""
    backlog = make(tmp_path)
    assert backlog.annotate("T-111", "T-111a merged; T-111b still open")
    text = backlog.path.read_text()
    assert "- [ ] **T-111** Run structure." in text          # unflipped
    assert "**Agent:** T-111a merged; T-111b still open" in text


def test_annotate_missing_id(tmp_path):
    assert make(tmp_path).annotate("T-999", "note") is False
