"""Defect-plan #2 item 5: the backlog writeback must not silently lose sub-tasks.

`set_status` matches ids exactly. When the planner splits backlog item **T-131**
into specs **T-131a** and **T-131b**, neither sub-id has a line, so the write
returned False, the caller discarded it, and NOTHING was written — no checkbox, no
`Agent:` note. Observed: T-131a merged at `4048bdf1eb` while the board still read
`[ ] T-131` with no trace of the run anywhere in the human's source of truth.
"""

import logging
from pathlib import Path
from types import SimpleNamespace

from orchestrator.core.config import Config
from orchestrator.nodes.integrator import finalize, write_back

PATTERN = r'^\s*-\s*\[(?P<status>[ x~!])\]\s*\*\*(?P<id>[A-Z]+-\d+)\*\*\s*(?P<title>.+)$'
CHARS = {"todo": " ", "in_progress": "~", "done": "x", "blocked": "!"}

BOARD = """# Backlog

### M4 — Core
- [ ] **T-131** Seed the chase camera at its settled pose.
- [ ] **T-140** Something else entirely.
"""


class _Store:
    def __init__(self, tasks):
        self.tasks = tasks              # id -> status
        self.events = []
        self.statuses = {}

    def all_tasks(self):
        return [{"id": tid, "status": st, **extra}
                for tid, (st, extra) in self.tasks.items()]

    def get_task(self, task_id):
        entry = self.tasks.get(task_id)
        return None if entry is None else {"id": task_id, "status": entry[0],
                                           **entry[1]}

    def set_task_status(self, task_id, status, retries=None):
        self.statuses[task_id] = status

    def task_cash_spend(self, task_id, run_id=None):
        return 0.0

    def log_event(self, run_id, task_id, kind, detail=""):
        self.events.append((kind, detail))


class _Git:
    feature = "agents/feature"

    async def aremove_worktree(self, name):
        pass

    async def adelete_branch(self, branch):
        pass


def _ctx(tmp_path, tasks):
    (tmp_path / "BACKLOG.md").write_text(BOARD)
    cfg = Config({
        "project": {"repo_path": str(tmp_path), "backlog_file": "BACKLOG.md"},
        "backlog": {"adapter": "markdown_checklist", "item_pattern": PATTERN,
                    "status_chars": CHARS},
        "projectmap": {"enabled": False},
    }, "p", tmp_path)
    return SimpleNamespace(cfg=cfg, store=_Store(tasks), git=_Git(), run_id="r")


def _board(ctx) -> str:
    return (Path(ctx.cfg.repo_path()) / "BACKLOG.md").read_text()


# ---- the direct path is unchanged -------------------------------------------

def test_an_exact_id_match_still_flips_and_notes(tmp_path):
    ctx = _ctx(tmp_path, {"T-140": ("done", {})})
    write_back(ctx, "T-140", {"id": "T-140"}, "done", "merged @ abc1234567")
    text = _board(ctx)
    assert "- [x] **T-140**" in text
    assert "**Agent:** merged @ abc1234567" in text


# ---- the decomposition fallback ---------------------------------------------

def test_a_finished_child_annotates_its_parent_without_completing_it(tmp_path):
    """T-131a is done, T-131b is not. The parent gets the news; its checkbox is the
    human's until every child lands."""
    ctx = _ctx(tmp_path, {"T-131a": ("done", {}), "T-131b": ("ready", {})})
    write_back(ctx, "T-131a", {"id": "T-131a"}, "done", "merged @ 4048bdf1eb")
    text = _board(ctx)
    assert "- [ ] **T-131**" in text                    # NOT flipped
    assert "T-131a done — merged @ 4048bdf1eb" in text
    assert "Still open: T-131b" in text


def test_the_parent_is_completed_once_every_child_is_done(tmp_path):
    ctx = _ctx(tmp_path, {"T-131a": ("done", {}), "T-131b": ("done", {})})
    write_back(ctx, "T-131b", {"id": "T-131b"}, "done", "merged @ cafebabe12")
    text = _board(ctx)
    assert "- [x] **T-131**" in text
    assert "all sub-tasks done (T-131a, T-131b)" in text


def test_an_explicit_parent_id_wins_over_the_derived_one(tmp_path):
    """The planner may name the parent, and ids it cannot be derived from ("split
    T-131 into T-500 and T-501") only work that way. Both the fallback target and
    the sibling grouping come from the field."""
    ctx = _ctx(tmp_path, {"T-500": ("done", {"parent_id": "T-131"}),
                          "T-501": ("ready", {"parent_id": "T-131"})})
    write_back(ctx, "T-500", {"id": "T-500", "parent_id": "T-131"}, "done", "merged")
    text = _board(ctx)
    assert "- [ ] **T-131**" in text                     # T-501 is still open
    assert "T-500 done — merged" in text
    assert "Still open: T-501" in text


def test_a_blocked_child_blocks_the_parent_item(tmp_path):
    """What would have happened without the decomposition: the human sees the item
    needs them, instead of an untouched checkbox and no note."""
    ctx = _ctx(tmp_path, {"T-131a": ("failed", {}), "T-131b": ("ready", {})})
    write_back(ctx, "T-131a", {"id": "T-131a"}, "blocked",
               "agent run failed — needs human")
    text = _board(ctx)
    assert "- [!] **T-131**" in text
    assert "T-131a: agent run failed — needs human" in text


def test_a_silent_no_op_is_logged_loudly(tmp_path, caplog):
    """The board disagreeing with the store is the worse half of this bug, so the
    unrecoverable case must at least be visible."""
    ctx = _ctx(tmp_path, {"T-777": ("done", {})})
    with caplog.at_level(logging.WARNING):
        write_back(ctx, "T-777", {"id": "T-777"}, "done", "merged")
    assert "no line for T-777" in caplog.text
    assert "no parent" in caplog.text


def test_an_unknown_parent_is_also_logged(tmp_path, caplog):
    ctx = _ctx(tmp_path, {"T-500a": ("done", {})})
    with caplog.at_level(logging.WARNING):
        write_back(ctx, "T-500a", {"id": "T-500a"}, "done", "merged")
    assert "parent T-500" in caplog.text


# ---- needs_human is a real outcome now (item 2's landing site) ---------------

async def test_finalize_records_an_unverifiable_task_as_needs_human(tmp_path):
    ctx = _ctx(tmp_path, {"T-140": ("running", {})})
    state = {"run_id": "r", "task_id": "T-140", "attempt": 1,
             "spec": {"id": "T-140", "title": "t"},
             "outcome": "needs_human",
             "blocked_reason": "scene verdict could not observe: the first frame",
             "candidates": [{"cand_id": "w", "attempt": 1,
                             "status": "visual_unverifiable",
                             "branch": "agents/wt/t-140-w"}]}
    out = await finalize(ctx, state)
    assert out["outcome"] == "needs_human"
    assert ctx.store.statuses["T-140"] == "needs_human"
    text = _board(ctx)
    assert "- [!] **T-140**" in text
    assert "could not observe: the first frame" in text


async def test_finalize_keeps_the_old_message_for_an_ordinary_failure(tmp_path):
    ctx = _ctx(tmp_path, {"T-140": ("running", {})})
    state = {"run_id": "r", "task_id": "T-140", "attempt": 2,
             "spec": {"id": "T-140", "title": "t"}, "candidates": []}
    out = await finalize(ctx, state)
    assert out["outcome"] == "failed"
    assert "agent run failed — needs human" in _board(ctx)
