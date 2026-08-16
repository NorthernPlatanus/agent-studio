"""The planner chat, driven end to end against a scripted planner.

No provider is ever called: `nodes.discuss.plan_or_ask` is replaced with a
function that returns pre-written envelopes, which is the same seam
`tests/test_discuss.py` uses for the CLI loop. What is under test here is the
adapter — frames, the replay cursor, the reply handshake, live settings, pinning,
and the mutual exclusion with jobs.
"""

from __future__ import annotations

import asyncio
import copy
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from orchestrator.api import deps
from orchestrator.api import discuss as discuss_mod
from orchestrator.api import jobs as api_jobs
from orchestrator.api.app import create_app
from orchestrator.api.discuss import DiscussManager, get_manager
from orchestrator.api.jobs import JobSupervisor
from orchestrator.core.config import Config, load_config
from orchestrator.core.errors import OrchestratorError
from orchestrator.nodes import discuss as discuss_node
from tests.api.fixtures import seed_store

PROJECT = seed_store.PROJECT

QUESTION = {"questions": [{"id": "q1", "q": "which store?", "why": "changes files"}],
            "assumptions": ["the switcher reads the allowlist"], "specs": []}
PROPOSAL = {"questions": [], "assumptions": [],
            "specs": [{"id": "T-900", "title": "project switcher",
                       "description": "d", "files_write": ["src/switch.ts"]}]}


class ScriptedPlanner:
    """Stands in for `plan_or_ask`, recording what each turn was asked with."""

    def __init__(self, envelopes):
        self.envelopes = list(envelopes)
        self.calls = []

    async def __call__(self, ctx, **kwargs):
        self.calls.append(kwargs)
        env = self.envelopes[min(len(self.calls) - 1, len(self.envelopes) - 1)]
        return copy.deepcopy(env)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    checkout = tmp_path / "checkout"
    (checkout / "src").mkdir(parents=True)
    (checkout / "src" / "switch.ts").write_text("export const x = 1;\n")
    (checkout / "secret.txt").write_text("not in the checkout tree of interest\n")
    return checkout


@pytest.fixture
def cfg(tmp_path: Path, repo: Path) -> Config:
    """A config whose store and checkout are both throwaway.

    `example`'s own profile leaves `repo_path` null on purpose, and a discuss
    session needs a checkout (`runner.make_context` builds `Git(repo_path)`), so
    the test supplies one rather than reaching for the machine-global overlay.
    """
    base = load_config(PROJECT, root=deps.REPO_ROOT)
    data = copy.deepcopy(base.as_dict())
    data["paths"]["state_dir"] = str(tmp_path / "state")
    # `example`'s profile leaves both of these null on purpose (they are the two
    # REQUIRED per-machine values), and `make_context` needs both.
    data["paths"]["work_dir"] = str(tmp_path / "worktrees")
    data["project"]["repo_path"] = str(repo)
    Path(data["paths"]["state_dir"]).mkdir(parents=True, exist_ok=True)
    seed_store.seed(Path(data["paths"]["state_dir"]) / f"{PROJECT}.sqlite3")
    return Config(data, PROJECT, deps.REPO_ROOT)


@pytest.fixture
def manager() -> DiscussManager:
    return DiscussManager()


@pytest.fixture
def sup() -> JobSupervisor:
    """A supervisor of this test's own, never the process-wide singleton — the
    module-level one would carry jobs between tests."""
    return JobSupervisor()


@pytest.fixture
def harmless_argv(monkeypatch):
    """Any job this test spawns is a python that prints and exits.

    The mutual-exclusion tests need a *real* in-flight job record to collide
    with, and no test in this repo may start an orchestrator process.
    """
    import sys
    child = [sys.executable, "-c", "import time; time.sleep(30)"]
    for name in ("run_argv", "plan_argv", "resume_argv", "import_backlog_argv",
                 "reconcile_argv"):
        monkeypatch.setattr(api_jobs, name, lambda *a, **k: list(child))


@pytest.fixture
def client(cfg: Config, manager: DiscussManager, sup: JobSupervisor, monkeypatch):
    monkeypatch.setattr(discuss_node, "plan_or_ask",
                        ScriptedPlanner([QUESTION, PROPOSAL]))
    registry = deps.ProjectRegistry(root=Path(cfg.paths.state_dir),
                                    configs={PROJECT: cfg})
    app = create_app()
    app.dependency_overrides[deps.get_registry] = lambda: registry
    app.dependency_overrides[get_manager] = lambda: manager
    app.dependency_overrides[api_jobs.get_supervisor] = lambda: sup
    with TestClient(app) as test_client:
        yield test_client


BASE = f"/api/projects/{PROJECT}/discuss"


def _start(client: TestClient, **body) -> dict:
    payload = {"request": "add a project switcher", "confirm": True, **body}
    response = client.post(BASE, json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def _await_status(client: TestClient, session_id: str, wanted: str,
                  tries: int = 200) -> dict:
    """Poll until the loop reaches a state. The session is a task on the same
    loop as the app, so it only advances while a request is not in flight."""
    for _ in range(tries):
        state = client.get(BASE).json()
        session = state["session"]
        if session and session["status"] == wanted:
            return session
        if session and session["status"] in ("failed", "aborted", "done") \
                and wanted not in ("failed", "aborted", "done"):
            pytest.fail(f"session ended as {session['status']}: {session['error']}")
    pytest.fail(f"session never reached {wanted}")


# ---- the happy path ------------------------------------------------------
def test_a_session_asks_then_proposes_then_applies(client: TestClient):
    started = _start(client)
    session_id = started["session_id"]

    asking = _await_status(client, session_id, "awaiting")
    assert asking["expects"] == "answer"
    kinds = [f["kind"] for f in asking["frames"]]
    # The assumption and the question are structured frames, not parsed prose.
    assert "assumption" in kinds and "question" in kinds
    question = next(f for f in asking["frames"] if f["kind"] == "question")
    assert question["data"]["q"] == "which store?"

    client.post(f"{BASE}/{session_id}/reply", json={"text": "sqlite"})
    deciding = _await_status(client, session_id, "awaiting")
    assert deciding["expects"] == "decision"
    preview = next(f for f in deciding["frames"] if f["kind"] == "specs_preview")
    assert preview["data"]["specs"][0]["id"] == "T-900"

    client.post(f"{BASE}/{session_id}/reply", json={"text": "y"})
    done = _await_status(client, session_id, "done")
    assert [s["id"] for s in done["applied"]] == ["T-900"]

    # Applied means written: the spec is in the store the panel reads.
    tasks = client.get(f"/api/projects/{PROJECT}/tasks").json()["tasks"]
    assert any(t["id"] == "T-900" for t in tasks)


def test_abort_at_the_preview_writes_nothing(client: TestClient):
    session_id = _start(client)["session_id"]
    _await_status(client, session_id, "awaiting")
    client.post(f"{BASE}/{session_id}/reply", json={"text": "sqlite"})
    _await_status(client, session_id, "awaiting")
    client.post(f"{BASE}/{session_id}/reply", json={"text": "abort"})

    session = _await_status(client, session_id, "aborted")
    assert session["applied"] == []
    tasks = client.get(f"/api/projects/{PROJECT}/tasks").json()["tasks"]
    assert not any(t["id"] == "T-900" for t in tasks)


# ---- the handshake -------------------------------------------------------
def test_a_reply_with_no_question_pending_is_refused():
    """A reply is not queued: one typed while the planner is still working would
    silently become the answer to whatever it asks next.

    Asserted on the session object rather than over HTTP because the scripted
    planner returns instantly — by the time a second request lands, the loop is
    already `awaiting` again, so the race this guards can't be staged over the
    wire.
    """
    from orchestrator.api.discuss import DiscussError, Session

    session = Session(session_id="s", project=PROJECT, request="r", started_at=0.0)
    assert session.status == "running"
    with pytest.raises(DiscussError) as caught:
        session.reply("too early")
    assert caught.value.status_code == 409
    assert "no question is pending" in caught.value.detail

    session.push({"kind": "awaiting", "expects": "answer"})
    session.reply("now it lands")
    assert session.status == "running"


def test_frames_replay_from_a_cursor(client: TestClient):
    session_id = _start(client)["session_id"]
    session = _await_status(client, session_id, "awaiting")
    last = session["frames"][-1]["seq"]

    assert client.get(BASE, params={"since": last}).json()["session"]["frames"] == []
    replayed = client.get(BASE, params={"since": 0}).json()["session"]["frames"]
    assert [f["seq"] for f in replayed] == list(range(1, last + 1))


def test_starting_a_second_session_is_refused(client: TestClient):
    _start(client)
    response = client.post(BASE, json={"request": "another", "confirm": True})
    assert response.status_code == 409


def test_starting_without_confirm_is_422(client: TestClient):
    assert client.post(BASE, json={"request": "hi"}).status_code == 422


def test_unknown_session_is_404(client: TestClient):
    assert client.post(f"{BASE}/nope/reply", json={"text": "x"}).status_code == 404


# ---- settings and pins ---------------------------------------------------
def test_settings_reach_the_next_planner_turn(client: TestClient, manager, monkeypatch):
    planner = ScriptedPlanner([QUESTION, QUESTION, PROPOSAL])
    monkeypatch.setattr(discuss_node, "plan_or_ask", planner)

    session_id = _start(client)["session_id"]
    _await_status(client, session_id, "awaiting")
    assert planner.calls[0]["effort"] is None

    client.post(f"{BASE}/{session_id}/settings",
                json={"note": "keep tasks under a day", "effort": "high",
                      "max_question_rounds": 0})
    client.post(f"{BASE}/{session_id}/reply", json={"text": "sqlite"})
    _await_status(client, session_id, "awaiting")

    # Re-read at the top of the turn, so a change lands on the next call rather
    # than the next session.
    assert planner.calls[1]["effort"] == "high"
    assert planner.calls[1]["discussion"] == "keep tasks under a day"


def test_an_attached_file_reaches_the_first_turn(client: TestClient, monkeypatch):
    """Attachments travel in the create request, because the first turn is the
    one they are for — and it is started by that same call."""
    planner = ScriptedPlanner([QUESTION, PROPOSAL])
    monkeypatch.setattr(discuss_node, "plan_or_ask", planner)

    session = _start(client, uploads=[{"name": "notes from triage.md",
                                       "text": "the timeline jitters on load"}])
    pin = session["pins"][0]
    # Named under `uploaded/`, and sanitized: the name becomes a markdown heading
    # in a provider prompt.
    assert pin["path"] == "uploaded/notes-from-triage.md"

    _await_status(client, session["session_id"], "awaiting")
    transcript = planner.calls[0]["transcript"]
    assert "the timeline jitters on load" in transcript
    # And the planner is told not to go looking for a file that does not exist.
    assert "do not try to open these paths" in transcript


def test_an_attachment_can_be_added_and_removed_mid_session(client: TestClient,
                                                            monkeypatch):
    planner = ScriptedPlanner([QUESTION, PROPOSAL])
    monkeypatch.setattr(discuss_node, "plan_or_ask", planner)
    session_id = _start(client)["session_id"]
    _await_status(client, session_id, "awaiting")

    added = client.post(f"{BASE}/{session_id}/pins",
                        json={"name": "run.log", "text": "TypeError: undefined"})
    assert added.status_code == 200
    assert [p["path"] for p in added.json()["pins"]] == ["uploaded/run.log"]

    client.post(f"{BASE}/{session_id}/reply", json={"text": "sqlite"})
    _await_status(client, session_id, "awaiting")
    assert "TypeError: undefined" in planner.calls[1]["transcript"]

    removed = client.post(f"{BASE}/{session_id}/pins/remove",
                          json={"path": "uploaded/run.log"})
    assert removed.json()["pins"] == []


def test_an_image_is_refused_rather_than_pinned_as_filler(client: TestClient):
    """The planner prompt is text. A PNG decoded with replacement is a page of
    U+FFFD that spends context and says nothing, so it is refused with why."""
    session_id = _start(client)["session_id"]
    png = "\ufffd".join("PNG" * 40)          # what `File.text()` makes of a PNG

    response = client.post(f"{BASE}/{session_id}/pins",
                           json={"name": "screenshot.png", "text": png})
    assert response.status_code == 415
    assert "text only" in response.json()["detail"]

    nul = client.post(f"{BASE}/{session_id}/pins",
                      json={"name": "a.bin", "text": "head\x00tail"})
    assert nul.status_code == 415 and "binary" in nul.json()["detail"]


def test_a_binary_attachment_is_refused_before_the_first_turn(client: TestClient,
                                                              monkeypatch):
    """At create time too — a 415 after the billable turn has started is a bill
    for a turn the operator did not get the attachment into."""
    planner = ScriptedPlanner([QUESTION])
    monkeypatch.setattr(discuss_node, "plan_or_ask", planner)

    response = client.post(BASE, json={
        "request": "fix the camera", "confirm": True,
        "uploads": [{"name": "shot.png", "text": "\ufffd".join("PNG" * 40)}]})
    assert response.status_code == 415
    assert planner.calls == []


def test_a_huge_attachment_is_truncated_not_refused(client: TestClient):
    """Truncation is reported, because a silently half-read file is worse than
    no attachment at all."""
    session_id = _start(client)["session_id"]
    body = {"name": "big.txt", "text": "x" * (discuss_mod.MAX_PIN_BYTES + 500)}

    pin = client.post(f"{BASE}/{session_id}/pins", json=body).json()["pins"][0]
    assert pin["truncated"] is True
    assert pin["bytes"] == discuss_mod.MAX_PIN_BYTES


def test_an_attachment_name_cannot_escape_its_prefix(client: TestClient):
    """The name is filesystem-supplied and ends up in a prompt heading."""
    session_id = _start(client)["session_id"]
    for name, expected in (("../../etc/passwd", "uploaded/passwd"),
                           ("C:\\Users\\np\\notes.txt", "uploaded/notes.txt"),
                           ("../..", "uploaded/upload.txt"),
                           ("a`b\nc.md", "uploaded/a-b-c.md")):
        pin = client.post(f"{BASE}/{session_id}/pins",
                          json={"name": name, "text": "hi"}).json()["pins"][-1]
        assert pin["path"] == expected, (name, pin["path"])


def test_pinning_a_repo_path_is_gone(client: TestClient):
    """The old shape — `{"path": …}`, read off disk — must not still be accepted
    under the same URL now that the body type there has changed."""
    session_id = _start(client)["session_id"]
    response = client.post(f"{BASE}/{session_id}/pins",
                           json={"path": "../../../etc/passwd"})
    assert response.status_code == 422


def test_revising_a_proposal_sends_the_note_and_comes_back(client: TestClient,
                                                           monkeypatch):
    """The panel's "Revise" sends the note itself — there is no `[y/edit/abort]`
    prompt in a chat to type `edit` at.

    The loop used to read that as a *choice*, discard it, and block on a second
    read it announced to nobody: `reply` 409s anything sent while the status is
    `running`, so the session sat wedged until the 30-minute idle TTL closed it,
    with the operator's revision thrown away. One of three buttons on the
    decision, and it was a dead end.
    """
    planner = ScriptedPlanner([PROPOSAL, PROPOSAL])
    monkeypatch.setattr(discuss_node, "plan_or_ask", planner)

    session_id = _start(client)["session_id"]
    assert _await_status(client, session_id, "awaiting")["expects"] == "decision"

    assert client.post(f"{BASE}/{session_id}/reply",
                       json={"text": "split T-900 in two"}).status_code == 200

    again = _await_status(client, session_id, "awaiting")
    assert again["expects"] == "decision"
    assert len(planner.calls) == 2
    # Verbatim and marked as an edit, so the planner reads it as a steer rather
    # than as an answer to a question it did not ask.
    assert "EDIT: split T-900 in two" in planner.calls[1]["transcript"]

    # And the session is still answerable, which is what "wedged" cost.
    assert client.post(f"{BASE}/{session_id}/reply",
                       json={"text": "abort"}).status_code == 200


def test_max_question_rounds_forces_a_proposal(client: TestClient, monkeypatch):
    """A planner that would ask forever still has to put something on the table."""
    planner = ScriptedPlanner([QUESTION, {**PROPOSAL, "questions": QUESTION["questions"]}])
    monkeypatch.setattr(discuss_node, "plan_or_ask", planner)

    session_id = _start(client, settings={"max_question_rounds": 1})["session_id"]
    _await_status(client, session_id, "awaiting")
    client.post(f"{BASE}/{session_id}/reply", json={"text": "sqlite"})

    deciding = _await_status(client, session_id, "awaiting")
    assert deciding["expects"] == "decision"
    # The unanswered questions are reported rather than dropped on the floor.
    note = next(f for f in deciding["frames"] if f["kind"] == "note")
    assert "max_question_rounds" in note["data"]["text"]


# ---- mutual exclusion with jobs -----------------------------------------
def test_a_job_cannot_start_while_a_session_is_open(client: TestClient,
                                                   harmless_argv):
    _start(client)
    response = client.post(f"/api/projects/{PROJECT}/jobs/import-backlog", json={})
    assert response.status_code == 409
    assert "discuss session" in response.json()["detail"]


def test_the_state_endpoint_reports_what_is_available(client: TestClient):
    state = client.get(BASE).json()
    assert state["session"] is None
    options = state["options"]
    assert options["efforts"] == ["low", "medium", "high", "xhigh", "max"]
    assert options["configured_provider"]
    assert options["max_pin_bytes"] > 0


def test_closing_a_session_frees_the_project(client: TestClient, harmless_argv):
    session_id = _start(client)["session_id"]
    _await_status(client, session_id, "awaiting")
    client.delete(f"{BASE}/{session_id}")
    _await_status(client, session_id, "aborted")
    # The closed conversation stays on screen — what frees up is the project.
    assert client.get(BASE).json()["session"]["status"] == "aborted"
    # And a job may start again.
    assert client.post(f"/api/projects/{PROJECT}/jobs/import-backlog",
                       json={}).status_code != 409


async def test_shutdown_closes_a_live_session(cfg: Config, manager: DiscussManager,
                                              monkeypatch):
    """A session outliving the process is the zombie `reconcile` exists for."""
    monkeypatch.setattr(discuss_node, "plan_or_ask", ScriptedPlanner([QUESTION]))
    session = manager.start(cfg, "hello")
    for _ in range(200):
        if session.status == "awaiting":
            break
        await asyncio.sleep(0.01)
    await manager.shutdown()
    assert session.status == "aborted"


async def test_a_frame_pushed_during_the_replay_is_not_lost():
    """The stream's subscriber must be attached before the backlog is replayed.

    `subscribe` used to be an async generator, and a generator's body does not
    run until its first `__anext__` — so `live = session.subscribe()` followed by
    a replay loop had **no** subscriber registered for the whole replay. Every
    `yield` in that loop writes to the socket and suspends, so a planner frame
    landing there went to an empty subscriber list and was gone: nothing
    re-fetches the session on a timer, so a dropped `awaiting` leaves the
    composer disabled and the question unrendered until a page reload.

    Attaching first can only ever duplicate, and `seq` already dedupes that.
    """
    from orchestrator.api.discuss import Session

    session = Session(session_id="s", project=PROJECT, request="hi", started_at=0.0)
    session.push({"kind": "note", "text": "one"})

    queue = session.attach()          # before the snapshot, as the router does
    replayed = session.since(0)
    for _ in replayed:
        await asyncio.sleep(0)        # what yielding a frame to the socket costs
        session.push({"kind": "awaiting", "expects": "answer"})

    seen = [f.seq for f in replayed]
    while not queue.empty():
        frame = queue.get_nowait()
        if frame.seq > max(seen):
            seen.append(frame.seq)

    assert seen == [f.seq for f in session.frames], (
        "a frame pushed during the replay never reached the subscriber")
    session.detach(queue)
    assert session._subscribers == []


# ---- a failed turn stays in the conversation -------------------------------

class FailingThenFine:
    """Fails the first turn the way a wedged CLI does, then proposes."""

    def __init__(self, envelope):
        self.envelope = envelope
        self.calls = 0

    async def __call__(self, ctx, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise OrchestratorError("claude CLI produced no output for 300s")
        return copy.deepcopy(self.envelope)


def test_a_failed_turn_leaves_the_session_open_for_a_retry(
        client: TestClient, monkeypatch):
    """This is the reported bug end to end: the turn dies, and the session used
    to go `failed`/`closed`, discarding the conversation. It must instead offer
    another attempt over the same session."""
    planner = FailingThenFine(PROPOSAL)
    monkeypatch.setattr(discuss_node, "plan_or_ask", planner)

    session_id = _start(client)["session_id"]
    offered = _await_status(client, session_id, "awaiting")
    assert offered["expects"] == "retry"
    failed = next(f for f in offered["frames"] if f["kind"] == "turn_failed")
    assert "300s" in failed["data"]["text"]
    assert offered["status"] != "failed"          # still live, still ours

    client.post(f"{BASE}/{session_id}/reply", json={"text": ""})
    deciding = _await_status(client, session_id, "awaiting")
    assert deciding["expects"] == "decision"      # it recovered on the retry
    client.post(f"{BASE}/{session_id}/reply", json={"text": "y"})
    done = _await_status(client, session_id, "done")
    assert [s["id"] for s in done["applied"]] == ["T-900"]


def test_progress_frames_do_not_evict_the_conversation(client: TestClient,
                                                       monkeypatch):
    """Progress is live-only. Retaining hundreds per turn would push the
    operator's own messages out of the replay log."""
    async def chatty(ctx, *, on_progress=None, **kwargs):
        for i in range(50):
            on_progress({"phase": "tool", "tool": "Read", "target": f"f{i}.ts"})
        return copy.deepcopy(PROPOSAL)

    monkeypatch.setattr(discuss_node, "plan_or_ask", chatty)
    session_id = _start(client)["session_id"]
    deciding = _await_status(client, session_id, "awaiting")
    kinds = [f["kind"] for f in deciding["frames"]]
    assert kinds.count("progress") == 1           # only the latest is retained
    assert "you" in kinds and "specs_preview" in kinds
    latest = next(f for f in deciding["frames"] if f["kind"] == "progress")
    assert latest["data"]["target"] == "f49.ts"


def test_a_frozen_session_is_not_reaped_as_idle():
    """A five-hour usage window is ten times the idle TTL. Reaping a frozen
    session would abort exactly the sessions the freeze exists to save."""
    manager = DiscussManager(idle_ttl_s=0.01)
    session = discuss_mod.Session(session_id="s1", project=PROJECT, request="r",
                                  started_at=time.time())
    manager._sessions["s1"] = session
    session.push({"kind": "thinking"})
    session.push({"kind": "limit_paused", "resets_at": time.time() + 3600,
                  "seconds": 3600, "limit_type": "five_hour"})
    session.push({"kind": "awaiting", "expects": "frozen"})
    session.last_activity = time.time() - 3600          # long "idle"

    manager.reap()
    assert session.live, "the freeze was reaped as an idle tab"
    assert not session._aborting

    # And once the window has passed, it is an ordinary idle session again.
    # `_aborting` rather than `status`: tearing down signals the loop through
    # its own inbox, and the status flips when the loop unwinds — there is no
    # loop here, only the manager.
    session.frozen_until = time.time() - 1
    manager.reap()
    assert session._aborting


def test_the_freeze_clears_when_the_turn_resumes():
    session = discuss_mod.Session(session_id="s2", project=PROJECT, request="r",
                                  started_at=time.time())
    session.push({"kind": "limit_paused", "seconds": 3600})
    assert session.frozen_until is not None
    session.push({"kind": "thinking"})                  # the retry started
    assert session.frozen_until is None


# ---- surviving a restart -------------------------------------------------
#
# Sessions are process-local and always will be: `run_discuss` is an awaited
# coroutine holding a provider subprocess. What the panel loses when the API
# restarts is not the loop — that is genuinely gone — but the *conversation*,
# and that is only lost because it was never written down. These cover writing
# it down and reading it back.


def _restart(manager: DiscussManager) -> None:
    """A fresh process, as far as the manager is concerned: no sessions in
    memory, the store untouched."""
    manager._sessions.clear()


def test_the_conversation_comes_back_after_a_restart(client: TestClient, manager):
    session_id = _start(client)["session_id"]
    asked = _await_status(client, session_id, "awaiting")
    original = [(f["kind"], f["data"].get("text") or f["data"].get("q"))
                for f in asked["frames"] if f["kind"] in ("you", "question")]

    _restart(manager)
    session = client.get(BASE).json()["session"]
    assert session is not None, "the planner page went blank on a restart"
    assert session["session_id"] == session_id
    restored = [(f["kind"], f["data"].get("text") or f["data"].get("q"))
                for f in session["frames"] if f["kind"] in ("you", "question")]
    assert restored == original
    assert original, "nothing was persisted to come back"


def test_a_restored_session_is_read_only(client: TestClient, manager):
    """It comes back `aborted`, not `awaiting`. Reporting the status it was
    caught at would put a composer under a question with no loop behind it, and
    the answer would 404 on a session id that no longer exists."""
    session_id = _start(client)["session_id"]
    _await_status(client, session_id, "awaiting")

    _restart(manager)
    session = client.get(BASE).json()["session"]
    assert session["status"] == "aborted"
    assert session["expects"] is None
    # And the transcript says why, rather than the conversation just stopping.
    notes = [f["data"]["text"] for f in session["frames"] if f["kind"] == "note"]
    assert any("restarted" in note for note in notes)
    assert client.post(f"{BASE}/{session_id}/reply",
                       json={"text": "the default one"}).status_code == 404


def test_a_finished_session_keeps_its_own_status_across_a_restart(client, manager):
    session_id = _start(client)["session_id"]
    _await_status(client, session_id, "awaiting")
    client.post(f"{BASE}/{session_id}/reply", json={"text": "the default one"})
    _await_status(client, session_id, "awaiting")
    client.post(f"{BASE}/{session_id}/reply", json={"text": "y"})
    _await_status(client, session_id, "done")

    _restart(manager)
    session = client.get(BASE).json()["session"]
    assert session["status"] == "done"
    assert [s["id"] for s in session["applied"]] == ["T-900"]
    # Nothing was interrupted, so nothing claims it was.
    assert not any("restarted" in f["data"].get("text", "")
                   for f in session["frames"] if f["kind"] == "note")


def test_the_flat_transcript_is_suppressed_once_the_frames_can_speak(client, manager):
    """`transcript` is the planner's payload — `ROLE: text`, flattened. Rendered
    beside the same conversation as frames it is a dump of what is already on
    screen, which is what the page had become."""
    session_id = _start(client)["session_id"]
    _await_status(client, session_id, "awaiting")
    _restart(manager)
    assert client.get(BASE).json()["transcript"] == ""


def test_a_store_written_before_frame_logs_still_shows_its_transcript(client, cfg):
    """The seeded store has a transcript and no frame log. That is every store
    that existed before this, and it must not come back empty."""
    state = client.get(BASE).json()
    assert state["session"] is None
    assert "rebuild the run timeline" in state["transcript"]


def test_a_live_session_outranks_the_stored_copy(client: TestClient):
    """Mid-turn the store is a frame or two behind. The session this process is
    driving is the truth."""
    session_id = _start(client)["session_id"]
    _await_status(client, session_id, "awaiting")
    session = client.get(BASE).json()["session"]
    assert session["status"] == "awaiting"      # not the terminal restored shape
    assert session["expects"] == "answer"


def test_progress_frames_are_never_written_to_the_store():
    """Hundreds arrive per turn and each would be an sqlite commit. The write is
    skipped for them, and the newest one the live log retains — so a mid-turn
    reconnect can show current activity — is left out of the payload too, since
    a restored conversation has no current activity to show."""
    session = discuss_mod.Session(session_id="s4", project=PROJECT, request="r",
                                  started_at=time.time())
    writes: list[dict] = []
    session._persist = writes.append

    session.push({"kind": "you", "text": "go"})
    assert len(writes) == 1
    for i in range(30):
        session.push({"kind": "progress", "phase": "tool", "tool": "Read",
                      "target": f"f{i}.ts"})
    assert len(writes) == 1, "a progress frame committed to sqlite"

    session.push({"kind": "note", "text": "a real frame"})
    assert len(writes) == 2
    assert [f["kind"] for f in writes[-1]["frames"]] == ["you", "note"]
    # The live log still holds the newest, for a reconnect mid-turn.
    assert [f.kind for f in session.frames] == ["you", "progress", "note"]


def test_a_store_that_cannot_be_written_does_not_end_the_conversation():
    """Losing recoverability is bad. Losing the conversation the operator is
    having, because sqlite was locked, is worse."""
    session = discuss_mod.Session(session_id="s3", project=PROJECT, request="r",
                                  started_at=time.time())
    def explode(_payload):
        raise RuntimeError("database is locked")
    session._persist = explode
    session.push({"kind": "you", "text": "still here"})
    assert [f.kind for f in session.frames] == ["you"]
