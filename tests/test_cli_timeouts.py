"""Idle timeouts, streaming, and retries on the CLI providers.

The bug these cover: `providers.claude_cli.timeout_s: 600` was a WALL-CLOCK cap
on a tool loop whose length the model decides. A measured planner turn runs 334s
one time and 539s the next for the identical prompt, so the cap killed working
calls at random — and killed them so completely that ten minutes of subscription
tokens produced zero bytes, because the old `--output-format json` buffers until
exit. The replacement is an idle clock over the event stream, plus retries for
the two failures that say nothing about the request.
"""

import asyncio
import json
import os

import pytest

from orchestrator.core.config import Section
from orchestrator.core.errors import LimitExhausted, OrchestratorError
from orchestrator.providers._process import CliTimeout, stream_cli
from orchestrator.providers.claude_cli import RESUME_NUDGE, ClaudeCliProvider
from tests.conftest import FakeCli, FakeProc, stream_json


# ---- _process.stream_cli -----------------------------------------------------

class DrippingStream:
    """A pipe that keeps producing, slowly. The point of the idle clock is that
    this process — alive, but slower overall than any wall-clock cap you would
    pick — must NOT be killed."""

    def __init__(self, chunks: list[bytes], gap: float):
        self._chunks = list(chunks)
        self._gap = gap

    async def read(self, _n: int = -1) -> bytes:
        if not self._chunks:
            return b""
        await asyncio.sleep(self._gap)
        return self._chunks.pop(0)


class _Silent:
    async def read(self, _n: int = -1) -> bytes:
        return b""


def _proc(monkeypatch, proc):
    async def fake_exec(*args, **kwargs):
        fake_exec.kwargs = kwargs
        return proc
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    return fake_exec


async def test_idle_timeout_kills_a_silent_process(monkeypatch):
    proc = FakeProc(stall=True)
    _proc(monkeypatch, proc)
    with pytest.raises(CliTimeout) as e:
        await stream_cli(["claude"], idle_timeout_s=0.05)
    assert e.value.kind == "idle" and e.value.retryable is True
    assert proc.killed                       # and it was actually reaped


async def test_a_slow_but_talking_process_is_never_killed(monkeypatch):
    """The regression that matters: total runtime far exceeds the idle window,
    and the call still completes because it never went quiet."""
    proc = FakeProc()
    proc.stdout = DrippingStream([b'{"a":1}\n'] * 10, gap=0.02)
    proc.stderr = _Silent()
    _proc(monkeypatch, proc)
    seen: list[str] = []
    run = await stream_cli(["claude"], idle_timeout_s=0.1,
                           on_line=lambda _t, line: seen.append(line))
    assert run.returncode == 0
    assert len(seen) == 10                   # ran 0.2s under a 0.1s idle window


async def test_total_ceiling_still_bounds_a_runaway(monkeypatch):
    proc = FakeProc()
    proc.stdout = DrippingStream([b'{"a":1}\n'] * 500, gap=0.01)
    proc.stderr = _Silent()
    _proc(monkeypatch, proc)
    with pytest.raises(CliTimeout) as e:
        await stream_cli(["claude"], idle_timeout_s=5, total_timeout_s=0.1)
    assert e.value.kind == "total" and e.value.retryable is False


async def test_stdin_is_devnull(monkeypatch):
    """An inherited stdin is how a CLI ends up blocked on a pipe nobody writes
    to. Verified against the real CLI, which does read stdin when it is open."""
    exec_ = _proc(monkeypatch, FakeProc(out="{}\n"))
    await stream_cli(["claude"], idle_timeout_s=1)
    assert exec_.kwargs["stdin"] is asyncio.subprocess.DEVNULL
    assert exec_.kwargs["start_new_session"] is True     # a killable process group


async def test_lines_are_framed_across_chunk_boundaries(monkeypatch):
    proc = FakeProc()
    proc.stdout = DrippingStream([b'{"a":', b'1}\n{"b":2}', b'\n'], gap=0.001)
    proc.stderr = _Silent()
    _proc(monkeypatch, proc)
    seen: list[str] = []
    await stream_cli(["claude"], idle_timeout_s=1,
                     on_line=lambda _t, line: seen.append(line))
    assert [json.loads(s) for s in seen] == [{"a": 1}, {"b": 2}]


async def test_a_line_larger_than_the_stream_limit_survives(monkeypatch):
    """Tool results embed whole files. `readline()` would raise on these, which
    is why framing is done over raw chunks instead."""
    big = json.dumps({"type": "result", "result": "x" * 200_000})
    _proc(monkeypatch, FakeProc(out=big + "\n"))
    seen: list[str] = []
    await stream_cli(["claude"], idle_timeout_s=1,
                     on_line=lambda _t, line: seen.append(line))
    assert len(seen) == 1 and json.loads(seen[0])["result"] == "x" * 200_000


async def test_a_final_line_without_a_newline_is_still_delivered(monkeypatch):
    _proc(monkeypatch, FakeProc(out='{"type":"result"}'))     # no trailing \n
    seen: list[str] = []
    await stream_cli(["claude"], idle_timeout_s=1,
                     on_line=lambda _t, line: seen.append(line))
    assert seen == ['{"type":"result"}']


async def test_a_raising_callback_cannot_fail_a_paid_for_call(monkeypatch):
    _proc(monkeypatch, FakeProc(out='{"a":1}\n'))

    def boom(_tag, _line):
        raise RuntimeError("the UI blew up")

    run = await stream_cli(["claude"], idle_timeout_s=1, on_line=boom)
    assert run.returncode == 0


# ---- claude_cli: argv, streaming, retries ------------------------------------

def _provider(monkeypatch, *specs, **pcfg):
    cli = FakeCli(*specs).install(monkeypatch)
    base = {"type": "claude_cli", "binary": "claude", "idle_timeout_s": 0.05,
            "timeout_s": 5, "retry_attempts": 2}
    return ClaudeCliProvider("claude_cli", Section({**base, **pcfg}),
                             Section({"mcp": {}})), cli


_OK = {"out": stream_json({"result": "ok", "usage": {"input_tokens": 3,
                                                     "output_tokens": 1}})}


async def test_streaming_flags_are_passed(monkeypatch):
    p, cli = _provider(monkeypatch, _OK)
    await p.complete(model="opus", system="S", user="U")
    argv = cli.argv[0]
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    # --verbose is mandatory with stream-json in print mode; the partial-message
    # flag is what keeps the final answer from being one long silence.
    assert "--verbose" in argv and "--include-partial-messages" in argv


async def test_the_result_event_is_parsed_out_of_the_stream(monkeypatch):
    p, _ = _provider(monkeypatch, _OK)
    res = await p.complete(model="opus", system="S", user="U")
    assert res.text == "ok" and res.input_tokens == 3


async def test_a_stream_with_no_result_event_is_an_error(monkeypatch):
    p, _ = _provider(monkeypatch, {"out": '{"type":"system","subtype":"init"}\n'})
    with pytest.raises(OrchestratorError, match="no result event"):
        await p.complete(model="opus", system="S", user="U")


class TalkThenStall:
    """Announces a session, then goes silent forever — a wedged CLI, exactly."""

    def __init__(self, data: bytes):
        self._data = data
        self._sent = False

    async def read(self, _n: int = -1) -> bytes:
        if not self._sent:
            self._sent = True
            return self._data
        await asyncio.sleep(3600)
        return b""                                           # pragma: no cover


async def test_an_idle_wedge_is_retried_by_resuming_the_dead_session(monkeypatch):
    """The whole point of learning the session id: the killed attempt already
    paid to read the repo, and --resume keeps that work. Verified against the
    real CLI — re-passing --session-id fails with 'already in use', and --resume
    on a SIGKILLed session succeeds."""
    wedged = FakeProc()
    wedged.stdout = TalkThenStall(
        json.dumps({"type": "system", "subtype": "init",
                    "session_id": "sess-x"}).encode() + b"\n")
    wedged.stderr = _Silent()
    good = FakeProc(out=_OK["out"])

    argvs: list[list[str]] = []
    queue = [wedged, good]

    async def fake_exec(*args, **kwargs):
        argvs.append(list(args))
        return queue.pop(0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    p = ClaudeCliProvider(
        "claude_cli",
        Section({"type": "claude_cli", "binary": "claude",
                 "idle_timeout_s": 0.05, "timeout_s": 5, "retry_attempts": 2}),
        Section({"mcp": {}}))

    res = await p.complete(model="opus", system="S", user="U")
    assert res.text == "ok"
    assert len(argvs) == 2                              # it retried
    assert argvs[1][argvs[1].index("--resume") + 1] == "sess-x"
    assert "--session-id" not in argvs[1]               # never re-pinned
    # The resumed session already holds the original prompt (verified against
    # the CLI), so the retry must not append a second copy of it.
    assert argvs[0][2] == "U"
    assert argvs[1][2] == RESUME_NUDGE


async def test_a_retry_with_nothing_to_resume_resends_the_real_prompt(monkeypatch):
    """The nudge is only meaningful to a session that has the prompt. Starting
    clean, the prompt has to be the prompt."""
    p, cli = _provider(monkeypatch, {"stall": True}, _OK)
    await p.complete(model="opus", system="S", user="U")
    assert cli.argv[1][2] == "U"


async def test_a_wedge_with_no_session_learned_retries_clean(monkeypatch):
    p, cli = _provider(monkeypatch, {"stall": True}, _OK)
    res = await p.complete(model="opus", system="S", user="U")
    assert res.text == "ok"
    assert "--resume" not in cli.argv[1]     # nothing to resume; starts fresh


async def test_a_transient_exit_is_retried(monkeypatch):
    p, cli = _provider(
        monkeypatch,
        {"rc": 1, "err": "API Error: 503 Service Unavailable"},
        _OK)
    res = await p.complete(model="opus", system="S", user="U")
    assert res.text == "ok" and len(cli.argv) == 2


async def test_a_limit_hit_is_never_retried(monkeypatch):
    """It has to reach the run's own pause/degrade ladder, and burning the
    retry budget against a weekly limit accomplishes nothing."""
    p, cli = _provider(monkeypatch, {"rc": 1, "err": "weekly limit reached"}, _OK)
    with pytest.raises(LimitExhausted):
        await p.complete(model="opus", system="S", user="U")
    assert len(cli.argv) == 1


async def test_a_deterministic_failure_is_never_retried(monkeypatch):
    p, cli = _provider(monkeypatch, {"rc": 2, "err": "unknown flag --nope"}, _OK)
    with pytest.raises(OrchestratorError):
        await p.complete(model="opus", system="S", user="U")
    assert len(cli.argv) == 1


async def test_blowing_the_absolute_ceiling_is_not_retried(monkeypatch):
    """Unlike a wedge, this call was working the whole time — a retry would do
    the same expensive thing again and hit the same ceiling."""
    p, cli = _provider(monkeypatch, {"stall": True}, _OK,
                       idle_timeout_s=5, timeout_s=0.05)
    with pytest.raises(OrchestratorError, match="absolute ceiling"):
        await p.complete(model="opus", system="S", user="U")
    assert len(cli.argv) == 1


async def test_retry_attempts_1_disables_retrying(monkeypatch):
    p, cli = _provider(monkeypatch, {"stall": True}, _OK, retry_attempts=1)
    with pytest.raises(OrchestratorError):
        await p.complete(model="opus", system="S", user="U")
    assert len(cli.argv) == 1


async def test_progress_reports_tool_calls_and_text(monkeypatch):
    events = [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Read",
             "input": {"file_path": "src/app.ts"}}]}},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Verified against main."}]}},
    ]
    p, _ = _provider(monkeypatch, {"out": stream_json(
        {"result": "ok", "usage": {}}, events=events)})
    seen: list[dict] = []
    await p.complete(model="opus", system="S", user="U", on_progress=seen.append)
    assert {"phase": "tool", "tool": "Read", "target": "src/app.ts"} in seen
    assert any(e["phase"] == "text" and "Verified" in e["text"] for e in seen)


async def test_progress_is_optional(monkeypatch):
    p, _ = _provider(monkeypatch, _OK)
    res = await p.complete(model="opus", system="S", user="U")   # no callback
    assert res.text == "ok"


async def test_tool_targets_are_shown_relative_to_the_repo(monkeypatch):
    """Absolute paths share the same long prefix on every line — measured live,
    it swallowed the whole activity row."""
    events = [{"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Read",
         "input": {"file_path": "/repo/src/features/debug/panel.ts"}}]}}]
    p, _ = _provider(monkeypatch, {"out": stream_json({"result": "ok", "usage": {}},
                                                      events=events)})
    seen: list[dict] = []
    await p.complete(model="opus", system="S", user="U", cwd="/repo",
                     on_progress=seen.append)
    assert {"phase": "tool", "tool": "Read",
            "target": "src/features/debug/panel.ts"} in seen


async def test_a_path_outside_the_repo_is_left_alone(monkeypatch):
    events = [{"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Read",
         "input": {"file_path": "/etc/hosts"}}]}}]
    p, _ = _provider(monkeypatch, {"out": stream_json({"result": "ok", "usage": {}},
                                                      events=events)})
    seen: list[dict] = []
    await p.complete(model="opus", system="S", user="U", cwd="/repo",
                     on_progress=seen.append)
    assert any(e.get("target") == "/etc/hosts" for e in seen)


# ---- against a real OS process ------------------------------------------------
# Everything above stubs the subprocess. These two do not: the process-group kill
# and the DEVNULL stdin are claims about the operating system, and a fake process
# cannot falsify either of them.

async def test_a_real_wedged_process_and_its_children_are_killed(tmp_path):
    """`proc.kill()` signals only the direct child. The CLI spawns ripgrep, node
    and MCP servers, and orphaning those leaves them holding the worktree."""
    marker = tmp_path / "child.pid"
    script = tmp_path / "wedge.sh"
    script.write_text(
        "#!/bin/sh\n"
        "sleep 300 &\n"                      # a grandchild that must also die
        f"echo $! > {marker}\n"
        "echo '{\"type\":\"system\"}'\n"     # talk once, then go quiet forever
        "sleep 300\n")
    script.chmod(0o755)

    # 2s, not a snappier number: the idle clock starts at spawn, and under the
    # full suite the shell may not be scheduled for hundreds of milliseconds —
    # a tighter window kills it before it has forked anything, and the test
    # passes vacuously or dies reading a marker that was never written.
    with pytest.raises(CliTimeout) as e:
        await stream_cli([str(script)], idle_timeout_s=2.0)
    assert e.value.kind == "idle"

    assert marker.exists(), "the script never got far enough to fork a child"
    # Poll rather than sleep a fixed interval: the signal lands asynchronously,
    # and on a loaded machine (the full suite) a fixed wait is a flaky test.
    child_pid = int(marker.read_text().strip())
    for _ in range(100):
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break                            # gone, not merely orphaned
        await asyncio.sleep(0.05)
    else:
        pytest.fail(f"the grandchild {child_pid} outlived the kill")


async def test_a_real_process_sees_a_closed_stdin(tmp_path):
    """The CLI reads stdin when it is open — measured: it waits, warns
    'no stdin data received in 3s', and only then proceeds. Inheriting the
    server's stdin is how that becomes an unbounded block instead."""
    script = tmp_path / "readstdin.sh"
    script.write_text("#!/bin/sh\ncat\necho DONE\n")     # blocks unless stdin is EOF
    script.chmod(0o755)

    seen: list[str] = []
    run = await stream_cli([str(script)], idle_timeout_s=5,
                           on_line=lambda _t, line: seen.append(line))
    assert run.returncode == 0 and seen == ["DONE"]
