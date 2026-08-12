"""Streaming subprocess plumbing shared by the CLI providers.

**Why this is not `asyncio.wait_for(proc.communicate(), timeout)`.**

That is what both CLI providers used to do, and it puts a WALL-CLOCK cap on a
call whose length is decided by how many tool round-trips the model chooses to
make. Measured on the real planner (demo-project, opus at effort=high): a healthy
turn is 24 tool turns / ~1.1M tokens read / 334s, and the same call has been
observed to run past 600s. A wall-clock cap sized for the median therefore fails
intermittently forever, and there is no number that fixes it — the distribution,
not the limit, is the problem.

What a cap can honestly detect is a call that has STOPPED PRODUCING. Both CLIs
can stream their events (`claude --output-format stream-json`, `codex exec
--json`), and both emit continuously while working — the Claude CLI even emits
`system/thinking_tokens` heartbeats while the model is thinking, so silence is
never just "still reasoning". So the timeout here is an IDLE timeout: the clock
resets on every byte from either stream, and only genuine silence kills the run.
A wedged call now dies in `idle_timeout_s` (faster than the old wall clock), and
a slow-but-working call is never killed at all.

`total_timeout_s` remains available as an absolute backstop against a runaway
loop that keeps emitting forever. It is deliberately loose — the idle clock is
the one meant to fire.

Two smaller correctness fixes ride along, both of which the old path got wrong:

* **stdin is `DEVNULL`.** It used to be inherited. A CLI that decides to read
  stdin then blocks forever against whatever the parent happened to be attached
  to — and under a server that is a pipe nobody writes to, or a terminal that
  raises SIGTTIN once the server is backgrounded.
* **The whole process group is killed.** `proc.kill()` signals only the direct
  child; the CLI's own children (ripgrep, node, MCP servers) survive it and keep
  holding the worktree. The child is spawned with `start_new_session=True` so
  there is a process group to signal.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass

from ..core.errors import OrchestratorError

log = logging.getLogger("orchestrator.providers.process")

#: Read granularity. Chunks, not `readline()`: a StreamReader raises once a
#: single line exceeds its buffer limit, and these streams carry tool results
#: that embed whole files. Line framing happens here instead, unbounded.
READ_CHUNK = 64 * 1024

#: StreamReader buffer. Only an upper bound on one `read()`; framing is ours.
STREAM_LIMIT = 8 * 1024 * 1024

#: Grace period for a process to exit after closing both pipes.
EXIT_GRACE_S = 30

#: How much of each stream to retain verbatim for error messages. The events the
#: caller wants are handed to `on_line` as they arrive; this is only the tail
#: that makes a failure readable, and it is bounded so a chatty CLI cannot grow
#: the orchestrator's memory by the size of its own log.
KEEP_BYTES = 64 * 1024


class CliTimeout(OrchestratorError):
    """A CLI run the runner gave up on. `kind` is 'idle' or 'total'.

    Separate from a plain OrchestratorError because the two kinds warrant
    different handling: an idle kill means the process wedged and produced
    nothing, which a retry can plausibly fix, while hitting the absolute backstop
    means it was working the whole time and would do it again.
    """

    def __init__(self, message: str, *, kind: str, elapsed: float):
        super().__init__(message)
        self.kind = kind
        self.elapsed = elapsed

    @property
    def retryable(self) -> bool:
        return self.kind == "idle"


@dataclass
class CliRun:
    """The outcome of one CLI invocation."""

    returncode: int
    #: Bounded head of each stream, for error messages (see KEEP_BYTES).
    #: NOT the place to read a CLI's output from — a long run is truncated here.
    #: Callers that need every event collect them through `on_line`.
    stdout: str = ""
    stderr: str = ""
    #: Wall-clock seconds the process ran.
    elapsed: float = 0.0

    @property
    def combined(self) -> str:
        return f"{self.stdout}\n{self.stderr}"


class _Head:
    """A bounded, order-preserving byte buffer: keeps the FIRST KEEP_BYTES.

    The head, not the tail: a CLI puts its error near where it failed, and a
    stream that runs long is truncated to the part that explains why.
    """

    def __init__(self, cap: int = KEEP_BYTES):
        self._parts: list[bytes] = []
        self._len = 0
        self._cap = cap

    def add(self, chunk: bytes) -> None:
        if self._len >= self._cap:
            return
        room = self._cap - self._len
        self._parts.append(chunk[:room])
        self._len += min(len(chunk), room)

    def text(self) -> str:
        return b"".join(self._parts).decode(errors="replace")


async def _pump(stream, queue: asyncio.Queue, tag: str) -> None:
    """Drain one pipe into the shared queue, ending with an EOF sentinel.

    Every chunk is an idle-clock reset, which is why both streams feed ONE queue:
    a CLI that only writes progress to stderr is just as alive as one that writes
    to stdout, and draining them separately would let the quiet stream's clock
    expire on a working process.
    """
    try:
        while True:
            chunk = await stream.read(READ_CHUNK)
            if not chunk:
                break
            queue.put_nowait((tag, chunk))
    except (BrokenPipeError, ConnectionResetError) as e:      # pragma: no cover
        log.debug("pipe %s closed early: %s", tag, e)
    finally:
        queue.put_nowait((tag, None))


def _kill_tree(proc) -> None:
    """SIGKILL the child's whole process group, then the child itself.

    The group is the point: the CLI spawns ripgrep/node/MCP servers, and killing
    only the direct child leaves those holding file handles in the worktree. Both
    steps are best-effort — the process may already be gone, and a test's fake
    process has no real pid at all.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError,
                             AttributeError, TypeError):
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    with contextlib.suppress(ProcessLookupError, OSError, AttributeError):
        proc.kill()


async def stream_cli(argv: list[str], *, cwd: str | None = None,
                     idle_timeout_s: float,
                     total_timeout_s: float | None = None,
                     on_line: Callable[[str, str], None] | None = None,
                     env: dict | None = None) -> CliRun:
    """Run `argv`, framing both streams into lines and enforcing an idle clock.

    `on_line(tag, line)` is called for every complete line as it arrives, with
    `tag` in {'stdout', 'stderr'}. It is where the caller parses events and
    reports progress; it must not block, and it is never allowed to fail the run
    (an exception in a progress callback would otherwise throw away a call that
    has already been paid for).

    Raises `CliTimeout` when the idle or total clock expires. Any other outcome —
    including a non-zero exit — returns a `CliRun` for the caller to interpret,
    because "what does exit 1 mean" is provider-specific and this layer has no
    business guessing.
    """
    started = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *argv, cwd=cwd, env=env,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=STREAM_LIMIT,
        start_new_session=True)

    queue: asyncio.Queue = asyncio.Queue()
    pumps = [asyncio.create_task(_pump(proc.stdout, queue, "stdout")),
             asyncio.create_task(_pump(proc.stderr, queue, "stderr"))]

    heads = {"stdout": _Head(), "stderr": _Head()}
    partial = {"stdout": b"", "stderr": b""}
    open_streams = 2
    expired: str | None = None

    def emit(tag: str, raw: bytes) -> None:
        if on_line is None:
            return
        line = raw.decode(errors="replace").strip()
        if not line:
            return
        try:
            on_line(tag, line)
        except Exception:                        # noqa: BLE001 — never fail the run
            log.exception("on_line callback raised; ignoring")

    deadline = started + total_timeout_s if total_timeout_s else None
    try:
        while open_streams:
            wait = idle_timeout_s
            if deadline is not None:
                wait = min(wait, max(0.0, deadline - time.monotonic()))
            try:
                tag, chunk = await asyncio.wait_for(queue.get(), wait)
            except (asyncio.TimeoutError, TimeoutError):
                expired = ("total" if deadline is not None
                           and time.monotonic() >= deadline else "idle")
                break

            if chunk is None:                    # EOF on this stream
                open_streams -= 1
                if partial[tag]:                 # a last line with no newline
                    emit(tag, partial[tag])
                    partial[tag] = b""
                continue

            heads[tag].add(chunk)
            buf = partial[tag] + chunk
            *lines, partial[tag] = buf.split(b"\n")
            for raw in lines:
                emit(tag, raw)
    finally:
        for pump in pumps:
            pump.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(*pumps, return_exceptions=True)

    elapsed = time.monotonic() - started
    if expired is not None:
        _kill_tree(proc)
        with contextlib.suppress(Exception):
            await proc.wait()                    # reap; avoid a zombie
        limit = total_timeout_s if expired == "total" else idle_timeout_s
        detail = ("produced no output for" if expired == "idle"
                  else "exceeded the absolute ceiling of")
        # No argv[0] here: the caller prefixes its own provider name, and the
        # binary is already in its message.
        raise CliTimeout(f"{detail} {limit:g}s (ran {elapsed:.0f}s)",
                         kind=expired, elapsed=elapsed)

    # Both pipes are closed, which for a well-behaved CLI means it is exiting —
    # but "means" is not "guarantees", and a bare `await proc.wait()` here would
    # be exactly the unbounded wait this module exists to remove. A process that
    # closed its output and then hung gets the same treatment as any other wedge.
    try:
        returncode = await asyncio.wait_for(proc.wait(), EXIT_GRACE_S)
    except (asyncio.TimeoutError, TimeoutError):
        _kill_tree(proc)
        with contextlib.suppress(Exception):
            await proc.wait()
        raise CliTimeout(
            f"closed its output but did not exit within {EXIT_GRACE_S:g}s",
            kind="idle", elapsed=time.monotonic() - started) from None

    return CliRun(returncode=returncode,
                  stdout=heads["stdout"].text(), stderr=heads["stderr"].text(),
                  elapsed=elapsed)
