"""Isolate the test suite from this machine's configuration.

`load_config()` layers `config/local.yaml` and then any `ORCH_*` environment
overrides on top of `config/default.yaml`. Both are per-machine, and neither is
something a unit test asked for. Concretely: a real `gate.install_cmd` in
local.yaml makes every worker test's `ensure_deps` shell out to an actual
`npm ci` inside a temp worktree with no package-lock, so ten tests fail with an
`IndexError` on an empty provider-call list — a suite whose result depends on
whose laptop it runs on, and a failure that says nothing about the real cause.

Tests that want a non-default config build it explicitly instead.
"""
import asyncio
import json
import os

import pytest

# ---- CLI subprocess fakes ----------------------------------------------------
# Shared because all four CLI-provider test modules need the same one, and
# because the providers now READ THE PIPES rather than calling communicate():
# a fake that only implements communicate() would pass while the real code path
# went untested. This one is stream-shaped, like the thing it stands in for.


class FakeStream:
    """The reader half of a pipe, handing out `data` in chunks then EOF."""

    def __init__(self, data: bytes, *, chunk: int = 4096):
        self._data = data
        self._pos = 0
        self._chunk = chunk

    async def read(self, n: int = -1) -> bytes:
        if self._pos >= len(self._data):
            return b""
        size = len(self._data) if n is None or n < 0 else min(n, self._chunk)
        out = self._data[self._pos:self._pos + size]
        self._pos += len(out)
        return out


class FakeProc:
    """A finished subprocess: both pipes readable, then a return code.

    `stall=True` makes it produce nothing and never exit, which is the only way
    to exercise the idle timeout without actually waiting on a real process.
    """

    pid = -1        # _kill_tree tolerates a pid that cannot be signalled

    def __init__(self, out: str = "", err: str = "", rc: int = 0,
                 stall: bool = False):
        self.stdout = _StalledStream() if stall else FakeStream(out.encode())
        self.stderr = _StalledStream() if stall else FakeStream(err.encode())
        self.returncode = rc
        self.killed = False

    async def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:
        self.killed = True


class _StalledStream:
    async def read(self, n: int = -1) -> bytes:
        await asyncio.sleep(3600)
        return b""                                          # pragma: no cover


def stream_json(payload: dict, *, session_id: str = "sess-1",
                events: list[dict] | None = None) -> str:
    """A claude CLI `--output-format stream-json` stdout carrying `payload` as
    its final `result` event, preceded by the `system/init` the real CLI always
    emits (that is where the provider learns the session id to resume)."""
    lines = [{"type": "system", "subtype": "init", "session_id": session_id}]
    lines += events or []
    lines.append({"type": "result", **payload})
    return "\n".join(json.dumps(line) for line in lines) + "\n"


class FakeCli:
    """Scripts `asyncio.create_subprocess_exec`: one spec per call, last sticky.

    Specs are kwargs for FakeProc rather than FakeProc instances on purpose — a
    process's pipes are CONSUMED by reading them, so handing the same instance to
    two calls would give the second one EOF, which is indistinguishable from a
    CLI that printed nothing. Building a fresh process per call is what really
    happens, and it keeps a retry test honest.

    `argv` records every command line; `procs` every process built.
    """

    def __init__(self, *specs: dict):
        self.specs = list(specs) or [{}]
        self.argv: list[list[str]] = []
        self.procs: list[FakeProc] = []

    def install(self, monkeypatch) -> "FakeCli":
        monkeypatch.setattr(asyncio, "create_subprocess_exec", self)
        return self

    async def __call__(self, *args, **kwargs) -> FakeProc:
        self.argv.append(list(args))
        spec = self.specs[min(len(self.procs), len(self.specs) - 1)]
        proc = FakeProc(**spec)
        self.procs.append(proc)
        return proc

    def flag(self, name: str, call: int = 0) -> str | None:
        """The value after `name` in call `call`'s argv, or None if absent."""
        argv = self.argv[call]
        return argv[argv.index(name) + 1] if name in argv else None


@pytest.fixture(autouse=True, scope="session")
def _isolate_machine_config():
    saved = {k: v for k, v in os.environ.items() if k.startswith("ORCH_")}
    for key in saved:
        del os.environ[key]
    os.environ["ORCH_SKIP_LOCAL_CONFIG"] = "1"
    yield
    os.environ.pop("ORCH_SKIP_LOCAL_CONFIG", None)
    os.environ.update(saved)
