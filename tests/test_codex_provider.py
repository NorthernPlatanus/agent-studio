"""Phase 1: CodexCliProvider — parse codex exec JSONL, map tokens/cost, and
raise the right control-flow errors. No real codex binary or network.
"""

import asyncio
from pathlib import Path

import pytest

from orchestrator.core.config import Section
from orchestrator.core.errors import LimitExhausted, OrchestratorError
from orchestrator.providers.codex_cli import CodexCliProvider

SUCCESS_JSONL = "\n".join([
    '{"type":"item.started","item":{"type":"agent_message"}}',
    '{"type":"item.completed","item":{"type":"agent_message","text":"HELLO FROM CODEX"}}',
    '{"type":"turn.completed","usage":{"input_tokens":123,"output_tokens":45}}',
])


class FakeProc:
    def __init__(self, out: str, err: str, rc: int):
        self._out = out.encode()
        self._err = err.encode()
        self.returncode = rc

    async def communicate(self):
        return self._out, self._err

    def kill(self):
        pass

    async def wait(self):
        return 0


def _patch(monkeypatch, out, err, rc):
    async def fake_exec(*args, **kwargs):
        return FakeProc(out, err, rc)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)


def _provider(monkeypatch, out="", err="", rc=0, pcfg=None, cfg=None):
    _patch(monkeypatch, out, err, rc)
    pcfg = Section(pcfg or {"type": "codex_cli", "binary": "codex",
                            "allowed_tools": "read-only", "auth": "subscription",
                            "timeout_s": 600})
    cfg = Section(cfg or {"mcp": {}, "worker_models": {}})
    return CodexCliProvider("codex_cli", pcfg, cfg)


async def test_parses_text_and_tokens(monkeypatch):
    p = _provider(monkeypatch, out=SUCCESS_JSONL, rc=0)
    res = await p.complete(model="gpt-5.6-sol", system="SYS", user="do it")
    assert res.text == "HELLO FROM CODEX"
    assert res.input_tokens == 123
    assert res.output_tokens == 45
    assert res.cost_usd == 0.0            # subscription auth: logged, not counted


async def test_limit_pattern_raises_limit_exhausted(monkeypatch):
    p = _provider(monkeypatch, out="", err="Error: usage limit reached, upgrade to continue", rc=1)
    with pytest.raises(LimitExhausted):
        await p.complete(model="m", system="", user="x")


async def test_generic_nonzero_raises_orchestrator_error(monkeypatch):
    p = _provider(monkeypatch, out="", err="boom: something went wrong", rc=1)
    with pytest.raises(OrchestratorError) as ei:
        await p.complete(model="m", system="", user="x")
    assert not isinstance(ei.value, LimitExhausted)


async def test_api_auth_prices_from_worker_models(monkeypatch):
    cfg = {"mcp": {}, "worker_models": {
        "codex_api": {"provider": "codex_cli", "model": "gpt-5.6-sol",
                      "input_per_mtok": 1.0, "output_per_mtok": 2.0}}}
    pcfg = {"type": "codex_cli", "binary": "codex", "allowed_tools": "read-only",
            "auth": "api", "timeout_s": 600}
    p = _provider(monkeypatch, out=SUCCESS_JSONL, rc=0, pcfg=pcfg, cfg=cfg)
    res = await p.complete(model="gpt-5.6-sol", system="", user="x")
    # 123/1e6*1.0 + 45/1e6*2.0
    assert res.cost_usd == pytest.approx(123 / 1e6 * 1.0 + 45 / 1e6 * 2.0)


def test_sandbox_mode_defaults_readonly(monkeypatch):
    # a Claude-style allowed_tools string must NOT widen the sandbox
    p = _provider(monkeypatch, pcfg={"type": "codex_cli", "binary": "codex",
                                     "allowed_tools": "Read,Grep,Glob"})
    assert p._sandbox_mode() == "read-only"


def test_explicit_sandbox_mode_passthrough(monkeypatch):
    p = _provider(monkeypatch, pcfg={"type": "codex_cli", "binary": "codex",
                                     "allowed_tools": "workspace-write"})
    assert p._sandbox_mode() == "workspace-write"


def test_mcp_off_by_default(monkeypatch):
    cfg = {"mcp": {"inspector": {"enabled": True, "command": "node srv.js"}},
           "worker_models": {}}
    p = _provider(monkeypatch, cfg=cfg)  # pcfg has no enable_mcp
    assert p._mcp_config_args() == []


def test_mcp_translated_when_enabled(monkeypatch):
    cfg = {"mcp": {"inspector": {"enabled": True, "command": "node srv.js"}},
           "worker_models": {}}
    pcfg = {"type": "codex_cli", "binary": "codex", "allowed_tools": "read-only",
            "enable_mcp": True}
    p = _provider(monkeypatch, cfg=cfg, pcfg=pcfg)
    args = p._mcp_config_args()
    assert args == ["-c", "mcp_servers.inspector.command=node srv.js"]
