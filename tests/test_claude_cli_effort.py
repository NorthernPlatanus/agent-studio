"""`providers.claude_cli.effort` -> `claude -p --effort <level>`.

Effort is the main lever on how many subscription tokens a smart-tier call
spends, and subscription tokens are the binding constraint — so the flag has to
actually reach the CLI, and a typo must not quietly leave it at the default.
"""

import pytest

from orchestrator.core.config import Section
from orchestrator.core.errors import OrchestratorError
from orchestrator.providers.claude_cli import EFFORT_LEVELS, ClaudeCliProvider
from tests.conftest import FakeCli, stream_json


def _provider(monkeypatch, captured: list, **pcfg) -> ClaudeCliProvider:
    cli = FakeCli({"out": stream_json({"result": "ok",
                                       "usage": {"input_tokens": 1}})})
    cli.argv = captured                          # tests assert on `captured`
    cli.install(monkeypatch)
    base = {"type": "claude_cli", "binary": "claude", "timeout_s": 600}
    return ClaudeCliProvider("claude_cli", Section({**base, **pcfg}),
                             Section({"mcp": {}}))


async def test_effort_is_passed_to_the_cli(monkeypatch):
    captured: list = []
    provider = _provider(monkeypatch, captured, effort="high")
    await provider.complete(model="claude-opus-5", system="s", user="u")
    argv = captured[0]
    assert "--effort" in argv
    assert argv[argv.index("--effort") + 1] == "high"


async def test_no_effort_configured_leaves_the_cli_default_alone(monkeypatch):
    captured: list = []
    provider = _provider(monkeypatch, captured)
    await provider.complete(model="claude-opus-5", system="s", user="u")
    assert "--effort" not in captured[0]


async def test_invalid_effort_fails_loudly(monkeypatch):
    """The CLI warns-and-ignores an unknown level, so validating here is the only
    thing standing between a typo and a silent change of tier behavior."""
    captured: list = []
    provider = _provider(monkeypatch, captured, effort="highest")
    with pytest.raises(OrchestratorError, match="not a valid level"):
        await provider.complete(model="claude-opus-5", system="s", user="u")
    assert captured == []           # refused before spawning the subprocess


@pytest.mark.parametrize("level", EFFORT_LEVELS)
async def test_every_documented_level_is_accepted(monkeypatch, level):
    captured: list = []
    provider = _provider(monkeypatch, captured, effort=level)
    await provider.complete(model="claude-opus-5", system="s", user="u")
    assert captured[0][captured[0].index("--effort") + 1] == level


# ---- per-role resolution (RunContext) ---------------------------------------

def _ctx():
    """A RunContext carrying only what role_effort/worker_effort read."""
    from pathlib import Path

    from orchestrator.core.config import Config
    from orchestrator.core.context import RunContext
    data = {
        "providers": {"claude_cli": {"type": "claude_cli", "effort": "low"},
                      "comet": {"type": "openai_compatible"}},
        "roles": {"smart_provider": "claude_cli",
                  "planner": {"provider": None, "model": "claude-opus-5",
                              "effort": "high"},
                  "reviewer": {"provider": None, "model": "claude-opus-5",
                               "effort": "medium"}},
        "worker_models": {"flash": {"provider": "comet", "model": "m"}},
        "run": {"escalate_model": "claude-opus-5", "escalate_effort": "medium",
                "degrade_model": "flash"},
    }
    cfg = Config(data, "p", Path("/tmp"))
    return RunContext(cfg=cfg, store=None, git=None, budget=None, run_id="r")


def test_each_role_gets_its_own_effort():
    ctx = _ctx()
    assert ctx.role_effort("planner") == "high"
    assert ctx.role_effort("reviewer") == "medium"
    assert ctx.worker_effort("senior") == "medium"


def test_cheap_workers_get_no_effort():
    """Chat-completions models have no effort dial; passing one would be noise."""
    assert _ctx().worker_effort("flash") is None


def test_role_without_effort_falls_back_to_the_tier_default():
    ctx = _ctx()
    ctx.cfg._data["roles"]["reviewer"]["effort"] = None
    assert ctx.role_effort("reviewer") == "low"      # providers.claude_cli.effort


def test_no_effort_anywhere_leaves_the_cli_default():
    ctx = _ctx()
    ctx.cfg._data["roles"]["planner"]["effort"] = None
    ctx.cfg._data["providers"]["claude_cli"]["effort"] = None
    assert ctx.role_effort("planner") is None


def test_degrade_mode_drops_effort():
    """Degrade reroutes to a cash worker model, where effort is meaningless."""
    ctx = _ctx()
    ctx.degraded = True
    assert ctx.role_effort("planner") is None
