"""Item 2 step 1: the CLI tiers must report prompt-cache telemetry.

Without this the `usage` ledger records 0/0 for exactly the tier we are rationed
on, so the smart-tier cache question cannot even be measured. Also covers the
`status` cache columns (item 12) that surface it.
"""

from orchestrator.core.config import Section
from orchestrator.ops.store import Store
from orchestrator.providers.claude_cli import ClaudeCliProvider, cache_tokens
from orchestrator.providers.codex_cli import CodexCliProvider
from tests.conftest import FakeCli, stream_json


def _claude(monkeypatch, payload: dict) -> ClaudeCliProvider:
    FakeCli({"out": stream_json(payload)}).install(monkeypatch)
    return ClaudeCliProvider(
        "claude_cli",
        Section({"type": "claude_cli", "binary": "claude", "timeout_s": 600}),
        Section({"mcp": {}}))


# ---- claude_cli --------------------------------------------------------------

def test_cache_tokens_sums_all_three_input_buckets():
    # The Anthropic shape splits input across three fields; `input_tokens` alone
    # is the UNCACHED remainder and badly understates the real prompt weight.
    total, hit, miss = cache_tokens({
        "input_tokens": 100, "cache_creation_input_tokens": 400,
        "cache_read_input_tokens": 9500})
    assert (total, hit, miss) == (10000, 9500, 500)
    assert hit + miss == total


def test_cache_tokens_absent_fields_are_zero_not_guessed():
    total, hit, miss = cache_tokens({"input_tokens": 42})
    assert (total, hit, miss) == (42, 0, 42)
    assert cache_tokens({}) == (0, 0, 0)


async def test_claude_cli_populates_cache_fields(monkeypatch):
    p = _claude(monkeypatch, {
        "result": "ok", "total_cost_usd": 0.0,
        "usage": {"input_tokens": 10, "output_tokens": 7,
                  "cache_creation_input_tokens": 90,
                  "cache_read_input_tokens": 900}})
    res = await p.complete(model="opus", system="S", user="U")
    assert res.input_tokens == 1000          # total prompt weight, not just fresh
    assert res.output_tokens == 7
    assert res.cache_hit_tokens == 900
    assert res.cache_miss_tokens == 100


async def test_claude_cli_old_payload_without_cache_fields(monkeypatch):
    p = _claude(monkeypatch, {"result": "ok", "usage": {"input_tokens": 5,
                                                        "output_tokens": 2}})
    res = await p.complete(model="opus", system="S", user="U")
    assert (res.input_tokens, res.cache_hit_tokens, res.cache_miss_tokens) == (5, 0, 5)


# ---- codex_cli ---------------------------------------------------------------

def _codex(monkeypatch, jsonl: str) -> CodexCliProvider:
    FakeCli({"out": jsonl}).install(monkeypatch)
    return CodexCliProvider(
        "codex_cli",
        Section({"type": "codex_cli", "binary": "codex", "allowed_tools": "read-only",
                 "auth": "subscription", "timeout_s": 600}),
        Section({"mcp": {}, "worker_models": {}}))


async def test_codex_cli_reports_cached_input(monkeypatch):
    # Codex counts cached bytes INSIDE input_tokens, so miss is the remainder.
    p = _codex(monkeypatch, '{"type":"turn.completed","usage":'
                            '{"input_tokens":1000,"output_tokens":20,'
                            '"cached_input_tokens":800}}')
    res = await p.complete(model="gpt-5.6-sol", system="S", user="U")
    assert res.input_tokens == 1000
    assert res.cache_hit_tokens == 800
    assert res.cache_miss_tokens == 200


async def test_codex_cli_nested_details_shape(monkeypatch):
    p = _codex(monkeypatch, '{"type":"turn.completed","usage":'
                            '{"input_tokens":50,"output_tokens":1,'
                            '"input_tokens_details":{"cached_tokens":30}}}')
    res = await p.complete(model="gpt-5.6-sol", system="S", user="U")
    assert (res.cache_hit_tokens, res.cache_miss_tokens) == (30, 20)


async def test_codex_cli_no_cache_field_is_zero(monkeypatch):
    p = _codex(monkeypatch, '{"type":"turn.completed","usage":'
                            '{"input_tokens":50,"output_tokens":1}}')
    res = await p.complete(model="gpt-5.6-sol", system="S", user="U")
    assert (res.cache_hit_tokens, res.cache_miss_tokens) == (0, 50)


# ---- the ledger surfaces it (item 12) ----------------------------------------

def test_usage_summary_carries_cache_columns(tmp_path):
    store = Store(tmp_path / "s.sqlite3")
    store.record_usage("r", "T-1", "reviewer", "claude_cli", "opus",
                       1000, 20, 0.0, False,
                       cache_hit_tokens=900, cache_miss_tokens=100)
    row = store.usage_summary()[0]
    assert row["cache_hit"] == 900 and row["cache_miss"] == 100
    store.close()
