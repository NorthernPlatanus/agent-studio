"""ops/pricing — the one price function, and the table the three callers share.

The two things worth testing here are the two things the old flat
`in_tok * pin + out_tok * pout` got wrong: the long-context cliff (which
re-prices the WHOLE request, so getting it wrong understates a big call by
~2x) and cache tiers (a hit is 10x cheaper than a fresh token, and a warm
prefix is the normal case in the worker loop, not the edge case).

The third thing tested is that neither of those changes anything for a model
whose entry declares no cache rates and no cliff — i.e. every entry that exists
in a profile written before presets did.
"""

from pathlib import Path

import pytest

from orchestrator.core.config import Config
from orchestrator.ops.pricing import build_price_table, price

# The luna_high shape PLUS a write rate. The shipped preset deliberately omits
# cache_write_per_mtok (see config/default.yaml) because `cache_miss` here means
# "input not served from cache", not "input written to cache" — but the function
# must still honour the key for a provider that reports a real creation count,
# so the unit tests exercise it.
LUNA = {
    "provider": "openai", "model": "gpt-5.6-luna",
    "input_per_mtok": 0.20, "output_per_mtok": 1.20,
    "cache_read_per_mtok": 0.02, "cache_write_per_mtok": 0.25,
    "long_context": {"threshold_in_tok": 272000,
                     "input_multiplier": 2.0, "output_multiplier": 1.5},
}
# A pre-presets entry: two flat rates and nothing else.
FLAT = {"provider": "cometapi", "model": "deepseek-v4-flash",
        "input_per_mtok": 0.12, "output_per_mtok": 0.24}


# ---- the 272K cliff ---------------------------------------------------------

def _cliff_entry() -> dict:
    """LUNA without the cache rates, so the cliff is measured on its own."""
    return {k: v for k, v in LUNA.items()
            if k not in ("cache_read_per_mtok", "cache_write_per_mtok")}


def test_just_below_the_threshold_is_short_context():
    entry = _cliff_entry()
    assert price(entry, 271_999, 1_000) == pytest.approx(
        271_999 / 1e6 * 0.20 + 1_000 / 1e6 * 1.20)


def test_exactly_at_the_threshold_is_still_short_context():
    """The provider's rule is '>272K', so the boundary token is not over it."""
    entry = _cliff_entry()
    assert price(entry, 272_000, 1_000) == pytest.approx(
        272_000 / 1e6 * 0.20 + 1_000 / 1e6 * 1.20)


def test_just_above_the_threshold_reprices_the_whole_request():
    entry = _cliff_entry()
    # 2x input / 1.5x output on EVERY token in the call, not on the 1 token of
    # overflow. Pricing the overflow only would give ~$0.0544 — a 2x understate.
    assert price(entry, 272_001, 1_000) == pytest.approx(
        272_001 / 1e6 * 0.20 * 2.0 + 1_000 / 1e6 * 1.20 * 1.5)


def test_the_cliff_nearly_doubles_a_call_that_barely_crosses_it():
    entry = _cliff_entry()
    below = price(entry, 271_999, 1_000)
    above = price(entry, 272_001, 1_000)
    assert above > below * 1.9        # a cliff, not a slope


def test_no_long_context_block_means_no_cliff():
    assert price(FLAT, 1_000_000, 1_000) == pytest.approx(
        1_000_000 / 1e6 * 0.12 + 1_000 / 1e6 * 0.24)


# ---- cache tiers ------------------------------------------------------------

def test_cache_hits_are_priced_at_the_read_rate():
    # 900 cached + 100 fresh: the hit tokens must not be charged at the fresh
    # input rate (which is what every caller did before this module existed).
    cost = price(LUNA, 1_000, 0, cache_hit=900, cache_miss=100)
    assert cost == pytest.approx(900 / 1e6 * 0.02 + 100 / 1e6 * 0.25)


def test_a_fully_cached_prompt_is_an_order_of_magnitude_cheaper():
    warm = price(LUNA, 100_000, 0, cache_hit=100_000, cache_miss=0)
    cold = price(LUNA, 100_000, 0, cache_hit=0, cache_miss=100_000)
    assert warm * 10 < cold


def test_flat_fallback_reproduces_todays_arithmetic():
    """An entry with no cache rates prices EXACTLY as it did before presets.

    This is the back-compat guarantee for every profile written so far: hit and
    miss both fall back to input_per_mtok, so the split cancels out.
    """
    in_tok, out_tok = 12_345, 678
    expected = in_tok / 1e6 * 0.12 + out_tok / 1e6 * 0.24
    assert price(FLAT, in_tok, out_tok) == pytest.approx(expected)
    assert price(FLAT, in_tok, out_tok, cache_hit=10_000,
                 cache_miss=2_345) == pytest.approx(expected)


def test_input_the_provider_did_not_split_is_charged_as_uncached():
    """No cache telemetry at all (hit == miss == 0) must not price input at 0."""
    assert price(FLAT, 1_000, 0) == pytest.approx(1_000 / 1e6 * 0.12)


# ---- unknown models ---------------------------------------------------------

def test_a_model_absent_from_the_table_prices_at_zero():
    """As today: an unknown model is never guessed at — a guessed number would
    flow straight into the pre-flight budget guard."""
    table = build_price_table(
        Config({"worker_models": {"w": FLAT}}, "p", Path("/tmp")), "cometapi")
    assert price(table.get("something-else"), 10_000, 10_000) == 0.0
    assert price(None, 10_000, 10_000) == 0.0
    assert price({}, 10_000, 10_000) == 0.0


# ---- the table --------------------------------------------------------------

def _cfg(**sections) -> Config:
    return Config(sections, "proj", Path("/tmp"))


def test_table_covers_presets_worker_models_and_roles():
    """The bug this fixes: a table built from worker_models alone prices every
    planner/reviewer call over HTTP at $0.00."""
    cfg = _cfg(
        presets={"luna_high": LUNA},
        worker_models={"w": FLAT},
        roles={"smart_provider": "openai",
               "planner": {"provider": None, "preset": "luna_high"},
               "reviewer": {"provider": "openai", "model": "gpt-5.6-terra",
                            "input_per_mtok": 1.0, "output_per_mtok": 2.0},
               "worker": {"default": "w", "candidates": ["w"]}},
    )
    openai_table = build_price_table(cfg, "openai")
    assert set(openai_table) == {"gpt-5.6-luna", "gpt-5.6-terra"}
    assert build_price_table(cfg, "cometapi")["deepseek-v4-flash"] == FLAT


def test_a_role_with_a_null_provider_lands_under_the_smart_provider():
    cfg = _cfg(
        presets={},
        roles={"smart_provider": "openai",
               "planner": {"provider": None, "model": "gpt-5.6-luna",
                           "input_per_mtok": 0.20, "output_per_mtok": 1.20}},
    )
    assert "gpt-5.6-luna" in build_price_table(cfg, "openai")


def test_an_inline_price_on_the_entry_beats_the_preset():
    cfg = _cfg(presets={"luna_high": LUNA},
               worker_models={"w": {"preset": "luna_high",
                                    "input_per_mtok": 0.99}})
    entry = build_price_table(cfg, "openai")["gpt-5.6-luna"]
    assert entry["input_per_mtok"] == 0.99
    assert entry["output_per_mtok"] == 1.20        # the preset still fills in


def test_unpriced_entries_never_clobber_a_priced_one():
    """roles.planner names the same model id with no prices; the preset's rates
    must survive, or the smart tier silently re-prices itself to 0.0."""
    cfg = _cfg(presets={"luna_high": LUNA},
               roles={"smart_provider": "openai",
                      "planner": {"provider": "openai", "model": "gpt-5.6-luna"}})
    assert build_price_table(cfg, "openai")["gpt-5.6-luna"]["input_per_mtok"] == 0.20


# ---- the SHIPPED presets, not just the function -----------------------------

def _shipped_presets() -> dict:
    import yaml
    root = Path(__file__).resolve().parents[1]
    data = yaml.safe_load((root / "config" / "default.yaml").read_text())
    return data["presets"]


@pytest.mark.parametrize("key", ["luna_high", "luna_med", "luna_xhigh"])
def test_a_cold_luna_call_is_charged_at_the_plain_input_rate(key):
    """Regression guard on the CONFIG, not the function.

    `cache_miss` is derived as `in_tok - cache_hit`, so it covers every input
    token the provider did not serve from cache — including the volatile suffix
    that is never cached at all. Pricing that at a write premium would inflate
    every cold call by 25%, and input dominates this workload, so the ledger's
    cash column would read high across the board. The shipped Luna presets
    therefore declare a read rate and no write rate.
    """
    entry = _shipped_presets()[key]
    assert "cache_write_per_mtok" not in entry
    cold = price(entry, 100_000, 10_000, cache_hit=0, cache_miss=100_000)
    assert cold == pytest.approx(100_000 / 1e6 * 0.20 + 10_000 / 1e6 * 1.20)


def test_the_shipped_luna_presets_still_price_a_cache_hit_cheaply():
    entry = _shipped_presets()["luna_high"]
    warm = price(entry, 100_000, 10_000, cache_hit=90_000, cache_miss=10_000)
    assert warm == pytest.approx(
        90_000 / 1e6 * 0.02 + 10_000 / 1e6 * 0.20 + 10_000 / 1e6 * 1.20)


def test_the_shipped_luna_presets_keep_the_cliff():
    entry = _shipped_presets()["luna_high"]
    assert entry["long_context"]["threshold_in_tok"] == 272_000
