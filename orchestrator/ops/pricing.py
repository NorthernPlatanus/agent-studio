"""The price table and the price function — one copy, three callers.

Until this module existed the same "model id -> $/Mtok" table was built THREE
times, from the same section, with three slightly different sets of bugs:
`openai_compatible` (recorded cost), `codex_cli._priced` (recorded cost under
`auth: api`), and `Budget` (the pre-flight estimate). Three copies of an
accounting rule is three chances for the ledger and the guard that reads it to
disagree about what a call costs.

Two things this models that the old flat `in_tok * pin + out_tok * pout` did not:

* **Cache tiers.** A cached input token is an order of magnitude cheaper than a
  fresh one ($0.02 vs $0.20/Mtok on the GPT-5.6 family). Pricing every input
  token at the fresh rate overstates a warm-cache call — and the whole worker
  loop is built around keeping a warm prefix, so that is the normal case, not
  the edge case. `cache_read_per_mtok` / `cache_write_per_mtok` are OPTIONAL:
  absent, both fall back to the flat input rate, which reproduces the old number
  exactly. Nothing regresses for a model whose preset declares no cache rates.

* **The long-context cliff.** Above a threshold the provider re-prices the
  ENTIRE request, not the overflow: >272K input tokens on GPT-5.6 costs 2x input
  and 1.5x output on every token in the call, including the first one. A model
  that prices the overflow only would understate a 300K-token call by nearly 2x,
  and the pre-flight budget guard would wave through a call it should have
  refused. `long_context: {threshold_in_tok, input_multiplier, output_multiplier}`.

The price shape lives on the ENTRY (a preset, or a worker_models/roles entry that
inlines the same keys), so adding a model with a different pricing structure is a
config edit, not a code change.
"""

from __future__ import annotations

from typing import Any

from ..core.presets import as_dict, resolve_entry, section

#: Keys that make an entry worth putting in the price table at all. An entry
#: with none of them prices at 0.0, which is indistinguishable from absence —
#: so it is left out, and a later entry for the same model id (a preset that
#: DOES carry prices) is not clobbered by it.
_PRICE_KEYS = ("input_per_mtok", "output_per_mtok",
               "cache_read_per_mtok", "cache_write_per_mtok")


def _f(entry: dict, key: str, default: float = 0.0) -> float:
    value = entry.get(key)
    return default if value is None else float(value)


def has_prices(entry: Any) -> bool:
    """True when the entry declares any price at all."""
    data = as_dict(entry)
    return any(data.get(k) is not None for k in _PRICE_KEYS)


def price(entry: Any, in_tok: int, out_tok: int,
          cache_hit: int = 0, cache_miss: int = 0) -> float:
    """Cost in USD of one call against `entry`'s price shape.

    `entry` is a resolved pricing entry (see `build_price_table`) or None/{} for
    a model the table does not know — which prices at 0.0, exactly as every
    caller did before: an unknown model must not be guessed at, and a guessed
    number would flow straight into the budget guard.

    `cache_hit` / `cache_miss` are the provider's own split of `in_tok` (the
    shapes differ per provider; each provider normalizes before calling here).
    Input the provider did not attribute is charged as uncached — the
    conservative side, and the side that reproduces the old flat number when a
    provider reports no cache telemetry at all.

    MISS IS PRICED AS A CACHE WRITE. An uncached prefix token inside a
    cache-enabled conversation is precisely a token that gets written to the
    cache, so `cache_write_per_mtok` is the rate for the miss side and
    `cache_read_per_mtok` for the hit side. The two rates are optional, and
    omitting them is how a preset says "no cache tiers": both fall back to
    `input_per_mtok` and the split cancels out. A preset that declares a write
    premium it does not want charged on cold input simply leaves the key out.
    """
    data = as_dict(entry)
    if not data:
        return 0.0
    p_in = _f(data, "input_per_mtok")
    p_out = _f(data, "output_per_mtok")
    # Absent cache rates fall back to the flat input rate for BOTH sides, so
    # hit + miss == in_tok reduces to in_tok * p_in — today's arithmetic.
    p_read = _f(data, "cache_read_per_mtok", p_in)
    p_write = _f(data, "cache_write_per_mtok", p_in)

    # Floats are tolerated, not truncated: Budget's pre-flight estimate divides
    # prompt chars by a chars-per-token constant and passes the fractional
    # result, and rounding that down would change the number the guard checks.
    in_tok = max(float(in_tok or 0), 0.0)
    out_tok = max(float(out_tok or 0), 0.0)
    hit = max(int(cache_hit or 0), 0)
    miss = max(int(cache_miss or 0), 0)
    # `in_tok` is the total prompt for every provider this repo talks to, but a
    # provider that reports cache tokens OUTSIDE its input count (the Anthropic
    # shape) would make hit + miss the larger number. Take the larger of the two
    # as the request's real input size so the cliff test below can't be dodged.
    total_in = max(in_tok, hit + miss)
    miss += max(total_in - hit - miss, 0)

    cost_in = hit / 1e6 * p_read + miss / 1e6 * p_write
    cost_out = out_tok / 1e6 * p_out

    lc = as_dict(data.get("long_context"))
    threshold = lc.get("threshold_in_tok")
    if threshold and total_in > int(threshold):
        # EXCEEDS, not reaches: OpenAI's cliff is ">272K input tokens", so a
        # request of exactly the threshold is still short-context. And the
        # multipliers apply to the WHOLE request, which is why they multiply the
        # totals here rather than an overflow slice.
        cost_in *= _f(lc, "input_multiplier", 1.0)
        cost_out *= _f(lc, "output_multiplier", 1.0)
    return cost_in + cost_out


def build_price_table(cfg: Any, provider_name: str) -> dict[str, dict]:
    """model id -> pricing entry, for everything `provider_name` might be asked
    to call.

    Built from `presets` ∪ `worker_models` ∪ `roles`, in that order, because a
    table built from `worker_models` alone (what all three copies used to do)
    silently prices every smart-tier call at 0.0 the moment a planner or a
    reviewer runs over HTTP instead of a CLI. A role is not a worker, but it
    spends the same dollars.

    Later sources win, so a `worker_models` entry with inline prices overrides
    the preset it also names — the entry is the more specific statement. Entries
    carrying no prices are skipped entirely (they price at 0.0 either way, and
    including them would let an unpriced `roles.planner` clobber a priced preset
    for the same model id).
    """
    table: dict[str, dict] = {}

    def add(entry: Any, provider_default: str | None = None) -> None:
        resolved = resolve_entry(cfg, entry)
        if not resolved:
            return
        provider = resolved.get("provider") or provider_default
        model = resolved.get("model")
        if provider != provider_name or not model or not has_prices(resolved):
            return
        table[str(model)] = resolved

    for preset in section(cfg, "presets").values():
        add(preset)
    for entry in section(cfg, "worker_models").values():
        add(entry)

    roles = section(cfg, "roles")
    # A role with `provider: null` inherits roles.smart_provider (see
    # RunContext.role_target) — resolve that here too, or a preset-less smart
    # role would never match its own provider and would price at 0.0 again.
    smart = roles.get("smart_provider") or "claude_cli"
    for entry in roles.values():
        data = as_dict(entry)
        if not data.get("model"):        # skips smart_provider (a string) and
            continue                     # roles.worker (a pool, not a binding)
        add(entry, provider_default=smart)
    return table
