"""Token/cost accounting with hard budgets.

Every LLM call is recorded in the store's usage ledger. Cash spend (paid API
providers) is checked against per-task and per-run budgets; subscription-auth
CLI usage (claude_cli, and codex_cli with auth:subscription) is logged for
visibility but exempt by default.

Whether a call counts as cash is generalized across CLI providers:
  * openai_compatible (and anything non-CLI) always counts.
  * claude_cli counts only if budget.count_claude_cli or budget.count_cli.
  * codex_cli counts if auth == api, else only under the count_cli/back-compat
    toggles (subscription = logged, not counted).
  * a per-provider `count: true|false` overrides the above.
"""

from __future__ import annotations

import logging

from ..core.errors import BudgetExceeded
from .store import Store

log = logging.getLogger("orchestrator.budget")


class Budget:
    # CLI provider types are subscription-metered by default (not cash).
    _CLI_TYPES = ("claude_cli", "codex_cli")

    # Rough bytes-per-token for the pre-flight estimate. Deliberately LOW (real
    # English/code averages ~3.5-4), so the estimate over-counts tokens and the
    # guard errs toward stopping early rather than overshooting.
    _CHARS_PER_TOKEN = 3.0

    def __init__(self, cfg, store: Store, run_id: str):
        self.store = store
        self.run_id = run_id
        self.per_task = float(cfg.budget.per_task_usd)
        self.per_run = float(cfg.budget.per_run_usd)
        self.count_claude_cli = bool(cfg.budget.get("count_claude_cli", False))
        self.count_cli = bool(cfg.budget.get("count_cli", False))
        self.assumed_max_output = int(
            cfg.budget.get("assumed_max_output_tokens", 8000))
        # Snapshot provider configs so record() can consult per-provider `auth`
        # and `count` without holding the whole Config.
        self._providers: dict[str, dict] = {}
        for name in (cfg.get("providers") or {}).keys():
            entry = cfg.providers.get(name)
            self._providers[name] = entry.as_dict() if entry is not None else {}
        # model id -> ($/Mtok in, $/Mtok out), for the pre-flight estimate only;
        # recorded cost still comes from the provider.
        self._prices: dict[str, tuple[float, float]] = {}
        for wm in (cfg.get("worker_models") or {}).keys():
            entry = cfg.worker_models.get(wm)
            if entry is None:
                continue
            self._prices[entry.get("model")] = (
                float(entry.get("input_per_mtok", 0) or 0),
                float(entry.get("output_per_mtok", 0) or 0))

    def _is_cash(self, provider: str, provider_type: str) -> bool:
        pcfg = self._providers.get(provider, {})
        if "count" in pcfg:                       # explicit per-provider override
            return bool(pcfg["count"])
        if provider_type not in self._CLI_TYPES:  # openai_compatible etc. = real cash
            return True
        if provider_type == "codex_cli" and pcfg.get("auth", "subscription") == "api":
            return True
        # subscription-auth CLI: counted only under the global toggles
        return self.count_cli or (provider_type == "claude_cli" and self.count_claude_cli)

    def record(self, *, task_id: str | None, role: str, provider: str,
               provider_type: str, model: str, input_tokens: int,
               output_tokens: int, cost_usd: float,
               cache_hit_tokens: int = 0, cache_miss_tokens: int = 0) -> None:
        cash = self._is_cash(provider, provider_type)
        self.store.record_usage(
            self.run_id, task_id, role, provider, model,
            input_tokens, output_tokens, cost_usd, cash,
            cache_hit_tokens, cache_miss_tokens)
        log.info("usage %s/%s %s: in=%d (cache_hit=%d) out=%d $%.4f%s",
                 role, model, task_id or "-", input_tokens, cache_hit_tokens,
                 output_tokens, cost_usd, "" if cash else " (subscription)")
        self.check(task_id)

    def estimate_and_check(self, *, task_id: str | None, provider: str,
                           provider_type: str, model: str,
                           prompt_chars: int,
                           max_output_tokens: int | None = None) -> float:
        """Pre-flight guard: refuse a call whose ESTIMATED cost would breach a cap.

        `record()` writes the row and then checks, so a single large call can
        overshoot `per_task_usd` / `per_run_usd` by an unbounded amount before the
        exception fires — mild on the cheap tier today, serious the moment a
        smart tier moves to metered billing. This runs before the call instead.

        The estimate is deliberately conservative (low chars-per-token, and the
        full configured output allowance assumed spent), so it stops early rather
        than late; `record()`'s post-hoc check stays as the exact backstop.
        Returns the estimate in USD (0.0 for subscription-metered calls, which
        are exempt and never blocked). Raises BudgetExceeded.
        """
        if not self._is_cash(provider, provider_type):
            return 0.0
        price_in, price_out = self._prices.get(model, (0.0, 0.0))
        if not price_in and not price_out:
            return 0.0            # no price table entry: nothing to estimate from
        in_tok = prompt_chars / self._CHARS_PER_TOKEN
        out_tok = max_output_tokens or self.assumed_max_output
        estimate = in_tok / 1e6 * price_in + out_tok / 1e6 * price_out

        run_spend = self.store.run_cash_spend(self.run_id)
        if run_spend + estimate > self.per_run:
            raise BudgetExceeded(
                f"run budget would be exceeded by the next call: "
                f"${run_spend:.2f} spent + ~${estimate:.2f} est > ${self.per_run:.2f}")
        if task_id:
            task_spend = self.store.task_cash_spend(task_id, run_id=self.run_id)
            if task_spend + estimate > self.per_task:
                raise BudgetExceeded(
                    f"task {task_id} budget would be exceeded by the next call: "
                    f"${task_spend:.2f} spent + ~${estimate:.2f} est > "
                    f"${self.per_task:.2f}")
        return estimate

    def check(self, task_id: str | None = None) -> None:
        run_spend = self.store.run_cash_spend(self.run_id)
        if run_spend > self.per_run:
            raise BudgetExceeded(
                f"run budget exceeded: ${run_spend:.2f} > ${self.per_run:.2f}")
        if task_id:
            # Scoped to THIS run: the cap is "what this run may spend on the
            # task", not a lifetime total that a re-run inherits already blown.
            task_spend = self.store.task_cash_spend(task_id, run_id=self.run_id)
            if task_spend > self.per_task:
                raise BudgetExceeded(
                    f"task {task_id} budget exceeded in run {self.run_id}: "
                    f"${task_spend:.2f} > ${self.per_task:.2f}")
