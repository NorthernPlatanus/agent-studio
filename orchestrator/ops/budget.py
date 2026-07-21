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

    def __init__(self, cfg, store: Store, run_id: str):
        self.store = store
        self.run_id = run_id
        self.per_task = float(cfg.budget.per_task_usd)
        self.per_run = float(cfg.budget.per_run_usd)
        self.count_claude_cli = bool(cfg.budget.get("count_claude_cli", False))
        self.count_cli = bool(cfg.budget.get("count_cli", False))
        # Snapshot provider configs so record() can consult per-provider `auth`
        # and `count` without holding the whole Config.
        self._providers: dict[str, dict] = {}
        for name in (cfg.get("providers") or {}).keys():
            entry = cfg.providers.get(name)
            self._providers[name] = entry.as_dict() if entry is not None else {}

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

    def check(self, task_id: str | None = None) -> None:
        run_spend = self.store.run_cash_spend(self.run_id)
        if run_spend > self.per_run:
            raise BudgetExceeded(
                f"run budget exceeded: ${run_spend:.2f} > ${self.per_run:.2f}")
        if task_id:
            task_spend = self.store.task_cash_spend(task_id)
            if task_spend > self.per_task:
                raise BudgetExceeded(
                    f"task {task_id} budget exceeded: "
                    f"${task_spend:.2f} > ${self.per_task:.2f}")
