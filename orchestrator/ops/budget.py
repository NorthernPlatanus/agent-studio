"""Token/cost accounting with hard budgets.

Every LLM call is recorded in the store's usage ledger. Cash spend (paid API
providers) is checked against per-task and per-run budgets; subscription
usage (claude_cli) is logged for visibility but exempt by default
(budget.count_claude_cli).
"""

from __future__ import annotations

import logging

from ..core.errors import BudgetExceeded
from .store import Store

log = logging.getLogger("orchestrator.budget")


class Budget:
    def __init__(self, cfg, store: Store, run_id: str):
        self.store = store
        self.run_id = run_id
        self.per_task = float(cfg.budget.per_task_usd)
        self.per_run = float(cfg.budget.per_run_usd)
        self.count_claude_cli = bool(cfg.budget.count_claude_cli)

    def record(self, *, task_id: str | None, role: str, provider: str,
               provider_type: str, model: str, input_tokens: int,
               output_tokens: int, cost_usd: float) -> None:
        cash = provider_type != "claude_cli" or self.count_claude_cli
        self.store.record_usage(
            self.run_id, task_id, role, provider, model,
            input_tokens, output_tokens, cost_usd, cash)
        log.info("usage %s/%s %s: in=%d out=%d $%.4f%s",
                 role, model, task_id or "-", input_tokens, output_tokens,
                 cost_usd, "" if cash else " (subscription)")
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
