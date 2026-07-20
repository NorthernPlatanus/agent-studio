"""RunContext — runtime services bound into graph nodes via closures.

Deliberately NOT part of LangGraph state: state stays plain data
(checkpoint-serializable); services are reconstructed on resume.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..ops.budget import Budget
from .config import Config
from ..ops.gitops import Git
from ..ops.store import Store


@dataclass
class RunContext:
    cfg: Config
    store: Store
    git: Git
    budget: Budget
    run_id: str
    dry_run: bool = False
    degraded: bool = False   # opus roles rerouted to run.degrade_model

    def role_target(self, role: str) -> tuple[str, str]:
        """Resolve (provider_name, model_id) for planner/reviewer, honoring
        degrade mode."""
        rcfg = self.cfg.roles.get(role)
        if self.degraded:
            wm = self.cfg.worker_models.get(self.cfg.run.degrade_model)
            if wm is None:
                raise ValueError("run.degrade_model is not configured")
            return wm.provider, wm.model
        return rcfg.provider, rcfg.model

    def worker_target(self, cand_id: str) -> tuple[str, str]:
        wm = self.cfg.worker_models.get(cand_id)
        if wm is None:
            raise ValueError(f"Unknown worker model key: {cand_id}")
        return wm.provider, wm.model
