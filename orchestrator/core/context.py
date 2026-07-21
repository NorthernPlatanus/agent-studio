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
        degrade mode.

        Provider precedence: an explicit role.provider wins; a null/missing
        provider inherits roles.smart_provider; final fallback is claude_cli.
        Use .get() throughout so a missing key never raises AttributeError.
        (degrade always targets a worker_models key regardless of the smart
        provider, which is why Codex-as-planner still degrades correctly.)
        """
        rcfg = self.cfg.roles.get(role)
        if self.degraded:
            wm = self.cfg.worker_models.get(self.cfg.run.degrade_model)
            if wm is None:
                raise ValueError("run.degrade_model is not configured")
            return wm.provider, wm.model
        provider = (rcfg.get("provider") if rcfg is not None else None) \
            or self.cfg.roles.get("smart_provider", "claude_cli")
        model = rcfg.get("model") if rcfg is not None else None
        return provider, model

    def worker_target(self, cand_id: str) -> tuple[str, str]:
        # "senior" is the escalation pseudo-candidate: the subscription smart
        # tier acting as an implementer. It resolves to the active smart_provider
        # and run.escalate_model (which must co-vary with smart_provider), NOT a
        # worker_models key — so escalation stays a $0-cash subscription senior.
        if cand_id == "senior":
            provider = self.cfg.roles.get("smart_provider", "claude_cli")
            model = self.cfg.run.get("escalate_model") or "opus"
            return provider, model
        wm = self.cfg.worker_models.get(cand_id)
        if wm is None:
            raise ValueError(f"Unknown worker model key: {cand_id}")
        return wm.provider, wm.model
