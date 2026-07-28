"""RunContext — runtime services bound into graph nodes via closures.

Deliberately NOT part of LangGraph state: state stays plain data
(checkpoint-serializable); services are reconstructed on resume.
"""

from __future__ import annotations

import asyncio
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
    _inspector_lock: asyncio.Lock | None = field(default=None, repr=False)

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

    def role_effort(self, role: str) -> str | None:
        """Reasoning effort for a smart-tier role, or None to leave the
        provider's own default alone.

        Effort is per-role because the roles are not equally hard: the planner
        decomposes a whole backlog item and its mistakes propagate into every
        downstream attempt, while the reviewer grades a diff against acceptance
        criteria that are already written down. Paying planner-grade effort for
        every review spends the binding constraint (subscription tokens) on the
        cheaper half of the problem.

        Precedence: roles.<role>.effort, else providers.<provider>.effort (the
        tier-wide default), else None. Degrade mode returns None — the degrade
        target is a cash worker model, and effort is a claude_cli concept.
        """
        if self.degraded:
            return None
        rcfg = self.cfg.roles.get(role)
        effort = rcfg.get("effort") if rcfg is not None else None
        return effort or self._provider_effort(self.role_target(role)[0])

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

    def worker_effort(self, cand_id: str) -> str | None:
        """Effort for a candidate. Only the `senior` pseudo-candidate has one —
        cheap workers are chat-completions models with no effort concept, so
        they get None and their provider ignores the field entirely."""
        if cand_id != "senior":
            return None
        return (self.cfg.run.get("escalate_effort")
                or self._provider_effort(self.worker_target(cand_id)[0]))

    def role_allowed_tools(self, role: str) -> str | None:
        """Tool allowlist for a role, or None to use the provider-wide one.

        Per-role because tool grants are not uniform across the tier: the
        verifier needs an inspector's `mcp__*` tools, and the planner and
        reviewer must keep the read-only `Read,Grep,Glob` set. The global `mcp:`
        section cannot express that — it applies to every claude_cli role at
        once."""
        rcfg = self.cfg.roles.get(role)
        return (rcfg.get("allowed_tools") if rcfg is not None else None) or None

    @property
    def inspector_lock(self) -> asyncio.Lock:
        """Serializes the visual-verify phase.

        The scene inspector is a SINGLETON: one dev server per port, one bridge
        port, one browser. Candidates are parallel and ephemeral, so without this
        two verifications would fight over the same ports and each would report
        facts about whichever app won the race — a confidently wrong verdict,
        worse than none.

        Scoped to the verify phase only, NOT the pipeline: workers and the
        deterministic gate need no browser, and serializing them would cost
        throughput for no consistency gain. Same shape and rationale as
        `Git.repo_lock`; created lazily so it binds to the running loop.
        """
        if self._inspector_lock is None:
            self._inspector_lock = asyncio.Lock()
        return self._inspector_lock

    def _provider_effort(self, provider_name: str) -> str | None:
        entry = (self.cfg.get("providers") or {}).get(provider_name)
        if entry is None:
            return None
        return self.cfg.providers.get(provider_name).get("effort")
