"""Orchestrator control-flow exceptions."""

import time


class OrchestratorError(Exception):
    """Base class."""


class LimitExhausted(OrchestratorError):
    """The Claude Code subscription limit is exhausted.

    Behavior is config-driven (run.on_limit_exhausted):
      pause   -> checkpoint everything and stop; `orchestrator resume` later.
      degrade -> re-route opus roles to run.degrade_model and continue.

    `resets_at` is a unix timestamp when the provider told us one, and None when
    it did not. The Claude CLI does: every call streams a `rate_limit_event`
    carrying `{status, resetsAt, rateLimitType}`, so the window's end is a known
    number rather than something to poll for. `limit_type` is that window's name
    (`five_hour` / `weekly` …), which matters because they reset on different
    clocks and a caller waiting out the wrong one waits far too long.

    Callers that can afford to wait (the planner chat, where one session can
    outlast a five-hour window) should freeze until `resets_at` and carry on.
    """

    def __init__(self, message: str, *, resets_at: float | None = None,
                 limit_type: str | None = None):
        super().__init__(message)
        self.resets_at = resets_at
        self.limit_type = limit_type

    @property
    def seconds_until_reset(self) -> float | None:
        """Seconds from now until the window reopens, floored at 0. None when
        the provider never told us — the caller must not invent a number."""
        if self.resets_at is None:
            return None
        return max(0.0, self.resets_at - time.time())


class BudgetExceeded(OrchestratorError):
    """A per-task or per-run cash budget was hit. Run pauses (resumable)."""


class DryRunViolation(OrchestratorError):
    """Something tried to spend money or mutate git during --dry-run."""


class PatchError(OrchestratorError):
    """A worker patch could not be applied (bad SEARCH block, path outside
    files_write, etc.). Counts as a worker failure -> retry with the error."""


class SessionLost(OrchestratorError):
    """A CLI provider could not resume the session it was asked to continue.

    Callers that sent an abbreviated payload (relying on the session to carry the
    context) MUST catch this and retry once with the full self-contained prompt —
    the provider has already dropped the dead session id, so the retry starts a
    fresh one. Never let this reach a task as a plain failure: the request was
    fine, only the continuation was lost."""


class PlannerNeedsInput(OrchestratorError):
    """The tech-lead planner returned clarifying questions instead of specs.
    One-shot `plan` prints these and exits non-zero; `discuss` handles them
    interactively."""

    def __init__(self, questions: list[dict], assumptions: list[str] | None = None):
        self.questions = questions
        self.assumptions = assumptions or []
        super().__init__(f"planner needs input: {len(questions)} question(s)")
