"""Orchestrator control-flow exceptions."""


class OrchestratorError(Exception):
    """Base class."""


class LimitExhausted(OrchestratorError):
    """The Claude Code subscription limit is exhausted.

    Behavior is config-driven (run.on_limit_exhausted):
      pause   -> checkpoint everything and stop; `orchestrator resume` later.
      degrade -> re-route opus roles to run.degrade_model and continue.
    """


class BudgetExceeded(OrchestratorError):
    """A per-task or per-run cash budget was hit. Run pauses (resumable)."""


class DryRunViolation(OrchestratorError):
    """Something tried to spend money or mutate git during --dry-run."""


class PatchError(OrchestratorError):
    """A worker patch could not be applied (bad SEARCH block, path outside
    files_write, etc.). Counts as a worker failure -> retry with the error."""
