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


class PlannerNeedsInput(OrchestratorError):
    """The tech-lead planner returned clarifying questions instead of specs.
    One-shot `plan` prints these and exits non-zero; `discuss` handles them
    interactively."""

    def __init__(self, questions: list[dict], assumptions: list[str] | None = None):
        self.questions = questions
        self.assumptions = assumptions or []
        super().__init__(f"planner needs input: {len(questions)} question(s)")
