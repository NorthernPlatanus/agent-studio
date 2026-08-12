"""Pydantic v2 response models — the contract with the UI.

Every field name and type here is a promise to `studio-web` (which generates its
TypeScript from `/openapi.json`), so a rename is a contract change and belongs in
`DEVDOCS/DECISIONS.md`.

Two conventions worth knowing before reading further:

- **Nothing is invented.** Each field traces to a column in `state/<p>.sqlite3`,
  to `TaskState`, or to a pure function in `engine/scheduler`. Where a figure is
  derived (`cache_hit_rate`, `*_per_completed_task`) it is `None` rather than 0
  when the inputs are missing, because "not reported" and "measured zero" are
  different answers and a dashboard that conflates them lies.
- **Task specs are open.** `TaskDetail.spec` is the raw planner blob: the planner
  may add keys, and dropping unknown ones would silently hide them from the only
  UI that shows specs. The promoted fields are the ones the UI filters on.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ..ops import liveness

TaskStatus = Literal["ready", "running", "needs_human", "done", "failed",
                     "rejected", "human_only", "needs_plan"]
RunStatus = Literal["running", "paused", "done", "aborted"]
UsageGroupBy = Literal["role", "model", "provider", "day"]
EventOrder = Literal["asc", "desc"]
# Which config layer supplied `project.repo_path` (see deps.repo_path_provenance).
RepoPathSource = Literal["profile", "global", "env"]


class Health(BaseModel):
    status: Literal["ok"]
    version: str


# ---- projects ------------------------------------------------------------
class Project(BaseModel):
    name: str
    profile_path: str | None = Field(
        None, description="profile that defines it, when it came from disk")
    has_store: bool
    store_path: str | None
    has_checkpoints: bool
    repo_path: str | None = Field(
        None, description="the EFFECTIVE value after the config merge; null when it "
                          "is set nowhere — such a project is readable but cannot "
                          "run jobs")
    repo_path_source: RepoPathSource | None = Field(
        None, description="which layer supplied repo_path: this project's own "
                          "'profile', the machine-global config/local.yaml "
                          "('global'), or 'env'. null when unset")
    runnable: bool = Field(
        description="a checkout resolves, so job endpoints that need one will not "
                    "409. Read together with repo_path_source: 'global' means the "
                    "checkout is not this project's own")
    runnable_detail: str | None = Field(
        None, description="why it cannot run, or the caveat when repo_path was "
                          "inherited rather than declared; null when the project's "
                          "own profile names its checkout")
    is_active: bool = Field(description="matches ORCH_PROJECT")


class Projects(BaseModel):
    projects: list[Project]
    active: str | None = Field(None, description="ORCH_PROJECT, if set and allowlisted")


# ---- shared money/token shapes -------------------------------------------
class ChannelTotals(BaseModel):
    calls: int
    in_tok: int
    out_tok: int
    cache_hit: int
    cache_miss: int
    cost: float = Field(description="USD; real for cash, notional quota for subscription")


class TokenChannels(BaseModel):
    """Both billing channels, either of which may be absent (no calls yet)."""
    cash: ChannelTotals | None = None
    subscription: ChannelTotals | None = None


# ---- tasks ---------------------------------------------------------------
class TaskListItem(BaseModel):
    id: str
    title: str
    status: TaskStatus | str
    milestone: str | None = None
    retries: int = 0
    cost_usd: float = 0.0
    updated_at: float | None = None
    domain: str | None = None
    risk: str | None = None
    complexity: str | None = None
    visual: bool | None = None
    agent_able: bool | None = None
    n_candidates: int | None = None
    parent_id: str | None = None
    deps: list[str] = Field(default_factory=list)


class Event(BaseModel):
    rowid: int
    ts: float
    run_id: str | None = None
    task_id: str | None = None
    kind: str
    detail: str | None = None


class TaskDetail(TaskListItem):
    files_read: list[str] = Field(default_factory=list)
    files_write: list[str] = Field(default_factory=list)
    acceptance: list[str] = Field(default_factory=list)
    cash_spend_usd: float = Field(
        description="lifetime cash across runs; tasks.cost_usd is the same figure "
                    "accumulated at write time and can lag a re-run")
    children: list[str] = Field(default_factory=list,
                                description="sub-task ids when this task was decomposed")
    spec: dict[str, Any] = Field(description="raw planner spec blob, unfiltered")
    events: list[Event] = Field(default_factory=list)


class Tasks(BaseModel):
    tasks: list[TaskListItem]
    total: int = Field(description="rows matching the filter")
    queue_stats: dict[str, int] = Field(
        description="status -> count for the FILTERED set")


# ---- candidates ----------------------------------------------------------
class Candidate(BaseModel):
    cand_id: str
    attempt: int
    status: str | None = Field(
        None, description="gate_passed | gate_failed | patch_failed | llm_failed | "
                          "visual_failed | visual_unverifiable | skipped")
    model: str | None = None
    branch: str | None = None
    worktree: str | None = None
    no_patch: bool = False
    error: str | None = None
    gate_log: str | None = Field(None, description="tail only, never the full log")
    notes: str | None = None
    has_diff: bool = False
    visual_facts: dict[str, Any] | None = None


class Candidates(BaseModel):
    task_id: str
    run_id: str | None = Field(None, description="run whose thread was read")
    source: Literal["checkpoint", "events", "none"]
    candidates: list[Candidate]


# ---- waves ---------------------------------------------------------------
class WaveTask(BaseModel):
    id: str
    title: str
    domain: str | None = None
    n_candidates: int
    candidates: list[str] = Field(description="worker_models keys that would run")
    files_write: list[str] = Field(default_factory=list)
    files_read: list[str] = Field(default_factory=list)


class Wave(BaseModel):
    index: int = Field(description="1-based, matches `run --dry-run` output")
    tasks: list[WaveTask]


class Waves(BaseModel):
    max_parallel_tasks: int
    default_n_candidates: int
    waves: list[Wave]
    unreachable: list[str] = Field(
        default_factory=list,
        description="ready tasks no wave can reach this run (unmet deps)")
    seam_missing_deps: list[str] = Field(
        default_factory=list,
        description="seam-domain tasks with no deps — almost always a planner mistake")
    queue_stats: dict[str, int]
    domain_stats: dict[str, int]


# ---- runs ---------------------------------------------------------------
class RunListItem(BaseModel):
    id: str
    started_at: float
    status: RunStatus | str
    note: str | None = None
    cost_usd: float = 0.0
    tokens: TokenChannels
    last_activity_at: float | None = Field(
        None, description="newest events/usage row this run wrote; null when it "
                          "wrote none. Not the same as started_at")
    stale: bool = Field(
        False,
        description="the row says `running` but nothing has happened for "
                    "`stale_after_s` — the process died without writing a terminal "
                    "status. Render it as stalled, not running; `reconcile` closes it")
    stale_after_s: float = Field(
        liveness.STALE_AFTER_S,
        description="the silence the `stale` flag was computed against")


class Runs(BaseModel):
    runs: list[RunListItem]


class RunDetail(RunListItem):
    task_ids: list[str]
    events: list[Event]


# ---- usage --------------------------------------------------------------
class UsageRow(BaseModel):
    key: str = Field(description="human label for the group")
    cash: bool = Field(description="False = subscription tier (logged, not billed)")
    role: str | None = None
    provider: str | None = None
    model: str | None = None
    day: str | None = Field(None, description="local YYYY-MM-DD, when group_by=day")
    calls: int
    in_tok: int
    out_tok: int
    cache_hit: int
    cache_miss: int
    cost: float
    cache_hit_rate: float | None = Field(
        None, description="hit/(hit+miss); null when the provider reported no cache "
                          "telemetry at all — never 0.0 for 'unknown'")


class Usage(BaseModel):
    group_by: UsageGroupBy
    rows: list[UsageRow]
    totals: TokenChannels


# ---- metrics ------------------------------------------------------------
class GateOutcome(BaseModel):
    cand_id: str
    first_attempt: bool
    passed: int
    failed: int
    pass_rate: float


class RoleTokens(BaseModel):
    role: str
    calls: int
    in_tok: int
    out_tok: int
    cache_hit: int
    in_tok_per_completed_task: float | None = None


class Metrics(BaseModel):
    completed_tasks: int
    gate_outcomes: list[GateOutcome]
    event_counts: dict[str, int] = Field(
        description="the kinds `orchestrator metrics` counts; 0-filled")
    subscription_tokens_by_role: list[RoleTokens]
    subscription_in_tok_per_completed_task: float | None = None
    cash_usd_per_completed_task: float | None = None
    queue_stats: dict[str, int]


# ---- events -------------------------------------------------------------
class Events(BaseModel):
    events: list[Event]
    order: EventOrder = Field(
        description="asc = oldest-first (page forward from a cursor); "
                    "desc = newest-first (the dashboard's recent-events tail)")
    next_since_rowid: int = Field(
        description="pass back as ?since_rowid= to continue; unchanged when the page "
                    "was empty. Always the HIGHEST rowid in the page, both orders")
    max_rowid: int = Field(description="newest row in the table, for lag display")


# ---- summary ------------------------------------------------------------
class Summary(BaseModel):
    project: str
    task_count: int
    queue_stats: dict[str, int]
    domain_stats: dict[str, int]
    active_run: RunListItem | None = Field(
        None, description="the running/paused run, if any")
    last_run: RunListItem | None = None
    totals: TokenChannels
    cash_spend_usd: float
    event_counts: dict[str, int]
    max_event_rowid: int


# ---- jobs (phase 3 owns the writes; these are the read shapes) -----------
#: Every command the supervisor can spawn. One alias, used by both the read shape
#: and the 202 body, because they are the same set: a command that can be spawned
#: is a command that will be listed, and splitting them let `reconcile` be
#: spawnable but unserializable — a 500 on its own 202, and on every `GET /jobs`
#: afterwards for as long as the record (or its sidecar) survived.
JobCommandName = Literal["run", "plan", "resume", "import-backlog", "reconcile"]


class Job(BaseModel):
    job_id: str
    project: str
    command: JobCommandName
    status: Literal["starting", "running", "exited", "stopped", "failed"]
    argv: list[str]
    pid: int | None = None
    run_id: str | None = None
    started_at: float
    ended_at: float | None = None
    exit_code: int | None = None
    log_path: str | None = None


class Jobs(BaseModel):
    jobs: list[Job]


class JobLog(BaseModel):
    job_id: str
    offset: int = Field(description="byte offset this chunk starts at")
    next_offset: int
    eof: bool
    text: str


# ---- job requests --------------------------------------------------------
# Every field below becomes an element of a subprocess argv, so each is
# constrained to a shape argparse cannot mistake for a flag. There is no shell
# (`Popen` takes a list), so quoting is not the risk; being read as an option is.
_ARGV_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _argv_safe_ids(v: list[str] | None) -> list[str] | None:
    if v is not None:
        for task_id in v:
            if not _ARGV_SAFE.match(task_id):
                raise ValueError(f"unsafe task id: {task_id!r}")
    return v


class RunRequest(BaseModel):
    """`run`. The only endpoint in the API that can spend subscription quota."""

    confirm: bool = Field(
        False, description="required for a real run — it spends quota and writes to "
                           "git worktrees. Not required when dry_run is true")
    dry_run: bool = Field(
        False, description="`--dry-run`: the runner prints the schedule and exits. "
                           "Zero tokens, zero git, so no confirmation is asked for")
    tasks: list[str] | None = Field(
        None, description="explicit task ids; null lets the scheduler pick")
    n: int | None = Field(
        None, ge=1, le=64,
        description="`--n`: override `n_candidates` (best-of-N) for EVERY task this "
                    "run dispatches, replacing each task's planner-set count. This "
                    "is a multiplier on LLM calls, not a cap on anything — there is "
                    "no cap-on-tasks flag. Null leaves each task's own spec alone")

    _safe_tasks = field_validator("tasks")(_argv_safe_ids)

    @model_validator(mode="after")
    def _spending_is_explicit(self) -> RunRequest:
        # PLAN §3.1 rule 6: no token-spending endpoint is reachable by accident.
        # A 422 rather than a 409 — the body itself is what is wrong.
        if not self.dry_run and not self.confirm:
            raise ValueError("a real run spends subscription quota: pass "
                             "{'confirm': true}, or {'dry_run': true} to preview")
        return self


class PlanRequest(BaseModel):
    """`plan`. Spends planner tokens, so it is confirm-gated too."""

    confirm: bool = Field(False, description="required — planning spends tokens")
    tasks: list[str] | None = None
    all_needs_plan: bool = Field(
        False, description="`--all-needs-plan`: plan every needs_plan task")
    limit: int | None = Field(None, ge=1, le=64)
    note: str = Field(
        "", max_length=2000,
        description="free-text steer, passed as the positional argument")

    _safe_tasks = field_validator("tasks")(_argv_safe_ids)

    @field_validator("note")
    @classmethod
    def _note_is_not_a_flag(cls, v: str) -> str:
        # argparse would read a leading "-" as an option even in the positional
        # slot, so the note is rejected rather than silently mangled.
        if v.startswith("-"):
            raise ValueError("note must not start with '-'")
        return v

    @model_validator(mode="after")
    def _spending_is_explicit(self) -> PlanRequest:
        if not self.confirm:
            raise ValueError("planning spends tokens: pass {'confirm': true}")
        return self


class ResumeRequest(BaseModel):
    """`resume`. Continues the paused run, so it spends whatever remains."""

    confirm: bool = Field(False, description="required — resuming continues spending")

    @model_validator(mode="after")
    def _spending_is_explicit(self) -> ResumeRequest:
        if not self.confirm:
            raise ValueError("resuming continues a run that spends quota: pass "
                             "{'confirm': true}")
        return self


class ReconcileRequest(BaseModel):
    """`reconcile`. Rewrites abandoned `running` rows to `aborted` — no LLM, no
    git, so no confirmation. The only mutation in the panel that costs nothing."""

    dry_run: bool = Field(
        False, description="list what would be closed and change nothing")


class ImportBacklogRequest(BaseModel):
    """`import-backlog`. Registers stubs from markdown — no LLM, no confirmation.

    An empty body is the whole point: this is the one job that costs nothing, so
    the UI can offer it as a plain button.
    """


# ---- planner chat (discuss) ---------------------------------------------
PlannerEffort = Literal["low", "medium", "high", "xhigh", "max"]
#: A generous character bound on an uploaded pin, so a body that could never fit
#: the byte cap is rejected before it is read. The real cap is
#: `api.discuss.MAX_PIN_BYTES`, applied in bytes after decoding.
MAX_PIN_CHARS = 256 * 1024
DiscussStatus = Literal["running", "awaiting", "done", "aborted", "failed"]
#: What the loop is blocked on. `answer` = the planner asked something;
#: `decision` = a spec preview is on the table and wants y / edit / abort.
DiscussExpects = Literal["answer", "decision"]
DiscussFrameKind = Literal["you", "thinking", "assumption", "question", "awaiting",
                           "note", "specs_preview", "applied", "aborted", "error",
                           "closed"]


class DiscussSettingsModel(BaseModel):
    """Session settings. Every field maps to a `plan_or_ask` argument or a config
    value the planner call reads — none of them is a preference merely stored.

    All of them can be changed mid-session; the loop re-reads them at the top of
    each turn, so a change lands on the next planner call, not the next session.
    """

    note: str = Field(
        "", max_length=4000,
        description="folded into every turn as the HUMAN NOTE block")
    only_ids: list[str] | None = Field(
        None, description="restrict the backlog excerpt the planner is shown "
                          "(`plan_or_ask(only_ids=)`); null = the whole backlog")
    effort: PlannerEffort | None = Field(
        None, description="overrides roles.planner.effort for this session. Higher "
                          "effort spends more subscription tokens per call")
    model: str | None = Field(
        None, description="overrides roles.planner.model for this session")
    session_reuse: bool | None = Field(
        None, description="overrides run.session_reuse. On, turn 2+ sends only the "
                          "newest human turn and the provider session supplies the "
                          "rest — far cheaper, and lost if the session drops")
    max_question_rounds: int = Field(
        0, ge=0, le=20,
        description="force a spec proposal after this many clarify rounds. "
                    "0 = no limit. Unanswered questions are reported, not dropped")

    _safe_ids = field_validator("only_ids")(_argv_safe_ids)


class PinnedFileInfo(BaseModel):
    """A file attached to every planner turn. The text is not echoed back — the
    UI already has it, and a 64KB blob per pin in every poll response is waste."""

    path: str = Field(
        description="a display name under `uploaded/`, not a location. The content "
                    "was sent by the operator and exists nowhere on disk")
    bytes: int
    truncated: bool = Field(
        description="the file exceeded the pin cap and only its head is in the prompt")


class UploadedPin(BaseModel):
    """File content sent from the operator's machine — a log, a spec, notes out
    of a tracker, or a source file they would rather point at than describe.

    Content, never a path into the checkout. Naming a path was the earlier
    design and is gone: it asked the operator to hand-type something the planner
    can usually find on its own, and it could not carry the case that matters
    most, a file that is not in the repo at all.

    Text in JSON, not a multipart body: the request that most needs an upload is
    `POST …/discuss`, which creates the session *and* starts the billable first
    turn, so a pin that arrives in a second request has already missed the turn
    it was for. Uploading through the same JSON keeps the staged case and the
    live case on one mechanism.

    The planner prompt is text; an image cannot reach it in any form, so a binary
    upload is refused with that reason rather than pinned as unreadable filler.
    """

    name: str = Field(min_length=1, max_length=255,
                      description="the original filename, reduced server-side to a "
                                  "safe display name under `uploaded/`")
    text: str = Field(max_length=MAX_PIN_CHARS,
                      description="the file's text. Over the per-pin cap it is "
                                  "truncated and reported as such")


class DiscussFrame(BaseModel):
    """One event from the loop. `seq` is the replay cursor: reconnect with
    `?since=<seq>` and the stream resumes exactly where it stopped."""

    seq: int
    ts: float
    kind: DiscussFrameKind | str
    # Required, not defaulted: a defaulted field is `not required` in the
    # generated schema, and every consumer then has to narrow a value the server
    # always sends. An empty dict is the empty case.
    data: dict[str, Any] = Field(
        description="kind-dependent: `question` carries id/q/why, `specs_preview` "
                    "carries the proposed specs, `you` carries text")


class DiscussSessionModel(BaseModel):
    """A session, whole. Every field is always present — see `DiscussFrame.data`
    for why none of these are defaulted."""

    session_id: str
    project: str
    request: str = Field(description="the operator's opening message")
    status: DiscussStatus
    expects: DiscussExpects | None = Field(
        description="set only while status is `awaiting`; null otherwise")
    started_at: float
    last_activity_at: float
    error: str | None = Field(description="null unless the session failed")
    applied: list[dict[str, Any]] = Field(
        description="specs written to the store, once approved; empty before that")
    settings: DiscussSettingsModel
    pins: list[PinnedFileInfo]
    frames: list[DiscussFrame] = Field(
        description="the conversation so far — from `?since=` when one was given, "
                    "otherwise all of it, for a cold load or a reconnect")


class DiscussOptions(BaseModel):
    """What this project's config actually permits, so the settings UI offers
    real choices instead of a hardcoded list that can drift from the harness."""

    efforts: list[PlannerEffort]
    models: list[str] = Field(description="known model ids for the planner role")
    configured_provider: str
    configured_model: str | None
    configured_effort: PlannerEffort | None
    configured_session_reuse: bool
    max_pin_bytes: int
    idle_ttl_s: float


class DiscussState(BaseModel):
    """`GET …/discuss` — everything a cold page load needs in one request."""

    project: str
    session: DiscussSessionModel | None = None
    transcript: str = Field(
        "", description="the persisted transcript of the last session "
                        "(`store.load_discussion`), for history when nothing is live")
    options: DiscussOptions
    blocked_by_job: str | None = Field(
        None, description="a job is in flight, so a session cannot start — its "
                          "command, for the message")


class StartDiscussRequest(BaseModel):
    """Opening a session spends planner tokens on its very first turn, so it is
    confirm-gated exactly like `plan`."""

    request: str = Field(min_length=1, max_length=8000,
                         description="the opening message / feature description")
    confirm: bool = Field(False, description="required — the planner call spends quota")
    settings: DiscussSettingsModel = Field(default_factory=DiscussSettingsModel)
    uploads: list[UploadedPin] = Field(
        default_factory=list, max_length=32,
        description="files sent from the operator's machine, attached from the first "
                    "turn. Here rather than in a follow-up request because the "
                    "first turn is the expensive one and is started by this call")

    @model_validator(mode="after")
    def _spending_is_explicit(self) -> StartDiscussRequest:
        if not self.confirm:
            raise ValueError("a discuss session spends subscription quota on its "
                             "first turn: pass {'confirm': true}")
        return self


class DiscussReplyRequest(BaseModel):
    text: str = Field(max_length=8000,
                      description="an answer, or y / edit / abort at the preview")


class PinRequest(BaseModel):
    """Names an existing pin, for removal. Not a location — see `PinnedFileInfo`."""

    path: str = Field(min_length=1, max_length=1024,
                      description="the pin's display path, as `PinnedFileInfo.path`")


# ---- live stream ---------------------------------------------------------
# These are never serialized by a handler — SSE frames are written by
# `sse_starlette`, not by a response model. They exist so the frame payloads are
# part of the generated contract instead of being hand-typed in `sse.ts`, which
# is the drift the generated-types rule exists to prevent.
StreamEventName = Literal["hello", "tasks", "runs", "usage", "events", "jobs",
                          "heartbeat"]


class StreamHello(BaseModel):
    """`event: hello` — the first frame, and the thing that makes reconnects cheap.

    A client adopts `event_rowid`, refetches once, and receives only deltas after
    that; without it a reconnect either replays the log or silently drops the
    rows written while it was away.
    """

    tasks: str
    runs: str
    usage: str
    events: str
    jobs: str
    event_rowid: int = Field(description="the client is caught up to this rowid")
    poll_interval_s: float = Field(description="server's tick, for staleness UI")


class StreamCursor(BaseModel):
    """`event: tasks|runs|usage|jobs` — a signal, not data. Refetch the entity.

    The cursor is an opaque digest: compare it for equality, never parse it.
    """

    cursor: str


class StreamEvents(BaseModel):
    """`event: events` — the one frame carrying rows, because the log is append-only."""

    events: list[Event]
    next_since_rowid: int
    truncated: bool = Field(
        description="the burst hit the server's per-frame cap; refetch from "
                    "next_since_rowid rather than trusting this frame to be complete")


class StreamHeartbeat(BaseModel):
    """`event: heartbeat` — only sent when a tick produced nothing else."""

    event_rowid: int


class StreamFrame(BaseModel):
    """Documentation-only envelope: which payload arrives with which event name.

    SSE carries the name in the frame's `event:` line, not inside `data`, so this
    cannot be a discriminated union — the client switches on the event name and
    then reads `data` as the matching member.
    """

    data: StreamHello | StreamEvents | StreamHeartbeat | StreamCursor


class JobAccepted(BaseModel):
    """202 body for every spawn. `run_id` is null until the child mints it."""

    job_id: str
    project: str
    command: JobCommandName
    argv: list[str] = Field(
        description="the exact command line spawned — the panel shows it so a "
                    "human can reproduce the job in a terminal")
    run_id: str | None = None
