# agent-studio

A **project-agnostic multi-agent development orchestrator** on LangGraph 1.x.
The core knows nothing about any particular language, framework, or repo —
every project-specific detail (paths, gate commands, worker models, coding
protocol) lives in `config/projects/<name>.yaml` and `config/prompts/`. See
`config/projects/example.yaml` for the template to copy when wiring up a
project.

## The economic model

Two tiers of intelligence, priced accordingly:

| Role | Model | Channel | Cost |
|---|---|---|---|
| Planner | Opus | Claude Code CLI (`claude -p`) | subscription (uses your weekly limit) |
| Reviewer | Opus | Claude Code CLI | subscription |
| Workers | DeepSeek / GLM / Kimi | CometAPI (OpenAI-compatible) | cents |
| Scheduler, Gate, Integrator | — | pure Python + git | **zero** |

The design maximizes quality-per-token:

- **Workers get a tiny, explicit context.** No filesystem, no tools. They see
  the task spec, a distilled protocol excerpt, and the exact files the planner
  listed — nothing else. The planner's `files_read` list *is* the worker's
  entire world; authoring it well is the single most important act in the
  system, which is why the planner is Opus.
- **The gate is deterministic and free.** typecheck/lint/test/build run as
  subprocesses in the candidate's worktree. Broken code never reaches the
  Opus reviewer; failure logs go straight back to the cheap worker.
- **The reviewer only sees green diffs.** Narrow context applies to Opus too:
  the reviewer sees spec + diff, never the queue; the planner sees the
  backlog, never the diffs.

## Architecture

```
 orchestrator plan  ──►  PLANNER (Opus/CLI): backlog ─► enriched specs ─► SQLite store
                                                          │
 orchestrator run   ──►  SCHEDULER (pure py): deps toposort + files_write-disjoint batches
                                                          │  per task, parallel
                                              ┌───────────▼───────────┐
                                              │  LangGraph task graph  │  thread_id = run:task
                                              │                        │  SqliteSaver checkpoints
        dispatch ──Send fan-out──► work_candidate(×N) ──► collect      │
           ▲    (candidates: deepseek/glm/kimi, each in its own        │
           │     git worktree; LLM ─► patch ─► GATE, zero tokens)      │
           ├── retries (≤ max_retries), feedback = raw gate log        │
           │                                        │ any green        │
           ├── review says "revise" ◄── REVIEWER (Opus/CLI, diff only) │
           │                                        │ approve          │
           │                              INTEGRATOR (pure py + git):  │
           │                              merge winner ─► feature      │
           │                              branch, backlog writeback,   │
           └──────────────────────────────worktree cleanup ─► finalize │
                                              └────────────────────────┘
```

Git ownership: **the orchestrator owns `agents/feature`** — candidate
branches merge into it automatically after green gate + approval. Merging
`feature → main` (and release tags) is a human/separate-agent decision, on
purpose.

## Setup

```bash
cd agent-studio
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env      # add your COMETAPI_KEY
# Claude Code CLI must be installed and logged in (subscription auth):
claude --version
```

Copy `config/projects/example.yaml` to `config/projects/<yourname>.yaml`, fill
in the real values (or leave paths null and set them in `config/local.yaml`,
see `config/local.yaml.example`), and verify worker model ids/prices against
your provider's current list — aggregator ids rotate.

## Usage

```bash
# 1. Register backlog items as stubs (no LLM):
python -m orchestrator import-backlog --project example

# 2. Enrich into executable specs (Opus reads the repo; this is where
#    files_read/files_write/deps/agent_able get authored):
python -m orchestrator plan --project example
#    ...or discuss & re-plan around a new idea:
python -m orchestrator plan --project example "add rate limiting to the API layer; fold into M4"

# 3. See what would happen (waves, candidates, file lists — zero tokens):
python -m orchestrator run --project example --dry-run

# 4. Execute:
python -m orchestrator run --project example
python -m orchestrator run --project example --tasks T-120,T-121 --n 3   # best-of-3

# 5. Observe / continue:
python -m orchestrator status --project example
python -m orchestrator resume --project example    # after a limit/budget pause
```

(`example` above is a placeholder project name — swap in whatever you named
your `config/projects/<name>.yaml`.)

## Graph UI (LangGraph Studio)

An n8n-style visual view of the task state machine — nodes, edges, live
runs, and state inspection at every step:

```bash
pip install -e ".[studio]"          # langgraph-cli[inmem]; needs Python 3.11+
ORCH_PROJECT=example langgraph dev  # local server + Studio in the browser
```

`langgraph dev` serves the graph from `orchestrator/studio.py` (wired via
`langgraph.json`) with hot reload and opens LangSmith Studio connected to
your local server. The Studio UI needs a free LangSmith login; set
`LANGSMITH_TRACING=false` in `.env` if you want nothing to leave your
machine. Without `ORCH_PROJECT` the graph is view-only topology; with a
project profile you can invoke a single task straight from the UI against
the real store/repo.

## Key behaviors

**Best-of-N** (`run.n_candidates` or `--n`, or per-task by the planner): the
same task goes to N cheap models in parallel worktrees; the free gate culls
broken candidates; the Opus reviewer picks the winner among green diffs. One
Opus call buys N independent implementations.

**Retries**: gate failure sends the raw log back to the same worker
(`max_retries` cap). Review "revise" retries only the winner with the review
notes. Review "reject" means the *spec* is broken — the task goes back to
humans/planner, not to a worker loop.

**`need_files`**: workers can't read the repo, but they can *ask*
(`need_files_rounds` cap, size-capped). The orchestrator arbitrates and logs
every request — a signal that the planner under-specified the task.

**Limits & budgets**: every LLM call lands in the SQLite usage ledger.
CometAPI spend is checked against `budget.per_task_usd` / `per_run_usd`.
Claude CLI usage is subscription-covered (logged, not counted, by default).
When the CLI reports limit exhaustion the run **checkpoints and pauses**
(`run.on_limit_exhausted: pause`); `resume` continues mid-task from the
LangGraph SQLite checkpoint. Set `degrade` to reroute Opus roles to a cheap
model instead of pausing.

**Backlog sync**: the human markdown backlog stays the editorial source of
truth. `plan` imports/enriches into SQLite (machine truth: deps, files,
retries, cost); the finalizer writes back — flipping only the checkbox char
and appending one `**Agent:** ...` note line, never rewriting your text.

**Visual/subjective tasks**: the planner marks subjective acceptance
(`agent_able: false` → `human_only`) — workers can't see rendered output, UI,
or anything else that needs a human eye. For richer *review* of such tasks,
the `mcp:` block in a project profile lets you wire up an extra MCP server
for the reviewer/planner roles (e.g. a live app inspector or devtools
bridge) — see `config/projects/example.yaml` for the shape; it's off by
default and entirely optional.

## Adapting to another project

The core is a skeleton; a new project touches zero Python:

1. Copy `config/projects/example.yaml` → `config/projects/<name>.yaml` — repo
   path, branches, backlog file + item regex, gate commands
   (`cargo check && cargo test`, `pytest`, whatever), providers, worker
   models + prices, budgets. Keep machine-specific paths out of it by
   setting them in `config/local.yaml` instead (copy from
   `config/local.yaml.example`, already gitignored).
2. `config/prompts/worker_protocol.md` — your project's distilled coding
   protocol (or point `prompts.dir` at a per-project prompts folder).
3. Optional: extra MCP servers for reviewer/planner in `mcp:`.

Extension points that *are* code, kept deliberately small:
- new provider type → one class in `orchestrator/providers/` + one registry line;
- new backlog format → one adapter class in `orchestrator/ops/backlog.py`;
- extra graph stages (e.g. a pixel-diff visual gate) → one node + one edge in `orchestrator/engine/graph.py`.

## Layout

```
config/
  default.yaml            generic skeleton defaults (project-agnostic)
  projects/example.yaml   TEMPLATE profile — copy & rename per project
  local.yaml.example      TEMPLATE for gitignored personal/machine overrides
  prompts/*.md            planner / worker / reviewer prompts (also templates)
orchestrator/
  cli.py __main__.py      entrypoint
  studio.py               LangGraph Studio entrypoint (graph UI, langgraph.json)
  core/                   foundation
    config.py             layered config (default < project < local < env)
    state.py              checkpoint-serializable state + reducers
    context.py errors.py  runtime-services closure + control-flow exceptions
  engine/                 the state machine
    graph.py              LangGraph task graph (Send fan-out via conditional
                          edges — the 1.x-safe pattern)
    runner.py             outer loop: batches, pause/resume, dry-run
    scheduler.py          zero-token wave planning (deps + files_write)
  ops/                    deterministic ops — zero tokens
    store.py backlog.py   SQLite machine truth <-> markdown human truth
    patch.py gitops.py    SEARCH/REPLACE application, worktree isolation
    gate.py budget.py     project self-checks, usage ledger + caps
  providers/              claude_cli (subscription) / openai_compatible
  nodes/                  planner, worker, reviewer, integrator (LLM roles)
tests/                    pure-logic unit tests (patch, scheduler, backlog, graph)
state/                    (gitignored) SQLite store + checkpoints
```

## Design notes / known tradeoffs

- Send objects are emitted **only from conditional edges** — returning them
  from node bodies is unsupported in LangGraph 1.x and breaks checkpoint
  serialization ([issue #6789](https://github.com/langchain-ai/langgraph/issues/6789)).
- The gate lives *inside* the candidate node (worker→patch→gate is one graph
  step). Coarser checkpoints, but it keeps per-candidate identity out of
  global state and the graph honest under parallel fan-out.
- Worktrees need their own dependency install (`gate.install_cmd`, skipped
  when `install_marker` exists). With npm that's an `npm ci` per fresh
  worktree per task — the integrator deletes worktrees on finalize; raise
  throughput by switching the project to pnpm if it hurts.
- Limit detection parses CLI error text (`LIMIT_PATTERNS` in
  `providers/claude_cli.py`) — inherently fragile; patterns are one regex to
  extend when the CLI's wording changes.
- SQLite checkpointing is fine for this scale (a handful of parallel tasks);
  it is not a multi-machine setup.
