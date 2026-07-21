<!--
  TEMPLATE — per-project documentation skeleton.

  This is the ONE committed per-project doc. At project init it is copied into
  the gitignored projects/<name>/ overlay and filled in there; the filled copy
  is NEVER committed to this harness repo (only the system itself lives in git).
  Replace every placeholder below.
-->

# <PROJECT NAME> — agent-studio project notes

## What this project is
<one paragraph: the target repo, its language/stack, what the agents build>

## Smart tier
- `roles.smart_provider`: `claude_cli` | `codex_cli`  (<which, and why>)
- `run.escalate_model`: <opus/sonnet for claude_cli, a gpt-5.6-* id for codex_cli>

## Workers
- Default worker: <worker_models key, e.g. ds_v4_flash> — <model id + price>
- Best-of-N: <when the planner raises n_candidates>
- Domains (if any): <domain -> worker/protocol overrides>

## Gate
- install: <gate.install_cmd, e.g. npm ci / pip install -e .[dev]>
- commands: <the exact checks, e.g. npm run typecheck && npm run lint && npm test>

## Protocol excerpt
<pointer to projects/<name>/prompts/worker_protocol.md — the distilled coding
rules workers must follow (language, style, test framework, no new deps, ...)>

## Visual gate (if used)
- inspector: <threejs | unity | blender>
- run_cmd / ready_probe: <how the app is started + when it's ready>
- assertions: <the scene-graph checks that define "actually renders">

## Backlog
- source of truth: <BACKLOG.md path + item regex if non-default>

## Notes / gotchas
<anything an agent should know: untracked doc prefixes, monorepo quirks, etc.>
