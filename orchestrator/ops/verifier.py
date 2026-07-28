"""Visual verdict — an agent inspects the candidate's RUNNING scene via MCP.

The deterministic gate proves "compiles, lints, tests pass". `visual_gate` proves
"the facts I thought to assert hold". Neither can answer *"is anything wrong here
that nobody thought to assert?"* — and for a rendering change that is the
question that matters. This phase answers it by starting the candidate's app and
letting a smart-tier agent drive a scene inspector over MCP.

Deliberate design points, each learned rather than assumed (all four verified
against `threejs-devtools-mcp@0.4.1` before this was written):

  * **Headless, no human.** The inspector launches its own browser via
    puppeteer-core when `HEADLESS=true`. Its README says "keep the browser tab
    open", which is true of the default interactive path and false of this one —
    an unattended run works.
  * **Ports are configurable**, via `BRIDGE_PORT` / `DEV_PORT` env, and the
    bridge falls back to a free port when its preferred one is taken. We still
    serialize (see `RunContext.inspector_lock`): the dev server is bound to one
    port by `run_cmd`, and two candidates would otherwise inspect each other's
    app.
  * **The MCP config is generated per call**, into the run's state dir, so the
    dev port travels in `env` rather than depending on the agent remembering to
    call `set_dev_port`.
  * **cwd is the CANDIDATE's worktree**, not `repo_path`. Pointing this at the
    primary checkout would grade one candidate's diff against another build's
    scene, which is worse than not checking at all.

Cost note: the inspector exposes ~60 tools, and their schemas are re-sent every
turn. A measured trivial two-tool query cost ~77k input tokens versus ~13k for a
plain call. That is why this runs ONCE per task on the winner rather than per
review round.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from ..core.errors import OrchestratorError
from .visualgate import FactSourceError, _spawn, _teardown, wait_for_probe

log = logging.getLogger("orchestrator.verifier")

VERDICT_RE = re.compile(r"\{.*\}", re.S)


def verify_enabled(cfg) -> bool:
    v = cfg.get("verify") or {}
    get = v.get if hasattr(v, "get") else (lambda k, d=None: d)
    return bool(get("enabled", False))


def verify_needed(cfg, spec: dict | None) -> bool:
    """A scene verdict is for specs that change what renders. A pure-logic task
    would pay the full browser + ~60-tool-schema cost to look at a scene its diff
    never touched, so `visual: true` gates this exactly as it gates visual_gate."""
    return verify_enabled(cfg) and bool((spec or {}).get("visual"))


def build_mcp_config(cfg, dest: Path) -> Path:
    """Write the inspector's MCP config, with the dev port baked into `env`.

    Generated rather than hand-maintained because the port is a runtime fact. The
    file lands in the state dir (not the repo) since it is derived, per-run data.
    """
    v = cfg.get("verify") or {}
    get = v.get if hasattr(v, "get") else (lambda k, d=None: d)
    env = {
        "HEADLESS": "true" if get("headless", True) else "false",
        "DEV_PORT": str(get("dev_port", 5199)),
        "BRIDGE_PORT": str(get("bridge_port", 9333)),
        # The injected on-page overlay is for humans watching a browser; in a
        # headless verify it is pure noise in every screenshot the agent takes.
        "THREEJS_DEVTOOLS_NO_OVERLAY": "1",
    }
    # mcp_env arrives as a Section when it came from YAML and a plain dict from a
    # test or a programmatic config; Section isn't subscriptable, so normalize.
    extra = get("mcp_env") or {}
    if hasattr(extra, "as_dict"):
        extra = extra.as_dict()
    env.update({str(k): str(vv) for k, vv in dict(extra).items()})
    payload = {"mcpServers": {str(get("mcp_server", "threejs")): {
        "command": str(get("mcp_command", "npx")),
        "args": list(get("mcp_args") or ["-y", "threejs-devtools-mcp"]),
        "env": env,
    }}}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return dest


def parse_verdict(text: str) -> dict:
    """Parse the agent's JSON verdict, failing CLOSED.

    An unparseable verdict is a failed verification, never a pass: the whole
    point of this phase is to catch what assertions miss, and "the verifier
    returned prose" is not evidence that the scene is right.
    """
    m = VERDICT_RE.search(text or "")
    if not m:
        return {"ok": False, "findings": ["verifier returned no JSON verdict: "
                                         + (text or "")[:500]]}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return {"ok": False, "findings": [f"verifier returned bad JSON ({e}): "
                                          + m.group(0)[:500]]}
    if not isinstance(data, dict) or "ok" not in data:
        return {"ok": False, "findings": ["verifier verdict has no `ok` field: "
                                          + m.group(0)[:500]]}
    findings = data.get("findings") or []
    if isinstance(findings, str):
        findings = [findings]
    return {"ok": bool(data["ok"]), "findings": [str(f) for f in findings][:20],
            "facts": data.get("facts") if isinstance(data.get("facts"), dict) else None}


def _fmt(template: str, port: int) -> str:
    """`{port}` is the only placeholder, so a stray brace in a shell command
    can't blow up with a KeyError."""
    return template.replace("{port}", str(port))


async def run_verification(ctx, spec: dict, cand: dict) -> dict:
    """Start the candidate's app, have the agent inspect it, return the verdict.

    Caller MUST hold `ctx.inspector_lock`. Always tears the app down, including
    on failure — a leaked dev server would hold the port and wedge every later
    verification in the run.
    """
    from ..providers import get_provider          # local: avoids an import cycle

    v = ctx.cfg.get("verify") or {}
    get = v.get if hasattr(v, "get") else (lambda k, d=None: d)
    port = int(get("dev_port", 5199))
    worktree = Path(cand["worktree"])
    run_cmd = get("run_cmd")
    if not run_cmd:
        raise OrchestratorError(
            "verify.enabled is true but verify.run_cmd is unset — there would be "
            "no running app to inspect. Set it, or disable verify.")

    mcp_path = build_mcp_config(
        ctx.cfg, Path(ctx.cfg.state_dir()) / "verify" / f"{ctx.run_id}-mcp.json")

    proc = None
    try:
        proc = _spawn(_fmt(run_cmd, port), worktree)
        probe = get("ready_probe")
        if probe:
            wait_for_probe(_fmt(probe, port), int(get("ready_timeout_s", 60)))

        provider_name, model = ctx.role_target("verifier")
        provider = get_provider(ctx.cfg, provider_name)
        system = ctx.cfg.prompt("verifier")
        result = await provider.complete(
            model=model, system=system,
            user=build_verify_prompt(spec, cand),
            cwd=str(worktree),                    # the candidate's code, not main
            effort=ctx.role_effort("verifier"),
            allowed_tools=ctx.role_allowed_tools("verifier"),
            mcp_config=str(mcp_path))
        ctx.budget.record(
            task_id=spec["id"], role="verifier", provider=provider_name,
            provider_type=provider.type, model=model,
            input_tokens=result.input_tokens, output_tokens=result.output_tokens,
            cost_usd=result.cost_usd, cache_hit_tokens=result.cache_hit_tokens,
            cache_miss_tokens=result.cache_miss_tokens)
        return parse_verdict(result.text)
    except FactSourceError as e:
        # The app never came up. That is a failed verification, not a pass.
        return {"ok": False, "findings": [f"app did not become ready: {e}"]}
    finally:
        _teardown(proc)


def build_verify_prompt(spec: dict, cand: dict) -> str:
    parts = [f"# TASK {spec['id']}: {spec['title']}", "", spec["description"], ""]
    if spec.get("acceptance"):
        parts += ["## Acceptance criteria", *[f"- {a}" for a in spec["acceptance"]], ""]
    parts += ["## The change under verification", "```diff", cand.get("diff", ""),
              "```", ""]
    facts = cand.get("visual_facts")
    if facts:
        parts += ["## Facts already measured by the deterministic visual gate",
                  "```json", json.dumps(facts, indent=2, sort_keys=True, default=str),
                  "```", ""]
    return "\n".join(parts)
