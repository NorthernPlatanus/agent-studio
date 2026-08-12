"""Candidate pipeline: minimal-context worker LLM -> patch -> gate.

One invocation = one candidate attempt. The worker sees ONLY:
  - the task spec (planner-written, self-contained)
  - the distilled protocol excerpt
  - the exact files listed in files_read (+ arbitrated need_files extras)
  - retry feedback (gate log / review notes), if any

No filesystem, no tools, no repo-wide context. The gate is a deterministic
subprocess in the candidate's own git worktree — zero tokens.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from ..core.context import RunContext
from ..core.errors import BudgetExceeded, LimitExhausted, OrchestratorError, PatchError
from ..ops.assetops import run_asset_op
from ..ops.gate import ensure_deps, run_gate
from ..ops import retrieval
from ..ops.patch import apply_response, parse_worker_response
from ..providers import get_provider
from ..core.state import Candidate

log = logging.getLogger("orchestrator.worker")

# Appended to every retry-feedback turn. Observed failure: a worker answered a
# `revise` with 831 and then 1,241 output tokens of prose — two attempts, two
# `patch_failed: worker returned no <file>/<edit> blocks`, two fix rounds gone and
# an escalation forced, having attempted nothing. The feedback turn reads as a
# discussion prompt ("here are the review notes"), several thousand tokens after
# the output contract in the frozen prefix, so the model answers in kind. Restate
# the contract where the instruction is, and name the one legal alternative to a
# patch so a genuinely blocked worker has somewhere to go besides prose.
RETRY_CONTRACT = """

## OUTPUT CONTRACT — unchanged, and it applies to this turn
Reply with the patch: `<file>` and/or `<edit>` blocks that implement the fix
above. No prose, no questions, no summary of what you would do — a reply without
blocks changes nothing and is thrown away. If you truly cannot patch without
seeing more code, reply with `<grep>`/`<read>`/`<ls>` requests ONLY and you will
get the results back, then patch on the next turn."""


def _read_task_file(ctx: RunContext, worktree: Path, rel: str) -> str | None:
    """Code comes from the candidate worktree (branch state); paths under
    project.untracked_doc_prefixes come from the primary checkout, because
    the target repo may gitignore its own docs. Delegates the safety rule to
    ops/retrieval so reads and retrieval share one allowlist."""
    path = retrieval.resolve_read_path(
        ctx.cfg.repo_path(), worktree,
        ctx.cfg.project.untracked_doc_prefixes, rel)
    if path is None or not path.exists():
        return None
    try:
        return path.read_text()
    except (UnicodeDecodeError, OSError):
        return None


def run_retrieval_requests(ctx: RunContext, worktree: Path, parsed, *,
                           round_no: int, max_matches: int, snippet_lines: int,
                           max_bytes: int) -> str:
    """Execute a round of read-only <grep>/<read>/<ls> requests and format the
    results as a volatile-suffix block. Never writes; never runs worker shell."""
    repo = ctx.cfg.repo_path()
    prefixes = ctx.cfg.project.untracked_doc_prefixes
    out: list[str] = [f"## RETRIEVAL RESULTS (round {round_no})"]
    for pat in parsed.grep:
        matches, status = retrieval.grep(
            worktree, pat, max_matches=max_matches, snippet_lines=snippet_lines)
        out.append(f"### grep: {pat} [{status}]")
        out += matches or ["(no matches)"]
    for rel in parsed.need_files:
        path = retrieval.resolve_read_path(repo, worktree, prefixes, rel)
        text, status = retrieval.read_file(path, max_bytes)
        if status == "ok":
            out += [f'<source path="{rel}">', text, "</source>"]
        else:
            out.append(f"### read: {rel} [{status}]")
    for rel in parsed.ls:
        path = retrieval.resolve_read_path(repo, worktree, prefixes, rel)
        names = retrieval.ls_dir(path)
        out.append(f"### ls: {rel}")
        out += names if names is not None else ["(not a directory)"]
    return "\n".join(out)


def _build_stable_prompt(ctx: RunContext, spec: dict, files: dict[str, str],
                         approach: str | None = None) -> str:
    """The FROZEN first turn: task spec + protocol + files in a fixed order.

    This is the cache prefix. It is built exactly once per candidate (attempt 1)
    and never mutated — feedback and retrieval results arrive as later message
    turns (the volatile suffix), so retries bill at the cached input rate. The
    caller is responsible for passing `files` already sorted-once; do NOT fold
    later-granted files back into this block (that would insert bytes mid-prefix
    and break byte-stability).

    `approach` is the per-candidate best-of-N hint (constant for a candidate
    across retries, so its cache does not break; it differs across candidates,
    which is the point). Appended at the very end so the shared task+protocol+
    files prefix stays byte-identical across candidates on one model."""
    parts = [f"# TASK {spec['id']}: {spec['title']}", "", spec["description"], ""]
    if spec.get("acceptance"):
        parts.append("## Acceptance criteria")
        parts += [f"- {a}" for a in spec["acceptance"]]
        parts.append("")
    if spec.get("notes_for_worker"):
        parts += ["## Task notes", spec["notes_for_worker"], ""]
    parts += ["## Files you may write",
              *[f"- {p}" for p in spec.get("files_write", [])], ""]
    parts += ["# PROTOCOL", ctx.cfg.prompt("worker_protocol"), ""]
    # Optional per-domain protocol excerpt (resolved via the Phase 0 two-layer
    # resolver: project overlay wins over shared). Injected in addition to the
    # generic protocol; missing files degrade gracefully.
    domain = spec.get("domain")
    domains = ctx.cfg.get("domains") or {}
    dcfg = domains.get(domain) if domain else None
    proto_name = dcfg.get("protocol") if dcfg is not None else None
    if proto_name:
        try:
            parts += [f"# DOMAIN PROTOCOL ({domain})",
                      ctx.cfg.prompt(proto_name), ""]
        except (FileNotFoundError, OSError):
            pass
    parts.append("# FILES (your complete view of the repository)")
    for rel, content in files.items():
        parts.append(f'<source path="{rel}">')
        parts.append(content if content is not None else "<< FILE DOES NOT EXIST YET >>")
        parts.append("</source>")
    if approach:
        parts += [f"# APPROACH\n{approach}", ""]
    return "\n".join(parts)


async def run_candidate(ctx: RunContext, payload: dict) -> dict:
    """The Send-mapped node body. Payload: {run_id, task_id, spec, cand_id,
    attempt, feedback, messages}. Returns {"candidates": [Candidate]} for the
    reducer. `messages` is this candidate's warm chat history from prior attempts
    (per-candidate, plain dicts); it is continued, never rebuilt, so the stable
    prefix (first turn) stays byte-identical and hits the prompt cache."""
    spec = payload["spec"]
    task_id = payload["task_id"]
    cand_id = payload["cand_id"]
    attempt = payload["attempt"]
    feedback = payload.get("feedback", "")
    # Per-candidate warm history (plain role/content dicts). Copy so we never
    # mutate the prior attempt's list in place (earlier turns stay frozen).
    messages: list[dict] = [dict(m) for m in (payload.get("messages") or [])]

    provider_name, model = ctx.worker_target(cand_id)
    provider = get_provider(ctx.cfg, provider_name)
    wt_name = f"{task_id}-{cand_id}".lower()

    # Per-candidate best-of-N knobs: sampling `params` + an `approach` hint,
    # pulled from the worker_models entry. The `senior` escalation pseudo-
    # candidate is NOT a worker_models key (it resolves to the subscription smart
    # tier), so it has neither — both stay None (no crash; the CLI senior ignores
    # params, and no approach is appended to its prefix).
    entry = ctx.cfg.worker_models.get(cand_id) if cand_id != "senior" else None
    edict = entry.as_dict() if entry is not None else {}
    params = edict.get("params")
    approach = edict.get("approach")

    cand: Candidate = {"cand_id": cand_id, "model": model, "attempt": attempt,
                       "status": "llm_failed", "worktree": "", "branch": "",
                       "diff": "", "gate_log": "", "error": "", "notes": "",
                       "messages": messages}
    try:
        # Fail closed BEFORE spending a call. persist_specs rejects a spec with no
        # files_write at plan time, but specs planned before that check exist in
        # the store; without an allowlist apply_response would let this candidate
        # write anywhere in the worktree.
        if not spec.get("files_write"):
            raise PatchError(
                f"spec {task_id} declares no files_write — refusing to run a worker "
                f"with no write allowlist. Re-plan the task "
                f"(`plan --tasks {task_id}`) so it lists the files it may write.")
        # Worktree: fresh on attempt 1, reused (with prior commits) on retries.
        if attempt == 1:
            worktree = await ctx.git.acreate_worktree(wt_name)
        else:
            worktree = ctx.git.work_dir / wt_name
            if not worktree.exists():
                worktree = await ctx.git.acreate_worktree(wt_name)
        cand["worktree"] = str(worktree)
        cand["branch"] = ctx.git.wt_branch(wt_name)
        await asyncio.to_thread(ensure_deps, worktree, ctx.cfg)

        system = ctx.cfg.prompt(
            "worker_system",
            full_file_max_lines=ctx.cfg.worker_output.full_file_max_lines)

        if not messages:
            # First turn for this candidate: read + sort the initial files ONCE;
            # that sorted block is the frozen prefix for the whole warm chain.
            files: dict[str, str | None] = {}
            for rel in sorted(dict.fromkeys(
                    list(spec.get("files_read") or []) + list(spec.get("files_write") or []))):
                files[rel] = _read_task_file(ctx, worktree, rel)
            messages.append({"role": "user",
                             "content": _build_stable_prompt(ctx, spec, files,
                                                             approach=approach)})
        if feedback:
            # Append feedback as a NEW turn after the (possibly just-built) frozen
            # prefix — never inserted before it. This is `if`, not `elif`, so a
            # fresh chain that is nonetheless dispatched WITH feedback still sees
            # it: the escalation senior has empty prior `messages` but carries the
            # prior-failure log as feedback, and an `elif` here would silently drop
            # that context. On attempt 1 `feedback` is always "" (no-op); on
            # retries `messages` is non-empty and this appends as before.
            messages.append({
                "role": "user",
                "content": f"## RETRY FEEDBACK — fix exactly this\n{feedback}"
                           + RETRY_CONTRACT})

        run = ctx.cfg.run
        # retrieval_rounds supersedes the legacy need_files_rounds (kept as fallback).
        rounds_left = int(run.get("retrieval_rounds", run.get("need_files_rounds", 3)))
        max_matches = int(run.get("retrieval_max_matches", 40))
        snippet_lines = int(run.get("retrieval_snippet_lines", 4))
        max_bytes = int(run.get("need_files_max_bytes", 65536))
        forced_final = False
        round_no = 0
        # cwd matters for CLI providers (the escalated senior): root its read
        # tools at the candidate worktree, not the orchestrator repo.
        cwd = str(worktree)
        while True:
            # Pre-flight: refuse the call if its estimated cost would blow a cap,
            # rather than discovering that after paying for it (ops/budget).
            ctx.budget.estimate_and_check(
                task_id=task_id, provider=provider_name,
                provider_type=provider.type, model=model,
                prompt_chars=len(system) + sum(len(m["content"]) for m in messages),
                max_output_tokens=(params or {}).get("max_tokens"))
            result = await provider.complete_chat(
                model=model, system=system, messages=messages, cwd=cwd,
                params=params, effort=ctx.worker_effort(cand_id))
            messages.append({"role": "assistant", "content": result.text})
            ctx.budget.record(
                task_id=task_id, role=f"worker:{cand_id}", provider=provider_name,
                provider_type=provider.type, model=model,
                input_tokens=result.input_tokens, output_tokens=result.output_tokens,
                cost_usd=result.cost_usd, cache_hit_tokens=result.cache_hit_tokens,
                cache_miss_tokens=result.cache_miss_tokens)
            parsed = parse_worker_response(result.text)

            # Retrieval-only output (no patch) => execute read-only and loop.
            # Results append as a NEW user turn (volatile suffix), never before
            # the frozen prefix.
            if parsed.has_retrieval and not parsed.touched_paths:
                if rounds_left <= 0:
                    # Intentional behavior change from the old raise-based
                    # need_files: instead of hard-failing, force ONE final
                    # implement attempt and record a retrieval_exhausted signal
                    # (a high count still flags an under-specified planner spec).
                    if not forced_final:
                        forced_final = True
                        messages.append({"role": "user", "content":
                            "## RETRIEVAL EXHAUSTED\nNo retrieval rounds remain. "
                            "Implement the task now with the files you already have."})
                        ctx.store.log_event(
                            ctx.run_id, task_id, "retrieval_exhausted",
                            f"{cand_id} attempt={attempt} rounds={round_no}")
                        continue
                    break  # still no patch after the forced final -> guarded below
                rounds_left -= 1
                round_no += 1
                messages.append({"role": "user", "content": run_retrieval_requests(
                    ctx, worktree, parsed, round_no=round_no,
                    max_matches=max_matches, snippet_lines=snippet_lines,
                    max_bytes=max_bytes)})
                ctx.store.log_event(
                    ctx.run_id, task_id, "retrieval",
                    f"{cand_id} round={round_no} grep={len(parsed.grep)} "
                    f"read={len(parsed.need_files)} ls={len(parsed.ls)}")
                continue
            break

        # Require an actual patch. A retrieval-only response (even after the
        # forced final attempt) has no touched_paths and must never be applied.
        if not parsed.touched_paths:
            # Flag it BEFORE raising: the except block returns this same dict, and
            # `no_patch` is what tells the router nothing was attempted, so the fix
            # round is refunded rather than spent on prose (bounded by
            # run.max_unproductive_attempts — see graph.unproductive_attempts).
            cand["no_patch"] = True
            raise PatchError("worker returned no <file>/<edit> blocks")

        cand["notes"] = parsed.plan[:2000]
        written = await asyncio.to_thread(
            apply_response, worktree, parsed, spec.get("files_write"))
        await ctx.git.acommit_all(
            worktree,
            f"wip({task_id}): {cand_id} attempt {attempt}\n\nRefs {task_id}")
        log.info("%s/%s attempt %d: applied %s", task_id, cand_id, attempt, written)

        # Deterministic asset op (ops/assetops): a named, human-authored command
        # from the profile — same trust level as gate.commands, no LLM string in
        # it. Runs AFTER the patch commit (so the tool sees the worker's changes)
        # and BEFORE the gate (so the gate builds against the processed asset).
        # Its outputs are committed here, which is what puts them in
        # diff_against_feature and therefore in front of the reviewer and into
        # the merge. A failure is a red gate: same status, same retry branch.
        asset = await asyncio.to_thread(run_asset_op, worktree, ctx.cfg, spec)
        if asset.ran:
            # Its own kind when red (counted by `metrics`): a rising
            # asset_op_failed means the tool or its inputs are broken, which no
            # amount of worker retrying can fix — that has to be visible without
            # reading candidate logs.
            ctx.store.log_event(
                ctx.run_id, task_id,
                "asset_op" if asset.passed else "asset_op_failed",
                f"{cand_id} attempt={attempt} op={asset.op} passed={asset.passed}"
                + ("" if asset.passed else f"\n{(asset.log_tail or '')[-2000:]}"))
        if asset.ran and not asset.passed:
            cand["status"] = "gate_failed"
            cand["gate_log"] = (f"[asset_op {asset.op}: {asset.cmd}]\n"
                                f"{asset.log_tail}")
            log.warning("%s/%s attempt %d asset_op FAILED: %s\n%s",
                        task_id, cand_id, attempt, asset.op,
                        (asset.log_tail or "")[-1500:])
            return {"candidates": [cand]}
        if asset.ran:
            await ctx.git.acommit_all(
                worktree,
                f"asset({task_id}): {asset.op} for {cand_id} attempt {attempt}"
                f"\n\nRefs {task_id}")

        gate = await asyncio.to_thread(run_gate, worktree, ctx.cfg)
        if gate.passed:
            cand["status"] = "gate_passed"
            cand["diff"] = await asyncio.to_thread(ctx.git.diff_against_feature, worktree)
        else:
            cand["status"] = "gate_failed"
            cand["gate_log"] = f"[{gate.failed_step}]\n{gate.log_tail}"
            # Which step went red is the single most useful thing about a failed
            # attempt, and `gate_log` is in-memory graph state only — without
            # this the run log and the store both say `passed=False` and nothing
            # more, so a post-mortem can't tell a typecheck error from a
            # timed-out build without re-running the whole task.
            log.warning("%s/%s attempt %d gate FAILED at: %s\n%s",
                        task_id, cand_id, attempt, gate.failed_step,
                        (gate.log_tail or "")[-1500:])
        ctx.store.log_event(
            ctx.run_id, task_id, "gate",
            f"{cand_id} attempt={attempt} passed={gate.passed}"
            + ("" if gate.passed else
               f" failed_step={gate.failed_step}\n{(gate.log_tail or '')[-2000:]}"))

    except (LimitExhausted, BudgetExceeded):
        raise  # run-level control flow: checkpoint & pause
    except PatchError as e:
        cand["status"] = "patch_failed"
        cand["error"] = str(e)[:4000]
        _log_candidate_failure(ctx, task_id, cand_id, attempt, cand)
    except OrchestratorError as e:
        cand["status"] = "llm_failed"
        cand["error"] = str(e)[:4000]
        _log_candidate_failure(ctx, task_id, cand_id, attempt, cand)

    return {"candidates": [cand]}


def _log_candidate_failure(ctx, task_id: str, cand_id: str, attempt: int,
                           cand: dict) -> None:
    """Surface a candidate's failure reason to the log AND the event store.

    Without this the reason lives only in `cand["error"]`, which is in-memory
    graph state: a run whose every attempt died before the LLM was even called
    (`ensure_deps` raising on a broken `npm ci`, say) printed three
    `dispatch attempt N` lines, an `escalated`, and `finalized: failed` — and
    nothing, anywhere, saying why. Logged at WARNING (an attempt failing is not
    itself fatal — the graph may still retry or escalate) and mirrored into the
    store so `status`/`metrics` can show it after the process is gone.
    """
    reason = str(cand.get("error", ""))[:1000]
    log.warning("%s/%s attempt %d %s: %s",
                task_id, cand_id, attempt, cand["status"], reason)
    ctx.store.log_event(ctx.run_id, task_id, "candidate_failed",
                        f"{cand_id} attempt={attempt} {cand['status']}: {reason}")
    if cand.get("no_patch"):
        # Its own kind so `metrics` can count it: a rising no_patch count means the
        # cheap tier is answering instructions conversationally, which is a prompt
        # problem, not a capability one — and the refunded fix rounds hide it from
        # the retry/escalation numbers.
        ctx.store.log_event(ctx.run_id, task_id, "no_patch",
                            f"{cand_id} attempt={attempt} (no blocks; fix round refunded)")
