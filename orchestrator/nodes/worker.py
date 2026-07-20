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
from ..ops.gate import ensure_deps, run_gate
from ..ops.patch import apply_response, parse_worker_response
from ..providers import get_provider
from ..core.state import Candidate

log = logging.getLogger("orchestrator.worker")


def _read_task_file(ctx: RunContext, worktree: Path, rel: str) -> str | None:
    """Code comes from the candidate worktree (branch state); paths under
    project.untracked_doc_prefixes come from the primary checkout, because
    the target repo may gitignore its own docs."""
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        # Workers can request extra files (need_files) — never let a request
        # escape the repo/worktree. Treat as missing -> refused.
        return None
    prefixes = ctx.cfg.project.untracked_doc_prefixes or []
    root = ctx.cfg.repo_path() if any(rel.startswith(p) for p in prefixes) else worktree
    path = root / rel
    if not path.exists():
        return None
    try:
        return path.read_text()
    except (UnicodeDecodeError, OSError):
        return None


def _build_user_prompt(ctx: RunContext, spec: dict, files: dict[str, str],
                       feedback: str) -> str:
    parts = [f"# TASK {spec['id']}: {spec['title']}", "", spec["description"], ""]
    if spec.get("acceptance"):
        parts.append("## Acceptance criteria")
        parts += [f"- {a}" for a in spec["acceptance"]]
        parts.append("")
    if spec.get("notes_for_worker"):
        parts += ["## Task notes", spec["notes_for_worker"], ""]
    parts += ["## Files you may write",
              *[f"- {p}" for p in spec.get("files_write", [])], ""]
    if feedback:
        parts += ["## RETRY FEEDBACK — fix exactly this",
                  "```", feedback, "```", ""]
    parts += ["# PROTOCOL", ctx.cfg.prompt("worker_protocol"), ""]
    parts.append("# FILES (your complete view of the repository)")
    for rel, content in files.items():
        parts.append(f'<source path="{rel}">')
        parts.append(content if content is not None else "<< FILE DOES NOT EXIST YET >>")
        parts.append("</source>")
    return "\n".join(parts)


async def run_candidate(ctx: RunContext, payload: dict) -> dict:
    """The Send-mapped node body. Payload: {run_id, task_id, spec, cand_id,
    attempt, feedback}. Returns {"candidates": [Candidate]} for the reducer."""
    spec = payload["spec"]
    task_id = payload["task_id"]
    cand_id = payload["cand_id"]
    attempt = payload["attempt"]
    feedback = payload.get("feedback", "")

    provider_name, model = ctx.worker_target(cand_id)
    provider = get_provider(ctx.cfg, provider_name)
    wt_name = f"{task_id}-{cand_id}".lower()

    cand: Candidate = {"cand_id": cand_id, "model": model, "attempt": attempt,
                       "status": "llm_failed", "worktree": "", "branch": "",
                       "diff": "", "gate_log": "", "error": "", "notes": ""}
    try:
        # Worktree: fresh on attempt 1, reused (with prior commits) on retries.
        if attempt == 1:
            worktree = await asyncio.to_thread(ctx.git.create_worktree, wt_name)
        else:
            worktree = ctx.git.work_dir / wt_name
            if not worktree.exists():
                worktree = await asyncio.to_thread(ctx.git.create_worktree, wt_name)
        cand["worktree"] = str(worktree)
        cand["branch"] = ctx.git.wt_branch(wt_name)
        await asyncio.to_thread(ensure_deps, worktree, ctx.cfg)

        files: dict[str, str | None] = {}
        for rel in dict.fromkeys(
                list(spec.get("files_read") or []) + list(spec.get("files_write") or [])):
            files[rel] = _read_task_file(ctx, worktree, rel)

        system = ctx.cfg.prompt(
            "worker_system",
            full_file_max_lines=ctx.cfg.worker_output.full_file_max_lines)

        rounds_left = int(ctx.cfg.run.need_files_rounds)
        max_extra = int(ctx.cfg.run.need_files_max_bytes)
        while True:
            user = _build_user_prompt(ctx, spec, files, feedback)
            result = await provider.complete(model=model, system=system, user=user)
            ctx.budget.record(
                task_id=task_id, role=f"worker:{cand_id}", provider=provider_name,
                provider_type=provider.type, model=model,
                input_tokens=result.input_tokens, output_tokens=result.output_tokens,
                cost_usd=result.cost_usd)
            parsed = parse_worker_response(result.text)

            if parsed.need_files and not parsed.touched_paths:
                if rounds_left <= 0:
                    raise PatchError("need_files budget exhausted; task spec is "
                                     "likely missing context (planner problem)")
                rounds_left -= 1
                granted, refused = [], []
                for rel in parsed.need_files:
                    content = _read_task_file(ctx, worktree, rel)
                    if content is not None and len(content) <= max_extra:
                        files[rel] = content
                        granted.append(rel)
                    else:
                        refused.append(rel)
                ctx.store.log_event(ctx.run_id, task_id, "need_files",
                                    f"{cand_id} granted={granted} refused={refused}")
                if refused:
                    feedback = (feedback + f"\nRefused files (missing or too large): "
                                f"{refused}. Proceed without them.").strip()
                continue
            break

        if parsed.is_empty:
            raise PatchError("worker returned no <file>/<edit> blocks")

        cand["notes"] = parsed.plan[:2000]
        written = await asyncio.to_thread(
            apply_response, worktree, parsed, spec.get("files_write"))
        await asyncio.to_thread(
            ctx.git.commit_all, worktree,
            f"wip({task_id}): {cand_id} attempt {attempt}\n\nRefs {task_id}")
        log.info("%s/%s attempt %d: applied %s", task_id, cand_id, attempt, written)

        gate = await asyncio.to_thread(run_gate, worktree, ctx.cfg)
        if gate.passed:
            cand["status"] = "gate_passed"
            cand["diff"] = await asyncio.to_thread(ctx.git.diff_against_feature, worktree)
        else:
            cand["status"] = "gate_failed"
            cand["gate_log"] = f"[{gate.failed_step}]\n{gate.log_tail}"
        ctx.store.log_event(ctx.run_id, task_id, "gate",
                            f"{cand_id} attempt={attempt} passed={gate.passed}")

    except (LimitExhausted, BudgetExceeded):
        raise  # run-level control flow: checkpoint & pause
    except PatchError as e:
        cand["status"] = "patch_failed"
        cand["error"] = str(e)[:4000]
    except OrchestratorError as e:
        cand["status"] = "llm_failed"
        cand["error"] = str(e)[:4000]

    return {"candidates": [cand]}
