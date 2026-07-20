# Project protocol excerpt — TEMPLATE

This file is copied into every worker's prompt as-is. It is the *only* place
project-specific coding rules live for the worker/reviewer roles — replace
every bullet below with the real conventions of the project you're
orchestrating (distilled from your own CONTRIBUTING.md / ADRs / style guide).
Keep it short: it costs prompt tokens on every single task.

- Language & framework: [e.g. "TypeScript, strict mode" / "Python 3.11,
  fully type-hinted" / "Go 1.22"].
- Architecture / folder conventions specific to this codebase: [e.g. layering
  rules, where business logic vs. I/O may live, module boundaries].
- Performance or safety constraints that apply to hot paths, if any: [e.g.
  "no allocations in the render loop", "no blocking calls in async handlers"].
- Tests: [test framework]. [What must have tests — e.g. "all pure logic",
  "bug fix implies a regression test"].
- Lint/format: [tools], config lives at [path]; match surrounding style.
- Dependency policy: [e.g. "no new dependencies without approval"].
- Where settled decisions live (ADRs/RFCs) and how workers should treat them
  if a task spec cites one.
