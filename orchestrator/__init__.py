"""Multi-agent LangGraph orchestrator skeleton.

Project-agnostic core: an expensive planner/reviewer (Claude Code
subscription via CLI), cheap workers (any OpenAI-compatible endpoint), a
deterministic gate, and git-worktree isolation. All project specifics live in
config/projects/<name>.yaml and config/prompts/.
"""

__version__ = "0.1.0"
