from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0     # provider-reported or price-table-derived
    model: str = ""
    raw: object = None


class LLMProvider(ABC):
    """One configured provider (an endpoint / a CLI), possibly many models."""

    type: str

    def __init__(self, name: str, pcfg, cfg):
        self.name = name
        self.pcfg = pcfg
        self.cfg = cfg

    @abstractmethod
    async def complete(self, *, model: str, system: str, user: str,
                       cwd: str | None = None) -> LLMResult:
        """Single-turn completion. `cwd` only matters for CLI providers with
        repo read access."""
