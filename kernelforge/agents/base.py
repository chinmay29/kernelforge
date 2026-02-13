"""Common interfaces for KernelForge multi-agent workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentResult:
    """Standard result payload for non-throwing agent execution."""

    ok: bool
    error: str = ""
    data: dict[str, Any] = field(default_factory=dict)


class Agent:
    """Base class for all coordinator-managed agents."""

    name: str = "agent"

    def run(self, inputs: dict[str, Any]) -> AgentResult:
        raise NotImplementedError
