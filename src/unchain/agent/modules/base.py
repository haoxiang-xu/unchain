from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..builder import AgentBuilder


class AgentModule(Protocol):
    name: str

    def configure(self, builder: "AgentBuilder") -> None:
        ...


@dataclass(frozen=True)
class BaseAgentModule:
    name: str

    def configure(self, builder: "AgentBuilder") -> None:
        """Leave the builder unchanged unless a module overrides this hook."""
