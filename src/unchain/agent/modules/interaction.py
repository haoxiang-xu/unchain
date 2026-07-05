from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...interaction.fyi import FyiChannel, FyiInjectionHarness
from .base import BaseAgentModule

if TYPE_CHECKING:
    from ..builder import AgentBuilder


@dataclass(frozen=True)
class InteractionModule(BaseAgentModule):
    fyi_channel: FyiChannel | None = None
    name: str = "interaction"

    def configure(self, builder: "AgentBuilder") -> None:
        if self.fyi_channel is not None:
            builder.add_harness(FyiInjectionHarness(channel=self.fyi_channel))
