from __future__ import annotations

import copy
from dataclasses import dataclass, field

from ..agent.modules.base import BaseAgentModule  # direct import, avoids agent/__init__.py
from .spec import CharacterSpec
from .harness import CharacterNarrativeHarness
from .tools import build_character_tools


@dataclass(frozen=True)
class CharacterModule(BaseAgentModule):
    """Agent module that injects character identity, narrative, and state management.

    Usage:
        spec = CharacterSpec.from_json(json.load(open("alice.json")))
        agent = Agent(
            name="alice",
            modules=(
                CharacterModule(spec=spec),
                MemoryModule(memory=memory_manager),  # recommended for state persistence
                ToolsModule(tools=(...,)),
            ),
        )
    """
    spec: CharacterSpec = field(default=None, repr=False)
    name: str = "character"

    def configure(self, builder) -> None:
        # Register the narrative harness (handles bootstrap, before_model, before_commit)
        builder.add_harness(CharacterNarrativeHarness(spec=self.spec))

        # Register character-specific tools
        for tool in build_character_tools():
            builder.add_tool(tool)

        # Apply payload defaults from spec (e.g., preferred temperature)
        if self.spec.payload_defaults:
            builder.set_payload_defaults(copy.deepcopy(dict(self.spec.payload_defaults)))
