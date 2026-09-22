from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...skills.activation import ActivationQueue
from ...skills.harness import SkillActivationHarness, SkillCatalogHarness
from ...skills.registry import SkillRegistry, SkillsConfig
from ...skills.tools import build_skill_tool
from .base import BaseAgentModule


@dataclass(frozen=True)
class SkillsModule(BaseAgentModule):
    """Discovers SKILL.md skills, renders their catalog, exposes the `skill` tool
    and projects activations into the durable `<active_skills>` block.

    The `skill` tool is registered only when the inventory has at least one
    model-invocable skill at configure time (decided once per run); the catalog
    block itself is rendered per step from the live inventory.
    """

    config: SkillsConfig | dict[str, Any] | None = None
    name: str = field(default="skills", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", SkillsConfig.coerce(self.config))

    def configure(self, builder) -> None:
        registry = SkillRegistry(self.config, runtime_toolkit=builder.toolkit)
        queue = ActivationQueue()
        inventory = registry.list()
        if any(skill.model_invocable for skill in inventory.skills):
            builder.add_tool(build_skill_tool(registry, queue, tool_name=self.config.tool_name))
        builder.add_harness(
            SkillCatalogHarness(
                registry=registry,
                tool_name=self.config.tool_name,
                description_max_length=self.config.catalog_description_max_length,
            )
        )
        # Resolve at execution time: module configuration order must not decide
        # whether the active Context V2 owner persists the skill snapshot.
        def journal_binding(context):
            runtime = getattr(builder, "context_runtime", None)
            return runtime.skill_activation_journal(context) if runtime is not None else None

        builder.add_harness(SkillActivationHarness(
            registry=registry, queue=queue, journal_binding=journal_binding,
        ))


__all__ = ["SkillsModule"]
