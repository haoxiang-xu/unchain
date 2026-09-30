from .base import AgentModule, BaseAgentModule
from .context import ContextModule, ContextShadowModule
from .context_composition import (
    ContextCompositionBootstrapModule,
    ContextCompositionBootstrapModuleError,
)
from .durability import DurabilityModule
from .interaction import InteractionModule
from .jobs import JobsModule
from .memory import MemoryModule
from .optimizers import OptimizersModule
from .policies import PoliciesModule
from .skills import SkillsModule
from .subagents import SubagentModule
from .tool_discovery import ToolDiscoveryModule
from .tool_optimizer import ToolOptimizerModule
from .tools import ToolsModule

__all__ = [
    "AgentModule",
    "BaseAgentModule",
    "CharacterModule",
    "ContextModule",
    "ContextCompositionBootstrapModule",
    "ContextCompositionBootstrapModuleError",
    "ContextShadowModule",
    "DurabilityModule",
    "InteractionModule",
    "JobsModule",
    "MemoryModule",
    "OptimizersModule",
    "PoliciesModule",
    "SkillsModule",
    "SubagentModule",
    "ToolDiscoveryModule",
    "ToolOptimizerModule",
    "ToolsModule",
]


def __getattr__(name):
    if name == "CharacterModule":
        from ...character.module import CharacterModule
        return CharacterModule
    raise AttributeError(name)
