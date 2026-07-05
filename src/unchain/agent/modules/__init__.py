from .base import AgentModule, BaseAgentModule
from .interaction import InteractionModule
from .memory import MemoryModule
from .optimizers import OptimizersModule
from .policies import PoliciesModule
from .subagents import SubagentModule
from .tool_discovery import ToolDiscoveryModule
from .tool_optimizer import ToolOptimizerModule
from .tools import ToolsModule

__all__ = [
    "AgentModule",
    "BaseAgentModule",
    "CharacterModule",
    "InteractionModule",
    "MemoryModule",
    "OptimizersModule",
    "PoliciesModule",
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
