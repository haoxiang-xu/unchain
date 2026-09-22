from __future__ import annotations

from ..tools.models import ToolPromptSpec
from ..tools.tool import Tool
from .activation import ActivationQueue
from .models import is_valid_skill_name
from .registry import SkillRegistry
from .rendering import render_skill_loaded_envelope


def build_skill_tool(
    registry: SkillRegistry,
    queue: ActivationQueue,
    *,
    tool_name: str = "skill",
) -> Tool:
    """Model-facing activation tool: validates, loads and queues one skill.

    The result is an envelope only; the full instructions are projected into
    the `<active_skills>` system block by `SkillActivationHarness` before the
    next model call, so repeated loads never duplicate a body in context.
    """

    def skill(name: str) -> str:
        """Activate one skill from the <available_skills> catalog by its exact name.

        Args:
            name: Exact skill name as listed in <available_skills>.
        """

        if not is_valid_skill_name(name):
            return f'Error: invalid skill name "{name}"'
        summary = registry.resolve(name)
        if summary is None or summary.name != name:
            return f'Error: unknown skill "{name}"'
        if not summary.model_invocable:
            return f'Error: skill "{name}" is not available for model invocation'
        loaded = registry.get(name)
        if loaded is None:
            return f'Error: skill "{name}" could not be loaded'
        status = queue.push(loaded, activation="tool")
        return render_skill_loaded_envelope(name=name, revision=loaded.revision, status=status)

    return Tool.from_callable(
        skill,
        name=tool_name,
        description="Activate a skill from the <available_skills> catalog by its exact name.",
        always_load=True,
        prompt_spec=ToolPromptSpec(
            purpose="Load the complete instructions of a skill listed in <available_skills> before acting on it.",
            when_to_use=(
                "The user names a skill, or the task clearly matches a skill's description.",
                "You need the full procedure behind a catalog summary.",
            ),
            when_not_to_use=(
                "The skill already appears in the <active_skills> system block.",
                "No listed skill matches the task.",
            ),
            examples=(f'{tool_name}(name="plan")',),
            advanced_tips=(
                "Activate every applicable skill first; their instructions then appear in <active_skills>.",
            ),
        ),
    )


__all__ = ["build_skill_tool"]
